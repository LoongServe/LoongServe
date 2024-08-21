import uuid
import asyncio
import math
import numpy as np
from typing import List
from ..io_struct import Batch, Req
from .profiler import Profiler
from loongserve.utils.infer_utils import calculate_time
from loongserve.longserve_server.io_struct import Req
from loongserve.longserve_server.io_struct import ReqRunStatus, FinishStatus

class ReqQueue:

    def __init__(self, args) -> None:
        self.max_total_tokens = args.max_total_token_num
        assert args.batch_max_tokens is not None
        self.batch_max_tokens = args.batch_max_tokens
        self.running_max_req_size = args.running_max_req_size
        self.waiting_req_list: List[Req] = []
        self.router_token_ratio = args.router_token_ratio
        self.router_max_new_token_len = args.router_max_new_token_len
        self.sp_world_size = args.sp_world_size
        self.pause_req_dict: dict[int, Req] = {}
        self.pause_req_used_tokens_list = np.zeros((self.sp_world_size,), dtype=np.int32)

        self.avg_decoding_time = args.avg_decoding_time

        self.max_num_ooe = args.max_num_ooe
        self.num_ooe = 0

        self.max_prefill_time = args.max_prefill_time
        self.disable_scale_up = args.disable_scale_up
        
    def append(self, req):
        self.waiting_req_list.append(req)
        return
    
    def back_to_wait_list(self, req_list:List[Req]):
        for req in req_list:
            if req.req_status in [ReqRunStatus.PAUSED_AND_KVKEEP, ReqRunStatus.PAUSED_AND_OFFLOAD]:
                self.pause_req_dict[req.request_id] = req
        self.waiting_req_list = req_list + self.waiting_req_list
        self.recalcu_pause_req_used_tokens_list()
        return 

    def _init_cache_list(self, current_batch_list:List[Batch], is_busy):
        self.cache_pause_reqs_used_tokens_list = self.pause_req_used_tokens_list
        self.cache_pause_reqs_num = len(self.pause_req_dict)
        if len(current_batch_list) > 0:
            self.cache_len_list = [req.get_tuple_tokens(is_busy, self.router_max_new_token_len) for batch in current_batch_list for req in batch.reqs]
        else:
            self.cache_len_list = []

    def _can_add_new_req(self, req:Req, is_busy: bool):
        self.cache_len_list.append(req.get_tuple_tokens(is_busy, self.router_max_new_token_len)) # hard to analysis
        new_cache_len_list = sorted(self.cache_len_list, key=lambda x: -x[1])
        self.cache_len_list.pop()
        
        left_out_len_array = np.array([e[1] for e in new_cache_len_list])
        has_run_len_array = np.array([e[0] for e in new_cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(new_cache_len_list) + 1, 1)
        
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        if req.req_status in [ReqRunStatus.PAUSED_AND_KVKEEP, ReqRunStatus.PAUSED_AND_OFFLOAD]:
            pause_reqs_used_tokens_num_delta = req.get_used_tokens_list().sum()
            pause_reqs_num_delta = 1
        else:
            pause_reqs_used_tokens_num_delta = 0
            pause_reqs_num_delta = 0

        ok_token_num = need_max_token_num <= self.max_total_tokens * self.sp_world_size - (self.cache_pause_reqs_used_tokens_list.sum() - pause_reqs_used_tokens_num_delta)
        ok_req_num = len(new_cache_len_list) + (self.cache_pause_reqs_num - pause_reqs_num_delta) <= self.running_max_req_size

        if ok_token_num and ok_req_num:
            return True
        else:
            return False
    
    def generate_new_req_list(self, current_batch_list:List[Batch], sum_finished_req_output_len: float, num_finished_reqs: float, profiler: Profiler):
        if len(self.waiting_req_list) == 0:
            return []

        exist_req_num = sum((len(batch.reqs) for batch in current_batch_list))
        exist_req_num += len(self.pause_req_dict)
        req_is_full = exist_req_num >= self.running_max_req_size
        if req_is_full:
            return []
        
        estimated_max_iterations = 10
        avg_num_iterations = sum_finished_req_output_len / num_finished_reqs
        inverted_preempted_decode_token_sum = 0
        cur_all_used_tokens_list = self.recalcu_pause_req_used_tokens_list().copy()
        for batch in current_batch_list:
            cur_all_used_tokens_list += batch.batch_used_tokens_list

            for req in batch.reqs:
                inverted_preempted_decode_token_sum += 1.0 / len(req.output_ids)
                estimated_max_iterations = max(estimated_max_iterations, avg_num_iterations - len(req.output_ids), 0.1 * len(req.output_ids))
        estimated_waiting_time = estimated_max_iterations * self.avg_decoding_time
        
        num_used_tokens = cur_all_used_tokens_list.sum()
        num_idle_slots = self.sp_world_size * self.max_total_tokens - num_used_tokens
        min_num_used_instances = (num_used_tokens + self.max_total_tokens - 1) // self.max_total_tokens
        max_num_idle_instances = self.sp_world_size - min_num_used_instances
        num_isolated_idle_tokens = self.max_total_tokens * max_num_idle_instances
        batch_max_tokens = min(self.batch_max_tokens, num_idle_slots)
        
        available_instances = np.nonzero(cur_all_used_tokens_list == 0)[0].tolist()
        num_idle_instances = len(available_instances)

        cur_token_ratio_list = cur_all_used_tokens_list / self.max_total_tokens
        is_busy = np.all(cur_token_ratio_list >= self.router_token_ratio)
        
        self._init_cache_list(current_batch_list, is_busy)
        can_run_list = []
        new_waiting_req_list = []
        req_prefill_sum = 0
        req_prefill_square_sum = 0
        is_ooe = False
        before_ooe_len = None
        allow_ooe = self.num_ooe < self.max_num_ooe
        
        undecided_req_list = []
        undecided_req_prefill_sum = 0
        undecided_req_prefill_square_sum = 0
        inverted_undecided_req_prefill_sum = 0
        undecided_cache_len_list = []
        
        can_append_undecided_req_list = True
        can_append_run_idx = 0

        need_break = False

        for i, req in enumerate(self.waiting_req_list):
            if req.finish_status.is_aborted() and req.req_status == ReqRunStatus.WAIT_IN_QUEUE: 
                continue

            req_first_router_need_tokens = req.get_first_router_need_tokens()
            new_req_prefill_sum = req_prefill_sum + req_first_router_need_tokens

            if self._can_add_new_req(req, is_busy)\
                and new_req_prefill_sum <= batch_max_tokens \
                and (not self.disable_scale_up or new_req_prefill_sum <= num_idle_instances * self.max_total_tokens):

                is_empty_undecided_req_list = len(undecided_req_list) == 0
                if (is_empty_undecided_req_list or allow_ooe) and new_req_prefill_sum <= num_isolated_idle_tokens:
                    new_req_prefill_square_sum = req_prefill_square_sum + req_first_router_need_tokens**2

                    cur_prefill_iteration_time = profiler.predict(max_num_idle_instances, new_req_prefill_sum, new_req_prefill_square_sum)
                    best_prefill_iteration_time = profiler.predict(self.sp_world_size, new_req_prefill_sum, new_req_prefill_square_sum)
                    potential_slowdown = cur_prefill_iteration_time - best_prefill_iteration_time

                    satisfy_prefill_time_limit = cur_prefill_iteration_time <= self.max_prefill_time

                    if potential_slowdown <= estimated_waiting_time\
                        and (len(can_run_list) == 0 or satisfy_prefill_time_limit):

                        if not satisfy_prefill_time_limit:
                            need_break = True

                        if not is_empty_undecided_req_list and not is_ooe:
                            before_ooe_len = len(can_run_list)
                            is_ooe = True

                        can_run_list.append(req)

                        self.cache_len_list.append(req.get_tuple_tokens(is_busy, self.router_max_new_token_len)) # hard to analysis
                        req_prefill_sum = new_req_prefill_sum
                        req_prefill_square_sum = new_req_prefill_square_sum

                        if req.req_status in [ReqRunStatus.PAUSED_AND_KVKEEP, ReqRunStatus.PAUSED_AND_OFFLOAD]:
                            self.cache_pause_reqs_num -= 1
                            self.pause_req_dict.pop(req.request_id)
                        
                        continue
                
                self.cache_len_list += undecided_cache_len_list
                can_add_new_req = self._can_add_new_req(req, is_busy)
                self.cache_len_list = self.cache_len_list[:-len(undecided_cache_len_list)]
                if can_append_undecided_req_list and can_add_new_req and new_req_prefill_sum + undecided_req_prefill_sum <= batch_max_tokens:
                    undecided_req_list.append(req)
                    undecided_req_prefill_sum += req_first_router_need_tokens
                    undecided_req_prefill_square_sum += req_first_router_need_tokens ** 2
                    inverted_undecided_req_prefill_sum += 1.0 / req_first_router_need_tokens

                    preempted_iteration_time = profiler.predict(
                        self.sp_world_size,
                        undecided_req_prefill_sum + req_prefill_sum,
                        undecided_req_prefill_square_sum + req_prefill_square_sum
                    )

                    satisfy_prefill_time_limit = preempted_iteration_time <= self.max_prefill_time
                    if preempted_iteration_time * inverted_preempted_decode_token_sum <= estimated_waiting_time * inverted_undecided_req_prefill_sum\
                        and (len(can_run_list) + len(undecided_req_list) == 1 or satisfy_prefill_time_limit):

                        if not satisfy_prefill_time_limit:
                            need_break = True
                        
                        can_append_run_idx = len(undecided_req_list)

                    continue
                else:
                    can_append_undecided_req_list = False
            
            if allow_ooe and not need_break:
                if not is_ooe:
                    before_ooe_len = len(can_run_list)
                    is_ooe = True
                new_waiting_req_list.append(req)
            else:
                new_waiting_req_list += self.waiting_req_list[i:]
                break
        
        can_run_list += undecided_req_list[:can_append_run_idx]
        new_waiting_req_list = undecided_req_list[can_append_run_idx:] + new_waiting_req_list

        if len(can_run_list) != 0:
            if is_ooe and before_ooe_len < len(can_run_list):
                self.num_ooe += 1
            else:
                self.num_ooe = 0
            self.waiting_req_list = new_waiting_req_list
            self.recalcu_pause_req_used_tokens_list()
            return can_run_list
        else:
            return []
    
    def _can_add_greedy_req(self, req:Req, is_busy, num_instances: int):
        self.cache_len_list.append(req.get_tuple_tokens(is_busy, self.router_max_new_token_len)) # hard to analysis
        new_cache_len_list = sorted(self.cache_len_list, key=lambda x: -x[1])
        self.cache_len_list.pop()
        
        left_out_len_array = np.array([e[1] for e in new_cache_len_list])
        has_run_len_array = np.array([e[0] for e in new_cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(new_cache_len_list) + 1, 1)
        
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        if req.req_status in [ReqRunStatus.PAUSED_AND_KVKEEP, ReqRunStatus.PAUSED_AND_OFFLOAD]:
            pause_reqs_used_tokens_num_delta = req.get_used_tokens_list().sum()
            pause_reqs_num_delta = 1
        else:
            pause_reqs_used_tokens_num_delta = 0
            pause_reqs_num_delta = 0

        ok_token_num = need_max_token_num <= self.max_total_tokens * num_instances - (self.cache_pause_reqs_used_tokens_list.sum() - pause_reqs_used_tokens_num_delta)
        ok_req_num = len(new_cache_len_list) + (self.cache_pause_reqs_num - pause_reqs_num_delta) <= self.running_max_req_size

        if ok_token_num and ok_req_num:
            return True
        else:
            return False
    
    def generate_greedy_req_list(self, current_batch_list:List[Batch], available_instances: List[int], iteration_left_time: float, profiler: Profiler):
        is_busy = True
        num_instances = len(available_instances)
        
        self._init_cache_list(current_batch_list, is_busy)
        can_run_list = []
        new_waiting_req_list = []
        req_prefill_sum = 0
        req_prefill_square_sum = 0
        for req in self.waiting_req_list:
            if req.finish_status.is_aborted() and req.req_status == ReqRunStatus.WAIT_IN_QUEUE: 
                continue

            req_first_router_need_tokens = req.get_first_router_need_tokens()
            new_req_prefill_sum = req_prefill_sum + req_first_router_need_tokens
            new_req_prefill_square_sum = req_prefill_square_sum + req_first_router_need_tokens**2
            iteration_time = profiler.predict(num_instances, new_req_prefill_sum, new_req_prefill_square_sum)

            if self._can_add_greedy_req(req, is_busy, num_instances)\
                and new_req_prefill_sum <= self.batch_max_tokens\
                and iteration_time <= iteration_left_time:
                can_run_list.append(req)

                self.cache_len_list.append(req.get_tuple_tokens(is_busy, self.router_max_new_token_len)) # hard to analysis
                req_prefill_sum = new_req_prefill_sum
                req_prefill_square_sum = new_req_prefill_square_sum
                
                if req.req_status in [ReqRunStatus.PAUSED_AND_KVKEEP, ReqRunStatus.PAUSED_AND_OFFLOAD]:
                    self.cache_pause_reqs_used_tokens_list -= req.get_used_tokens_list()
                    self.cache_pause_reqs_num -= 1
                    self.pause_req_dict.pop(req.request_id)
            else:
                new_waiting_req_list.append(req)

        if len(can_run_list) != 0:
            self.waiting_req_list = new_waiting_req_list
            self.recalcu_pause_req_used_tokens_list()
            return can_run_list
        else:
            return []

    def _can_add_new_req_fixed_sp(self, req:Req, is_busy):
        self.cache_len_list.append(req.get_tuple_tokens(is_busy, self.router_max_new_token_len)) # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])
        
        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
        
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        if req.req_status in [ReqRunStatus.PAUSED_AND_KVKEEP, ReqRunStatus.PAUSED_AND_OFFLOAD]:
            self.cache_pause_reqs_used_tokens_list -= req.get_used_tokens_list()
            self.cache_pause_reqs_num -= 1

        ok_token_num = need_max_token_num <= self.max_total_tokens * self.sp_world_size - self.cache_pause_reqs_used_tokens_list.sum()
        ok_req_num = len(self.cache_len_list) + self.cache_pause_reqs_num <= self.running_max_req_size

        if ok_token_num and ok_req_num:
            return True
        else:
            return False

    def generate_new_req_list_fixed_sp(self, current_batch_list:List[Batch]):
        exist_req_num = sum((len(batch.reqs) for batch in current_batch_list))
        exist_req_num += len(self.pause_req_dict)
        req_is_full = exist_req_num >= self.running_max_req_size
        if req_is_full:
            return []
        
        cur_all_used_tokens_list = self.recalcu_pause_req_used_tokens_list().copy()
        for batch in current_batch_list:
            cur_all_used_tokens_list += batch.batch_used_tokens_list
        
        
        cur_token_ratio_list = cur_all_used_tokens_list / self.max_total_tokens
        is_busy = np.any(cur_token_ratio_list >= self.router_token_ratio)
        
        self._init_cache_list(current_batch_list, is_busy)
        can_run_list = []
        new_batch_first_router_need_tokens = 0
        per_instance_need_tokens = 0
        aborted_count = 0
        for req in self.waiting_req_list:
            if req.finish_status.is_aborted() and req.req_status == ReqRunStatus.WAIT_IN_QUEUE: 
                aborted_count += 1
                continue

            req_first_router_need_tokens = req.get_first_router_need_tokens()
            if self._can_add_new_req_fixed_sp(req, is_busy)\
                and new_batch_first_router_need_tokens + req_first_router_need_tokens <= self.batch_max_tokens\
                and per_instance_need_tokens + math.ceil(req_first_router_need_tokens / self.sp_world_size) + cur_all_used_tokens_list.max() <= self.max_total_tokens:
                can_run_list.append(req)
                new_batch_first_router_need_tokens += req_first_router_need_tokens
                per_instance_need_tokens += math.ceil(req_first_router_need_tokens / self.sp_world_size)
                if req.req_status in [ReqRunStatus.PAUSED_AND_KVKEEP, ReqRunStatus.PAUSED_AND_OFFLOAD]:
                    self.pause_req_dict.pop(req.request_id)
            else:
                break

        if len(can_run_list) != 0:
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count:]
            self.recalcu_pause_req_used_tokens_list()
            return can_run_list
        else:
            return []
        
    def recalcu_pause_req_used_tokens_list(self):
        used_tokens_list = np.zeros((self.sp_world_size,), dtype=np.int32)
        for req_id, req_obj in self.pause_req_dict.items():
            used_tokens_list += req_obj.get_used_tokens_list()
        self.pause_req_used_tokens_list = used_tokens_list
        return self.pause_req_used_tokens_list

