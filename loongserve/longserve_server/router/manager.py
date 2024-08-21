import os
import time
import math
import uuid
import uvloop
import asyncio
import socket
import numpy as np
from functools import partial
from typing import Dict, List, Optional, Tuple, Union
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio

from ..sampling_params import SamplingParams
from ..io_struct import Req, NormalReq, Batch, ReqDetokenizationState
from .model_infer.model_rpc import ModelRpcServer
from .req_queue import ReqQueue
from loongserve.utils.infer_utils import calculate_time
from ..io_struct import BatchTokenIdOut, AbortReq, ReqRunStatus, FinishStatus
from .stats import Stats
from .pause_strategy import Fcfs, select_paused_reqs
from .profiler import Profiler
from loongserve.utils.log_utils import init_logger
import rnccl

from ..detokenization.manager import DeTokenizationManager

import longserve_c_scheduler

os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)


class RouterManager:

    def __init__(self, args, router_port, detokenization_port, httpserver_port, model_rpc_ports):
        print(f"Starting the router on node {socket.gethostname()}")
        self.args = args
        self.model_weightdir = args.model_dir
        self.total_world_size = args.total_world_size
        self.tp_world_size = args.tp_world_size
        self.sp_world_size = args.sp_world_size
        assert self.total_world_size == self.tp_world_size * self.sp_world_size
        self.local_world_size = args.local_world_size
        assert self.total_world_size % self.local_world_size == 0
        self.local_sp_world_size = self.local_world_size // self.tp_world_size
        self.load_way = args.load_way
        self.mode = args.mode
        self.max_total_token_num = args.max_total_token_num
        self.max_mig_len = args.max_mig_len
        self.use_fixed_sp = args.use_fixed_sp
        self.disable_scale_up = args.disable_scale_up
        if self.disable_scale_up:
            logger.info("disable_scale_up is True, disable scale up")
            self._schedule_decode_batch_list = partial(RouterManager._schedule_decode_batch_list_without_scale_up, self)

        if self.use_fixed_sp:
            logger.info("use_fixed_sp is True, use fixed sequence parallelism")
            self.step = partial(RouterManager._step_fixed_sp, self)
        else:
            logger.info("use_fixed_sp is False, use elastic sequence parallelism")
            self.step = partial(RouterManager._step, self)

        self.pause_strategy = Fcfs()
        self.running_batch_list: List[Batch] = []
        self.eos_id = args.eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = args.max_wait_tokens

        self.min_comp_bound_decoding_batch_size = args.min_comp_bound_decoding_batch_size

        self.sum_finished_req_output_len = 512
        self.num_finished_reqs = 1
        
        context = zmq.asyncio.Context(2)
        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{router_port}")
        
        self.num_detokenizers = args.num_detokenizers
        self.detokenizer_id = 0

        self.detokenization_rpcs: List[DeTokenizationManager] = []
        for i in range(self.num_detokenizers):
            self.detokenization_rpcs.append(
                ray.remote(
                    num_cpus=0,
                    num_gpus=0
                )(DeTokenizationManager).options(
                    name=f"detokenizer {i}",
                    resources={f"{ray.state.current_node_id()}": 0.001} # Make sure it runs on the same node
                ).remote(
                    detokenizer_id=i,
                    model_weightdir=args.model_dir,
                    tokenizor_mode=args.tokenizer_mode,
                    httpserver_port=httpserver_port,
                    trust_remote_code=args.trust_remote_code,
                    skip_special_tokens=not args.no_skipping_special_tokens,
                    spaces_between_special_tokens=not args.no_spaces_between_special_tokens,
                )
            )

        self.stats_tool = Stats(not args.disable_log_stats, args.with_log_trace, args.log_stats_interval)
        
        self._scale_up_counter = 0
        self._multi_master_counter = 0

        self.profiler = Profiler(args)
        return

    async def wait_to_model_ready(self):
        print("Starting workers")

        self.model_rpcs: List[List[Union[ModelRpcServer, None]]] = [[None for tp_rank in range(self.tp_world_size)] for sp_rank in range(self.sp_world_size)]

        if self.sp_world_size > self.local_world_size or self.tp_world_size > self.local_world_size:
            # Schedule arbitrary workers into one node
            print("Scheduling arbitrary workers into one node")
            for sp_rank in range(self.sp_world_size):
                for tp_rank in range(self.tp_world_size):
                    total_rank = sp_rank * self.tp_world_size + tp_rank
                    rpc_model: ModelRpcServer = ray.remote(
                        num_cpus=0,
                        num_gpus=1,
                    )(ModelRpcServer).options(name=f"worker {total_rank}").remote()
                    self.model_rpcs[sp_rank][tp_rank] = rpc_model

        elif self.sp_world_size * self.tp_world_size <= self.local_world_size:
            # Schedule all workers into one node
            print("Scheduling all workers into one node")
            placement_group = ray.util.placement_group(
                bundles = [{"CPU": 0, "GPU": 1} for _ in range(self.sp_world_size * self.tp_world_size)],
                strategy="STRICT_PACK"
            )
            ray.get(placement_group.ready(), timeout=60)
            for sp_rank in range(self.sp_world_size):
                for tp_rank in range(self.tp_world_size):
                    total_rank = sp_rank * self.tp_world_size + tp_rank
                    rpc_model = ray.remote(
                        num_cpus=0,
                        num_gpus=1
                    )(ModelRpcServer).options(
                        name=f"worker {total_rank}",
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=placement_group,
                            placement_group_capture_child_tasks=True
                        )
                    ).remote()
                    self.model_rpcs[sp_rank][tp_rank] = rpc_model
        elif self.sp_world_size >= self.tp_world_size:
            # Schedule all workers that are in the same SP group into one node
            print("Scheduling all workers in the same SP group into one node...")
            for tp_rank in range(self.tp_world_size):
                placement_group = ray.util.placement_group(
                    bundles = [{"CPU": 0, "GPU": 1} for _ in range(self.sp_world_size)],
                    strategy="STRICT_PACK"
                )
                ray.get(placement_group.ready(), timeout=60)
                for sp_rank in range(self.sp_world_size):
                    total_rank = sp_rank * self.tp_world_size + tp_rank
                    rpc_model = ray.remote(
                        num_cpus=0,
                        num_gpus=1
                    )(ModelRpcServer).options(
                        name=f"worker {total_rank}",
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=placement_group,
                            placement_group_capture_child_tasks=True
                        )
                    ).remote()
                    self.model_rpcs[sp_rank][tp_rank] = rpc_model
        else:
            # Schedule all workers that are in the same TP group into one node
            print("Scheduling all workers in the same TP group into one node...")
            for sp_rank in range(self.sp_world_size):
                placement_group = ray.util.placement_group(
                    bundles = [{"CPU": 0, "GPU": 1} for _ in range(self.tp_world_size)],
                    strategy="STRICT_PACK"
                )
                ray.get(placement_group.ready(), timeout=60)
                for tp_rank in range(self.tp_world_size):
                    total_rank = sp_rank * self.tp_world_size + tp_rank
                    rpc_model = ray.remote(
                        num_cpus=0,
                        num_gpus=1
                    )(ModelRpcServer).options(
                        name=f"worker {total_rank}",
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=placement_group,
                            placement_group_capture_child_tasks=True
                        )
                    ).remote()
                    self.model_rpcs[sp_rank][tp_rank] = rpc_model

        print("Initializing models")
        nccl_host = ray.get(self.model_rpcs[0][0].get_ip.remote())
        print(f"{nccl_host=}")
        init_model_ret = []
        # async init model process
        for tp_rank in range(self.tp_world_size):
            sp_rnccl_unique_id = rnccl.get_nccl_unique_id()
            for sp_rank in range(self.sp_world_size):
                total_rank = sp_rank * self.tp_world_size + tp_rank
                kvargs = {
                    "total_world_size": self.total_world_size,
                    "total_rank": total_rank,
                    "tp_world_size": self.tp_world_size,
                    "tp_rank": tp_rank,
                    "sp_world_size": self.sp_world_size,
                    "sp_rank": sp_rank,
                    "sp_rnccl_unique_id": sp_rnccl_unique_id,

                    "weight_dir" : self.model_weightdir,
                    "load_way" : self.load_way,
                    "max_total_token_num" : self.max_total_token_num,
                    "mode" : self.mode,
                    "max_req_num" : self.args.running_max_req_size + 8,
                    "max_seq_length" : self.args.max_req_total_len + 8, # 留一点余量
                    "host": nccl_host,
                    "nccl_port" : self.args.nccl_port,
                    "max_mig_len": self.max_mig_len,
                }
                init_model_ret.append(self.model_rpcs[sp_rank][tp_rank].exposed_init_model.remote(kvargs))

        await asyncio.gather(*init_model_ret)

        print("Models are ready")

        self.req_queue = ReqQueue(self.args)   
        return
    
    def to_detokenization_processes(self, send_obj: BatchTokenIdOut):
        ret = self.detokenization_rpcs[self.detokenizer_id].handle_loop.remote(send_obj)
        self.detokenizer_id = (self.detokenizer_id + 1) % self.num_detokenizers
        return ret

    async def add_req(
        self,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        request_id: str,
    ):  
        req = NormalReq(request_id, prompt_ids, sampling_params, self.sp_world_size)
        self.req_queue.append(req)
        rets = []
        for i in range(self.num_detokenizers):
            rets.append(self.detokenization_rpcs[i].handle_loop.remote(req.to_req_detokenization_state()))
        await asyncio.gather(*rets)
        return

    async def abort(self, request_id):
        if len(self.running_batch_list) > 0:
            for batch in self.running_batch_list:
                for req in batch.reqs:
                    if req.request_id == request_id:
                        req.finish_status = FinishStatus.FINISHED_ABORT
        for req in self.req_queue.waiting_req_list:
            if req.request_id == request_id:
                req.finish_status = FinishStatus.FINISHED_ABORT
        return

    async def loop_for_fwd(self,):
        last_log_time = time.time()
        while True:
            await self.step()
            if len(self.running_batch_list) == 0:
                await asyncio.sleep(0.005)  # 5 ms
            
            cur_log_time = time.time()
            if cur_log_time - last_log_time > self.stats_tool.log_stats_interval:
                # Print statistics every log_stats_interval seconds
                total_used_tokens_list = self._get_total_used_tokens_list(self.running_batch_list)
                total_used_tokens = total_used_tokens_list.sum()
                batch_size_list = [len(batch.reqs) for batch in self.running_batch_list]
                total_batch_size = sum(batch_size_list)
                token_ratio = total_used_tokens / (self.max_total_token_num * self.sp_world_size)
                logger.debug(
                    f"current batch size: {total_batch_size} ({batch_size_list})\n"
                    f"paused req num: {len(self.req_queue.pause_req_dict)}\n"
                    f"token used ratio: {token_ratio}\n"
                    f"total_used_tokens_list: {total_used_tokens_list}"
                )
                last_log_time = cur_log_time
                self.stats_tool.print_stats(cur_log_time, self.num_finished_reqs - 1)

    async def _step_fixed_sp(self):
        """
        事件处理循环
        """
        if len(self.running_batch_list) == 0:
            new_req_list = self.req_queue.generate_new_req_list_fixed_sp(self.running_batch_list)
            if len(new_req_list) > 0:
                # prefill batch starts
                new_batch = Batch(uuid.uuid4().hex, new_req_list, self.sp_world_size, list(range(self.sp_world_size)))
                self.running_batch_list.append(new_batch)
                rets = self._prefill_batch(self.running_batch_list[0])
                await self._post_forward_batch(self.running_batch_list, rets)
                self.running_batch_list = self._filter_running_batch(self.running_batch_list)
                # prefill batch finishes
                self.has_wait_tokens = 0
            return

        if self.has_wait_tokens >= self.max_wait_tokens:
            new_req_list = self.req_queue.generate_new_req_list_fixed_sp(self.running_batch_list)
            if len(new_req_list) > 0:
                # prefill batch starts
                new_mini_batch = Batch(uuid.uuid4().hex, new_req_list, self.sp_world_size, list(range(self.sp_world_size)))
            else:
                new_mini_batch = None
            self.has_wait_tokens = 0
            if new_mini_batch is not None:
                rets = self._prefill_batch(new_mini_batch)
                await self._post_forward_batch([new_mini_batch], rets)
                if not new_mini_batch.is_clear():
                    await self._merge_batch(self.running_batch_list[0], new_mini_batch)
                    self.running_batch_list[0].merge(new_mini_batch)
                # prefill batch finishes
                return

        if self._can_decode_fixed_sp(self.running_batch_list[0]):
            batch_size = len(self.running_batch_list[0].reqs)
            # decoding batch starts
            rets = self._decode_batch(self.running_batch_list[0], num_sp_master_ranks=1, mini_batch_range_list=[(0, batch_size)], mini_batch_size_list=[batch_size])
            await self._post_forward_batch(self.running_batch_list, rets)
            self.running_batch_list = self._filter_running_batch(self.running_batch_list)
            # decoding batch finishes
            self.has_wait_tokens += 1
            return
        else:
            # pause strategy
            paused_req_batches = select_paused_reqs(self.running_batch_list, self.pause_strategy, self.req_queue, self.max_total_token_num)
            for req, batch in paused_req_batches:
                await self._pause_reqs(batch, [req])

            logger.debug(f"pasued req num: {len(self.req_queue.pause_req_dict)}")
            self.has_wait_tokens = 0
            return
        return

    def _can_decode_fixed_sp(self, batch: Batch):
        total_used_tokens_list = self._get_total_used_tokens_list([batch])
        batch.occupied_instances.sort(key=lambda x: total_used_tokens_list[x])
        batch_size = len(batch.reqs)
        return total_used_tokens_list[batch.occupied_instances[0]] + batch_size <= self.max_total_token_num

    async def _step(self):
        """
        事件处理循环
        """
        is_idle = (len(self.running_batch_list) == 0)
        if is_idle or self.has_wait_tokens >= self.max_wait_tokens:
            new_req_list = self.req_queue.generate_new_req_list(self.running_batch_list, self.sum_finished_req_output_len, self.num_finished_reqs, self.profiler)
            if len(new_req_list) > 0:
                self.has_wait_tokens = 0
                # prefill batch starts
                self.running_batch_list = await self._schedule_new_req_list_with_decoding(new_req_list, self.running_batch_list)
                return
            elif is_idle:
                return

        total_used_tokens_list = self._get_total_used_tokens_list(self.running_batch_list)
        can_decode, self.running_batch_list, rets = await self._schedule_decode_batch_list(self.running_batch_list, total_used_tokens_list, True)
        if can_decode:
            await self._post_forward_batch(self.running_batch_list, rets)
            self.running_batch_list = self._filter_running_batch(self.running_batch_list)
            # decoding batch finishes
            self.has_wait_tokens += 1
            return
        else:
            print(f"execute offloading, total_used_tokens_list: {total_used_tokens_list}, pause_req_used_tokens_list: {self.req_queue.pause_req_used_tokens_list}, max_total_token_num: {self.max_total_token_num}")
            # pause strategy
            paused_req_batches = select_paused_reqs(self.running_batch_list, self.pause_strategy, self.req_queue, self.max_total_token_num) # FIXME
            for req, batch in paused_req_batches:
                await self._pause_reqs(batch, [req])
            self.running_batch_list = self._filter_running_batch(self.running_batch_list)
            logger.debug(f"pasued req num: {len(self.req_queue.pause_req_dict)}")
            self.has_wait_tokens = 0
            return
        return

    async def _post_forward_batch(self, running_batch_list: List[Batch], rets: list):
        """
            filter out finished requests, remove empty batches, and send results to the detokenization process
        """
        ans = await asyncio.gather(*rets)
        handle_finish_req_rets = []
        if self.total_world_size != 1:
            start = 0
            for batch in running_batch_list:
                req_to_out_status_list = []

                for logical_sp_rank, sp_rank in enumerate(batch.occupied_instances):
                    ans_i = ans[start + logical_sp_rank * self.tp_world_size]
                    req_to_out_status_list.append((logical_sp_rank, sp_rank, ans_i))
                
                self._update_out_status_to_batch(batch, req_to_out_status_list)
                unfinished_req_ids, finished_req_ids, sum_finished_req_output_len = batch.mark_and_get_finished_req_and_preupdate_status(self.eos_id)
                
                self.sum_finished_req_output_len += sum_finished_req_output_len
                self.num_finished_reqs += len(finished_req_ids)
                
                self._send_to_detokenization_proc(batch, req_to_out_status_list)
                batch.filter_out_finished_req(unfinished_req_ids, finished_req_ids)
                handle_finish_req_rets.append(self._handle_finish_req(batch, unfinished_req_ids, finished_req_ids))

                start += len(batch.occupied_instances) * self.tp_world_size
        else:
            assert len(ans) == len(running_batch_list)
            for ans_i, batch in zip(ans, running_batch_list):
                req_to_out_status_list = [(0, batch.occupied_instances[0], ans_i)]

                self._update_out_status_to_batch(batch, req_to_out_status_list)
                unfinished_req_ids, finished_req_ids, sum_finished_req_output_len = batch.mark_and_get_finished_req_and_preupdate_status(self.eos_id)
                
                self.sum_finished_req_output_len += sum_finished_req_output_len
                self.num_finished_reqs += len(finished_req_ids)
                
                self._send_to_detokenization_proc(batch, req_to_out_status_list)
                batch.filter_out_finished_req(unfinished_req_ids, finished_req_ids)
                handle_finish_req_rets.append(self._handle_finish_req(batch, unfinished_req_ids, finished_req_ids))
        removed_instances_list = await asyncio.gather(*handle_finish_req_rets)

        final_removed_instances = []
        for removed_instances in removed_instances_list:
            final_removed_instances += removed_instances
        return final_removed_instances
    
    async def _run_independent_batch_list(self, independent_batch_list: list[Batch], rets: list, is_prefill: bool, global_iteration_time: float, iteration_start_time: float):
        """
            A coroutine to manage an independent batch list until this macro iteration finishes
        """
        idle_instances = []
        need_count = is_prefill
        while True:
            idle_instances += await self._post_forward_batch(independent_batch_list, rets)
            independent_batch_list = self._filter_running_batch(independent_batch_list)

            if is_prefill:
                for batch in independent_batch_list:
                    removed_instances = [sp_rank for sp_rank in batch.occupied_instances if batch.batch_used_tokens_list[sp_rank] == 0]
                    if len(removed_instances) > 0:
                        await self._scale_down_batch(batch, removed_instances)
                        idle_instances += removed_instances
                
                # prefill batch finishes
                
                if need_count:
                    self.num_running_prefill_batch -= 1
                    assert self.num_running_prefill_batch >= 0
                    need_count = False
                is_prefill = False
            else:
                pass
                # decoding batch finishes
            
            if len(independent_batch_list) == 0:
                assert len(idle_instances) > 0
                
                while self.num_running_prefill_batch > 0:
                    iteration_elapsed_time = (time.perf_counter() - iteration_start_time) * 1000 # milliseconds
                    iteration_left_time = global_iteration_time - iteration_elapsed_time
                    if iteration_left_time <= 0:
                        return independent_batch_list
                    
                    greedy_req_list = self.req_queue.generate_greedy_req_list([], idle_instances, iteration_left_time, self.profiler)
                    
                    if len(greedy_req_list) > 0:
                        total_used_tokens_list = self._get_total_used_tokens_list(independent_batch_list)
                        independent_batch_list, rets, _ = self._schedule_new_req_list(greedy_req_list, idle_instances, total_used_tokens_list)

                        idle_instances = []
                        is_prefill = True

                        break

                    await asyncio.sleep(0.01) # 10ms
                
                if is_prefill:
                    continue

            if self.num_running_prefill_batch == 0:
                return independent_batch_list
            
            total_used_tokens_list = self._get_total_used_tokens_list(independent_batch_list)
            can_decode, independent_batch_list, rets = await self._schedule_decode_batch_list(independent_batch_list, total_used_tokens_list, False)
            if not can_decode:
                logger.debug(f"blocked in _run_independent_batch_list, total_used_tokens_list: {total_used_tokens_list}, max_total_token_num: {self.max_total_token_num}")
                return independent_batch_list

    async def _dispatch_independ_batch_list(self, prefill_batch_list: list[Batch], prefill_rets: list, decode_batch_list: list[Batch], decode_rets: list, global_iteration_time: float):
        """
            Launch multiple coroutines to manage independent batch lists
        """
        iteration_start_time = time.perf_counter() # seconds
        self.num_running_prefill_batch = len(prefill_batch_list)
        final_rets = []
        if len(decode_batch_list) > 0:
            final_rets.append(self._run_independent_batch_list(decode_batch_list, decode_rets, False, global_iteration_time, iteration_start_time))
        
        start = 0
        for batch in prefill_batch_list:
            num_instances = len(batch.occupied_instances)
            final_rets.append(self._run_independent_batch_list([batch], prefill_rets[start:start + num_instances * self.tp_world_size], True, global_iteration_time, iteration_start_time))
            start += num_instances * self.tp_world_size
        final_independent_batches_list = await asyncio.gather(*final_rets)
        running_batch_list = []
        for independent_batches in final_independent_batches_list:
            running_batch_list += independent_batches
        return running_batch_list
    
    async def _schedule_new_req_list_with_decoding(self, new_req_list: List[Req], running_batch_list: List[Batch]):
        """
            Schedule a macro iteration with requests in the prefill and decoding phase
        """
        num_new_tokens = 0
        prefill_len_square_sum = 0
        avg_input_token_latency_factor = 0
        for req in new_req_list:
            first_router_need_tokens = req.get_first_router_need_tokens()
            num_new_tokens += first_router_need_tokens
            prefill_len_square_sum += first_router_need_tokens ** 2
            avg_input_token_latency_factor += 1 / first_router_need_tokens

        total_used_tokens_list: np.array = self._get_total_used_tokens_list(running_batch_list)
        available_instances = np.nonzero(total_used_tokens_list == 0)[0].tolist()
        num_idle_instances = len(available_instances)

        num_current_busy_instance = self.sp_world_size - num_idle_instances
        num_min_busy_instances = (np.sum(total_used_tokens_list) + self.max_total_token_num - 1) // self.max_total_token_num
        num_max_idle_instances = self.sp_world_size - num_min_busy_instances
        if num_min_busy_instances < num_current_busy_instance and num_new_tokens <= num_max_idle_instances * self.max_total_token_num and not self.disable_scale_up:
            running_batch_list, total_used_tokens_list, num_idle_instances, available_instances = await self._minimize_decoding_occupied_instances(running_batch_list, total_used_tokens_list, num_idle_instances, num_max_idle_instances, available_instances, num_new_tokens, prefill_len_square_sum)
        
        pending_decode_batch_list: List[Batch] = []
        num_needed_slots = num_new_tokens - num_idle_instances * self.max_total_token_num
        if num_needed_slots > 0:
            
            decode_batch_token_tuple_list = [(batch, len(batch.occupied_instances) * self.max_total_token_num - batch.batch_used_tokens_list.sum()) for batch in running_batch_list]
            decode_batch_token_tuple_list.sort(key=lambda x: x[1])
            
            while num_needed_slots > 0:
                batch, num_idle_token = decode_batch_token_tuple_list.pop()
                pending_decode_batch_list.append(batch)
                num_needed_slots -= num_idle_token
                available_instances += batch.occupied_instances
                num_idle_instances += len(batch.occupied_instances)
            running_batch_list = [batch for batch, _ in decode_batch_token_tuple_list]
        
        running_decode_batch_list = []
        for batch in running_batch_list:
            num_idle_token = len(batch.occupied_instances) * self.max_total_token_num - batch.batch_used_tokens_list.sum()
            
            cur_max_prefill_iter_time = self.profiler.predict(num_idle_instances, num_new_tokens, prefill_len_square_sum)
            nxt_max_prefill_iter_time = self.profiler.predict(num_idle_instances + len(batch.occupied_instances), num_new_tokens, prefill_len_square_sum)
            
            prefill_speedup = (cur_max_prefill_iter_time - nxt_max_prefill_iter_time) * avg_input_token_latency_factor
            avg_output_token_latency_factor = len(batch.reqs) / np.sum([len(req.output_ids) for req in batch.reqs])
            decode_slowdown = nxt_max_prefill_iter_time * len(batch.reqs) * avg_output_token_latency_factor
            
            if prefill_speedup < decode_slowdown:
                running_decode_batch_list.append(batch)
            else:
                pending_decode_batch_list.append(batch)
                available_instances += batch.occupied_instances
                num_idle_instances += len(batch.occupied_instances)
            
        can_decode, running_decode_batch_list, decode_rets = await self._schedule_decode_batch_list(running_decode_batch_list, total_used_tokens_list, False)

        if not can_decode:
            extra_idle_instances = []
            while num_new_tokens <= (len(available_instances) - 1) * self.max_total_token_num and not can_decode:
                extra_idle_instances.append(available_instances.pop())
                can_decode, running_decode_batch_list, decode_rets = await self._schedule_decode_batch_list(running_decode_batch_list, total_used_tokens_list, False, extra_idle_instances)
            
            if not can_decode:
                print(f"\nprefill only start, len(new_req_list): {len(new_req_list)}, running decode batch: {len(running_decode_batch_list)}, batch size list: {[len(batch.reqs) for batch in running_decode_batch_list]}, pending decode batch: {len(pending_decode_batch_list)}, total_used_tokens_list: {total_used_tokens_list}\n")
                pending_decode_batch_list += running_decode_batch_list
                running_decode_batch_list = []
                available_instances = list(range(self.sp_world_size))
        

        prefill_batch_list, prefill_rets, global_iteration_time = self._schedule_new_req_list(new_req_list, available_instances, total_used_tokens_list)

        new_batch_list: List[Batch] = await self._dispatch_independ_batch_list(prefill_batch_list, prefill_rets, running_decode_batch_list, decode_rets, global_iteration_time)

        if len(pending_decode_batch_list) > 0:
            for new_batch in new_batch_list:
                new_batch_occupied_instances_set = set(new_batch.occupied_instances)
                new_running_batch_list = [new_batch]
                for batch in pending_decode_batch_list:
                    if len(new_batch_occupied_instances_set.intersection(set(batch.occupied_instances))) > 0:
                        await self._merge_batch(new_batch, batch)
                        new_batch.merge(batch)
                    else:
                        new_running_batch_list.append(batch)
                
                pending_decode_batch_list = new_running_batch_list
            return pending_decode_batch_list
        else:
            return new_batch_list
    
    def _get_migration_time(self, num_migrated_tokens: int):
        # TODO: support different model and hardware configs
        num_transferred_bytes = num_migrated_tokens * 32 * 2 * 32 * 128 * 2 # num_tokens * num_layers * (k+v) * num_heads * head_dim * fp16
        transfer_rate = 400 * 1024 * 1024 * 1024 # 400GB/s
        return num_transferred_bytes / (transfer_rate * self.tp_world_size) * 1000 # ms
    
    async def _minimize_decoding_occupied_instances(self, running_batch_list: List[Batch], total_used_tokens_list: np.array, num_idle_instances: int, num_max_idle_instances: int, available_instances: list[int], num_new_tokens: int, prefill_len_square_sum: int):
        """
            Migrate key-value caches to leave more idle instances for prefill if it is beneficial
        """
        instance_batch_mapping = {sp_rank: batch for batch in running_batch_list for sp_rank in batch.occupied_instances}
        instance_batch_tuple_list = sorted(instance_batch_mapping.items(), key=lambda x: total_used_tokens_list[x[0]], reverse=True)
        num_min_busy_instances = self.sp_world_size - num_max_idle_instances
        start = num_min_busy_instances - 1
        removed_batch_set = set()
        while num_idle_instances < num_max_idle_instances:
            src_sp_rank, _ = instance_batch_tuple_list.pop()
            src_batch = instance_batch_mapping[src_sp_rank]
            
            if num_new_tokens <= num_idle_instances * self.max_total_token_num:
                cur_max_prefill_iter_time = self.profiler.predict(num_idle_instances, num_new_tokens, prefill_len_square_sum)
                nxt_max_prefill_iter_time = self.profiler.predict(num_idle_instances + 1, num_new_tokens, prefill_len_square_sum)
                prefill_speedup = cur_max_prefill_iter_time - nxt_max_prefill_iter_time
                
                num_migrated_tokens = total_used_tokens_list[src_sp_rank]
                migration_time = self._get_migration_time(num_migrated_tokens)
                
                if prefill_speedup < migration_time:
                    break

            while total_used_tokens_list[src_sp_rank] > 0:
                dst_sp_rank, _ = instance_batch_tuple_list[start]
                dst_batch = instance_batch_mapping[dst_sp_rank]
                total_migration_len = min(total_used_tokens_list[src_sp_rank], self.max_total_token_num - total_used_tokens_list[dst_sp_rank])
                if total_migration_len == 0:
                    start -= 1
                    continue

                if src_batch.batch_id != dst_batch.batch_id:
                    await self._merge_batch(dst_batch, src_batch)
                    dst_batch.merge(src_batch)

                    removed_batch_set.add(src_batch.batch_id)
                    
                    for sp_i in src_batch.occupied_instances:
                        instance_batch_mapping[sp_i] = dst_batch

                    src_batch = dst_batch
                
                src_batch.batch_used_tokens_list[src_sp_rank] -= total_migration_len
                dst_batch.batch_used_tokens_list[dst_sp_rank] += total_migration_len
                total_used_tokens_list[src_sp_rank] -= total_migration_len
                total_used_tokens_list[dst_sp_rank] += total_migration_len

                b_request_ids = []
                b_migration_len = []

                for req in src_batch.reqs:
                    if req.cur_kv_len_list[src_sp_rank] == 0:
                        continue
                    b_request_ids.append(req.request_id)

                    migration_len = min(total_migration_len, req.cur_kv_len_list[src_sp_rank])
                    b_migration_len.append(migration_len)

                    req.cur_kv_len_list[src_sp_rank] -= migration_len
                    req.cur_kv_len_list[dst_sp_rank] += migration_len
                    total_migration_len -= migration_len
            
                await self._migrate_batch(src_batch, b_request_ids, b_migration_len, src_sp_rank, dst_sp_rank)
            
            assert src_batch.batch_used_tokens_list[src_sp_rank] == 0
            await self._scale_down_batch(src_batch, [src_sp_rank])
            available_instances.append(src_sp_rank)
            num_idle_instances += 1
        
        running_batch_list = [batch for batch in running_batch_list if batch.batch_id not in removed_batch_set]
        return running_batch_list, total_used_tokens_list, num_idle_instances, available_instances
    
    def _schedule_new_req_list(self, new_req_list: List[Req], available_instances: List[int], total_used_tokens_list: np.array):
        """
            Schedule requests in the prefill phase
        """
        new_req_list.sort(key=lambda x: x.get_first_router_need_tokens(), reverse=True)
        num_new_req = len(new_req_list)
        available_instances.sort(key=lambda x: (total_used_tokens_list[x], x // self.local_sp_world_size))
        num_instances = len(available_instances)
        
        min_iteration_times = np.full((num_new_req + 1, num_instances + 1), fill_value=float("inf"), dtype=np.float64)
        min_iteration_times_index = np.full((num_new_req + 1, num_instances + 1, 2), fill_value=-1, dtype=np.int32)

        req_input_prefix_sum = np.array([req.get_first_router_need_tokens() for req in new_req_list], dtype=np.int64)
        req_input_square_prefix_sum = req_input_prefix_sum ** 2
        inverted_req_input_prefix_sum = 1.0 / req_input_prefix_sum.astype(np.float64)
        req_input_prefix_sum = np.cumsum(req_input_prefix_sum)
        req_input_square_prefix_sum = np.cumsum(req_input_square_prefix_sum)
        inverted_req_input_prefix_sum = np.cumsum(inverted_req_input_prefix_sum)

        left_token_prefix_sum = np.array([self.max_total_token_num - total_used_tokens_list[sp_rank] for sp_rank in available_instances], dtype=np.int64)
        left_token_prefix_sum = np.cumsum(left_token_prefix_sum)
        
        assert left_token_prefix_sum[-1] >= req_input_prefix_sum[-1], f"{left_token_prefix_sum[-1]=}, {req_input_prefix_sum[-1]=}"
        
        longserve_c_scheduler.minimize_prefill_iteration_time(
            num_new_req, num_instances, min_iteration_times, min_iteration_times_index, req_input_prefix_sum, req_input_square_prefix_sum, inverted_req_input_prefix_sum, left_token_prefix_sum, self.profiler.predictor_parameters
        )
        
        global_min_iteration_time = float("inf")
        global_num_used_instances = -1
        for num_used_instances in range(1, num_instances + 1):
            if min_iteration_times[num_new_req, num_used_instances] < global_min_iteration_time:
                global_min_iteration_time = min_iteration_times[num_new_req, num_used_instances]
                global_num_used_instances = num_used_instances
        
        new_batch_list = []
        rets = []
        global_num_served_req = num_new_req
        global_max_iteration_time = 0
        while global_num_served_req > 0:
            last_batch_size, last_used_instances = min_iteration_times_index[global_num_served_req, global_num_used_instances]
            
            if last_batch_size == -1 or last_used_instances == -1:
                print(f"global_num_served_req: {global_num_served_req}, global_num_used_instances: {global_num_used_instances}, last_batch_size: {last_batch_size}, last_used_instances: {last_used_instances}, min_iteration_times: {min_iteration_times}, min_iteration_times_index: {min_iteration_times_index}", flush=True)
                assert False

            num_prev_served_req = global_num_served_req - last_batch_size
            req_input_sum = req_input_prefix_sum[global_num_served_req - 1]\
                        - (req_input_prefix_sum[num_prev_served_req - 1] if (num_prev_served_req > 0) else 0)
            req_input_square_sum = req_input_square_prefix_sum[global_num_served_req - 1]\
                        - (req_input_square_prefix_sum[num_prev_served_req - 1] if (num_prev_served_req > 0) else 0)
            single_iteration_time = self.profiler.predict(last_used_instances, req_input_sum, req_input_square_sum)
            global_max_iteration_time = max(global_max_iteration_time, single_iteration_time)


            new_batch = Batch(uuid.uuid4().hex, new_req_list[global_num_served_req - last_batch_size:global_num_served_req], self.sp_world_size, available_instances[global_num_used_instances - last_used_instances:global_num_used_instances])
            need_context_migration, migration_plan = self._get_batch_prefill_migration_plan(new_batch, total_used_tokens_list)
            new_batch_list.append(new_batch)
            rets += self._prefill_batch(
                    new_batch,
                    need_context_migration=need_context_migration,
                    migration_plan=migration_plan
            )

            global_num_served_req -= last_batch_size
            global_num_used_instances -= last_used_instances
        assert global_num_served_req == 0 and global_num_used_instances == 0
        return new_batch_list, rets, global_max_iteration_time

    def _print_all_reqs(self, batch_list: list[Batch]):
        print("---------", flush=True)
        for batch in batch_list:
            print(f"{batch.batch_id}, batch_used_tokens_list: {batch.batch_used_tokens_list}, occupied_instances: {batch.occupied_instances}, batch size: {len(batch.reqs)}", flush=True)
            for req in batch.reqs:
                print(f"request_id: {req.request_id}, cur_kv_len_list: {req.cur_kv_len_list}", flush=True)
        print("---------", flush=True)
        return
    
    def _get_batch_prefill_migration_plan(self, batch: Batch, total_used_tokens_list: np.array):
        """
            Generate a context migration plan for a prefill batch
        """
        if len(batch.occupied_instances) == 1:
            return False, None

        batch.occupied_instances.sort(key=lambda x: total_used_tokens_list[x], reverse=True)

        num_instances = len(batch.occupied_instances)
        batch_size = len(batch.reqs)
        
        migration_plan = [[[0 for _ in range(batch_size)] for __ in range(2)] for ___ in range(num_instances)]
        cur_logical_sp_rank = 0
        cur_sp_rank = batch.occupied_instances[cur_logical_sp_rank]
        new_batch_used_tokens = 0
        
        for i, req in enumerate(batch.reqs):
            prefill_len = req.input_len + len(req.output_ids)

            start = 0
            while prefill_len > 0:
                while new_batch_used_tokens + total_used_tokens_list[cur_sp_rank] == self.max_total_token_num:
                    cur_logical_sp_rank += 1
                    assert cur_logical_sp_rank < num_instances
                    cur_sp_rank = batch.occupied_instances[cur_logical_sp_rank]
                    new_batch_used_tokens = 0
                
                left_tokens = self.max_total_token_num - total_used_tokens_list[cur_sp_rank] - new_batch_used_tokens
                used_tokens = min(prefill_len, left_tokens)
                end = start + used_tokens
                migration_plan[cur_logical_sp_rank][0][i] = start
                migration_plan[cur_logical_sp_rank][1][i] = end

                start = end
                prefill_len -= used_tokens
                new_batch_used_tokens += used_tokens
        
        need_context_migration = True
        return need_context_migration, migration_plan

    async def _schedule_decode_batch_list_without_scale_up(self, decode_batch_list: List[Batch], total_used_tokens_list: np.array, can_use_extra_instances: bool, extra_idle_instances: List[int] = []):
        can_use_extra_instances = False
        extra_idle_instances = []
        
        for batch in decode_batch_list:
            while True:
                decode_need_tokens = len(batch.reqs)
                idle_tokens = self.max_total_token_num - total_used_tokens_list[batch.occupied_instances[0]] - decode_need_tokens
                
                if idle_tokens >= 0:
                    break
                
                paused_req_batches = select_paused_reqs([batch], self.pause_strategy, self.req_queue, self.max_total_token_num)
                for req, batch in paused_req_batches:
                    await self._pause_reqs(batch, [req])
                    
                    total_used_tokens_list = self._get_total_used_tokens_list(decode_batch_list)
                    
                    print(f"due to disabling scale-up operations, execute offloading, total_used_tokens_list: {total_used_tokens_list}, pause_req_used_tokens_list: {self.req_queue.pause_req_used_tokens_list}, max_total_token_num: {self.max_total_token_num}, batch.occupied_instances: {batch.occupied_instances}, output_len: {len(req.output_ids)}, input_len: {req.input_len}", flush=True)
        
        decode_batch_list = self._filter_running_batch(decode_batch_list)
        
        rets = []
        for batch in decode_batch_list:
            decode_need_tokens = len(batch.reqs)
            num_sp_master_ranks = 1
            mini_batch_range_list = [(0, decode_need_tokens)]
            mini_batch_size_list = [decode_need_tokens]
            
            assert decode_need_tokens + total_used_tokens_list[batch.occupied_instances[0]] <= self.max_total_token_num
            
            rets += self._decode_batch(
                batch,
                num_sp_master_ranks=num_sp_master_ranks,
                mini_batch_range_list=mini_batch_range_list, mini_batch_size_list=mini_batch_size_list
            )
        
        return True, decode_batch_list, rets
    
    async def _schedule_decode_batch_list(self, decode_batch_list: List[Batch], total_used_tokens_list: np.array, can_use_extra_instances: bool, extra_idle_instances: List[int] = []):
        """
            Schedule a set of decoding batches in an iteration
        """
        assert not can_use_extra_instances or len(extra_idle_instances) == 0

        can_decode_batch_tuple_list: list[Tuple[list[Batch], int]] = []
        cannot_decode_batch_tuple_list: list[Tuple[Batch, int]] = []
        total_idle_tokens = 0
        for batch in decode_batch_list:
            num_instances = len(batch.occupied_instances)
            decode_need_tokens = len(batch.reqs)
            num_used_tokens = np.sum(total_used_tokens_list[batch.occupied_instances])
            idle_tokens = self.max_total_token_num * num_instances - num_used_tokens - decode_need_tokens
            total_idle_tokens += idle_tokens
            if idle_tokens >= 0:
                can_decode_batch_tuple_list.append(([batch], idle_tokens))
            else:
                cannot_decode_batch_tuple_list.append((batch, idle_tokens))
        
        if can_use_extra_instances:
            extra_idle_instances = np.nonzero(total_used_tokens_list == 0)[0].tolist()
        num_extra_idle_instances = len(extra_idle_instances)
        total_idle_tokens += num_extra_idle_instances * self.max_total_token_num
        if total_idle_tokens < 0:
            return False, decode_batch_list, []
        
        # decoding batch starts

        # merge cannot decode batches
        if len(cannot_decode_batch_tuple_list) > 0:
            cannot_decode_batch_tuple_list.sort(key=lambda x: x[1])
            can_decode_batch_tuple_list.sort(key=lambda x: x[1])

            for cannot_decode_batch, cannot_decode_idle_tokens in cannot_decode_batch_tuple_list:
                cannot_decode_batches = [cannot_decode_batch]
                while len(can_decode_batch_tuple_list) > 0:
                    can_decode_batches, can_decode_idle_tokens = can_decode_batch_tuple_list.pop()
                    cannot_decode_batches += can_decode_batches
                    cannot_decode_idle_tokens += can_decode_idle_tokens
                    if cannot_decode_idle_tokens >= 0:
                        can_decode_batch_tuple_list.append((cannot_decode_batches, cannot_decode_idle_tokens))
                        break
                
                if cannot_decode_idle_tokens < 0:
                    assert len(extra_idle_instances) > 0
                    num_used_extra_instances = (-cannot_decode_idle_tokens + self.max_total_token_num - 1) // self.max_total_token_num
                    assert num_used_extra_instances <= num_extra_idle_instances
                    added_instances = extra_idle_instances[:num_used_extra_instances]
                    extra_idle_instances = extra_idle_instances[num_used_extra_instances:]
                    num_extra_idle_instances -= num_used_extra_instances
                    await self._scale_up_batch(cannot_decode_batch, added_instances)
                    cannot_decode_idle_tokens += num_used_extra_instances * self.max_total_token_num
                    can_decode_batch_tuple_list.append((cannot_decode_batches, cannot_decode_idle_tokens))
            
            new_decode_batch_list = []
            for can_decode_batches, _ in can_decode_batch_tuple_list:
                if len(can_decode_batches) > 1:
                    for batch in can_decode_batches[1:]:
                        await self._merge_batch(can_decode_batches[0], batch)
                        can_decode_batches[0].merge(batch)
                new_decode_batch_list.append(can_decode_batches[0])
            decode_batch_list = new_decode_batch_list
                
        # get decode plan
        rets = []
        for batch in decode_batch_list:
            batch.occupied_instances.sort(key=lambda x: total_used_tokens_list[x], reverse=True)

            num_sp_master_ranks = 0
            mini_batch_range_list = []
            mini_batch_size_list = []
            decode_need_tokens = len(batch.reqs)
            busy_instances = []
            idle_instances = []
            removed_instances = []

            start = 0
            idx = 0
            while idx < len(batch.occupied_instances):
                sp_rank = batch.occupied_instances[idx]

                left_tokens = self.max_total_token_num - total_used_tokens_list[sp_rank]
                if left_tokens > 0:
                    num_left_instances = len(batch.occupied_instances) - len(busy_instances) - len(removed_instances) - len(idle_instances)
                    
                    # add extra instances when the decoding batch is compute bound
                    added_instances = []
                    while decode_need_tokens // num_left_instances > self.min_comp_bound_decoding_batch_size and len(extra_idle_instances) > 0:
                        added_instances.append(extra_idle_instances.pop())
                        num_left_instances += 1
                    if len(added_instances) > 0:
                        await self._scale_up_batch(batch, added_instances)
                    
                    mini_batch_size = min(decode_need_tokens, left_tokens, max(decode_need_tokens // num_left_instances, self.min_comp_bound_decoding_batch_size))
                    if mini_batch_size > 0:
                        num_sp_master_ranks += 1
                        mini_batch_size_list.append(mini_batch_size)
                        mini_batch_range_list.append((start, start + mini_batch_size))
                        start += mini_batch_size
                        decode_need_tokens -= mini_batch_size
                        idle_instances.append(sp_rank)
                    elif batch.batch_used_tokens_list[sp_rank] > 0:
                        idle_instances.append(sp_rank)
                    else:
                        removed_instances.append(sp_rank)
                elif batch.batch_used_tokens_list[sp_rank] > 0:
                    busy_instances.append(sp_rank)
                else:
                    removed_instances.append(sp_rank)
                
                idx += 1
            
            assert decode_need_tokens == 0, f"decode_need_tokens: {decode_need_tokens}, num_sp_master_ranks: {num_sp_master_ranks}, mini_batch_size_list: {mini_batch_size_list}, mini_batch_range_list: {mini_batch_range_list}, total_used_tokens_list: {total_used_tokens_list}, batch.batch_used_tokens_list: {batch.batch_used_tokens_list}, batch.occupied_instances: {batch.occupied_instances}, busy_instances: {busy_instances}, idle_instances: {idle_instances}, removed_instances: {removed_instances}\n"
        
            if len(removed_instances) > 0:
                assert False
                await self._scale_down_batch(batch, removed_instances)
            
            batch.occupied_instances = idle_instances + busy_instances

            rets += self._decode_batch(
                batch,
                num_sp_master_ranks=num_sp_master_ranks,
                mini_batch_range_list=mini_batch_range_list, mini_batch_size_list=mini_batch_size_list
            )
        return True, decode_batch_list, rets

    async def _scale_up_batch(self, batch: Batch, added_instances: List[int]):
        reqs = [r.to_empty_rpc_obj() for r in batch.reqs]
        rets = [self.model_rpcs[sp_rank][tp_rank].exposed_add_batch.remote(batch.batch_id, reqs) for sp_rank in added_instances for tp_rank in range(self.tp_world_size)]
        
        if self.disable_scale_up:
            self._scale_up_counter += 1
            print(f"_scale_up_batch, batch_id: {batch.batch_id}, occupied_instances: {batch.occupied_instances}, added_instances: {added_instances}, _scale_up_counter: {self._scale_up_counter}", flush=True)
            import traceback
            traceback.print_stack()
            assert False
        
        batch.occupied_instances += added_instances
        await asyncio.gather(*rets)
        return
    
    async def _scale_down_batch(self, batch: Batch, removed_instances: List[int]):
        rets = [self.model_rpcs[sp_rank][tp_rank].exposed_remove_batch.remote(batch.batch_id) for sp_rank in removed_instances for tp_rank in range(self.tp_world_size)]
        removed_instances = set(removed_instances)
        batch.occupied_instances = [sp_rank for sp_rank in batch.occupied_instances if sp_rank not in removed_instances]
        await asyncio.gather(*rets)
        return

    def _prefill_batch(self, batch:Batch, need_context_migration: bool=False, migration_plan: np.array=None):
        """
            Execute a prefill batch
        """
        num_instances = len(batch.occupied_instances)
        max_token_num = np.sum(((req.input_len + len(req.output_ids) + num_instances - 1) // num_instances for req in batch.reqs))
        rets = []

        async def prefill_batch_wrapper(sp_rank: int, tp_rank: int, batch: Batch, reqs, global_input_kvargs):
            if tp_rank == 0:
                self.stats_tool.on_prompt_batch_start(sp_rank, batch)
            result = await self.model_rpcs[sp_rank][tp_rank].exposed_prefill_batch.remote(batch.batch_id, reqs, global_input_kvargs)
            if tp_rank == 0:
                self.stats_tool.on_prompt_batch_finish(sp_rank, batch)
            return result

        for logical_sp_rank, sp_rank in enumerate(batch.occupied_instances):
            if need_context_migration:
                kv_cache_index_begin, kv_cache_index_end = migration_plan[logical_sp_rank]
                reqs = [r.to_rpc_obj(logical_sp_rank, num_instances, need_context_migration, input_start, input_end) for r, input_start, input_end in zip(batch.reqs, kv_cache_index_begin, kv_cache_index_end)]
            else:
                kv_cache_index_begin, kv_cache_index_end = None, None
                reqs = [r.to_rpc_obj(logical_sp_rank, num_instances, need_context_migration, None, None) for r in batch.reqs]

            input_ids = [req.generate_stripped_input_ids(logical_sp_rank, num_instances) for req in batch.reqs]
            should_decode = {
                req.request_id
                    for req in batch.reqs if (req.input_len + len(req.output_ids) - 1) % num_instances == logical_sp_rank 
                }
            
            for tp_rank in range(self.tp_world_size):
                if tp_rank == 0:
                    global_input_kvargs = (
                        input_ids,                  # "input_ids"
                        max_token_num,              # "max_token_num"
                        batch.occupied_instances,   # "occupied_instances"
                        logical_sp_rank,            # "logical_sp_rank"
                        should_decode,              # "should_decode"
                        need_context_migration,     # "need_context_migration"
                        kv_cache_index_begin,       # "kv_cache_index_begin"
                        kv_cache_index_end,         # "kv_cache_index_end"
                    )
                else:
                    global_input_kvargs = (
                        max_token_num,              # "max_token_num"
                        batch.occupied_instances,   # "occupied_instances"
                        logical_sp_rank,            # "logical_sp_rank"
                        need_context_migration,     # "need_context_migration"
                    )
                rets.append(prefill_batch_wrapper(sp_rank, tp_rank, batch, reqs, global_input_kvargs))
        return rets

    def _decode_batch(self, batch:Batch, num_sp_master_ranks: int, mini_batch_range_list: list[int], mini_batch_size_list: list[int]):
        """
            Execute a decoding batch
        """
        batch_size = len(batch.reqs)
        assert num_sp_master_ranks == len(mini_batch_range_list) == len(mini_batch_size_list)

        total_first_token_global_idx = [req.input_len + len(req.output_ids) - 1 for req in batch.reqs]
        
        async def decoding_batch_wrapper(sp_rank: int, tp_rank: int, batch: Batch, global_input_kvargs):
            if tp_rank == 0:
                self.stats_tool.on_decoding_batch_start(sp_rank, batch)
            result = await self.model_rpcs[sp_rank][tp_rank].exposed_decode_batch.remote(batch.batch_id, global_input_kvargs, cuda_sync_afterwards = self.stats_tool.with_log_trace)
            if tp_rank == 0:
                self.stats_tool.on_decoding_batch_finish(sp_rank, batch)
            return result
        
        rets = []
        for logical_sp_rank, sp_rank in enumerate(batch.occupied_instances):
            if logical_sp_rank < num_sp_master_ranks:
                mini_batch_start, mini_batch_end = mini_batch_range_list[logical_sp_rank]

                input_ids = [
                    req.output_ids[-1] for req in batch.reqs[mini_batch_start:mini_batch_end]
                ]

                first_token_global_idx = total_first_token_global_idx[mini_batch_start:mini_batch_end]
            else:
                input_ids = None
                first_token_global_idx = None

            
            for tp_rank in range(self.tp_world_size):
                if tp_rank == 0:
                    global_input_kvargs = (
                        input_ids,                  # "input_ids"
                        first_token_global_idx,     # "first_token_global_idx"

                        num_sp_master_ranks,        # "num_sp_master_ranks"
                        mini_batch_size_list,       # "mini_batch_size_list"
                        
                        batch.occupied_instances,   # "occupied_instances"
                        logical_sp_rank,            # "logical_sp_rank"
                    )
                else:
                    global_input_kvargs = (
                        num_sp_master_ranks,        # "num_sp_master_ranks"
                        mini_batch_size_list,       # "mini_batch_size_list"
                        
                        batch.occupied_instances,   # "occupied_instances"
                        logical_sp_rank,            # "logical_sp_rank"
                    )
                rets.append(decoding_batch_wrapper(sp_rank, tp_rank, batch, global_input_kvargs))
        return rets

    async def _filter_batch(self, batch: Batch, unfinished_req_ids, finished_req_ids: List):
        rets = [self.model_rpcs[sp_rank][tp_rank].exposed_filter_batch.remote(batch.batch_id, unfinished_req_ids, finished_req_ids) for sp_rank in batch.occupied_instances for tp_rank in range(self.tp_world_size)]
        await asyncio.gather(*rets)
        return

    async def _merge_batch(self, batch1: Batch, batch2: Batch):
        batch1_instance_set = set(batch1.occupied_instances)
        batch2_instance_set = set(batch2.occupied_instances)
        diff_batch2_batch1 = list(batch2_instance_set.difference(batch1_instance_set))
        diff_batch1_batch2 = list(batch1_instance_set.difference(batch2_instance_set))
        if len(diff_batch2_batch1) > 0:
            await self._scale_up_batch(batch1, diff_batch2_batch1)
        if len(diff_batch1_batch2) > 0:
            await self._scale_up_batch(batch2, diff_batch1_batch2)
        rets = [self.model_rpcs[sp_rank][tp_rank].exposed_merge_batch.remote(batch1.batch_id, batch2.batch_id) for sp_rank in batch1.occupied_instances for tp_rank in range(self.tp_world_size)]
        await asyncio.gather(*rets)
        return

    async def _remove_batch(self, batch: Batch):
        rets = [self.model_rpcs[sp_rank][tp_rank].exposed_remove_batch.remote(batch.batch_id) for sp_rank in batch.occupied_instances for tp_rank in range(self.tp_world_size)]
        await asyncio.gather(*rets)
        return
    
    async def _pause_reqs(self, batch: Batch, pasue_reqs: List[Req]):
        pasue_reqs_info = [(r.request_id, r.req_status) for r in pasue_reqs]
        rets = [self.model_rpcs[sp_rank][tp_rank].exposed_pause_reqs.remote(batch.batch_id, pasue_reqs_info) for sp_rank in batch.occupied_instances for tp_rank in range(self.tp_world_size)]
        await asyncio.gather(*rets)
        return

    async def _migrate_batch(self, batch: Batch, request_ids: List[int], b_migration_len: List[int], src_sp_rank: int, dst_sp_rank: int):
        rets = []
        for tp_rank in range(self.tp_world_size):
            rets.append(self.model_rpcs[src_sp_rank][tp_rank].exposed_migrate_batch.remote(batch.batch_id, False, request_ids, b_migration_len, dst_sp_rank))
            rets.append(self.model_rpcs[dst_sp_rank][tp_rank].exposed_migrate_batch.remote(batch.batch_id, True, request_ids, b_migration_len, src_sp_rank))
        await asyncio.gather(*rets)
        return

    async def _handle_finish_req(self, batch: Batch, unfinished_req_ids, finished_req_ids):
        if len(finished_req_ids) != 0:
            if batch.is_clear():
                await self._remove_batch(batch)
                return batch.occupied_instances
            else:
                await self._filter_batch(batch, unfinished_req_ids, finished_req_ids)

                removed_instances = [sp_rank for sp_rank in batch.occupied_instances if batch.batch_used_tokens_list[sp_rank] == 0]
                if len(removed_instances) > 0:
                    await self._scale_down_batch(batch, removed_instances)
                return removed_instances
        return []

    def _filter_running_batch(self, running_batch_list: List[Batch]):
        new_running_batch_list = []
        for batch in running_batch_list:
            if not batch.is_clear():
                new_running_batch_list.append(batch)
        return new_running_batch_list

    
    def _update_out_status_to_batch(self, batch: Batch, req_to_out_status_list: list[Tuple[int, int, dict]]):
        new_batch_used_tokens_list = np.zeros((self.sp_world_size,), dtype=np.int32)
        for logical_sp_rank, sp_rank, req_to_out_status in req_to_out_status_list:
            for req_id, cur_kv_len, new_token_id, new_gen_metadata in req_to_out_status:
                req : Req = batch.id_to_reqs[req_id]
                req.req_status = ReqRunStatus.RUNNING
                req.cur_kv_len_list[sp_rank] = cur_kv_len
                if new_token_id is not None:
                    req.output_ids.append(new_token_id)
                    req.output_metadata_list.append(new_gen_metadata)
        
        for req in batch.reqs:
            used_tokens_list = req.get_used_tokens_list()
            new_batch_used_tokens_list += used_tokens_list
        
        batch.batch_used_tokens_list = new_batch_used_tokens_list
        return

    def _get_total_used_tokens_list(self, batch_list: List[Batch]):
        total_used_tokens_list = np.zeros((self.sp_world_size,), dtype=np.int32)
        for batch in batch_list:
            total_used_tokens_list += batch.batch_used_tokens_list
        return total_used_tokens_list
        
    def _send_to_detokenization_proc(self, batch: Batch, req_ans_list: List[Tuple[int, int, Dict[int, Tuple[ReqRunStatus, int, Optional[int], Optional[dict]]]]]):
        batch_out = BatchTokenIdOut()
        for logical_sp_rank, sp_rank, req_ans in req_ans_list:
            for req_id, cur_kv_len, new_token_id, new_gen_metadata in req_ans:
                if new_token_id is not None:
                    req = batch.id_to_reqs[req_id]
                    batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.finish_status.value))

        self.to_detokenization_processes(batch_out)
        return

    async def loop_for_netio_req(self, recv_req):
        if isinstance(recv_req, tuple) and len(recv_req) == 3:
            prompt_ids, sampling_params, request_id= recv_req
            await self.add_req(prompt_ids, sampling_params, request_id)
        elif isinstance(recv_req, AbortReq):
            abort_req = recv_req
            request_id = abort_req.req_id
            await self.abort(request_id)
            for i in range(self.num_detokenizers):
                self.detokenization_rpcs[i].handle_loop.remote(abort_req)
        else:
            assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for sp_rank in range(self.sp_world_size):
            for model_rpc in self.model_rpcs[sp_rank]:
                model_rpc.rpc_server_process.kill()
        for sp_rank in range(self.sp_world_size):
            for model_rpc in self.model_rpcs[sp_rank]:
                model_rpc.rpc_server_process.join()
        return

def start_router_process(args, router_port, detokenization_port, httpserver_port, model_rpc_ports, pipe_writer):
    try:
        router = RouterManager(
            args,
            router_port=router_port,
            detokenization_port=detokenization_port,
            httpserver_port=httpserver_port,
            model_rpc_ports=model_rpc_ports)
    
        asyncio.run(router.wait_to_model_ready())
    except Exception as e:
        import traceback
        import sys
        etype, evalue, tb = sys.exc_info()
        err_str = '\n'.join(traceback.format_exception(etype, evalue, tb))
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    pipe_writer.send('init ok')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return
