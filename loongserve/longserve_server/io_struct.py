from .sampling_params import SamplingParams
from typing import Dict, List, Optional, Tuple, Union
import asyncio
import enum
import numpy as np

class ReqRunStatus(enum.Enum):
    WAIT_IN_QUEUE = 0
    RUNNING = 1
    PAUSED_AND_KVKEEP = 2
    PAUSED_AND_OFFLOAD = 3
    RERUNNING_FROM_KVKEEP = 4
    RERUNNING_FROM_OFFLOAD = 5

class FinishStatus(enum.Enum):
    NO_FINISH = 0
    FINISHED_STOP = 1
    FINISHED_LENGTH = 2
    FINISHED_ABORT = 3

    def is_finished(self):
        return 1 <= self.value <= 3
    
    def is_aborted(self):
        return self == FinishStatus.FINISHED_ABORT

    def get_finish_reason(self):
        if self == FinishStatus.FINISHED_STOP:
            finish_reason = "stop"
        elif self == FinishStatus.FINISHED_LENGTH:
            finish_reason = "length"
        elif self == FinishStatus.FINISHED_ABORT:
            finish_reason = "abort"
        else:
            finish_reason = None
        return finish_reason

class Req:
    def __init__(self, request_id, prompt_ids, sample_params: SamplingParams, sp_world_size: int):
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.input_len = len(prompt_ids)
        self.max_output_len = sample_params.max_new_tokens
        self.sample_params = sample_params
        self.sp_world_size = sp_world_size
        self.output_ids = []
        self.output_metadata_list = []

        self.req_status = ReqRunStatus.WAIT_IN_QUEUE
        self.finish_status = FinishStatus.NO_FINISH
        self.cur_kv_len_list = np.zeros((self.sp_world_size,), dtype=np.int32)
        return
    
    def to_rpc_obj(self, logical_sp_rank: int, num_instances: int, need_context_migration: bool, input_start: int, input_end: int):
        if need_context_migration:
            prompt_len = input_end - input_start
        else:
            prompt_len = (self.input_len + len(self.output_ids) - logical_sp_rank + num_instances - 1) // num_instances
        return (self.request_id,                # "request_id"
                prompt_len,                     # "prompt_len"
                self.sample_params.to_dict(),   # "sampling_param"
                self.req_status,                # "req_status"
            )
    
    def to_empty_rpc_obj(self):
        return (self.request_id,                # "request_id"
                0,                              # "prompt_len"
                self.sample_params.to_dict(),   # "sampling_param"
                self.req_status,                # "req_status"
            )
    
    def generate_ranged_input_ids(self, input_start: int, input_end: int):
        ranged_input_ids = self.prompt_ids[input_start:min(input_end, self.input_len)] if input_start < self.input_len else []
        ranged_input_ids += self.output_ids[max(input_start-self.input_len, 0):input_end-self.input_len] if input_end >= self.input_len else []
        assert len(ranged_input_ids) == input_end - input_start
        return ranged_input_ids

    def generate_stripped_input_ids(self, input_start: int, input_step: int):
        output_start = (input_start + input_step - self.input_len % input_step) % input_step
        stripped_input_ids = self.prompt_ids[input_start::input_step] + self.output_ids[output_start::input_step]
        return stripped_input_ids
    
    def to_req_detokenization_state(self):
        out = ReqDetokenizationState(self.request_id, self.prompt_ids, self.max_output_len, self.sample_params.ignore_eos)
        return out
    
    def stop_sequences_matched(self):
        for stop_token_ids in self.sample_params.stop_sequences:
            stop_len = len(stop_token_ids)
            if stop_len > 0:
                if len(self.output_ids) >= stop_len:
                    if all(self.output_ids[-(stop_len - i)] == stop_token_ids[i] for i in range(stop_len)):
                        return True
        return False

    def __repr__(self):
        return (f"(request_id(n={self.request_id}, "
                f"prompt_ids={self.prompt_ids}, "
                f"seq_len={self.input_len + len(self.output_ids)}, "
                f"cur_kv_len_list={self.cur_kv_len_list}) ")
    
    def get_used_tokens_list(self):
        return np.maximum(0, self.cur_kv_len_list)

    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        raise Exception("need to impl")
    
    def get_first_router_need_tokens(self):
        raise Exception("need to impl")

class NormalReq(Req):
    def __init__(self, request_id, prompt_ids, sample_params: SamplingParams, sp_world_size: int):
        super().__init__(request_id, prompt_ids, sample_params, sp_world_size)
        return
    
    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        """
        普通continues batch调度模式, 先prefill 后 decode 的估计方式 的实现
        """
        has_out_len = len(self.output_ids)
        if self.sample_params.ignore_eos:
            cur_max_new_token_len = self.max_output_len
        elif is_busy:
            cur_max_new_token_len = self.max_output_len
        else:
            cur_max_new_token_len = min(self.max_output_len, max(int(1.1 * has_out_len), router_max_new_token_len))

        if self.req_status == ReqRunStatus.RUNNING:
            return (self.input_len + has_out_len, max(0, cur_max_new_token_len - has_out_len - 1))
        elif self.req_status == ReqRunStatus.WAIT_IN_QUEUE:
            return (self.input_len + 1,  max(0, cur_max_new_token_len - 1 - 1))
        elif self.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
            return (self.input_len + has_out_len + 1, max(0, cur_max_new_token_len - has_out_len - 1 - 1))
        elif self.req_status == ReqRunStatus.PAUSED_AND_KVKEEP:
            return (self.input_len + has_out_len, max(0, cur_max_new_token_len - has_out_len - 1))
        else:
            assert False, "error state"
        return
    
    def get_first_router_need_tokens(self):
        if self.req_status == ReqRunStatus.WAIT_IN_QUEUE:
            return self.input_len
        elif self.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
            return self.input_len + len(self.output_ids)
        elif self.req_status == ReqRunStatus.PAUSED_AND_KVKEEP:
            return 0
        else:
            assert False, f"error state: {self.req_status}"

class ReqDetokenizationState:
    def __init__(
        self,
        request_id: str,
        prompt_ids: List[int],
        max_output_len: int,
        ignore_eos: bool,
    ) -> None:
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.output_ids = []
        self.output_tokens = []
        self.output_str = ""
        self.sub_texts = []
        self.current_sub_text = []
        self.max_output_len = max_output_len
        self.ignore_eos = ignore_eos
        self.gen_metadata = {}

class Batch:
    def __init__(self, batch_id, reqs: List[Req], sp_world_size: int, occupied_instances: List[int]):
        self.batch_id = batch_id
        self.reqs = reqs
        self.id_to_reqs = {req.request_id: req for req in reqs}
        self.sp_world_size = sp_world_size
        self.occupied_instances = occupied_instances

        self.batch_used_tokens_list = np.zeros((self.sp_world_size,), dtype=np.int32)
        for req in self.reqs:
            self.batch_used_tokens_list += req.get_used_tokens_list()
        return

    def input_tokens(self):
        batch_input_tokens = 0
        for req in self.reqs:
            batch_input_tokens += req.input_len
        return batch_input_tokens

    def mark_and_get_finished_req_and_preupdate_status(self, eos_id):
        unfinished_req_ids, finished_req_ids, sum_finished_req_output_len = [], [], 0
        for req in self.reqs:
            if req.stop_sequences_matched():
                req.finish_status = FinishStatus.FINISHED_STOP
            elif len(req.output_ids) >= 1 and req.output_ids[-1] == eos_id and req.sample_params.ignore_eos is False:
                req.finish_status = FinishStatus.FINISHED_STOP
            elif len(req.output_ids) >= req.max_output_len:
                req.finish_status = FinishStatus.FINISHED_LENGTH

            if req.finish_status.is_finished():
                sum_finished_req_output_len += len(req.output_ids)
                finished_req_ids.append(req.request_id)
                self.batch_used_tokens_list -= req.get_used_tokens_list()
            else:
                unfinished_req_ids.append(req.request_id)
    
        return unfinished_req_ids, finished_req_ids, sum_finished_req_output_len
    
    def filter_out_finished_req(self, unfinished_req_ids, finished_req_ids):
        # update batch
        if len(finished_req_ids) != 0:
            self.reqs = [self.id_to_reqs[req_id] for req_id in unfinished_req_ids]
            self.id_to_reqs = {req.request_id: req for req in self.reqs}
        return
    
    def pop_req(self, req_id):
        self.reqs = [req for req in self.reqs if req.request_id != req_id]
        req = self.id_to_reqs[req_id]
        self.id_to_reqs.pop(req_id)
        self.batch_used_tokens_list -= req.get_used_tokens_list()
        return

    def is_clear(self):
        return len(self.reqs) == 0

    def merge(self, mini_batch):
        for _req in mini_batch.reqs:
            self.reqs.append(_req)
        self.id_to_reqs = {req.request_id: req for req in self.reqs}
        self.batch_used_tokens_list += mini_batch.batch_used_tokens_list
        return

    def __repr__(self):
        return (f"(batch_id={self.batch_id}, "
                f"reqs=({self.reqs}), "
                f"occupied_instances={self.occupied_instances}, "
                f"batch_used_tokens_list={self.batch_used_tokens_list}) ")
        
class BatchTokenIdOut:
    def __init__(self):
        self.reqs_infs: List[Tuple[str, int, Dict, int]] = []  # [req_id, new_token_id, gen_metadata, finish_status]

class BatchStrOut:
    def __init__(self):
        self.reqs_infs: List[Tuple[str, str, Dict, int]] = [] # [req_id, token_str, gen_metadata, finish_status]
        
class AbortReq:
    def __init__(self, req_id):
        self.req_id = req_id
        
