import time
import torch
import numpy as np
import collections

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from loongserve.common.longserve_req_manager import LongServeReqManager
from loongserve.common.longserve_mem_manager import LongServeMemoryManager
from loongserve.utils.infer_utils import mark_start, mark_end
from loongserve.longserve_server.io_struct import ReqRunStatus


requests_mapping = {}

class InferSamplingParams:

    def __init__(
        self,
        do_sample: bool = False,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        vocab_size: int = -1,
    ) -> None:
        self.do_sample = do_sample
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        if self.top_k == -1:
            self.top_k = vocab_size
        return


class InferReq:
    
    def __init__(
        self,
        r_id,
        out_token_id_count={},
        sampling_param: Optional[InferSamplingParams]=None,
        req_idx=-1,
        prompt_len=0,
        req_status=None,
    ) -> None:
        self.r_id = r_id
        self.out_token_id_count = out_token_id_count
        self.sampling_param = sampling_param
        self.req_idx = req_idx
        self.prompt_len = prompt_len
        self.req_status = req_status
        self.cur_kv_len = 0
        return


@dataclass
class InferBatch:
    batch_id: int
    request_ids: List
    req_manager: LongServeReqManager
    
    @classmethod
    @torch.inference_mode()
    def init_batch(cls, batch_id, requests, dtype: torch.dtype, device: torch.device, req_manager:LongServeReqManager, vocab_size: int):

        request_ids = []
        need_alloc_size = len([r for r in requests if r[0] not in requests_mapping])
        nopad_b_req_idx = req_manager.alloc(need_alloc_size)
        nopad_b_req_idx = nopad_b_req_idx.cpu().numpy()
        
        index = 0
        for r in requests:
            # request id -> idx in list mapping
            r_id, prompt_len, sampling_param, req_status = r

            if r_id not in requests_mapping.keys():
                # postprocessor
                sampling_param["vocab_size"] = vocab_size
                assert req_status in (ReqRunStatus.WAIT_IN_QUEUE, ReqRunStatus.PAUSED_AND_OFFLOAD) or (prompt_len == 0 and req_status == ReqRunStatus.RUNNING), f"should not exist {req_status} {prompt_len}"
                r_obj = InferReq(r_id,
                                out_token_id_count=collections.defaultdict(int), 
                                sampling_param=InferSamplingParams(**sampling_param),
                                req_idx=nopad_b_req_idx[index], 
                                prompt_len=prompt_len,
                                req_status=req_status,)
                requests_mapping[r_id] = r_obj
                index += 1
            else:
                if requests_mapping[r_id].req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
                    r_obj : InferReq = requests_mapping[r_id]
                    r_obj.prompt_len = prompt_len
                    r_obj.req_status = ReqRunStatus.RERUNNING_FROM_OFFLOAD
                elif requests_mapping[r_id].req_status == ReqRunStatus.PAUSED_AND_KVKEEP:
                    r_obj : InferReq = requests_mapping[r_id]
                    r_obj.req_status = ReqRunStatus.RERUNNING_FROM_KVKEEP
                else:
                    assert False, f"should not exist {requests_mapping[r_id].req_status}"
            
            request_ids.append(r_id)
            
            r_obj.req_status = ReqRunStatus.RUNNING

        return cls(
            batch_id=batch_id,
            request_ids=request_ids,
            req_manager=req_manager,
        )
    
    @torch.inference_mode()
    def free_self(self):
        free_req_index = []
        free_token_index = []
        for request_id in self.request_ids:
            req : InferReq = requests_mapping.pop(request_id)
            free_req_index.append(req.req_idx)
            free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][:req.cur_kv_len])
            
        free_token_index = torch.cat(free_token_index, dim=-1) if len(free_token_index) > 0 else None
        self.req_manager.free(free_req_index, free_token_index)
        if len(requests_mapping) == 0:
            requests_mapping.clear()
        return
    
    @torch.inference_mode()
    def filter(self, request_ids: List[str], finished_request_ids: List[str]):
        if len(requests_mapping) == 0:
            raise ValueError("Batch must have at least one request")
        if len(request_ids) == len(self):
            return self
        if len(request_ids) == 0:
            self.free_self()
            return InferBatch(
                batch_id=self.batch_id,
                request_ids=[],
                req_manager=self.req_manager
            )
        free_req_index = []
        free_token_index = []
        for request_id in finished_request_ids:
            req : InferReq = requests_mapping.pop(request_id)
            free_req_index.append(req.req_idx)
            free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][:req.cur_kv_len])
        free_token_index = torch.cat(free_token_index, dim=-1)
        self.req_manager.free(free_req_index, free_token_index)
        
        return InferBatch(
            batch_id=self.batch_id,
            request_ids=request_ids,
            req_manager=self.req_manager,
        )

    @torch.inference_mode()
    def pause_reqs(self, pause_reqs: List[str]):
        free_req_index = []
        free_token_index = []
        for request_id, pause_way in pause_reqs:
            req : InferReq = requests_mapping.pop(request_id)
            req.req_status = pause_way
            self.request_ids.remove(request_id)
            if pause_way == ReqRunStatus.PAUSED_AND_OFFLOAD:
                free_req_index.append(req.req_idx)
                free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][:req.cur_kv_len])
                req.cur_kv_len = 0
                req.prompt_len = 0
        free_token_index = torch.cat(free_token_index, dim=-1)
        self.req_manager.free(free_req_index, free_token_index)
        return self

    @classmethod
    @torch.inference_mode()
    def merge(cls, batch1, batch2):
        request_ids = batch1.request_ids + batch2.request_ids
        
        return InferBatch(
            batch_id=batch1.batch_id,
            request_ids=request_ids,
            req_manager=batch1.req_manager,
        )

    def __len__(self):
        return len(self.request_ids)
    
