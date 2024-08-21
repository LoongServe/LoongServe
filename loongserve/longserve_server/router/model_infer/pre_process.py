import torch
import torch.distributed as dist
import numpy as np
from .infer_batch import requests_mapping, InferReq, InferBatch
from loongserve.longserve_server.io_struct import ReqRunStatus
from loongserve.utils.infer_utils import calculate_time

#@calculate_time(show=True, min_cost_ms=1)
def prepare_prefill_inputs(batch: InferBatch, global_input_kvargs: tuple, worker):
    run_reqs, not_run_reqs = [], []

    is_driver_worker = (worker.tp_rank == 0)
    need_broadcast = worker.tp_world_size > 1
    if is_driver_worker:
        input_ids, max_token_num, occupied_instances, logical_sp_rank, should_decode, need_context_migration, nopad_kv_cache_index_begin, nopad_kv_cache_index_end = global_input_kvargs
    else:
        max_token_num, occupied_instances, logical_sp_rank, need_context_migration = global_input_kvargs

    logical_sp_peer_ranks = [sp_i * worker.tp_world_size + worker.tp_rank for sp_i in occupied_instances]

    nopad_b_req_idx = []
    if is_driver_worker:
        start_loc = 0
        nopad_b_start_loc = []
        nopad_b_seq_len = []
        nopad_input_ids = []

    for i, request_id in enumerate(batch.request_ids):
        req : InferReq = requests_mapping[request_id]
        assert req.req_status == ReqRunStatus.RUNNING
        if req.cur_kv_len != 0: 
            not_run_reqs.append(req)
            continue
        
        run_reqs.append(req)
        nopad_b_req_idx.append(req.req_idx)

        if is_driver_worker:
            nopad_b_start_loc.append(start_loc)
            
            input_id = input_ids[i]
            seq_len = len(input_id)
            if seq_len > 0:
                nopad_input_ids.append(input_id)
            
            nopad_b_seq_len.append(seq_len)
            start_loc += seq_len
    
    if len(run_reqs) >= 1:
        nopad_batch_size = len(run_reqs)
        nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device='cuda')

        src_rank = worker.sp_rank * worker.tp_world_size
        if is_driver_worker:
            nopad_input_ids = np.concatenate(nopad_input_ids, dtype=np.int64)
            nopad_input_ids = torch.tensor(nopad_input_ids, dtype=torch.int64, device='cuda')
            nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device='cuda')
            nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device='cuda')

            if need_broadcast:
                dist.broadcast(nopad_b_seq_len, src=src_rank, group=worker.tp_group, async_op=False)
            nopad_total_token_num = torch.sum(nopad_b_seq_len).item()
        else:
            nopad_b_start_loc = torch.empty((nopad_batch_size,), dtype=torch.int32, device='cuda')
            nopad_b_seq_len = torch.empty((nopad_batch_size,), dtype=torch.int32, device='cuda')

            if need_broadcast:
                dist.broadcast(nopad_b_seq_len, src=src_rank, group=worker.tp_group, async_op=False)
            nopad_total_token_num = torch.sum(nopad_b_seq_len).item()

            nopad_input_ids = torch.empty((nopad_total_token_num,), dtype=torch.int64, device='cuda')
        
        if need_broadcast:
            dist.broadcast(nopad_input_ids, src=src_rank, group=worker.tp_group, async_op=False)
            dist.broadcast(nopad_b_start_loc, src=src_rank, group=worker.tp_group, async_op=False)

        nopad_max_len_in_batch = torch.max(nopad_b_seq_len).item()

        nopad_first_token_global_idx = torch.tensor([logical_sp_rank], dtype=torch.int32, device="cuda").repeat(nopad_batch_size)

        if need_context_migration:
            if is_driver_worker:
                nopad_kv_cache_index_begin = torch.tensor(nopad_kv_cache_index_begin, dtype=torch.int32, device="cuda")
                nopad_kv_cache_index_end = torch.tensor(nopad_kv_cache_index_end, dtype=torch.int32, device="cuda")
            else:
                nopad_kv_cache_index_begin = torch.empty((nopad_batch_size,), dtype=torch.int32, device="cuda")
                nopad_kv_cache_index_end = torch.empty((nopad_batch_size,), dtype=torch.int32, device="cuda")
            
            if need_broadcast:
                dist.broadcast(nopad_kv_cache_index_begin, src=src_rank, group=worker.tp_group, async_op=False)
                dist.broadcast(nopad_kv_cache_index_end, src=src_rank, group=worker.tp_group, async_op=False)

        else:
            nopad_kv_cache_index_begin, nopad_kv_cache_index_end = None, None

        kwargs = {
            "batch_size": nopad_batch_size,
            "total_token_num": nopad_total_token_num,
            "max_token_num": max_token_num,
            "max_len_in_batch": nopad_max_len_in_batch,
            "input_ids": nopad_input_ids,
            "b_req_idx": nopad_b_req_idx,
            "b_start_loc": nopad_b_start_loc,
            "b_seq_len": nopad_b_seq_len,
            "first_token_global_idx": nopad_first_token_global_idx,
            "sp_master_rank": None,
            "logical_sp_peer_ranks": logical_sp_peer_ranks,
            "logical_sp_rank": logical_sp_rank,
            "newkv_alloc_sp_rank": None,
            "peer_sp_master_rank_list": None,
            "peer_query_buffer_range_list": None,
            "peer_batch_size": None,
            "peer_max_len_in_batch": None,
            "peer_b_req_idx": None,
            "peer_b_seq_len": None,
            "need_context_migration": need_context_migration,
            "kv_cache_index_begin": nopad_kv_cache_index_begin,
            "kv_cache_index_end": nopad_kv_cache_index_end,
            "multimodal_params": None,
            "is_prefill": True,
        }
        return kwargs, run_reqs, not_run_reqs
    else:
        return {}, run_reqs, not_run_reqs
    
def prepare_decode_inputs(batch:InferBatch, global_input_kvargs:dict, worker):
    run_reqs, peer_run_reqs, not_run_reqs = [], [], []

    is_driver_worker = (worker.tp_rank == 0)
    need_broadcast = worker.tp_world_size > 1
    if is_driver_worker:
        input_ids, first_token_global_idx, num_sp_master_ranks, mini_batch_size_list, occupied_instances, logical_sp_rank = global_input_kvargs
    else:
        num_sp_master_ranks, mini_batch_size_list, occupied_instances, logical_sp_rank = global_input_kvargs

    logical_sp_peer_ranks = [sp_i * worker.tp_world_size + worker.tp_rank for sp_i in occupied_instances]

    start = 0
    mini_batch_range_list = []
    for mini_batch_size in mini_batch_size_list:
        end = start + mini_batch_size
        mini_batch_range_list.append((start, end))
        start = end
    
    is_master_rank = (logical_sp_rank < num_sp_master_ranks)
    mini_batch_size = mini_batch_size_list[logical_sp_rank] if is_master_rank else 0
    mini_batch_start, mini_batch_end = mini_batch_range_list[logical_sp_rank] if is_master_rank else (len(batch.request_ids), len(batch.request_ids))

    master_request_ids = batch.request_ids[mini_batch_start:mini_batch_end] if is_master_rank else None
    peer_request_ids = batch.request_ids[:mini_batch_start] + batch.request_ids[mini_batch_end:] if is_master_rank else batch.request_ids

    if master_request_ids is not None:
        nopad_batch_size = len(master_request_ids)
        nopad_total_token_num = len(master_request_ids)
        nopad_b_req_idx = []
        nopad_b_seq_len = []

        for request_id in master_request_ids:
            req : InferReq = requests_mapping[request_id]
            assert req.req_status == ReqRunStatus.RUNNING
            run_reqs.append(req)
            nopad_b_req_idx.append(req.req_idx)
            req.prompt_len += 1
            seq_len = req.prompt_len
            assert req.cur_kv_len == seq_len - 1, f"{request_id}: {req.cur_kv_len} != {seq_len - 1}"
            nopad_b_seq_len.append(seq_len)
        
        if is_driver_worker:
            input_ids = torch.tensor(input_ids, dtype=torch.int64, device='cuda')
            nopad_first_token_global_idx = torch.tensor(first_token_global_idx, dtype=torch.int32, device="cuda")
        else:
            input_ids = torch.empty((nopad_batch_size,), dtype=torch.int64, device='cuda')
            nopad_first_token_global_idx = torch.empty((nopad_batch_size,), dtype=torch.int32, device="cuda")

        if need_broadcast:
            src_rank = worker.sp_rank * worker.tp_world_size
            dist.broadcast(input_ids, src=src_rank, group=worker.tp_group, async_op=False)
            dist.broadcast(nopad_first_token_global_idx, src=src_rank, group=worker.tp_group, async_op=False)

        nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device='cuda')
        nopad_b_start_loc = None
        nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device='cuda')
        nopad_max_len_in_batch = torch.max(nopad_b_seq_len).item()
    else:
        nopad_batch_size = 0
        nopad_total_token_num = None
        nopad_max_len_in_batch = 0
        input_ids = None
        nopad_b_req_idx = None
        nopad_b_start_loc = None
        nopad_b_seq_len = None
        nopad_first_token_global_idx = None
    
    if peer_request_ids is not None and len(peer_request_ids) > 0:
        nopad_peer_batch_size = len(peer_request_ids)
        nopad_peer_b_req_idx = []
        nopad_peer_b_seq_len = []
        for request_id in peer_request_ids:
            req: InferReq = requests_mapping[request_id]
            assert req.req_status == ReqRunStatus.RUNNING
            peer_run_reqs.append(req)
            nopad_peer_b_req_idx.append(req.req_idx)
            seq_len = req.prompt_len
            assert req.cur_kv_len == seq_len, f"{req.cur_kv_len} != {seq_len}"
            nopad_peer_b_seq_len.append(seq_len)
        
        nopad_peer_b_req_idx = torch.tensor(nopad_peer_b_req_idx, dtype=torch.int32, device='cuda')
        nopad_peer_b_seq_len = torch.tensor(nopad_peer_b_seq_len, dtype=torch.int32, device='cuda')
        nopad_peer_max_len_in_batch = torch.max(nopad_peer_b_seq_len).item()
    else:
        nopad_peer_batch_size = None
        nopad_peer_max_len_in_batch = 0
        nopad_peer_b_req_idx = None
        nopad_peer_b_seq_len = None
    
    assert len(run_reqs) + len(peer_run_reqs) == len(batch)
    
    if len(run_reqs) + len(peer_run_reqs) >= 1:
        sp_master_rank = logical_sp_peer_ranks[logical_sp_rank] if is_master_rank else None

        peer_sp_master_rank_list = [sp_i for logical_sp_i, sp_i in enumerate(logical_sp_peer_ranks[:num_sp_master_ranks]) if logical_sp_i != logical_sp_rank]
        if len(peer_sp_master_rank_list) == 0:
            peer_sp_master_rank_list = None
        peer_query_buffer_range_list = [
                (
                    mini_batch_range_list[logical_sp_i][0] - (mini_batch_size if logical_sp_i > logical_sp_rank else 0),
                    mini_batch_range_list[logical_sp_i][1] - (mini_batch_size if logical_sp_i > logical_sp_rank else 0)
                )
                    for logical_sp_i in range(num_sp_master_ranks) if logical_sp_i != logical_sp_rank
        ]

        kwargs = {
            "batch_size": nopad_batch_size,
            "total_token_num": nopad_total_token_num,
            "max_token_num": None,
            "max_len_in_batch": nopad_max_len_in_batch,
            "input_ids": input_ids,
            "b_req_idx": nopad_b_req_idx,
            "b_start_loc": nopad_b_start_loc,
            "b_seq_len": nopad_b_seq_len,
            "first_token_global_idx": nopad_first_token_global_idx,
            "sp_master_rank": sp_master_rank,
            "logical_sp_peer_ranks": logical_sp_peer_ranks,
            "logical_sp_rank": logical_sp_rank,
            "newkv_alloc_sp_rank": None,
            "peer_sp_master_rank_list": peer_sp_master_rank_list,
            "peer_query_buffer_range_list": peer_query_buffer_range_list,
            "peer_batch_size": nopad_peer_batch_size,
            "peer_max_len_in_batch": nopad_peer_max_len_in_batch,
            "peer_b_req_idx": nopad_peer_b_req_idx,
            "peer_b_seq_len": nopad_peer_b_seq_len,
            "need_context_migration": False,
            "kv_cache_index_begin": None,
            "kv_cache_index_end": None,
            "multimodal_params": None,
            "is_prefill": False            
        }
        return kwargs, run_reqs, peer_run_reqs, not_run_reqs
    else:
        return {}, run_reqs, peer_run_reqs, not_run_reqs