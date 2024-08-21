import torch

import triton
import triton.language as tl

@triton.jit
def __fwd_kernel_pre_copy_and_register_kv(
    kv_range_begin,     # [batch_size, ]
    kv_range_end,       # [batch_size, ]
    new_kv_cache_len,   # [batch_size, ]

    batch_size,
    kv_cache_index_begin,   # [batch_size, ]
    kv_cache_index_end,     # [batch_size, ]
    kv_first_token_global_idx,  # [batch_size, ]
    num_logical_sp_peers,

    BLOCK_SIZE: tl.constexpr
):
    offs = tl.program_id(0)*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cur_kv_cache_index_begin = tl.load(kv_cache_index_begin + offs, mask=offs < batch_size)
    cur_kv_cache_index_end = tl.load(kv_cache_index_end + offs, mask=offs < batch_size)
    cur_kv_first_token_global_idx = tl.load(kv_first_token_global_idx + offs, mask=offs < batch_size)

    cur_kv_range_begin = tl.cdiv(cur_kv_cache_index_begin - cur_kv_first_token_global_idx, num_logical_sp_peers)
    cur_kv_range_end = tl.cdiv(cur_kv_cache_index_end - cur_kv_first_token_global_idx, num_logical_sp_peers)
    cur_new_kv_cache_len = cur_kv_range_end - cur_kv_range_begin
    
    tl.store(kv_range_begin + offs, cur_kv_range_begin, mask=offs < batch_size)
    tl.store(kv_range_end + offs, cur_kv_range_end, mask=offs < batch_size)
    tl.store(new_kv_cache_len + offs, cur_new_kv_cache_len, mask=offs < batch_size)

@torch.inference_mode()
def pre_copy_and_register_kv(
    kv_cache_index_begin: torch.Tensor,   # [batch_size, ]
    kv_cache_index_end: torch.Tensor,     # [batch_size, ]
    kv_first_token_global_idx: torch.Tensor,  # [batch_size, ]
    num_logical_sp_peers: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Equivalent to the following python code:

        kv_range_begin = (infer_state.kv_cache_index_begin - kv_first_token_global_idx + (num_logical_sp_peers - 1)) // num_logical_sp_peers
        kv_range_end = (infer_state.kv_cache_index_end - kv_first_token_global_idx + (num_logical_sp_peers - 1)) // num_logical_sp_peers
        new_kv_cache_len = kv_range_end - kv_range_begin
    """
    batch_size = kv_cache_index_begin.shape[0]
    kv_range_begin = torch.empty_like(kv_cache_index_begin)
    kv_range_end = torch.empty_like(kv_cache_index_begin)
    new_kv_cache_len = torch.empty_like(kv_cache_index_begin)

    BLOCK_SIZE = 64
    grid = ((batch_size+BLOCK_SIZE-1)//BLOCK_SIZE, )
    __fwd_kernel_pre_copy_and_register_kv[grid](
        kv_range_begin, kv_range_end, new_kv_cache_len,
        batch_size,
        kv_cache_index_begin, kv_cache_index_end, kv_first_token_global_idx, num_logical_sp_peers,
        BLOCK_SIZE
    )

    return (kv_range_begin, kv_range_end, new_kv_cache_len)
    
@triton.jit
def __fwd_kernel_destindex_copy_and_register_kv(
    new_kv_cache_len,       # [batch_size,]
    kv_range_begin,         # [batch_size,]
    kv_range_end,           # [batch_size,]
    new_kv_cache_len_sum,   # [batch_size,]
    mem_index,              # [alloc_token_num,]
    kv_b_start_loc,         # [batch_size,]
    kv,                     # [max_token_num, num_head, head_dim]
    total_kv_cache,         # [_, num_head, head_dim]
    b_req_idx,              # [batch_size,]
    cur_kv_cache_index,     # [batch_size,]
    req_to_token_indexs,    # [max_request_num, max_token_num]
    stride_kv_bs, stride_kv_h, stride_kv_d,
    stride_total_kv_cache_bs, stride_total_kv_cache_h, stride_total_kv_cache_d,
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    num_used_mem_index,
    head_num,
    should_register_kv: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    loop_start = tl.program_id(1) * BLOCK_N

    cur_new_kv_cache_len = tl.load(new_kv_cache_len + cur_batch)
    cur_kv_range_begin = tl.load(kv_range_begin + cur_batch)
    cur_kv_range_end = tl.load(kv_range_end + cur_batch)

    if (cur_new_kv_cache_len <= 0 or loop_start >= cur_new_kv_cache_len) or (cur_kv_range_begin < 0 or cur_kv_range_end < 0):
        return

    cur_mem_index_start = tl.load(new_kv_cache_len_sum + cur_batch - 1, mask=cur_batch>0, other=0) + num_used_mem_index
    cur_mem_index_ptr = mem_index + cur_mem_index_start

    cur_kv_b_start_loc = tl.load(kv_b_start_loc + cur_batch)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    stride_kv_bs = stride_kv_bs.to(tl.int64)
    stride_total_kv_cache_bs = stride_total_kv_cache_bs.to(tl.int64)
    cur_kv_ptrs = kv + (cur_kv_b_start_loc + cur_kv_range_begin) * stride_kv_bs + offs_h[:, None] * stride_kv_h + offs_d[None, :] * stride_kv_d

    total_kv_cache_ptrs = total_kv_cache + offs_h[:, None] * stride_total_kv_cache_h + offs_d[None, :] * stride_total_kv_cache_d

    if should_register_kv:
        cur_b_req_idx = tl.load(b_req_idx + cur_batch)
        cur_kv_cache_index_start = tl.load(cur_kv_cache_index + cur_batch)
        req_to_token_indexs_ptr = req_to_token_indexs + cur_b_req_idx * stride_req_to_tokens_b + cur_kv_cache_index_start * stride_req_to_tokens_s

    loop_end = tl.where(loop_start + BLOCK_N < cur_new_kv_cache_len, loop_start + BLOCK_N, cur_new_kv_cache_len)
    for start_n in range(loop_start, loop_end):
        cur_kv = tl.load(cur_kv_ptrs + start_n * stride_kv_bs, mask=offs_h[:, None] < head_num, other=0.0)
        cur_mem_index = tl.load(cur_mem_index_ptr + start_n)
        
        tl.store(total_kv_cache_ptrs + cur_mem_index * stride_total_kv_cache_bs, cur_kv, mask=offs_h[:, None] < head_num)
        if should_register_kv:
            tl.store(req_to_token_indexs_ptr + start_n * stride_req_to_tokens_s, cur_mem_index)
    
@torch.inference_mode()
def destindex_copy_and_register_kv(
    batch_size: int,
    new_kv_cache_len: torch.Tensor,
    kv_range_begin: torch.Tensor,
    kv_range_end: torch.Tensor,
    new_kv_cache_len_sum: torch.Tensor,
    kv: torch.Tensor,
    kv_b_start_loc: torch.Tensor,
    cur_kv_cache_index: torch.Tensor,
    mem_index: torch.Tensor,
    total_kv_cache: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    b_req_idx: torch.Tensor,
    num_used_mem_index: int,
    max_len_in_batch: int,
    should_register_kv: bool
):  
    assert new_kv_cache_len.shape[0] == kv_range_begin.shape[0] == kv_range_end.shape[0] == new_kv_cache_len_sum.shape[0] == kv_b_start_loc.shape[0] == b_req_idx.shape[0] == cur_kv_cache_index.shape[0] == batch_size
    assert kv.shape[1] == total_kv_cache.shape[1] and kv.shape[2] == total_kv_cache.shape[2]
    head_num = kv.shape[1]
    head_dim = kv.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(head_num)

    BLOCK_N = 256

    grid = (batch_size, (max_len_in_batch + BLOCK_N - 1) // BLOCK_N)
    __fwd_kernel_destindex_copy_and_register_kv[grid](
        new_kv_cache_len,
        kv_range_begin,
        kv_range_end,
        new_kv_cache_len_sum,
        mem_index,
        kv_b_start_loc,
        kv,
        total_kv_cache,
        b_req_idx,
        cur_kv_cache_index,
        req_to_token_indexs,
        kv.stride(0), kv.stride(1), kv.stride(2),
        total_kv_cache.stride(0), total_kv_cache.stride(1), total_kv_cache.stride(2),
        req_to_token_indexs.stride(0), req_to_token_indexs.stride(1),
        num_used_mem_index,
        head_num,
        should_register_kv,
        BLOCK_HEAD,
        BLOCK_DMODEL=head_dim,
        BLOCK_N=BLOCK_N,
    )