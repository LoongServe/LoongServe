import torch
import triton
import triton.language as tl

@triton.jit()
def _fwd_sender_kernel(
	send_buf,	# [num_layers, num_tokens_sum, 2*head_num, head_dim]
	stride_buf_layer,	# Not marked as tl.constexpr to avoid re-compile
	stride_buf_token,	# Not marked as tl.constexpr to avoid re-compile
	stride_buf_head: tl.constexpr,
	stride_buf_headdim: tl.constexpr,
	req_to_token_indexes,	# [max_request_num, max_sequence_length]
	stride_rtti_reqid: tl.constexpr,
	stride_rtti_tokenidx: tl.constexpr,
	mem_state,		# [memory_manager_size]

	kv_cache,		# [num_layers, kvcache_size, 2*head_num, head_dim]
	stride_kvc_layer,	# Not marked as tl.constexpr to be converted to tl.int64
	stride_kvc_token: tl.constexpr,
	stride_kvc_head: tl.constexpr,
	stride_kvc_headdim: tl.constexpr,

	request_ids,	# [batch_size]
	num_tokens,		# [batch_size]
	num_tokens_cumsum,	# [batch_size]
	b_seq_len,		# [batch_size]

	num_layers: tl.constexpr,
	kvcache_size: tl.constexpr,
	num_heads: tl.constexpr,
	head_dim: tl.constexpr
):
	"""
	The Triton kernel for the sender during decoding stage migration

	This kernel performs the following jobs:
	- Set mem_state of migrated tokens to 0
	- Gather K/Vs from kv_buffer to send_buf

	grid: (batch, migrating_token_idx, 2*num_heads)
	"""
	batch_idx = tl.program_id(0)
	migrating_token_idx = tl.program_id(1)
	cur_head = tl.program_id(2)

	cur_num_tokens = tl.load(num_tokens + batch_idx)
	if migrating_token_idx >= cur_num_tokens:
		return
	
	cur_b_seq_len = tl.load(b_seq_len + batch_idx)
	cur_token_idx_in_req = cur_b_seq_len - cur_num_tokens + migrating_token_idx

	cur_request_id = tl.load(request_ids + batch_idx)
	cur_token_idx_in_kv_cache = tl.load(req_to_token_indexes + cur_request_id*stride_rtti_reqid + cur_token_idx_in_req*stride_rtti_tokenidx)
	cur_token_idx_in_send_buf = tl.load(num_tokens_cumsum + batch_idx - 1, mask=batch_idx>0, other=0) + migrating_token_idx
	cur_token_idx_in_kv_cache = cur_token_idx_in_kv_cache.to(tl.int64)
	cur_token_idx_in_send_buf = cur_token_idx_in_send_buf.to(tl.int64)

	tl.store(mem_state + cur_token_idx_in_kv_cache, 0)

	stride_kvc_layer = stride_kvc_layer.to(tl.int64)
	stride_buf_layer = stride_buf_layer.to(tl.int64)
	kvc_ptrs = kv_cache + cur_token_idx_in_kv_cache*stride_kvc_token + cur_head*stride_kvc_head + tl.arange(0, head_dim)*stride_kvc_headdim
	send_buf_ptrs = send_buf + cur_token_idx_in_send_buf*stride_buf_token + cur_head*stride_buf_head + tl.arange(0, head_dim)*stride_buf_headdim
	for layer in tl.static_range(num_layers):
		# Need a for-loop here since tl.arange() only accepts power-of-two
		cur_kvc_ptrs = kvc_ptrs + layer*stride_kvc_layer
		cur_send_buf_ptrs = send_buf_ptrs + layer*stride_buf_layer
		tl.store(cur_send_buf_ptrs, tl.load(cur_kvc_ptrs))

def decoding_stage_migration_sender_kernel(
	send_buf: torch.Tensor,
	req_to_token_indexes: torch.Tensor,
	mem_state: torch.Tensor,
	kv_cache: torch.Tensor,
	request_ids: torch.Tensor,
	num_tokens: torch.Tensor,
	b_seq_len: torch.Tensor
):
	num_tokens_max = torch.max(num_tokens)
	num_tokens_cumsum = torch.cumsum(num_tokens, dim=0)
	batch_size = request_ids.shape[0]
	num_layers = kv_cache.shape[0]
	kvcache_size = kv_cache.shape[1]
	num_heads = kv_cache.shape[2] // 2
	head_dim = kv_cache.shape[3]

	_fwd_sender_kernel[(batch_size, num_tokens_max, 2*num_heads)](
		send_buf, send_buf.stride(0), send_buf.stride(1), send_buf.stride(2), send_buf.stride(3),
		req_to_token_indexes, req_to_token_indexes.stride(0), req_to_token_indexes.stride(1),
		mem_state,
		kv_cache, kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), kv_cache.stride(3),
		request_ids, num_tokens, num_tokens_cumsum, b_seq_len,
		num_layers, kvcache_size, num_heads, head_dim
	)
	
@triton.jit()
def _fwd_receiver_kernel(
	recv_buf,	# [num_layers, num_tokens_sum, 2*head_num, head_dim]
	stride_buf_layer,	# Not marked as tl.constexpr to avoid re-compile
	stride_buf_token,	# Not marked as tl.constexpr to avoid re-compile
	stride_buf_head: tl.constexpr,
	stride_buf_headdim: tl.constexpr,
	req_to_token_indexes,	# [max_request_num, max_sequence_length]
	stride_rtti_reqid: tl.constexpr,
	stride_rtti_tokenidx: tl.constexpr,
	kv_cache,		# [num_layers, kvcache_size, 2*head_num, head_dim]
	stride_kvc_layer,	# Not marked as tl.constexpr to be converted to tl.int64
	stride_kvc_token: tl.constexpr,
	stride_kvc_head: tl.constexpr,
	stride_kvc_headdim: tl.constexpr,
	alloc_mem,		# [num_tokens_sum]

	request_ids,	# [batch_size]
	num_tokens,		# [batch_size]
	num_tokens_cumsum,	# [batch_size]
	b_seq_len,		# [batch_size]

	num_layers: tl.constexpr,
	kvcache_size: tl.constexpr,
	num_heads: tl.constexpr,
	head_dim: tl.constexpr
):
	"""
	The Triton kernel for the receiver during decoding stage migration

	This kernel performs the following jobs:
	- Save recv_buf to kv_cache
	- Modify req_to_token_indexes

	grid: (batch, migrating_token_idx, 2*num_heads)
	"""
	batch_idx = tl.program_id(0)
	migrating_token_idx = tl.program_id(1)
	cur_head = tl.program_id(2)

	cur_num_tokens = tl.load(num_tokens + batch_idx)
	if migrating_token_idx >= cur_num_tokens:
		return
	
	cur_b_seq_len = tl.load(b_seq_len + batch_idx)
	cur_token_idx_in_recv_buf = tl.load(num_tokens_cumsum + batch_idx - 1, mask=batch_idx>0, other=0) + migrating_token_idx
	cur_token_idx_in_kv_cache = tl.load(alloc_mem + cur_token_idx_in_recv_buf)
	cur_token_idx_in_recv_buf = cur_token_idx_in_recv_buf.to(tl.int64)
	cur_token_idx_in_kv_cache = cur_token_idx_in_kv_cache.to(tl.int64)

	cur_request_id = tl.load(request_ids + batch_idx)
	cur_token_idx_in_req = cur_b_seq_len + migrating_token_idx
	tl.store(req_to_token_indexes + cur_request_id*stride_rtti_reqid + cur_token_idx_in_req*stride_rtti_tokenidx, cur_token_idx_in_kv_cache)

	stride_kvc_layer = stride_kvc_layer.to(tl.int64)
	stride_buf_layer = stride_buf_layer.to(tl.int64)
	kvc_ptrs = kv_cache + cur_token_idx_in_kv_cache*stride_kvc_token + cur_head*stride_kvc_head + tl.arange(0, head_dim)*stride_kvc_headdim
	recv_buf_ptrs = recv_buf + cur_token_idx_in_recv_buf*stride_buf_token + cur_head*stride_buf_head + tl.arange(0, head_dim)*stride_buf_headdim
	for layer in range(num_layers):
		cur_kvc_ptrs = kvc_ptrs + layer*stride_kvc_layer
		cur_recv_buf_ptrs = recv_buf_ptrs + layer*stride_buf_layer
		tl.store(cur_kvc_ptrs, tl.load(cur_recv_buf_ptrs))

def decoding_stage_migration_receiver_kernel(
	recv_buf: torch.Tensor,
	req_to_token_indexes: torch.Tensor,
	kv_cache: torch.Tensor,
	alloc_mem: torch.Tensor,
	request_ids: torch.Tensor,
	num_tokens: torch.Tensor,
	b_seq_len: torch.Tensor
):
	num_tokens_max = torch.max(num_tokens)
	num_tokens_cumsum = torch.cumsum(num_tokens, dim=0)
	batch_size = request_ids.shape[0]
	num_layers = kv_cache.shape[0]
	kvcache_size = kv_cache.shape[1]
	num_heads = kv_cache.shape[2] // 2
	head_dim = kv_cache.shape[3]

	_fwd_receiver_kernel[(batch_size, num_tokens_max, 2*num_heads)](
		recv_buf, recv_buf.stride(0), recv_buf.stride(1), recv_buf.stride(2), recv_buf.stride(3),
		req_to_token_indexes, req_to_token_indexes.stride(0), req_to_token_indexes.stride(1),
		kv_cache, kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), kv_cache.stride(3),
		alloc_mem,
		request_ids, num_tokens, num_tokens_cumsum, b_seq_len,
		num_layers, kvcache_size, num_heads, head_dim
	)
