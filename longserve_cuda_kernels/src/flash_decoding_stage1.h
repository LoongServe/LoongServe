#pragma once

#include <cuda_fp16.h>

namespace LongServe {

void flash_decoding_stage1(
	float* mid_out,					// [num_blocks, batch_size, num_q_heads, head_dim]
	float* mid_out_logsumexp,		// [num_blocks, batch_size, num_q_heads]
	const half* __restrict__ q,		// [batch_size, num_q_heads, head_dim]
	const half* __restrict__ k,		// [_, num_kv_heads, head_dim]
	const half* __restrict__ v,		// [_, num_kv_heads, head_dim]
	int64_t k_stride_0, int64_t k_stride_1,
	int64_t v_stride_0, int64_t v_stride_1,
	const float sm_scale,
	const int32_t* __restrict__ req_to_tokens,	// [max_req_num, max_seq_len]
	const int32_t* __restrict__ b_req_idx,		// [batch_size]
	const int32_t* __restrict__ b_seqlen,		// [batch_size]
	int64_t batch_size,
	int64_t num_q_heads,
	int64_t num_kv_heads,
	int64_t num_blocks,
	int64_t head_dim,
	int64_t max_seq_len,
	int64_t seq_block_size,
	int64_t max_len_in_batch
);

}
