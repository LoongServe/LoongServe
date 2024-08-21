#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include "flash_decoding_stage1.h"
#include "reduction.cuh"
#include "common.h"

namespace LongServe {

static constexpr int64_t WARP_SIZE = 32;
static constexpr int64_t NUM_WARPS = 4;
static constexpr int64_t THREAD_BLOCK_SIZE = WARP_SIZE*NUM_WARPS;

struct half4 {
	half a0, a1, a2, a3;
};
struct float4 {
	float a0, a1, a2, a3;
};

template<
	int64_t BLOCK_SEQ,
	int64_t HEAD_DIM
>
__global__ void flash_decoding_stage1_kernel(
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
	int64_t max_seq_len
) {
	// Grid: [batch_size, num_q_heads, ceil(max_len_in_batch / BLOCK_SEQ)]
	// Block: [THREAD_BLOCK_SIZE]
	int64_t cur_batch = blockIdx.x;
	int64_t cur_q_head = blockIdx.y;
	int64_t cur_block = blockIdx.z;
	int64_t cur_kv_head = cur_q_head / (num_q_heads / num_kv_heads);
	
	int64_t cur_seq_len = b_seqlen[cur_batch];
	int64_t cur_req_idx = b_req_idx[cur_batch];
	int64_t cur_start_token_idx = cur_block * BLOCK_SEQ;
	int64_t cur_end_token_idx = min(cur_start_token_idx + BLOCK_SEQ, cur_seq_len);

	if (cur_end_token_idx <= cur_start_token_idx) {
		return;
	}

	int64_t warp_id = threadIdx.x / WARP_SIZE;
	int64_t lane_id = threadIdx.x % WARP_SIZE;

	// Load Q
	half4 q4 = ((const half4*)q)[INDEX_3D(0, num_q_heads, HEAD_DIM/4, cur_batch, cur_q_head, lane_id)];
	float q0 = (float)q4.a0;
	float q1 = (float)q4.a1;
	float q2 = (float)q4.a2;
	float q3 = (float)q4.a3;

	// Load req_to_tokens, and initialize attn_score
	__shared__ int32_t cur_req_to_tokens[BLOCK_SEQ];
	__shared__ float attn_score[BLOCK_SEQ];
	#pragma unroll
	for (int64_t i = threadIdx.x; i < BLOCK_SEQ; i += THREAD_BLOCK_SIZE) {
		cur_req_to_tokens[i] = req_to_tokens[cur_req_idx*max_seq_len + cur_start_token_idx + i];
		attn_score[i] = -1e20f;
	}
	#pragma unroll
	for (int64_t i = threadIdx.x; i < HEAD_DIM; i += THREAD_BLOCK_SIZE) {
		mid_out[INDEX_4D(0, batch_size, num_q_heads, HEAD_DIM, cur_block, cur_batch, cur_q_head, i)] = 0.0f;
	}
	__syncthreads();

	// Calculate the attn_score
	#pragma unroll 8
	for (int64_t token_idx = cur_start_token_idx+warp_id; token_idx < cur_end_token_idx; token_idx += NUM_WARPS) {
		int64_t token_index = cur_req_to_tokens[token_idx - cur_start_token_idx];
		half4 k4 = ((const half4*)k)[token_index*k_stride_0/4 + cur_kv_head*k_stride_1/4 + lane_id];
		float local_score = q0 * (float)k4.a0 + q1 * (float)k4.a1 + q2 * (float)k4.a2 + q3 * (float)k4.a3;
		float global_score = warpReduceSum(local_score);
		if (lane_id == 0) {
			attn_score[token_idx - cur_start_token_idx] = global_score * sm_scale;
		}
	}
	__syncthreads();

	__shared__ float s_sum_exp;
	if (warp_id == 0) {
		// Softmax
		float max_logit = -1e20;
		#pragma unroll
		for (int64_t i = lane_id; i < BLOCK_SEQ; i += WARP_SIZE) {
			max_logit = max(max_logit, attn_score[i]);
		}
		max_logit = warpReduceMax(max_logit);
		float sum_exp = 0.0f;
		#pragma unroll
		for (int64_t i = lane_id; i < BLOCK_SEQ; i += WARP_SIZE) {
			attn_score[i] = __expf(attn_score[i] - max_logit);
			sum_exp += attn_score[i];
		}
		sum_exp = warpReduceSum(sum_exp);
		if (lane_id == 0) {
			mid_out_logsumexp[INDEX_3D(0, batch_size, num_q_heads, cur_block, cur_batch, cur_q_head)] = __logf(sum_exp) + max_logit;
			s_sum_exp = sum_exp;
		}
	}
	__syncthreads();

	// Calculate attn_score * V
	float sum_exp = s_sum_exp;
	float my_out1 = 0.0f;
	float my_out2 = 0.0f;
	float my_out3 = 0.0f;
	float my_out4 = 0.0f;
	#pragma unroll 8
	for (int64_t token_idx = warp_id; token_idx < cur_end_token_idx-cur_start_token_idx; token_idx += NUM_WARPS) {
		int64_t token_index = cur_req_to_tokens[token_idx];
		half4 v4 = ((const half4*)v)[token_index*v_stride_0/4 + cur_kv_head*v_stride_1/4 + lane_id];
		float score = attn_score[token_idx];
		my_out1 += score * (float)v4.a0;
		my_out2 += score * (float)v4.a1;
		my_out3 += score * (float)v4.a2;
		my_out4 += score * (float)v4.a3;
	}
	my_out1 /= sum_exp;
	my_out2 /= sum_exp;
	my_out3 /= sum_exp;
	my_out4 /= sum_exp;
	atomicAdd(&mid_out[INDEX_4D(0, batch_size, num_q_heads, HEAD_DIM, cur_block, cur_batch, cur_q_head, lane_id*4)], my_out1);
	atomicAdd(&mid_out[INDEX_4D(0, batch_size, num_q_heads, HEAD_DIM, cur_block, cur_batch, cur_q_head, lane_id*4+1)], my_out2);
	atomicAdd(&mid_out[INDEX_4D(0, batch_size, num_q_heads, HEAD_DIM, cur_block, cur_batch, cur_q_head, lane_id*4+2)], my_out3);
	atomicAdd(&mid_out[INDEX_4D(0, batch_size, num_q_heads, HEAD_DIM, cur_block, cur_batch, cur_q_head, lane_id*4+3)], my_out4);
}

#define LAUNCH_FLASH_DECODING_STAGE1_KERNEL(BLOCK_SEQ, HEAD_DIM) \
	flash_decoding_stage1_kernel<BLOCK_SEQ, HEAD_DIM><<<grid, block>>>( \
		mid_out, \
		mid_out_logsumexp, \
		q, \
		k, \
		v, \
		k_stride_0, k_stride_1, \
		v_stride_0, v_stride_1, \
		sm_scale, \
		req_to_tokens, \
		b_req_idx, \
		b_seqlen, \
		batch_size, \
		num_q_heads, \
		num_kv_heads, \
		num_blocks, \
		max_seq_len \
	)

#define DISPATCH_BLOCK_SEQ(HEAD_DIM) \
	switch (seq_block_size) { \
		case 1: LAUNCH_FLASH_DECODING_STAGE1_KERNEL(1, HEAD_DIM); break; \
		case 2: LAUNCH_FLASH_DECODING_STAGE1_KERNEL(2, HEAD_DIM); break; \
		case 4: LAUNCH_FLASH_DECODING_STAGE1_KERNEL(4, HEAD_DIM); break; \
		case 8: LAUNCH_FLASH_DECODING_STAGE1_KERNEL(8, HEAD_DIM); break; \
		case 16: LAUNCH_FLASH_DECODING_STAGE1_KERNEL(16, HEAD_DIM); break; \
		case 32: LAUNCH_FLASH_DECODING_STAGE1_KERNEL(32, HEAD_DIM); break; \
		case 64: LAUNCH_FLASH_DECODING_STAGE1_KERNEL(64, HEAD_DIM); break; \
		case 128: LAUNCH_FLASH_DECODING_STAGE1_KERNEL(128, HEAD_DIM); break; \
		case 256: LAUNCH_FLASH_DECODING_STAGE1_KERNEL(256, HEAD_DIM); break; \
		default: assert_whenever(false); break; \
	}

// NOTE head dim = 32 is not supported since we require HEAD_DIM/2 % THREAD_BLOCK_SIZE == 0
// NOTE head_dim = 64 is not supported since we hardcode some values in the kernel to gain potential speedup
#define DISPATCH_HEAD_DIM() \
	switch (head_dim) { \
		case 128: DISPATCH_BLOCK_SEQ(128); break; \
		default: assert_whenever(false); break; \
	}

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
) {
	assert_whenever (k_stride_0%2 == 0 && k_stride_1%2 == 0);
	assert_whenever (v_stride_0%2 == 0 && v_stride_1%2 == 0);
	assert_whenever (head_dim == 128);
	dim3 grid(batch_size, num_q_heads, (max_len_in_batch+seq_block_size-1)/seq_block_size);
	dim3 block(THREAD_BLOCK_SIZE);

	DISPATCH_HEAD_DIM();
}

}