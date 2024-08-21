#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include "flash_decoding_stage2.h"
#include "reduction.cuh"
#include "common.h"

namespace LongServe {

template<
	typename O_TYPE,
	int64_t BLOCK_SEQ,
	int64_t HEAD_DIM
>
__global__ void flash_decoding_stage2_kernel(
	O_TYPE* __restrict__ o,		// [batch_size, num_heads, head_dim]
	float* __restrict__ out_logexpsum,		// [batch_size, num_heads]
	const int32_t* __restrict__ b_seqlen,
	const float* __restrict__ mid_o,		// [num_seq_blocks, batch_size, num_heads, head_dim]
	const float* __restrict__ mid_o_logexpsum,	// [num_seq_blocks, batch_size, num_heads]
	int64_t batch_size,
	int64_t num_heads,
	int64_t num_seq_blocks
) {
	// Grid: [batch_size, num_heads]
	// Block: [BLOCK_DMODEL]
	int64_t cur_batch = blockIdx.x;
	int64_t cur_head = blockIdx.y;
	int64_t cur_batch_seq_len = b_seqlen[cur_batch];
	int64_t cur_num_seq_blocks = cur_batch_seq_len <= 0 ? 0 : (cur_batch_seq_len + BLOCK_SEQ - 1) / BLOCK_SEQ;
	int64_t thread_id = threadIdx.x;

	float sum_exp = 0.0;
	float max_logit = -1e20;
	float acc = 0;

	#pragma unroll(4)
	for (int64_t seq_index = 0; seq_index < cur_num_seq_blocks; ++seq_index) {
		float tv = mid_o[INDEX_4D(0, batch_size, num_heads, HEAD_DIM, seq_index, cur_batch, cur_head, thread_id)];
		float tlogit = mid_o_logexpsum[INDEX_3D(0, batch_size, num_heads, seq_index, cur_batch, cur_head)];
		float new_max_logit = max(tlogit, max_logit);

		float scale_for_old = __expf(max_logit - new_max_logit);
		float exp_logit = __expf(tlogit - new_max_logit);
		acc = acc * scale_for_old + tv * exp_logit;
		sum_exp = sum_exp * scale_for_old + exp_logit;
		max_logit = new_max_logit;
	}

	if (cur_num_seq_blocks) {
		if constexpr (std::is_same_v<O_TYPE, float>) {
			o[INDEX_3D(0, num_heads, HEAD_DIM, cur_batch, cur_head, thread_id)] = acc / sum_exp;
		} else {
			o[INDEX_3D(0, num_heads, HEAD_DIM, cur_batch, cur_head, thread_id)] = __float2half(acc / sum_exp);
		}
		out_logexpsum[INDEX_2D(0, num_heads, cur_batch, cur_head)] = max_logit + __logf(sum_exp);
	}
}

#define LAUNCH_FLASH_DECODING_STAGE2_KERNEL(BLOCK_SEQ, HEAD_DIM) \
	flash_decoding_stage2_kernel<O_TYPE, BLOCK_SEQ, HEAD_DIM><<<grid, block>>>( \
		o, out_logexpsum, b_seqlen, mid_o, mid_o_logexpsum, batch_size, num_heads, num_seq_blocks \
	);

#define DISPATCH_ON_BLOCK_SEQ(HEAD_DIM) \
	switch (block_seq) { \
		case 1: \
			LAUNCH_FLASH_DECODING_STAGE2_KERNEL(1, HEAD_DIM) \
			break; \
		case 2: \
			LAUNCH_FLASH_DECODING_STAGE2_KERNEL(2, HEAD_DIM) \
			break; \
		case 4: \
			LAUNCH_FLASH_DECODING_STAGE2_KERNEL(4, HEAD_DIM) \
			break; \
		case 8: \
			LAUNCH_FLASH_DECODING_STAGE2_KERNEL(8, HEAD_DIM) \
			break; \
		case 16: \
			LAUNCH_FLASH_DECODING_STAGE2_KERNEL(16, HEAD_DIM) \
			break; \
		case 32: \
			LAUNCH_FLASH_DECODING_STAGE2_KERNEL(32, HEAD_DIM) \
			break; \
		case 64: \
			LAUNCH_FLASH_DECODING_STAGE2_KERNEL(64, HEAD_DIM) \
			break; \
		case 128: \
			LAUNCH_FLASH_DECODING_STAGE2_KERNEL(128, HEAD_DIM) \
			break; \
		case 256: \
			LAUNCH_FLASH_DECODING_STAGE2_KERNEL(256, HEAD_DIM) \
			break; \
		default: \
			assert_whenever(false); \
			break; \
	}

#define DISPATCH_ON_HEAD_DIM() \
	switch (head_dim) { \
		case 32: \
			DISPATCH_ON_BLOCK_SEQ(32) \
			break; \
		case 64: \
			DISPATCH_ON_BLOCK_SEQ(64) \
			break; \
		case 128: \
			DISPATCH_ON_BLOCK_SEQ(128) \
			break; \
		default: \
			assert_whenever(false); \
			break; \
	}

template<typename O_TYPE>
void flash_decoding_stage2(
	O_TYPE* o,
	float* out_logexpsum,
	const int32_t* b_seqlen,
	const float* mid_o,
	const float* mid_o_logexpsum,
	int64_t batch_size,
	int64_t num_heads,
	int64_t num_seq_blocks,
	int64_t head_dim,
	int64_t block_seq
) {
	dim3 grid(batch_size, num_heads);
	dim3 block(head_dim);

	DISPATCH_ON_HEAD_DIM();
}

#define INSTANTIATE(O_TYPE) \
	template void flash_decoding_stage2( \
		O_TYPE* o, \
		float* out_logexpsum, \
		const int32_t* b_seqlen, \
		const float* mid_o, \
		const float* mid_o_logexpsum, \
		int64_t batch_size, \
		int64_t num_heads, \
		int64_t num_seq_blocks, \
		int64_t head_dim, \
		int64_t block_seq \
	);

INSTANTIATE(float)
INSTANTIATE(half)

}