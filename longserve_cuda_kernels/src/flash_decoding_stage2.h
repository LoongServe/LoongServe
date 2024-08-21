#pragma once

#include <cuda_fp16.h>

namespace LongServe {

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
);

}
