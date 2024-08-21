#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace LongServe {

void rotary_emb(
	half* x,
	const half* cos,
	const half* sin,
	int64_t x_stride_0,
	int64_t x_stride_1,
	int64_t x_stride_2,
	int64_t num_tokens,
	int64_t num_heads,
	int64_t head_dim
);

}