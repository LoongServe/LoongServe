#pragma once

#include <cuda_fp16.h>

namespace LongServe {

void silu_and_mul_inplace(
	half* io,
	int64_t num_tokens,
	int64_t ffn_inter_dim
);

}