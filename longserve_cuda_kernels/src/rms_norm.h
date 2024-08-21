#pragma once

#include <cuda_fp16.h>

namespace LongServe {

void rms_norm(
	half* output,
	const half* input,
	const half* weight,
	const int64_t num_tokens,
	const int64_t hidden_size,
	const float eps
);

}