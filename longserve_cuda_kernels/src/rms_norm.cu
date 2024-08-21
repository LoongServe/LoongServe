#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include "rms_norm.h"
#include "reduction.cuh"
#include "common.h"

namespace LongServe {

__global__ void rms_norm_kernel(
	half* __restrict__ output,
	const __restrict__ half* input,
	const __restrict__ half* weight,
	const int64_t hidden_size,
	const float eps
) {
	// grid: [num_tokens]
	// block: [min(hidden_size, 2048)]
	// Step 1. Every thread computes some part of the sum of squares
	float square_sum = 0.0;
	__shared__ float inv_rms_shared;
	float inv_rms;
	#pragma unroll 4
	for (int64_t i = threadIdx.x; i < hidden_size/2; i += blockDim.x) {
		const half2 a = ((const half2*)input)[blockIdx.x*hidden_size/2 + i];
		square_sum += __low2float(a) * __low2float(a) + __high2float(a) * __high2float(a);
	}
	// Step 2. Sum the squares across threads
	square_sum = blockReduceSum(square_sum);
	// Step 3. Compute the inverse root mean square
	if (threadIdx.x == 0) {
		inv_rms_shared = rsqrtf(square_sum / hidden_size + eps);
	}
	__syncthreads();
	inv_rms = inv_rms_shared;
	// Step 4. Compute the output
	#pragma unroll 4
	for (int64_t i = threadIdx.x; i < hidden_size/2; i += blockDim.x) {
		const half2 a = ((const half2*)input)[blockIdx.x*hidden_size/2 + i];
		const half2 b = ((const half2*)weight)[i];
		((half2*)output)[blockIdx.x*hidden_size/2 + i] = half2 {
			__low2float(a) * __low2float(b) * inv_rms,
			__high2float(a) * __high2float(b) * inv_rms
		};
	}
}

void rms_norm(
	half* output,			// [num_tokens, hidden_size]
	const half* input,		// [num_tokens, hidden_size]
	const half* weight,		// [num_tokens, hidden_size]
	const int64_t num_tokens,
	const int64_t hidden_size,
	const float eps
) {
	assert_whenever(hidden_size % 2 == 0);
	const int64_t block_size = std::min(hidden_size/2, 1024L);
	const int64_t grid_size = num_tokens;
	rms_norm_kernel<<<grid_size, block_size>>>(output, input, weight, hidden_size, eps);
}

}