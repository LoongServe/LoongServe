#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include "silu_and_mul.h"
#include "reduction.cuh"
#include "common.h"

namespace LongServe {

static constexpr int64_t BLOCK_SIZE = 256;

__device__ __forceinline__ half silu(const half &x) {
	float xf = __half2float(x);
	return __float2half(xf / (1 + __expf(-xf)));
}

__global__ void silu_and_mul_inplace_kernel(
	half* io,				// [num_tokens, ffn_inter_dim*2]
	int64_t num_tokens,
	int64_t ffn_inter_dim
) {
	// grid_size: [num_tokens, ffn_inter_dim//BLOCK_SIZE/2]
	// block_size: [BLOCK_SIZE]
	int64_t token_idx = blockIdx.x;
	int64_t dim_idx = blockIdx.y * BLOCK_SIZE + threadIdx.x;
	if (dim_idx < ffn_inter_dim/2) {
		half2 gate2 = ((half2*)io)[token_idx*ffn_inter_dim + dim_idx];
		half2 up2 = ((half2*)io)[token_idx*ffn_inter_dim + dim_idx + ffn_inter_dim/2];
		((half2*)io)[token_idx*ffn_inter_dim + dim_idx] = half2 {
			silu(gate2.x) * up2.x,
			silu(gate2.y) * up2.y
		};
	}
}

void silu_and_mul_inplace(
	half* io,
	int64_t num_tokens,
	int64_t ffn_inter_dim
) {
	assert_whenever(ffn_inter_dim % 2 == 0);
	dim3 grid_size(num_tokens, cdiv(ffn_inter_dim, 2*BLOCK_SIZE));
	dim3 block_size(BLOCK_SIZE);
	silu_and_mul_inplace_kernel<<<grid_size, block_size>>>(io, num_tokens, ffn_inter_dim);
}

}
