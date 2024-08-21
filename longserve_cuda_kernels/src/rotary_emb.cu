#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include "rotary_emb.h"
#include "reduction.cuh"
#include "common.h"

namespace LongServe {

__global__ void rotary_emb_kernel(
	half* x,			// [num_tokens, num_heads, head_dim]
	const half* cos,	// [num_tokens, head_dim/2]
	const half* sin,	// [num_tokens, head_dim/2]
	int64_t x_stride_0,
	int64_t x_stride_1,
	int64_t x_stride_2,
	int64_t num_tokens,
	int64_t num_heads,
	int64_t head_dim
) {
	// grid: [num_tokens, num_heads]
	// block: [head_dim/2/2]
	int64_t token_id = blockIdx.x;
	int64_t head_id = blockIdx.y;
	int64_t dim_id = threadIdx.x;
	half2 x1 = ((half2*)x)[token_id*x_stride_0/2 + head_id*x_stride_1/2 + dim_id*x_stride_2];
	half2 x2 = ((half2*)x)[token_id*x_stride_0/2 + head_id*x_stride_1/2 + (dim_id+head_dim/4)*x_stride_2];
	half2 cur_cos = ((half2*)cos)[INDEX_2D(num_tokens, head_dim/4, token_id, dim_id)];
	half2 cur_sin = ((half2*)sin)[INDEX_2D(num_tokens, head_dim/4, token_id, dim_id)];
	half2 new_x1 = __hfma2(x1, cur_cos, __hneg2(__hmul2(x2, cur_sin)));
	half2 new_x2 = __hfma2(x1, cur_sin, __hmul2(x2, cur_cos));
	((half2*)x)[token_id*x_stride_0/2 + head_id*x_stride_1/2 + dim_id*x_stride_2] = new_x1;
	((half2*)x)[token_id*x_stride_0/2 + head_id*x_stride_1/2 + (dim_id+head_dim/4)*x_stride_2] = new_x2;
}

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
) {
	assert_whenever(head_dim%4 == 0);
	dim3 grid(num_tokens, num_heads);
	dim3 block(head_dim/2/2);
	rotary_emb_kernel<<<grid, block>>>(x, cos, sin, x_stride_0, x_stride_1, x_stride_2, num_tokens, num_heads, head_dim);
}

}
