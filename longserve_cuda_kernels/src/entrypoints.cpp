#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cmath>

#include <torch/extension.h>

#include "common.h"
#include "rms_norm.h"
#include "silu_and_mul.h"
#include "rotary_emb.h"
#include "flash_decoding_stage1.h"
#include "flash_decoding_stage2.h"

torch::Tensor rms_norm(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const float eps
) {
    assert_whenever(input.dim() == 2);
    assert_whenever(weight.dim() == 1);
    assert_whenever(input.size(1) == weight.size(0));
    assert_whenever(input.scalar_type() == torch::kHalf);
    assert_whenever(weight.scalar_type() == torch::kHalf);
    assert_whenever(input.is_contiguous());
    assert_whenever(weight.is_contiguous());
    torch::Tensor output = torch::empty_like(input);
    assert_whenever(output.is_contiguous());
    LongServe::rms_norm(
        (half*) output.data_ptr(),
        (const half*) input.data_ptr(),
        (const half*) weight.data_ptr(),
        input.size(0),
        input.size(1),
        eps
    );
    return output;
}

void silu_and_mul_inplace(
    torch::Tensor &io
) {
    assert_whenever(io.dim() == 2);
    assert_whenever(io.scalar_type() == torch::kHalf);
    assert_whenever(io.is_contiguous());
    LongServe::silu_and_mul_inplace(
        (half*) io.data_ptr(),
        io.size(0),
        io.size(1)/2
    );
}

static void rotary_emb_single(
    torch::Tensor &x,
    const torch::Tensor &cos,
    const torch::Tensor &sin
) {
    assert(x.dim() == 3);
    int64_t num_tokens = x.size(0);
    int64_t num_heads = x.size(1);
    int64_t head_dim = x.size(2);
    assert_whenever(cos.sizes() == sin.sizes());
    assert_whenever(cos.dim() == 2 && cos.size(0) == num_tokens && cos.size(1) == head_dim/2);
    assert_whenever(cos.is_contiguous());
    assert_whenever(sin.is_contiguous());
    LongServe::rotary_emb(
        (half*) x.data_ptr(),
        (const half*) cos.data_ptr(),
        (const half*) sin.data_ptr(),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        num_tokens,
        num_heads,
        head_dim
    );
}

void rotary_emb(
    torch::Tensor &q,
    torch::Tensor &k,
    const torch::Tensor &cos,
    const torch::Tensor &sin
) {
    rotary_emb_single(q, cos, sin);
    rotary_emb_single(k, cos, sin);
}

torch::Tensor ffn_block(
    torch::Tensor &input_embds,
    const torch::Tensor &norm_weight,
    const float norm_eps,
    const torch::Tensor &gate_up_proj,
    const torch::Tensor &down_proj
) {
    torch::Tensor input_normed = rms_norm(input_embds, norm_weight, norm_eps);
    torch::Tensor up_gate_out = torch::mm(input_normed, gate_up_proj);
    silu_and_mul_inplace(up_gate_out);
    torch::mm_outf(up_gate_out.slice(1, 0, up_gate_out.size(-1)/2), down_proj, input_normed);
    return input_normed;
}

void flash_decoding_stage1(
    torch::Tensor &mid_o,
    torch::Tensor &mid_o_logexpsum,
    const torch::Tensor &q,
    const torch::Tensor &k,
    const torch::Tensor &v,
    const torch::Tensor &req_to_tokens,
    const torch::Tensor &b_req_idx,
    const torch::Tensor &b_seqlen,
    int64_t max_len_in_batch,
    int64_t seq_block_size
) {
    assert_whenever(mid_o.is_contiguous());
    assert_whenever(mid_o_logexpsum.is_contiguous());
    assert_whenever(q.is_contiguous());
    assert_whenever(k.stride(2) == 1);
    assert_whenever(v.stride(2) == 1);
    assert_whenever(req_to_tokens.is_contiguous());
    assert_whenever(b_req_idx.is_contiguous());
    assert_whenever(b_seqlen.is_contiguous());

    assert_whenever(mid_o.dim() == 4);
    assert_whenever(mid_o_logexpsum.dim() == 3);
    assert_whenever(q.dim() == 3);
    assert_whenever(k.dim() == 3);
    assert_whenever(v.dim() == 3);
    assert_whenever(req_to_tokens.dim() == 2);
    assert_whenever(b_req_idx.dim() == 1);
    assert_whenever(b_seqlen.dim() == 1);

    assert_whenever(mid_o.scalar_type() == torch::kFloat);
    assert_whenever(mid_o_logexpsum.scalar_type() == torch::kFloat);
    assert_whenever(q.scalar_type() == torch::kHalf);
    assert_whenever(k.scalar_type() == torch::kHalf);
    assert_whenever(v.scalar_type() == torch::kHalf);
    assert_whenever(req_to_tokens.scalar_type() == torch::kInt32);
    assert_whenever(b_req_idx.scalar_type() == torch::kInt32);
    assert_whenever(b_seqlen.scalar_type() == torch::kInt32);

    float sm_scale = 1.0f / sqrtf(q.size(2));
    LongServe::flash_decoding_stage1(
        (float*) mid_o.data_ptr(),
        (float*) mid_o_logexpsum.data_ptr(),
        (const half*) q.data_ptr(),
        (const half*) k.data_ptr(),
        (const half*) v.data_ptr(),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        sm_scale,
        (const int32_t*) req_to_tokens.data_ptr(),
        (const int32_t*) b_req_idx.data_ptr(),
        (const int32_t*) b_seqlen.data_ptr(),
        mid_o.size(1),
        mid_o.size(2),
        k.size(1),
        mid_o.size(0),
        mid_o.size(3),
        req_to_tokens.size(1),
        seq_block_size,
        max_len_in_batch
    );
}

void flash_decoding_stage2(
    torch::Tensor &o,
    torch::Tensor &out_logexpsum,
    const torch::Tensor &b_seqlen,
    const torch::Tensor &mid_o,
    const torch::Tensor &mid_o_logexpsum,
    int64_t block_seq
) {
    assert_whenever(o.is_contiguous());
    assert_whenever(out_logexpsum.is_contiguous());
    assert_whenever(b_seqlen.is_contiguous());
    assert_whenever(mid_o.is_contiguous());
    assert_whenever(mid_o_logexpsum.is_contiguous());
    
    assert_whenever(o.dim() == 3);
    assert_whenever(out_logexpsum.dim() == 2);
    assert_whenever(b_seqlen.dim() == 1);
    assert_whenever(mid_o.dim() == 4);
    assert_whenever(mid_o_logexpsum.dim() == 3);

    assert_whenever(o.dtype() == torch::kHalf || o.dtype() == torch::kFloat);
    assert_whenever(out_logexpsum.dtype() == torch::kFloat);
    assert_whenever(b_seqlen.dtype() == torch::kInt32);
    assert_whenever(mid_o.dtype() == torch::kFloat);
    assert_whenever(mid_o_logexpsum.dtype() == torch::kFloat);

    if (o.dtype() == torch::kHalf) {
        LongServe::flash_decoding_stage2(
            (half*) o.data_ptr(),
            (float*) out_logexpsum.data_ptr(),
            (const int32_t*) b_seqlen.data_ptr(),
            (const float*) mid_o.data_ptr(),
            (const float*) mid_o_logexpsum.data_ptr(),
            mid_o.size(1),
            mid_o.size(2),
            mid_o.size(0),
            mid_o.size(3),
            block_seq
        );
    } else if (o.dtype() == torch::kFloat) {
        LongServe::flash_decoding_stage2(
            (float*) o.data_ptr(),
            (float*) out_logexpsum.data_ptr(),
            (const int32_t*) b_seqlen.data_ptr(),
            (const float*) mid_o.data_ptr(),
            (const float*) mid_o_logexpsum.data_ptr(),
            mid_o.size(1),
            mid_o.size(2),
            mid_o.size(0),
            mid_o.size(3),
            block_seq
        );
    } else {
        assert_whenever (false);
    }
}

namespace py = pybind11;

PYBIND11_MODULE(longserve_cuda_kernels, m) {
    m.doc() = "Some CUDA kernels for LongServe";

    m.def("hello_world", [&]() {printf("Hello world from longserve_cuda_kernels!\n"); },
    "A function for verifying the model is correctly loaded");

    // Kernels
    m.def("rms_norm", &rms_norm);
    m.def("silu_and_mul_inplace", &silu_and_mul_inplace);
    m.def("rotary_emb", &rotary_emb);

    // Decoding kernels
    m.def("flash_decoding_stage1", &flash_decoding_stage1);
    m.def("flash_decoding_stage2", &flash_decoding_stage2);

    // Layers
    m.def("ffn_block", &ffn_block);
}
