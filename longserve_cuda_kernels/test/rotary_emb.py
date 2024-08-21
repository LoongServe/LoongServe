import torch
import longserve_cuda_kernels
import common

def rotary_emb(x, cos, sin):
    seq_len, h, dim = x.shape
    x0 = x[:, :, 0 : dim // 2]
    x1 = x[:, :, dim // 2 : dim]
    cos = cos.view((seq_len, 1, dim // 2))
    sin = sin.view((seq_len, 1, dim // 2))
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return torch.cat((o0, o1), dim=-1)

if __name__ == "__main__":
    common.set_up_env()
    
    SEQ_LEN = 128
    H = 32
    D = 128
    
    x_shape = (SEQ_LEN, H, D)
    q = -2.3 + 0.5 * torch.randn(x_shape)
    k = -2.3 + 0.5 * torch.randn(x_shape)
    
    cos_shape = (SEQ_LEN, D // 2)
    cos = -1.2 + 0.5 * torch.randn(cos_shape)
    sin = -2.0 + 0.5 * torch.randn(cos_shape)
    
    std_output_q = rotary_emb(q, cos, sin)
    std_output_k = rotary_emb(k, cos, sin)
    ans_output_q = q.clone()
    ans_output_k = k.clone()
    longserve_cuda_kernels.rotary_emb(ans_output_q, ans_output_k, cos, sin)
    
    assert common.check_allclose(ans_output_q, std_output_q)
    
    print("PASS")