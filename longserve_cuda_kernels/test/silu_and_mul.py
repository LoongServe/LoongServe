import torch
import longserve_cuda_kernels
import common

def silu_and_mul(x: torch.Tensor):
    ffn_inter_dim = x.shape[1]//2
    gate = x[:, :ffn_inter_dim]
    up = x[:, ffn_inter_dim:]
    return torch.nn.functional.silu(gate) * up

if __name__ == "__main__":
    common.set_up_env()
    
    batch_size = 105
    ffn_inter_dim = 1144
    input = torch.randn(batch_size, ffn_inter_dim*2)
    
    std_output = silu_and_mul(input)
    longserve_cuda_kernels.silu_and_mul_inplace(input)
    ans_output = input[:, :ffn_inter_dim]
    
    assert common.check_allclose(ans_output, std_output)
    
    print("PASS")