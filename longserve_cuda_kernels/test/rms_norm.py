import torch
import longserve_cuda_kernels
import common

def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float):
    t = input.pow(2).mean(-1, keepdim=True) + eps
    return input * torch.rsqrt(t) * weight

if __name__ == "__main__":
    common.set_up_env()
    
    batch_size = 105
    hidden_size = 1144
    input = torch.randn(batch_size, hidden_size)
    weight = torch.randn(hidden_size)
    eps = 1e-6
    
    std_output = rms_norm(input, weight, eps)
    ans_output = longserve_cuda_kernels.rms_norm(input, weight, eps)
    
    assert common.check_allclose(ans_output, std_output)
    
    print("PASS")