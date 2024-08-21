import torch
import random
import numpy as np

def set_up_env():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float16)

def check_allclose(ans, std):
	if ans.shape != std.shape:
		print("Shape mismatch!")
		print("ans.shape:", ans.shape)
		print("std.shape:", std.shape)
		return False
	abs_err = torch.abs(ans - std)
	rel_err = abs_err / (torch.max(torch.abs(std), torch.abs(ans)) + 1e-2)
	max_abs_err = torch.max(abs_err)
	max_rel_err = torch.max(rel_err)
	if not (abs_err < 1e-2 + (1e-2) * torch.max(torch.abs(std), torch.abs(ans))).all():
		print("ans:", ans)
		print("std:", std)
		print(f"max abs err: {max_abs_err} at pos {torch.argmax(abs_err)} where ans={ans.flatten()[torch.argmax(abs_err)]} and std={std.flatten()[torch.argmax(abs_err)]}")
		print(f"max rel err: {max_rel_err} at pos {torch.argmax(rel_err)} where ans={ans.flatten()[torch.argmax(rel_err)]} and std={std.flatten()[torch.argmax(rel_err)]}")
		return False
	return True