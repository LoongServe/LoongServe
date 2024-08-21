"""
This file convert a pytorch .bin file, which contains FP32 tensors, to a .bin
file with FP16 tensors.
"""
import sys
import tqdm
import torch

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print(f"Usage: {sys.argv[0]} <filename>")
	filename = sys.argv[1]
	print(f"Converting file {filename} to fp16...")

	weight_dict = torch.load(filename)
	for k, v in tqdm.tqdm(weight_dict.items()):
		if v is not None and v.dtype == torch.float32:
			weight_dict[k] = v.to(torch.float16)
	torch.save(weight_dict, filename)
