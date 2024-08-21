import os, sys
import random
import argparse
import tqdm

from lib_benchmark_serving.structs import Dataset, TestRequest

def find_nearest_prompt_len(avail_prompt_lens: list[int], target_len: int) -> int:
	min_delta = 1e18
	result_len = 0
	for cur_len in avail_prompt_lens:
		delta = abs(cur_len - target_len)
		if delta < min_delta:
			min_delta = delta
			result_len = cur_len
		if cur_len >= target_len:
			break
	return result_len

def main(args: argparse.Namespace):
	print(args)
	len2reqs: dict[int, list[TestRequest]] = {}	# length -> requests
	for dataset_path in args.datasets:
		dataset = Dataset.load(dataset_path)
		for req in dataset.data:
			prompt_len = req.prompt_len
			if prompt_len not in len2reqs:
				len2reqs[prompt_len] = []
			len2reqs[prompt_len].append(req)
	
	new_reqs = []
	lens = list(range(10, 200000))	# Prune too short requests, otherwise requests with short length will be sampled too frequently
									# Prune too long requests to make TP2DP4 runnable
	weights = [1/(x+args.zipf_beta)**args.zipf_alpha for x in lens]

	prompt_lens = random.choices(
		lens,
		weights=weights,
		k=args.num_prompts
	)
	avail_prompt_lens = list(len2reqs.keys())
	avail_prompt_lens.sort()
	for i in tqdm.tqdm(range(args.num_prompts)):
		nearest_prompt_len = find_nearest_prompt_len(avail_prompt_lens, prompt_lens[i])
		req = random.choice(len2reqs[nearest_prompt_len])
		new_reqs.append(req)
	
	dataset_name = f"{str(args.datasets)}-zipf-{args.zipf_alpha}-{args.zipf_beta}"
	dataset = Dataset(
		dataset_name,
		new_reqs
	)
	print(f"Saving dataset {dataset_name} to {args.output} (with a = {args.zipf_alpha}, b = {args.zipf_beta})")
	dataset.dump(args.output)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Mix multiple datasets with Zipf distribution")
	parser.add_argument("--output", type=str, required=True, help="Path to the output dataset")
	parser.add_argument("--datasets", type=str, required=True, nargs='+', help="List of datasets, separated by space")
	parser.add_argument("--num-prompts", type=int, required=True, help="Number of prompts to sample")
	parser.add_argument("--zipf-alpha", type=float, default=1, help="The alpha parameter of the Zipf distribution")
	parser.add_argument("--zipf-beta", type=float, default=0, help="The alpha parameter of the Zipf distribution")
	args = parser.parse_args()

	random.seed(0)
	main(args)
	