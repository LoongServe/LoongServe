import os, sys
import random

from lib_benchmark_serving.structs import Dataset

def print_usage():
	print(f"Usage {sys.argv[0]} <output_path> <dataset1:number1> [dataset2:number2] [dataset3:number3] ...")
	sys.exit(1)

if __name__ == "__main__":
	if len(sys.argv) <= 2:
		print_usage()
	random.seed(0)

	dataset_and_num_reqs = []
	for arg in sys.argv[2:]:
		if arg.count(":") != 1:
			print_usage()
		dataset, num_req = arg.split(':')
		num_req = int(num_req)
		print(f"Dataset: {dataset}\nNum of requests: {num_req}")
		dataset_and_num_reqs.append((dataset, num_req))

	# Sample the dataset
	selected_dataset = []
	for (dataset_name, num_reqs) in dataset_and_num_reqs:
		dataset = Dataset.load(dataset_name)
		filtered_data = [req for req in dataset.data if req.prompt_len >= 16 and req.output_len >= 4]
		if len(filtered_data) < num_reqs:
			print(f"Warning: dataset {dataset_name} only has {len(filtered_data)} requests (after being filtered), less than {num_reqs} required")
		selected_dataset.extend(random.sample(filtered_data, num_reqs))
	random.shuffle(selected_dataset)

	# Generate the dataset name
	dataset_name = "|".join([
		f"{dataset[dataset.rfind('/')+1:]}:{num_reqs}"
		for (dataset, num_reqs) in dataset_and_num_reqs
	])
	print(dataset_name)

	new_dataset = Dataset(
		dataset_name,
		selected_dataset
	)
	print(f"Writing output dataset to {sys.argv[1]}")
	new_dataset.dump(sys.argv[1])