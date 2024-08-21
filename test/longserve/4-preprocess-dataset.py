"""
Preprocess dataset. This includes:
1) Read the dataset, which may store in different formats.
2) Filter out the requests that are not suitable.
3) Perform tokenization on the input and output to get their length (in #tokens)
4) Store the preprocessed dataset to a file.
"""
from typing import List, Tuple
import json
import random
import os, sys
import argparse
import tqdm

import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers import AutoTokenizer

from lib.common import MODEL_PATH_LWM
from lib_benchmark_serving.structs import TestRequest, Dataset

def read_dataset(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    name: str,
    args: argparse.Namespace,
) -> Dataset:
    """
    read_dataset: Read the given dataset and return a list of TestRequest.
    """
    if name.lower() == "sharegpt":
        # Load the dataset.
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        
        result: List[TestRequest] = []
        for data in tqdm.tqdm(dataset):
            num_conversations = len(data["conversations"])
            
            # Filter out the conversations with less than args.sharegpt_min_turns turns.
            if num_conversations < args.sharegpt_min_turns or \
                num_conversations < args.sharegpt_min_prompt_turns + 1:
                continue
                
            num_prompt_turns = random.randint(
                args.sharegpt_min_prompt_turns,
                min(num_conversations - 1, args.sharegpt_max_prompt_turns)
            )
            
            prompt = "\n".join([data["conversations"][i]["value"] for i in range(num_prompt_turns)])
            completion = data["conversations"][num_prompt_turns]["value"]
            prompt_token_ids = tokenizer(prompt).input_ids
            completion_token_ids = tokenizer(completion).input_ids
            
            prompt_len = len(prompt_token_ids)
            output_len = len(completion_token_ids)
            if prompt_len < 4 or output_len < 4:
                # Prune too short sequences.
                continue
            if prompt_len + output_len >= args.sharegpt_max_len:
                # Prune too long sequences.
                continue
            if output_len >= args.sharegpt_max_output_len:
                # Prune too long outputs.
                continue
            
            result.append(TestRequest(prompt, prompt_len, output_len))
        
        return Dataset(name, result)

    elif name.lower() == "longbench":
        # find all .jsonl files under the dataset_path
        files = []
        for root, dirs, filenames in os.walk(dataset_path):
            for filename in filenames:
                if filename.endswith(".jsonl"):
                    files.append(os.path.join(root, filename))
        
        filtered_dataset = []
        for file in tqdm.tqdm(files):
            with open(file, "r") as f:
                for line in f.readlines():
                    if line.strip() == "": continue
                    data = json.loads(line)
                    
                    context = data["context"]
                    context_token_ids = tokenizer(context).input_ids
                    answer_token_ids = tokenizer(data["answers"][0]).input_ids
                    
                    filtered_dataset.append(TestRequest(
                        tokenizer.decode(context_token_ids),
                        len(context_token_ids),
                        len(answer_token_ids)
                    ))
                    
        return Dataset(name, filtered_dataset)

    elif name.lower() == "leval":
        files = []
        for root, dirs, filenames in os.walk(dataset_path):
            for filename in filenames:
                if filename.endswith(".jsonl"):
                    files.append(os.path.join(root, filename))

        num_lines = sum([
            sum([
                len(json.loads(line)["instructions"])
                for line in open(filename, "r").readlines()
            ])
            for filename in files
        ])

        dataset = []
        pbar = tqdm.tqdm(total=num_lines)
        for file in files:
            print(f"Processing {file}")
            with open(file, "r") as f:
                for line in f.readlines():
                    if line.strip() == "": continue
                    data = json.loads(line)

                    input = data["input"]
                    input_len = len(tokenizer.tokenize(input))
                    for (instruction, output) in zip(data["instructions"], data["outputs"]):
                        prompt_len = input_len + len(tokenizer.tokenize(instruction))
                        output_len = len(tokenizer.tokenize(output))
                        dataset.append(TestRequest(
                            input+instruction,
                            prompt_len,
                            output_len
                        ))
                        pbar.update(1)

        pbar.close()
        return Dataset(name, dataset)

    elif name.lower() == "lv-eval":
        files = []
        for root, dirs, filenames in os.walk(dataset_path):
            for filename in filenames:
                if filename.endswith(".jsonl"):
                    files.append(os.path.join(root, filename))
        
        dataset = []
        for file in tqdm.tqdm(files):
            print(f"Processing {file}")
            with open(file, "r") as f:
                for line in f.readlines():
                    if line.strip() == "": continue
                    data = json.loads(line)
                    input = data["context"] + data["input"]
                    output = " ".join(data["answers"])
                    input_len = len(tokenizer.tokenize(input))
                    output_len = len(tokenizer.tokenize(output))
                    dataset.append(TestRequest(
                        input,
                        input_len,
                        output_len
                    ))

        return Dataset(name, dataset)
    
    else:
        raise ValueError(
            f"Unsupported dataset name: {name}, we currently support shareGPT, alpaca, and mmlu."
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=MODEL_PATH_LWM)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sharegpt-min-turns", type=int, default=3)
    parser.add_argument("--sharegpt-min-prompt-turns", type=int, default=1)
    parser.add_argument("--sharegpt-max-prompt-turns", type=int, default=100000)
    parser.add_argument("--sharegpt-max-len", type=int, default=200000)
    parser.add_argument("--sharegpt-max-output-len", type=int, default=1000)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, trust_remote_code=args.trust_remote_code)
    
    dataset = read_dataset(args.dataset_path, tokenizer, args.dataset, args)
    print(f"Loaded {len(dataset.data)} TestRequests from dataset {args.dataset_path}")
    
    dataset.dump(args.output_path)
    print(f"Saved to {args.output_path}")