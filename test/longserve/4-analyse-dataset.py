"""
Analyse the target dataset
"""
import os, sys
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt
from lib_benchmark_serving.structs import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the dataset (produced by 4-preprocess-dataset.py)")
    parser.add_argument("--output-path", type=str, default="/tmp/dataset-distrib.png",
                        help="Path to the output image")
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    dataset = Dataset.load(dataset_path)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 15))
    
    def draw_hist_graph(ax, data):
        ax.hist(data, bins=100)
    
    prompt_lens = [req.prompt_len for req in dataset.data]
    output_lens = [req.output_len for req in dataset.data]

    draw_hist_graph(ax0, prompt_lens)
    ax0.set_title("Prompt len")

    draw_hist_graph(ax1, prompt_lens)
    ax1.set_xscale("log")
    ax1.set_title("Prompt len (log)")

    draw_hist_graph(ax2, output_lens)
    ax2.set_title("Output len")
    
    plt.savefig(args.output_path)
    
    print(f"Num of requests: {len(dataset.data)}")
    pencentages = list(range(0, 101, 5))
    print(f"Prompt len mean: {np.mean(prompt_lens)}")
    print(f"sqrt(Prompt len*2 mean): {math.sqrt(np.mean([x*x for x in prompt_lens]))}")
    for p in pencentages:
        print(f"Prompt len p{p}: {np.percentile(prompt_lens, p)}")
    print(f"Output len mean: {np.mean(output_lens)}")
    for p in pencentages:
        print(f"Output len p{p}: {np.percentile(output_lens, p)}")
    