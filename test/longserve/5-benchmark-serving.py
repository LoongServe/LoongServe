"""
The client during serving performance benchmarking
"""
import sys, os
import argparse
import time
import random
import asyncio
from typing import Optional
from tqdm.asyncio import tqdm

import numpy as np

from lib.common import EXP_RESULT_PATH, bcolors
from lib_benchmark_serving.structs import TestRequest, Dataset, ReqResult, dump_req_result_list
import lib_benchmark_serving.backend_request_func as req_backed
from lib_benchmark_serving.metrics import BenchmarkMetrics

async def run_some_requests(
    backend: str,
    api_url: str,
    requests: list[TestRequest],
    timestamps: list[float],
    output_descrip_str: str,
) -> list[ReqResult]:
    """
    Issue a bunch of requests on their corresponding timestamps, and return the ReqResults
    """

    outputs = []
    last_print_outputs_len = 0
    last_print_time = time.time()

    async def run_one_request(
        backend: str,
        api_url: str,
        request: TestRequest,
        timestamp: float,
        issued_pbar: tqdm,
        first_token_generated_pbar: tqdm,
        finished_pbar: tqdm
    ) -> Optional[ReqResult]:
        """
        Issue one request on the given timestamp, and then return the ReqResult
        """
        await asyncio.sleep(timestamp)
        issued_pbar.update(1)
        issued_pbar.refresh()
        request_func = req_backed.ASYNC_REQUEST_FUNCS[backend]
        output = await request_func(api_url, request, first_token_generated_pbar, finished_pbar)

        if output is None:
            return
        outputs.append(output)
        nonlocal last_print_outputs_len
        nonlocal last_print_time
        if len(outputs)-last_print_outputs_len > len(requests)*0.1 or \
            time.time() - last_print_time > 60:
            last_print_outputs_len = len(outputs)
            last_print_time = time.time()
            part_metrics = BenchmarkMetrics.from_req_results(outputs)
            print(f"\n\n\n{output_descrip_str}")
            print(f"TFT Gap: {issued_pbar.n - first_token_generated_pbar.n}")
            print(f"FIN Gap: {issued_pbar.n - finished_pbar.n}")
            print(part_metrics)
            issued_pbar.refresh()
            first_token_generated_pbar.refresh()
            finished_pbar.refresh()
    
    pbar_args = {
        "ncols": 90,
        "smoothing": 0.05,
    }
    issued_pbar = tqdm(total=len(requests), desc="Iss", colour="#FFC0CB", **pbar_args)
    first_token_generated_pbar = tqdm(total=len(requests), desc="TFT", colour="#ffffff", **pbar_args)
    finished_pbar = tqdm(total=len(requests), desc="Fin", colour="#66ccff", **pbar_args)
    tasks = []
    for (request, timestamp) in zip(requests, timestamps):
        task = asyncio.create_task(run_one_request(backend, api_url, request, timestamp, issued_pbar, first_token_generated_pbar, finished_pbar))
        tasks.append(task)
    
    await asyncio.gather(*tasks)
        
    outputs = [x for x in outputs if x is not None]
    issued_pbar.close()
    first_token_generated_pbar.close()
    finished_pbar.close()
    return outputs


def benchmark_serving(
    backend: str,
    api_url: str,
    dataset: Dataset,
    req_timestamps: list[float],
    num_prompts: int,
    request_rate: float
) -> list[ReqResult]:
    """
    Perform online serving benchmark under the given num_prompts and request_rate
    """
    # Generate requests and timestamps
    if len(dataset.data) < num_prompts:
        print(f"Warning: dataset only has {len(dataset.data)} requests, but we are asked to process {num_prompts} prompts")
        while len(dataset.data) < num_prompts:
            dataset.data += dataset.data
    if len(req_timestamps) < num_prompts:
        print(f"Error: req_timestamps only has {len(req_timestamps)} requests, but we are asked to process {num_prompts} prompts")
        sys.exit(1)
    requests = dataset.data[:num_prompts]
    timestamps = np.array(req_timestamps[:num_prompts])

    # Scale timestamps to [0, num_prompts/request_rate]
    timestamps -= timestamps[0]
    timestamps *= (num_prompts/request_rate) / timestamps[-1]
    timestamps = timestamps.tolist()

    output_descrip_str = f"{backend}, {dataset.dataset_name}, ({num_prompts}, {request_rate})"
    benchmark_result = asyncio.run(run_some_requests(backend, api_url, requests, timestamps, output_descrip_str))
    return benchmark_result


def generate_poisson_process(num_data_points: int = 40000, lam: float = 1) -> list[float]:
    """
    Generate a list of timestamps that follows a Poisson process
    """
    result = []
    t = 0
    for _ in range(num_data_points):
        t += np.random.exponential(1/lam)
        result.append(t)
    return result


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    port = args.port if args.port is not None else req_backed.BACKEND_TO_PORTS[backend]
    api_url = f"http://{args.host}:{port}{req_backed.BACKEND_TO_ENDPOINT[backend]}"

    dataset = Dataset.load(args.dataset)
    if args.dataset != "synthesised":
        random.shuffle(dataset.data)
    print(f"Loaded dataset {dataset.dataset_name} ({len(dataset.data)} requests)")

    if not args.uniform_distrib:
        req_timestamps = generate_poisson_process()
        print("Using Poisson distribution")
    else:
        req_timestamps = list(map(float, range(0, 100000)))
        print("Using uniform distribution")

    num_prompts_and_request_rates = eval(args.num_prompts_req_rates)
    for (num_prompts, request_rate) in num_prompts_and_request_rates:
        print(f"{bcolors.OKGREEN}==============================================={bcolors.ENDC}")
        print(f"{bcolors.OKGREEN}Running on {num_prompts=} {request_rate=}{bcolors.ENDC}")
        result = benchmark_serving(
            backend,
            api_url,
            dataset,
            req_timestamps,
            num_prompts,
            request_rate
        )
        metrics = BenchmarkMetrics.from_req_results(result)
        print(metrics)
        if not args.dont_save:
            exp_result_filename = f"{args.exp_result_prefix}-{num_prompts}-{request_rate}"
            if args.uniform_distrib:
                exp_result_filename += "-uniform"
            exp_result_filename += ".exp"

            exp_result_dir = os.path.join(args.exp_result_root, args.exp_result_dir)
            os.makedirs(exp_result_dir, exist_ok=True)
            exp_result_path = os.path.join(exp_result_dir, exp_result_filename)

            dump_req_result_list(result, exp_result_path)
        time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=list(req_backed.ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True
    )
    parser.add_argument(
        "--num-prompts-req-rates",
        type=str,
        required=True,
        help="[(num_prompts, request_rate), ...] where num_prompts is the number of prompts to process and request_rate is the number of requests per second.",
    )
    parser.add_argument(
        "--dont-save",
        action="store_true",
        help="If this flag is set, then we don't save the benchmark results to disk",
    )
    parser.add_argument(
        "--exp-result-root",
        type=str,
        default=EXP_RESULT_PATH,
        help="Experiment result will be stored under folder <exp-result-root>/<exp-result-dir>"
    )
    parser.add_argument(
        "--exp-result-dir",
        type=str,
        default=None,
        help="Experiment result will be stored under folder <exp-result-root>/<exp-result-dir> (default exp-result-dir: <dataset[:dataset.index('.')]>)"
    )
    parser.add_argument(
        "--exp-result-prefix",
        type=str,
        default=None,
        help="Exp result file will be named as <exp-result-prefix>-<num-prompts>-<req-rate>.exp (default exp-result-prefix: <backend>)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )
    parser.add_argument(
        "--uniform-distrib",
        action="store_true",
        help="Use uniform distribution instead of Poisson"
    )

    args = parser.parse_args()
    if args.exp_result_dir == None:
        args.exp_result_dir = os.path.basename(args.dataset[:args.dataset.rindex('.')])
    if args.exp_result_prefix == None:
        args.exp_result_prefix = args.backend

    main(args)
