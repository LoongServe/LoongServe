import argparse
import os
import json
from transformers import AutoTokenizer
import random
from typing import Tuple, Dict, List, Optional
import asyncio
from tqdm.asyncio import tqdm
from lib_benchmark_serving.structs import TestRequest, Dataset, ReqResult, dump_req_result_list
import time
import lib_benchmark_serving.backend_request_func as req_backed
from lib_benchmark_serving.metrics import BenchmarkMetrics

def parse_json_file(filename) -> Tuple[List[Dict], Dict]:
    print("Parse JSON file begins")
    with open(filename, 'r') as file:
        first_line = file.readline().strip()
        first_line = first_line.replace("\'", "\"")
        json_list = json.loads(first_line)
        
        other_json = {}
        for _ in range(2, 13):
            line = file.readline().strip()
            if line != "{" and line != "}":
                key, value = line.strip().split(': ')
                other_json[key] = float(value)

    return json_list, other_json

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
            print(f"output is None!")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default="localhost"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/home/hadoop-hdpmlpserving/dolphinfs_hdd_hadoop-hdpmlpserving/wubingyang/LWM-Text-1M"
    )
    parser.add_argument(
        "--trace",
        type=str,
        default="/home/hadoop-hdpmlpserving/dolphinfs_hdd_hadoop-hdpmlpserving/wubingyang/loongserve-leval/longserve-100-0.1.exp"
    )
    parser.add_argument(
        "--exp-result-root",
        type=str,
        default="/home/hadoop-hdpmlpserving/dolphinfs_hdd_hadoop-hdpmlpserving/wubingyang/trace-exp-result",
        help="Experiment result will be stored under folder <exp-result-root>/<exp-result-dir>"
    )
    parser.add_argument(
        "--exp-result-dir",
        type=str,
        default='leval',
        help="Experiment result will be stored under folder <exp-result-root>/<exp-result-dir> (default exp-result-dir: <dataset[:dataset.index('.')]>)"
    )
    parser.add_argument(
        "--exp-result-prefix",
        type=str,
        default="loongserve",
        help="Exp result file will be named as <exp-result-prefix>-<num-prompts>-<req-rate>.exp (default exp-result-prefix: <backend>)"
    )

    args = parser.parse_args()
    if args.exp_result_dir == None:
        assert False
    
    json_list, other_json = parse_json_file(args.trace)

    max_prompt_len = max([int(obj["prompt_len"]) for obj in json_list])
    print(f"max_prompt_len = {max_prompt_len}")

    print(f"Build dataset")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    token_vocab = list(tokenizer.get_vocab().keys())
    prompt_tokens = [token_vocab[random.randint(0, len(token_vocab) - 1)] for i in range(max_prompt_len)]

    for obj in json_list:
        prompt_len = obj["prompt_len"]
        prompt_text = tokenizer.convert_tokens_to_string(prompt_tokens[:prompt_len])
        obj["prompt_text"] = prompt_text
    
    json_list.sort(key=lambda obj: float(obj["issue_time"]))

    backend = "longserve"
    api_url = f"http://{args.host}:{args.port}{req_backed.BACKEND_TO_ENDPOINT[backend]}"

    requests = [TestRequest(prompt=obj["prompt_text"], prompt_len=obj["prompt_len"], output_len=obj["output_len"]) for obj in json_list]
    first_issue_time = float(json_list[0]["issue_time"])
    timestamps = [float(obj["issue_time"]) - first_issue_time for obj in json_list]
    
    output_descrip_str = f"{backend}, {args.trace[args.trace.rindex('/'):]}, ({len(requests)}, {len(requests) / timestamps[-1]})"

    print(f"Start execution")
    benchmark_result = asyncio.run(run_some_requests(backend, api_url, requests, timestamps, output_descrip_str))

    exp_result_filename = f"{args.trace[args.trace.rindex('/')+1:]}"

    exp_result_dir = os.path.join(args.exp_result_root, args.exp_result_dir)
    os.makedirs(exp_result_dir, exist_ok=True)
    exp_result_path = os.path.join(exp_result_dir, exp_result_filename)
    print(f"result_path: {exp_result_path}")

    dump_req_result_list(benchmark_result, exp_result_path)