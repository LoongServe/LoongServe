"""
Adapted from vLLM (https://github.com/vllm-project/vllm/blob/main/benchmarks/backend_request_func.py)
"""
import json
import os, sys
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp
from tqdm.asyncio import tqdm

from .structs import TestRequest, ReqResult

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

async def async_request_vllm(
    api_url: str,
    request: TestRequest,
    first_token_generated_pbar: tqdm,
    finished_pbar: tqdm
) -> ReqResult:
    assert api_url.endswith("generate")
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "prompt": request.prompt,
            "n": 1,
            "best_of": 1,
            "use_beam_search": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": request.output_len,
            "ignore_eos": True,
            "stream": True,
        }

        issue_time = time.perf_counter()
        first_token_time = 0
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for data in response.content.iter_any():
                        if first_token_time == 0:
                            first_token_generated_pbar.update(1)
                            first_token_generated_pbar.refresh()
                            first_token_time = time.perf_counter()

                    # When streaming, '\0' is appended to the end of the response.
                    complete_time = time.perf_counter()
                    body = data.decode("utf-8").strip("\0")
                else:
                    print(response)
                    print(response.status)
                    print(response.reason)
                    sys.exit(1)
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError) as e:
            print(e)
            sys.exit(1)

        finished_pbar.update(1)
        finished_pbar.refresh()
        return ReqResult.from_http_request_result(
            request.prompt_len,
            request.output_len,
            issue_time,
            first_token_time,
            complete_time
        )


async def async_request_lightllm(
    api_url: str,
    request: TestRequest,
    first_token_generated_pbar: tqdm,
    finished_pbar: tqdm
) -> Optional[ReqResult]:
    assert api_url.endswith("generate_stream")
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "inputs": request.prompt,
            "parameters": {
                "do_sample": False,
                "ignore_eos": True,
                "max_new_tokens": request.output_len,
            }
        }
        
        issue_time = time.perf_counter()
        first_token_time = 0
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for data in response.content.iter_any():
                        if first_token_time == 0:
                            first_token_generated_pbar.update(1)
                            first_token_generated_pbar.refresh()
                            first_token_time = time.perf_counter()

                    complete_time = time.perf_counter()
                else:
                    print(response)
                    print(response.status)
                    print(response.reason)
                    sys.exit(1)
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError) as e:
            print(f"Warning: Exception encountered while sending request to the server: {e}")
            finished_pbar.update(1)
            finished_pbar.refresh()
            return None
        
        finished_pbar.update(1)
        finished_pbar.refresh()
        return ReqResult.from_http_request_result(
            request.prompt_len,
            request.output_len,
            issue_time,
            first_token_time,
            complete_time
        )
        
async def async_request_deepspeed(
    api_url: str,
    request: TestRequest,
    first_token_generated_pbar: tqdm,
    finished_pbar: tqdm
) -> Optional[ReqResult]:
    assert api_url.endswith("generate")
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "prompt": request.prompt,
            "max_tokens": request.output_len,
            "min_new_tokens": request.output_len,
            "max_new_tokens": request.output_len,
            "stream": True,
            "max_length": int((request.prompt_len + request.output_len)*1.2+10) # *1.2 to prevent tokenization error
        }
        
        issue_time = time.perf_counter()
        first_token_time = 0
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for data in response.content.iter_any():
                        if first_token_time == 0:
                            first_token_generated_pbar.update(1)
                            first_token_generated_pbar.refresh()
                            first_token_time = time.perf_counter()

                    complete_time = time.perf_counter()
                else:
                    print(response)
                    print(response.status)
                    print(response.reason)
                    sys.exit(1)
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError) as e:
            print(f"Warning: Exception encountered while sending request to the server: {e}")
            finished_pbar.update(1)
            finished_pbar.refresh()
            return None
        
        finished_pbar.update(1)
        finished_pbar.refresh()
        return ReqResult.from_http_request_result(
            request.prompt_len,
            request.output_len,
            issue_time,
            first_token_time,
            complete_time
        )
        
ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_vllm,
    "vllm-sw": async_request_vllm,
    "lightllm-sf": async_request_lightllm,
    "longserve": async_request_lightllm,
    "deepspeed": async_request_deepspeed,
    "distserve": async_request_vllm,
    "longserve-fixsp": async_request_lightllm
}

BACKEND_TO_PORTS = {
    "vllm": 8100,
    "vllm-sw": 8200,
    "lightllm-sf": 8300,
    "longserve": 8400,
    "deepspeed": 8500,
    "distserve": 8600,
    "longserve-fixsp": 8700,
    "dummy": 8900,
}

BACKEND_TO_ENDPOINT = {
    "vllm": "/generate",
    "vllm-sw": "/generate",
    "lightllm-sf": "/generate_stream",
    "longserve": "/generate_stream",
    "deepspeed": "/generate",
    "longserve-fixsp": "/generate_stream",
    "distserve": "/generate",
}