import os, sys
import argparse
import aiohttp
import asyncio
import json

from lib_benchmark_serving.backend_request_func import BACKEND_TO_PORTS, BACKEND_TO_ENDPOINT

prompts = [
    "To be or not to be",
    "One two three four five",
    "Xin Jin is",
    "A shoulder for the past, let out the",
    "Life blooms like a flower, far",
    "Genshin Impact is"
]

async def main(args: argparse.Namespace):
    async def task(prompt: str):
        if "longserve" in args.backend or "lightllm" in args.backend:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "do_sample": False,
                    "ignore_eos": False,
                    "max_new_tokens": 100,
                }
            }
        elif "vllm" in args.backend:
            payload = {
                "prompt": prompt,
                "n": 1,
                "best_of": 1,
                "use_beam_search": False,
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 100,
                "ignore_eos": True,
                "stream": True,
            }
        elif "deepspeed" in args.backend:
            payload = {
                "prompt": prompt,
                "max_tokens": 100,
                "min_new_tokens": 100,
                "max_new_tokens": 100,
                "stream": False,
                "max_length": 1000
            }
        else:
            assert False, f"Unknown backend: {args.backend}"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            generated_text = ""
            async with session.post(url=url, json=payload) as response:
                async for data in response.content.iter_any():
                    generated_text += data.decode("utf-8")
            print(f"{prompt} | {generated_text}")

    url = f"http://{args.host}:{BACKEND_TO_PORTS[args.backend]}{BACKEND_TO_ENDPOINT[args.backend]}"
    tasks = []
    for prompt in prompts:
        tasks.append(asyncio.create_task(task(prompt)))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        choices=list(BACKEND_TO_PORTS.keys())
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost"
    )

    asyncio.run(main(parser.parse_args()))
