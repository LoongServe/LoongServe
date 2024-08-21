import zmq
import zmq.asyncio
import asyncio
import uvloop
import rpyc
import time
import hashlib

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from .tokenization.manager import TokenizationManager
from ..io_struct import BatchStrOut, AbortReq, FinishStatus
from ..sampling_params import SamplingParams
from ..tokenizer import get_tokenizer
from ..router.manager import RouterManager

from typing import List

import os
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"
import ray

class HttpServerManager:
    def __init__(
        self,
        args,
        router_rpc: RouterManager,
        router_port,
        httpserver_port,
    ):
        self.args = args
        self.router_rpc = router_rpc
        self.num_detokenizers = args.num_detokenizers
        context = zmq.asyncio.Context(self.num_detokenizers)

        self.recv_from_detokenization = context.socket(zmq.PULL)
        self.recv_from_detokenization.bind(f"tcp://127.0.0.1:{httpserver_port}")
        
        self.num_tokenizers = args.num_tokenizers
        self.enable_remote_tokenizers = (self.num_tokenizers > 0)
        
        if self.enable_remote_tokenizers:
            self.tokenizer_id = 0

            self.tokenization_rpcs: List[TokenizationManager] = []
            for i in range(self.num_tokenizers):
                self.tokenization_rpcs.append(
                    ray.remote(
                        num_cpus=0,
                        num_gpus=0,
                    )(TokenizationManager).options(
                        name=f"tokenization manager {i}",
                        resources={f"{ray.state.current_node_id()}": 0.001} # Make sure it runs on the same node
                    ).remote(
                        i, args
                    )
                )
        else:
            self.tokenizer = get_tokenizer(
                args.model_dir, args.tokenizer_mode, trust_remote_code=args.trust_remote_code
            )

        self.req_id_to_out_inf = {}  # value type (out_str, metadata, finished, event)

        self.total_token_num = args.max_total_token_num
        self.max_req_input_len = args.max_req_input_len
        self.max_req_total_len = args.max_req_total_len
        self.sp_world_size = args.sp_world_size

        print(f"HTTP server manager init ok", flush=True)

        return

    async def to_tokenization_process(self, text):
        if self.enable_remote_tokenizers:
            ret = self.tokenization_rpcs[self.tokenizer_id].encode.remote(text)
            self.tokenizer_id = (self.tokenizer_id + 1) % self.num_tokenizers
            return await ret
        else:
            return self.tokenizer.encode(text)

    async def generate(self, prompt, sampling_params: SamplingParams, request_id):
        prompt_ids = await self.to_tokenization_process(prompt)

        prompt_tokens = len(prompt_ids)

        if prompt_tokens > self.max_req_input_len:
            # use long_truncation_mode to truncate long input len req.
            if self.args.long_truncation_mode is None:
                raise ValueError(
                    f"the input prompt token len {prompt_tokens} is too long > {self.max_req_input_len}"
                )
            elif self.args.long_truncation_mode == "head":
                prompt_ids = prompt_ids[-self.max_req_input_len:]
                prompt_tokens = len(prompt_ids)
            elif self.args.long_truncation_mode == "center":
                prompt_ids = prompt_ids[0:self.max_req_input_len // 2] + prompt_ids[-(self.max_req_input_len - self.max_req_input_len // 2):]
                prompt_tokens = len(prompt_ids)
                assert prompt_tokens == self.max_req_input_len
            else:
                assert False, "error args"

        req_total_len = prompt_tokens + sampling_params.max_new_tokens
        if req_total_len > self.max_req_total_len:
            raise ValueError(
                f"the req token total len (input len + output len) is too long > max_req_total_len:{self.max_req_total_len}"
            )
        if req_total_len + 1 > self.total_token_num * self.sp_world_size:
            raise ValueError(
                f"the req token total len + 1 (input len + output len + 1) is too long > max_total_token_num:{self.total_token_num}"
            )
        
        await sampling_params.stop_sentences_to_token_ids(self.to_tokenization_process)

        req_status = ReqStatus(request_id)
        self.req_id_to_out_inf[request_id] = req_status
  
        self.router_rpc.loop_for_netio_req.remote((prompt_ids, sampling_params, request_id))

        while True:
            out_str, metadata, finish_status = await req_status.out_token_info_queue.get()
            metadata["prompt_tokens"] = prompt_tokens
            yield out_str, metadata, finish_status

            if finish_status.is_finished():
                try:
                    del self.req_id_to_out_inf[request_id]
                except:
                    pass
                return
        return

    async def abort(self, request_id):
        abort_req = AbortReq(req_id=request_id)
        self.router_rpc.loop_for_netio_req.remote(abort_req)
        try:
            req = self.req_id_to_out_inf[request_id]
            del self.req_id_to_out_inf[request_id]
        except:
            pass
        return

    async def handle_loop(self):
        while True:
            recv_ans: BatchStrOut = await self.recv_from_detokenization.recv_pyobj()
            assert isinstance(
                recv_ans, BatchStrOut
            ), f"error recv type {type(recv_ans)}"
            for req_id, text, metadata, finish_status in recv_ans.reqs_infs:
                finish_status = FinishStatus(finish_status)
                try:
                    if not finish_status.is_aborted():
                        req_status : ReqStatus = self.req_id_to_out_inf[req_id]
                        req_status.out_token_info_queue.put_nowait((text, metadata, finish_status))
                    else:
                        del self.req_id_to_out_inf[req_id]
                except:
                    pass
        return

class ReqStatus:
    def __init__(self, req_id) -> None:
        self.req_id = req_id
        self.out_token_info_queue = asyncio.Queue()
