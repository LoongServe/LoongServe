import time, os
from multiprocessing import Pipe
import multiprocessing

import numpy as np
import torch

from transformers import AutoTokenizer
import mii

from lib.structs import *
from lib.sut import SystemUnderTest, get_input_ids

class DeepSpeedSUT(SystemUnderTest):
    def __init__(
        self,
        worker_param: WorkerParam
    ):
        assert worker_param.sp_world_size == 1

        self.worker_param = worker_param
        self.tokenizer = AutoTokenizer.from_pretrained(self.worker_param.model_dir, use_fast=False)

        self.input_conn_list = []
        self.output_conn_list = []
        self.workers = []
        for rank_id in range(self.worker_param.tp_world_size):
            input_conn = Pipe(duplex=False)
            output_conn = Pipe(duplex=False)

            proc = multiprocessing.Process(
                target = worker_routine,
                args = (worker_param, rank_id, input_conn[0], output_conn[1])
            )
            proc.start()
            self.workers.append(proc)
            self.input_conn_list.append(input_conn[1])
            self.output_conn_list.append(output_conn[0])

    def __del__(self):
        for i in range(self.worker_param.tp_world_size):
            self.input_conn_list[i].send(None)
        for proc in self.workers:
            proc.join()

    def inference(
        self,
        input_param: InputParam
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], float, list[float]]:
        input_ids = get_input_ids(self.worker_param.model_dir, input_param.input_len*input_param.batch_size)
        prompt_token_ids = input_ids.view(input_param.batch_size, input_param.input_len).tolist()
        prompts = [
            self.tokenizer.decode(prompt_token_ids[batch_idx])
            for batch_idx in range(input_param.batch_size)
        ]

        for worker_id in range(self.worker_param.tp_world_size):
            self.input_conn_list[worker_id].send((input_param, prompts))
        
        predict_texts = []
        prefill_time_usages = []
        decoding_time_usages = []
        for worker_id in range(self.worker_param.tp_world_size):
            cur_predict_texts, cur_prefill_time_usage, cur_decoding_time_usage = self.output_conn_list[worker_id].recv()
            prefill_time_usages.append(cur_prefill_time_usage)
            decoding_time_usages.append(cur_decoding_time_usage)
            if worker_id == 0:
                predict_texts = cur_predict_texts

        predict_ids = self.tokenizer(predict_texts).input_ids
        prefill_time_usage = np.min(prefill_time_usages)
        decoding_time_usage = np.min(decoding_time_usages)
        return input_ids, predict_ids, predict_texts, prefill_time_usage, [decoding_time_usage for _ in range(input_param.output_len-1)]

@torch.inference_mode()
def worker_routine(worker_param: WorkerParam, rank: int, input_conn, output_conn):
    import torch
    import torch.distributed as dist
    world_size = worker_param.tp_world_size
    rank_id = rank

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "28195"
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:28195', rank=rank_id, world_size=world_size)
    torch.cuda.set_device(rank_id)

    import torch.distributed as dist
    dist.barrier()
    torch.cuda.empty_cache()

    import deepspeed
    engine = mii.pipeline(
        worker_param.model_dir,
        {
            "tensor_parallel": worker_param.tp_world_size,
            "max_length": worker_param.max_seq_len,
        }
    )
    
    while True:
        input_data = input_conn.recv()
        if input_data is None:
            print(f"{rank_id} `None` received. Exiting...")
            return
        
        input_param, prompts = input_data

        dist.barrier()
        torch.cuda.synchronize()

        prefill_only_start_time = time.perf_counter()
        _ = engine(
            prompts,
            max_new_tokens = 1,
            min_new_tokens = 1,
            ignore_eos = True,
            do_sample = False
        )
        prefill_only_end_time = time.perf_counter()

        dist.barrier()
        torch.cuda.synchronize()

        both_stage_start_time = time.perf_counter()
        responses = engine(
            prompts,
            max_new_tokens = input_param.output_len,
            min_new_tokens = input_param.output_len,
            ignore_eos = True,
            do_sample = False
        )
        both_stage_end_time = time.perf_counter()

        torch.cuda.synchronize()

        prefill_stage_time = prefill_only_end_time-prefill_only_start_time
        decoding_stage_time = (both_stage_end_time-both_stage_start_time-prefill_stage_time) / (input_param.output_len-1) if input_param.output_len > 1 else 0
        prefill_stage_time *= 1000
        decoding_stage_time *= 1000
        output_conn.send((
            [resp.generated_text for resp in responses],
            prefill_stage_time,
            decoding_stage_time
        ))
