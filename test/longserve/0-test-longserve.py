"""
Test the correctness of LongServe
"""
import os
import sys
import random
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from lib.structs import *
from lib.sut_longserve import LongServeSUT

@torch.inference_mode()
def test(
    model_dir: str,
    mode: list[str],
    sp_world_size: int,
    tp_world_size: int,
    batch_size: int,
    input_len: int,
    output_len: int,
    num_sp_master: int,
    num_decoding_stage_migration: int,
    need_context_migration: bool
):
    # Run LongServe model
    worker_param = WorkerParam(
        model_dir = model_dir,
        mode = mode,
        sp_world_size = sp_world_size,
        tp_world_size = tp_world_size,
        max_total_token_num = batch_size*(input_len+output_len+1) + 1,
        max_req_num = batch_size,
        max_seq_len = input_len+output_len+1
    )
    input_param = InputParam(
        batch_size = batch_size,
        input_len = input_len,
        output_len = output_len,
        num_sp_master = num_sp_master,
        need_context_migration = need_context_migration,
        num_decoding_stage_migration = num_decoding_stage_migration
    )

    sut = LongServeSUT(worker_param)
    input_ids, predict_ids, predict_texts, _, _ = sut.inference(input_param)
    print(f"predict_texts: {predict_texts}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    for i in range(batch_size):
        ans_output_token_ids = predict_ids[i]
        ans_output_tokens = tokenizer.convert_ids_to_tokens(ans_output_token_ids.tolist())
        prompt_suffix_token_ids = input_ids[i*input_len : (i+1)*input_len][-100:]
        prompt_suffix = "".join(tokenizer.convert_ids_to_tokens(prompt_suffix_token_ids))
        print(f"Batch #{i}:")
        print(f"\tprompt suffix: {prompt_suffix}")
        print(f"\tans: {ans_output_token_ids} '{ans_output_tokens}'")
    
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model_dir = os.environ.get("LWM_WEIGHT_PATH", "Env `LWM_WEIGHT_PATH` is not set!") if len(sys.argv) == 1 else sys.argv[1]
    test(
        model_dir = model_dir,
        mode = ["_token_decode_attention_overlapped"],
        sp_world_size = 1,
        tp_world_size = 1,
        batch_size = 100,
        input_len = 100,
        output_len = 16,
        num_sp_master = 1,
        need_context_migration = True,
        num_decoding_stage_migration = 10,
    )
