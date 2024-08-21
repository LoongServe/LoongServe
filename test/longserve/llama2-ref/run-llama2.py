import tqdm

import torch

from model_infer import test_model_inference
from db import RecordManager

params = [
    {
		"model_path": "/mnt/petrelfs/wubingyang/Chinese-Llama-2-7b-fp16", 
        "world_size": world_size,
        "max_total_token_num": 101000*world_size,
		"batch_size": 1,
		"input_len": input_len,
		"output_len": 16,
		"mode": ["triton_flashdecoding"]
    }
    for world_size in [1, 2, 4, 8]
    for input_len in [10, 100, 1000, 10000, 20000, 50000, 100000, 200000]
    if not (input_len == 200000 and world_size <= 2)
]
skip_duplicated = True
store_into_db = True

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    record_manager = RecordManager()
    for param in tqdm.tqdm(params):
        print(f"--------------------")
        print(f"{param}")
        if skip_duplicated and record_manager.check_if_record_exists(**param):
            print(f"Skipped")
            continue
        prefill_time_usage, decoding_time_usage = test_model_inference(**param)
        print(f"Prefill: {prefill_time_usage}")
        print(f"Decoding: {decoding_time_usage}")
        if store_into_db:
            record_manager.update_or_insert_record(
                **param,
                prefill_time_usage=prefill_time_usage,
                decoding_time_usage=decoding_time_usage
            )