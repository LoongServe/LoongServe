"""
Benchmark LongServe on a batch of requests, which have identical input&output length
"""
import sys, os
import random
import numpy as np
import torch

from lib.common import *
from lib.structs import *
from lib.run_test_param_group import run_test_params

example_testing_params = [
    TestParamGroup(
        worker_param = WorkerParam(
            model_dir = os.environ.get("LWM_WEIGHT_PATH", "Env `LWM_WEIGHT_PATH` is not set!"),
            mode = ["_token_decode_attention_overlapped"],
            sp_world_size = 1,
            tp_world_size = 1,
            max_total_token_num = 101000,
            max_req_num = 16,
            max_seq_len = 101000
        ),
        input_params = [
            InputParam(
                batch_size = 1,
                input_len = 100,
                output_len = 16,
                num_sp_master = 1,
                need_context_migration = False,
                num_decoding_stage_migration = 0
            )
        ]
    )
]

def get_ae_toy_example_params() :
    """
    Params for a toy example for artifact evaluation (AE)
    """
    return [TestParamGroup(
        worker_param = WorkerParam(
            model_dir = os.environ.get("LWM_WEIGHT_PATH", "Env `LWM_WEIGHT_PATH` is not set!"),
            mode = ["_token_decode_attention_overlapped"],
            sp_world_size = 4,
            tp_world_size = 2,
            max_total_token_num = 210000,
            max_req_num = 1024,
            max_seq_len = 210000
        ),
        input_params = [
            InputParam(
                batch_size = 1,
                input_len = 100,
                output_len = 16,
                num_sp_master = 1,
                need_context_migration = False,
                num_decoding_stage_migration = 0
            )
        ]
    )]

def get_sp_vs_tp_testing_params():
    """
    Params for testing the performance of SP/TP when batch_size = 1
    """
    worker_params = [
        WorkerParam(
            model_dir = os.environ.get("LWM_WEIGHT_PATH", "Env `LWM_WEIGHT_PATH` is not set!"),
            mode = ["_token_decode_attention_overlapped"],
            sp_world_size = sp_world_size,
            tp_world_size = tp_world_size,
            max_total_token_num = 101000 * tp_world_size,
            max_req_num = 128,
            max_seq_len = 410000
        ) for (sp_world_size, tp_world_size) in
        [
            (1, 1),
            (2, 1), (1, 2),
            (4, 1), (2, 2), (1, 4),
            (8, 1), (4, 2), (2, 4), (1, 8)
        ]
    ]
    testing_params = [
        TestParamGroup(
            worker_param = worker_param,
            input_params = [
                InputParam(
                    batch_size = 1,
                    input_len = input_len,
                    output_len = 16,
                    num_sp_master = 1,
                    need_context_migration = False,
                    num_decoding_stage_migration = 0
                )
                for input_len in [10, 100, 1000, 10000, 20000, 50000, 100000, 200000, 400000]
                if not (input_len >= 200000 and worker_param.sp_world_size * worker_param.tp_world_size <= 2)
                if not (input_len >= 400000 and worker_param.sp_world_size * worker_param.tp_world_size <= 4)
            ]
        )
        for worker_param in worker_params
    ]
    return testing_params

def get_time_with_batch_size_params(enable_multi_node: bool):
    """
    Params for testing the performance of LongServe
    """
    worker_params = [
        WorkerParam(
            model_dir = os.environ.get("LWM_WEIGHT_PATH", "Env `LWM_WEIGHT_PATH` is not set!"),
            mode = ["_token_decode_attention_overlapped"],
            sp_world_size = sp_world_size,
            tp_world_size = tp_world_size,
            max_total_token_num = 101000 * tp_world_size,
            max_req_num = 1024,
            max_seq_len = 510000
        ) for (sp_world_size, tp_world_size) in
        [
            (1, 1),
            (2, 1), (1, 2),
            (4, 1), (2, 2), (1, 4),
            (8, 1), (4, 2), (2, 4), (1, 8),
            (3, 1), (3, 2),
            (5, 1),
            (6, 1),
            (7, 1),
        ] + ([
            (5, 2),
            (6, 2),
            (7, 2),
            (8, 2)
        ] if enable_multi_node else [])
    ]
    testing_params = [
        TestParamGroup(
            worker_param = worker_param,
            input_params = [
                InputParam(
                    batch_size = batch_size,
                    input_len = input_len,
                    output_len = 16,
                    num_sp_master = 1,
                    need_context_migration = False,
                    num_decoding_stage_migration = 0
                )
                for batch_size in [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 196, 256, 384, 512, 768, 1024]
                for input_len in [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 400000, 500000]
                if batch_size*(input_len+16*worker_param.sp_world_size) <= (worker_param.max_total_token_num-100)*worker_param.sp_world_size
                if not (input_len >= 200000 and worker_param.sp_world_size*worker_param.tp_world_size <= 2)
                if not (input_len >= 400000 and worker_param.sp_world_size*worker_param.tp_world_size <= 4)
                if not (worker_param.sp_world_size == 1 and batch_size*input_len >= 640000)
                if not (worker_param.sp_world_size*worker_param.tp_world_size == 2 and input_len*batch_size >= 196000)
            ]
        )
        for worker_param in worker_params
    ]
    return testing_params

def get_tp_benchmark_params():
    """
    Params that satisfy TP = 1, used for baseline benchmarking across different
    serving systems
    """
    worker_params = [
        WorkerParam(
            model_dir = os.environ.get("LWM_WEIGHT_PATH", "Env `LWM_WEIGHT_PATH` is not set!"),
            mode = ["_token_decode_attention_overlapped"],
            sp_world_size = 1,
            tp_world_size = tp_world_size,
            max_total_token_num = 101000 * tp_world_size,
            max_req_num = 64,
            max_seq_len = 101000 if tp_world_size == 1 else 401000
        ) for tp_world_size in [1, 2, 4, 8]
    ]
    testing_params = [
        TestParamGroup(
            worker_param = worker_param,
            input_params = [
                InputParam(
                    batch_size = batch_size,
                    input_len = input_len,
                    output_len = 16,
                    num_sp_master = 1,
                    need_context_migration = False,
                    num_decoding_stage_migration = 0
                )
                for input_len in [10, 100, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 400000]
                for batch_size in [1, 2, 4, 8, 16, 32, 48]
                if batch_size*input_len <= worker_param.max_total_token_num
                if not (input_len >= 200000 and worker_param.tp_world_size <= 2)
                if not (batch_size*input_len >= 640000)
            ]
        )
        for worker_param in worker_params
    ]
    return testing_params

def get_scale_up_params():
    """
    Get params for scale-up benchmark
    """
    test_params = [
        TestParamGroup(
            worker_param=WorkerParam(
                model_dir = os.environ.get("LWM_WEIGHT_PATH", "Env `LWM_WEIGHT_PATH` is not set!"),
                mode = ["_token_decode_attention_overlapped"],
                sp_world_size = 4,
                tp_world_size = 2,
                max_total_token_num = 101000 * 2,
                max_req_num = 1024,
                max_seq_len = 201000
            ),
            input_params=[
                InputParam(
                    batch_size = batch_size,
                    input_len = input_len,
                    output_len = 16,
                    num_sp_master = num_sp_master,
                    need_context_migration = False,
                    num_decoding_stage_migration = 0
                )
                for (batch_size, input_len) in [
                    (1024, 10), (256, 100), (64, 1000), (16, 10000), (4, 50000), (2, 100000), (1, 200000)
                ]
                for num_sp_master in [1, 2, 4]
            ]
        )
    ]
    return test_params

def get_scale_down_params():
    """
    Get params for scale-down benchmark
    """
    test_params = [
        TestParamGroup(
            worker_param=WorkerParam(
                model_dir = os.environ.get("LWM_WEIGHT_PATH", "Env `LWM_WEIGHT_PATH` is not set!"),
                mode = ["_token_decode_attention_overlapped"],
                sp_world_size = 4,
                tp_world_size = 2,
                max_total_token_num = 101000 * 2,
                max_req_num = 1024,
                max_seq_len = 201000
            ),
            input_params=[
                InputParam(
                    batch_size = batch_size,
                    input_len = input_len,
                    output_len = 16,
                    num_sp_master = 1,
                    need_context_migration = need_context_migration,
                    num_decoding_stage_migration = 0
                )
                for (batch_size, input_len) in [
                    (1024, 10), (256, 100), (64, 1000), (16, 10000), (4, 50000), (2, 100000), (1, 200000)
                ]
                for need_context_migration in [True]
            ]
        )
    ]
    return test_params

def get_ae_figure2_params():
    """
    Get params for figure 2 in artifact evaluation
    """
    tp_sizes = [1, 2, 4, 8]
    batch_size_input_lens = [
        (16, 10),
        (16, 50),
        (16, 100),
        (16, 500),
        (1, 100),
        (1, 1000),
        (1, 10000),
        (1, 100000),
    ]
    test_params = [
        TestParamGroup(
            worker_param=WorkerParam(
                model_dir = os.environ.get("LWM_WEIGHT_PATH", "Env `LWM_WEIGHT_PATH` is not set!"),
                mode = ["_token_decode_attention_overlapped"],
                sp_world_size = 1,
                tp_world_size = tp_size,
                max_total_token_num = 101000,
                max_req_num = 16,
                max_seq_len = 101000
            ),
            input_params = [
                InputParam(
                    batch_size = batch_size,
                    input_len = input_len,
                    output_len = 16,
                    num_sp_master = 1,
                    need_context_migration = False,
                    num_decoding_stage_migration = 0
                )
                for (batch_size, input_len) in batch_size_input_lens
            ]
        )
        for tp_size in tp_sizes
    ]
    return test_params

def get_ae_figure3_params():
    """
    Get params for figure 3 in artifact evaluation
    """
    sp_and_tp_world_sizes = [
		(1, 8),
		(2, 4),
		(4, 2),
	]
    batch_size_and_input_lens = [
		(512, 1000, [1, 2, 4]),
		(128, 5000, [1, 1, 2]),
		(64, 10000, [1, 1, 2]),
		(16, 50000, [1, 1, 1]),
		(4, 100000, [1, 1, 1]),
		(1, 500000, [1, 1, 1])
	]
    test_params = [
        TestParamGroup(
            worker_param=WorkerParam(
                model_dir = os.environ.get("LWM_WEIGHT_PATH", "Env `LWM_WEIGHT_PATH` is not set!"),
                mode = ["_token_decode_attention_overlapped"],
                sp_world_size = sp_size,
                tp_world_size = tp_size,
                max_total_token_num = 810000//sp_size,
                max_req_num = 512,
                max_seq_len = 501000
            ),
            input_params = [
                InputParam(
                    batch_size = batch_size,
                    input_len = input_len,
                    output_len = 16,
                    num_sp_master = num_sp_masters[sp_tp_index],
                    need_context_migration = False,
                    num_decoding_stage_migration = 0
                )
                for (batch_size, input_len, num_sp_masters) in batch_size_and_input_lens
            ]
        )
        for sp_tp_index, (sp_size, tp_size) in enumerate(sp_and_tp_world_sizes)
    ]
    return test_params

def get_ae_figure13_params():
    """
    Get params for figure 13 in artifact evaluation
    """
    batch_size_and_input_lens = [
        (1024, 10), (256, 100), (64, 1000), (16, 10000), (4, 50000), (2, 100000), (1, 200000)
    ]
    scale_down_test_params = [
        TestParamGroup(
            worker_param=WorkerParam(
                model_dir = os.environ.get("LWM_WEIGHT_PATH", "Env `LWM_WEIGHT_PATH` is not set!"),
                mode = ["_token_decode_attention_overlapped"],
                sp_world_size = 4,
                tp_world_size = 2,
                max_total_token_num = 210000,
                max_req_num = 1024,
                max_seq_len = 200100
            ),
            input_params = [
                InputParam(
                    batch_size = batch_size,
                    input_len = input_len,
                    output_len = 16,
                    num_sp_master = 1,
                    need_context_migration = need_context_migration,
                    num_decoding_stage_migration = 0
                )
                for (batch_size, input_len) in batch_size_and_input_lens
                for need_context_migration in [False, True]
            ]
        )
    ]
    scale_up_test_params = [
        TestParamGroup(
            worker_param=WorkerParam(
                model_dir = os.environ.get("LWM_WEIGHT_PATH", "Env `LWM_WEIGHT_PATH` is not set!"),
                mode = ["_token_decode_attention_overlapped"],
                sp_world_size = 4,
                tp_world_size = 2,
                max_total_token_num = 210000 // 4,
                max_req_num = 1024,
                max_seq_len = 200100
            ),
            input_params = [
                InputParam(
                    batch_size = batch_size,
                    input_len = input_len,
                    output_len = 16,
                    num_sp_master = num_sp_master,
                    need_context_migration = False,
                    num_decoding_stage_migration = 0
                )
                for (batch_size, input_len) in batch_size_and_input_lens
                for num_sp_master in [1, 2, 4]
            ]
        )
    ]
    return scale_down_test_params + scale_up_test_params

def run_longserve(test_params: list[TestParamGroup], **kwargs):
    from lib.sut_longserve import LongServeSUT
    run_test_params(LongServeSUT, LOONGSERVE_DB_IDENTICAL_REQ_PATH, test_params, **kwargs)

def run_vllm(test_params: list[TestParamGroup], **kwargs):
    import ray
    from lib.sut_vllm import VLLMSUT
    ray.init()
    run_test_params(VLLMSUT, VLLM_DB_IDENTICAL_REQ_PATH, test_params, **kwargs)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    
    if len(sys.argv) == 1:
        print(f"Usage: {sys.argv[0]} <test_group_name>")
        sys.exit(1)
    
    test_group_candidates = {
        "longserve-sp-vs-tp": lambda : run_longserve(get_sp_vs_tp_testing_params()),
        "longserve-time-with-bs": lambda : run_longserve(get_time_with_batch_size_params(False)),
        "longserve-tp-base": lambda : run_longserve(get_tp_benchmark_params()),
        "longserve-scale-down": lambda : run_longserve(get_scale_down_params(), warmup_rounds=2, measure_rounds=8, skip_duplicated=False),
        "longserve-scale-up": lambda : run_longserve(get_scale_up_params(), warmup_rounds=2, measure_rounds=8, skip_duplicated=False),
        "longserve-example": lambda : run_longserve(example_testing_params, warmup_rounds=1, measure_rounds=1, skip_duplicated=False, store_into_db=False),

        "longserve-ae-toy-example": lambda: run_longserve(get_ae_toy_example_params(), warmup_rounds=1, measure_rounds=1, skip_duplicated=False, store_into_db=False),
        "longserve-ae-figure2": lambda: run_longserve(get_ae_figure2_params()),
        "longserve-ae-figure3": lambda: run_longserve(get_ae_figure3_params()),
        "longserve-ae-figure13": lambda: run_longserve(get_ae_figure13_params()),
        "longserve-ae-analytical-model-single-node": lambda: run_longserve(get_time_with_batch_size_params(False)),
        "longserve-ae-analytical-model-multi-node": lambda: run_longserve(get_time_with_batch_size_params(True)),

        "vllm-tp-base": lambda : run_vllm(get_tp_benchmark_params()),
        "vllm-example": lambda : run_vllm(example_testing_params, warmup_rounds=1, measure_rounds=1, skip_duplicated=False, store_into_db=False)
    }
    select_test_group = sys.argv[1]
    if select_test_group not in test_group_candidates:
        print(f"Wrong test group name! Available test group names: {list(test_group_candidates.keys())}")
        sys.exit(1)

    print(f"Selected test group: {select_test_group}")
    test_group_candidates[select_test_group]()
    