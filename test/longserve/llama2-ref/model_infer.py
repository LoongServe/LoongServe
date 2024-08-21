import time
import numpy as np
from multiprocessing import Queue
import multiprocessing

from loongserve.models.llama.model import LlamaTpPartModel

def test_model_inference(world_size, model_path, max_total_token_num: int, batch_size, input_len, output_len, mode):
    ans_queue = Queue()
    workers = []
    for rank_id in range(world_size):
        model_kvargs = {
            "tp_rank": rank_id,
            "world_size": world_size,
            "weight_dir": model_path,
            "max_total_token_num": max_total_token_num,
            "load_way": "HF",
            "mode": mode,
            "max_req_num": batch_size,
            "max_seq_length": (input_len + output_len),
            "user_defined_max_position_embeddings": (input_len + output_len),
        }
        
        proc = multiprocessing.Process(target=tppart_model_infer, args=(LlamaTpPartModel, model_kvargs, batch_size, input_len, output_len, ans_queue))
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()

    assert not ans_queue.empty()

    prefill_time_costs = []
    decoding_time_costs = []
    while not ans_queue.empty():
        cur_prefill_time_cost, cur_decoding_time_costs = ans_queue.get()
        prefill_time_costs.append(cur_prefill_time_cost)
        decoding_time_costs.append(cur_decoding_time_costs)

    return np.mean(prefill_time_costs), np.median(np.mean(decoding_time_costs, axis=1))


def tppart_model_infer(model_class, model_kvargs, batch_size, input_len, output_len, ans_queue):
    import torch
    import torch.distributed as dist
    rank_id = model_kvargs["tp_rank"]
    world_size = model_kvargs["world_size"]

    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:28765', rank=rank_id, world_size=world_size)
    torch.cuda.set_device(rank_id)

    import torch.distributed as dist
    dist.barrier()
    torch.cuda.empty_cache()

    model_part = model_class(model_kvargs)

    # warm up
    test_data = np.vstack([np.arange(5, input_len + 5) for _ in range(batch_size)])
    test_data = test_data.reshape(-1)
    test_data = torch.from_numpy(test_data).cuda()

    b_req_idx = model_part.req_manager.alloc(batch_size).int()
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        b_start_loc[i] = i * input_len
        b_seq_len[i] = input_len

    total_token_num = input_len * batch_size
    logics = model_part.forward(batch_size, 
                                total_token_num, 
                                input_len, 
                                test_data,
                                b_req_idx,
                                b_start_loc,
                                b_seq_len,
                                is_prefill=True)
    prob_out = torch.softmax(logics, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    for i in range(output_len):
        b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        total_token_num += batch_size
        b_seq_len += 1
        logics = model_part.forward(batch_size, total_token_num, input_len + i + 1, torch.from_numpy(
            predict_ids).cuda().reshape(-1), b_req_idx, b_start_loc, b_seq_len, is_prefill=False)
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()

    model_part.mem_manager.free_all()
    model_part.req_manager.free_all()
    
    if rank_id == 0:
        print("can use mem size:", model_part.mem_manager.can_use_mem_size)
        print("can use req size:", model_part.req_manager.can_use_req_size)
        
    b_req_idx = None
    b_start_loc = None
    b_seq_len = None
    
    b_req_idx = model_part.req_manager.alloc(batch_size).int()
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        b_start_loc[i] = i * input_len
        b_seq_len[i] = input_len

    dist.barrier()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()

    total_token_num = batch_size * input_len
    logics = model_part.forward(batch_size, total_token_num, input_len, test_data,
                                                 b_req_idx, b_start_loc, b_seq_len, is_prefill=True)
    prob_out = torch.softmax(logics, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    end_event.record()
    torch.cuda.synchronize()
    prefill_time_cost = start_event.elapsed_time(end_event)

    decoding_time_costs = []
    for i in range(output_len):
        b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        total_token_num += batch_size
        b_seq_len += 1

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        logics = model_part.forward(batch_size, total_token_num, input_len + i + 1, torch.from_numpy(
            predict_ids).cuda().reshape(-1), b_req_idx, b_start_loc, b_seq_len, is_prefill=False)
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()

        end_event.record()
        torch.cuda.synchronize()
        decoding_time_costs.append(start_event.elapsed_time(end_event))

    ans_queue.put((prefill_time_cost, decoding_time_costs))

    return


