import math
import time
import random

import numpy as np
import torch
import rnccl
from transformers import AutoTokenizer

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from lib.sut import SystemUnderTest, get_input_ids
from lib.worker import LongServeWorker
from lib.structs import *

class LongServeSUT(SystemUnderTest):
    """
    The system under testing. It is intended to be recreated when model parameters
    or worker parameters change, and then fed with a bunch of different input
    parameters.

    Upon being fed with input parameters, it returns both the predict result and
    time usage, making it suitable for both correctness verification and performance
    testing.
    """

    # Only one instance of LongServeSUT should exist concurrently
    instance_count: int = 0

    def __init__(
        self,
        worker_param: WorkerParam
    ):
        self.instance_count += 1
        assert self.instance_count == 1, "Only one instance of SystemUnderTest should exist concurrently"
        
        self.model_dir = worker_param.model_dir
        self.sp_world_size = worker_param.sp_world_size
        self.tp_world_size = worker_param.tp_world_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=False)
        self.total_world_size = self.sp_world_size * self.tp_world_size

        print("Creating workers...")
        assert self.sp_world_size * self.tp_world_size <= 16, "world size > 16 is not implemented"
        self.workers = [None for _ in range(self.total_world_size)]
        if self.total_world_size <= 8:
            # Schedule all workers on one node
            print("Scheduling all workers on one node...")
            placement_group = ray.util.placement_group(
                bundles = [{"CPU": 16, "GPU": 1} for _ in range(self.total_world_size)],
                strategy="STRICT_PACK"
            )
            ray.get(placement_group.ready(), timeout=20)
            for sp_rank_id in range(self.sp_world_size):
                for tp_rank_id in range(self.tp_world_size):
                    total_rank = sp_rank_id * self.tp_world_size + tp_rank_id
                    worker = LongServeWorker.options(
                        name=f"Worker sp={sp_rank_id} tp={tp_rank_id}",
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=placement_group,
                            placement_group_capture_child_tasks=True
                        )
                    ).remote()
                    self.workers[total_rank] = worker
        elif self.tp_world_size > self.sp_world_size and self.tp_world_size <= 8:
            # Schedule all workers in the same TP group on one node
            print("Scheduling all workers in the same TP group on one node...")
            for sp_rank_id in range(self.sp_world_size):
                placement_group = ray.util.placement_group(
                    bundles = [{"CPU": 16, "GPU": 1} for _ in range(self.tp_world_size)],
                    strategy="STRICT_PACK",
                )
                ray.get(placement_group.ready(), timeout=20)
                for tp_rank_id in range(self.tp_world_size):
                    total_rank = sp_rank_id * self.tp_world_size + tp_rank_id
                    worker = LongServeWorker.options(
                        name=f"Worker sp={sp_rank_id} tp={tp_rank_id}",
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=placement_group,
                            placement_group_capture_child_tasks=True
                        )
                    ).remote()
                    self.workers[total_rank] = worker
        elif self.tp_world_size <= self.sp_world_size and self.sp_world_size <= 8:
            # Schedule all workers in the same SP group on one node
            print("Scheduling all workers in the same SP group on one node...")
            for tp_rank_id in range(self.tp_world_size):
                placement_group = ray.util.placement_group(
                    bundles = [{"CPU": 16, "GPU": 1} for _ in range(self.sp_world_size)],
                    strategy="STRICT_PACK",
                )
                ray.get(placement_group.ready(), timeout=20)
                for sp_rank_id in range(self.sp_world_size):
                    total_rank = sp_rank_id * self.tp_world_size + tp_rank_id
                    worker = LongServeWorker.options(
                        name=f"Worker sp={sp_rank_id} tp={tp_rank_id}",
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=placement_group,
                            placement_group_capture_child_tasks=True
                        )
                    ).remote()
                    self.workers[total_rank] = worker
        else:
            print(f"Warning: Cannot schedule all workers in the same TP/SP group onto one node.")
            for sp_rank_id in range(self.sp_world_size):
                for tp_rank_id in range(self.tp_world_size):
                    total_rank = sp_rank_id * self.tp_world_size + tp_rank_id
                    worker = LongServeWorker.options(
                        name=f"Worker sp={sp_rank_id} tp={tp_rank_id}"
                    ).remote()
                    self.workers[total_rank] = worker
        
        print("Initializing models...")
        nccl_host = ray.get(self.workers[0].get_ip.remote())
        nccl_port = 14250
        init_model_rets = []
        for tp_rank_id in range(self.tp_world_size):
            sp_rnccl_unique_id = rnccl.get_nccl_unique_id()
            for sp_rank_id in range(self.sp_world_size):
                total_rank = sp_rank_id * self.tp_world_size + tp_rank_id
                model_kvargs = {
                    # LongServe related parameters
                    "total_rank": total_rank,
                    "total_world_size": self.total_world_size,
                    "tp_rank": tp_rank_id,
                    "tp_world_size": self.tp_world_size,
                    "tp_group": None,
                    "sp_rank": sp_rank_id,
                    "sp_world_size": self.sp_world_size,
                    "sp_rnccl_unique_id": sp_rnccl_unique_id,

                    # common parameters
                    "world_size": self.tp_world_size,
                    "weight_dir": self.model_dir,
                    "max_total_token_num": worker_param.max_total_token_num,
                    "load_way": "HF",
                    "mode": worker_param.mode,
                    "max_req_num": worker_param.max_req_num,
                    "max_seq_length": worker_param.max_seq_len,

                    "user_defined_max_position_embeddings": worker_param.max_seq_len,
                }
                init_model_rets.append(self.workers[total_rank].init_model.remote(nccl_host, nccl_port, model_kvargs))
        
        ray.get(init_model_rets)
    
    def __del__(self):
        self.instance_count -= 1
        assert self.instance_count == 0
        del self.workers
        time.sleep(2)

    @torch.inference_mode()
    def inference(
        self,
        input_param: InputParam
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], float, list[float]]:   # Return: [input_ids, predict_ids, predict_texts, prefill_time_usage, decoding_time_usages]
        """
        Run the system under a specific set of input parameters, and return the
        time consumption of the prefill phase and the decoding phase, respectively
        """
        batch_size = input_param.batch_size
        input_len = input_param.input_len
        output_len = input_param.output_len
        num_sp_master = input_param.num_sp_master

        prefill_time_usage = None
        decoding_time_usages = [None for _ in range(output_len-1)]
        predict_ids_list = []

        input_ids = get_input_ids(self.model_dir, batch_size*input_len)
        b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cpu")
        b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cpu")
        for i in range(batch_size):
            b_start_loc[i] = i * input_len
            b_seq_len[i] = input_len
        
        global_to_local_req_idx_table = [None for _ in range(self.total_world_size)]
        global_to_local_seq_len_table = [None for _ in range(self.total_world_size)]
        global_to_logical_sp_rank_table = [None for _ in range(self.total_world_size)]

        # Prefill Phase
        worker_rets = []
        # Send input data to workers
        for total_rank_id in range(self.total_world_size):
            sp_rank_id = total_rank_id // self.tp_world_size
            tp_rank_id = total_rank_id % self.tp_world_size

            local_batch_size = batch_size

            logical_sp_peer_ranks = [(self.sp_world_size - i - 1) * self.tp_world_size + tp_rank_id for i in range(self.sp_world_size)]
            logical_sp_rank = logical_sp_peer_ranks.index(total_rank_id)
            global_to_logical_sp_rank_table[total_rank_id] = logical_sp_rank

            # The current worker will have tokens `global_start`, `global_start+sp_world_size` ...
            global_start = logical_sp_rank
            local_b_start_loc = torch.empty((local_batch_size,), dtype=b_start_loc.dtype, device="cpu")
            local_b_seq_len = torch.empty((local_batch_size,), dtype=b_seq_len.dtype, device="cpu")
            local_first_token_global_idx = torch.full((local_batch_size,), fill_value=global_start, dtype=b_start_loc.dtype, device="cpu")

            start = 0
            for i in range(local_batch_size):
                local_b_start_loc[i] = start
                cur_seq_len = (b_seq_len[i] - global_start + self.sp_world_size - 1) // self.sp_world_size
                local_b_seq_len[i] = cur_seq_len
                start += cur_seq_len
            local_total_token_num = int(start)

            if input_param.need_context_migration:
                if sp_rank_id == 0:
                    kv_cache_index_begin = torch.full((local_batch_size, ), fill_value=0, dtype=torch.int32, device="cpu")
                    kv_cache_index_end = torch.full((local_batch_size, ), fill_value=input_len, dtype=torch.int32, device="cpu")
                else:
                    kv_cache_index_begin = torch.full((local_batch_size, ), fill_value=0, dtype=torch.int32, device="cpu")
                    kv_cache_index_end = torch.full((local_batch_size, ), fill_value=0, dtype=torch.int32, device="cpu")
                global_to_local_seq_len_table[total_rank_id] = kv_cache_index_end - kv_cache_index_begin
            else:
                kv_cache_index_begin = kv_cache_index_end = None
                global_to_local_seq_len_table[total_rank_id] = local_b_seq_len.clone()

            local_input_ids = torch.zeros((local_total_token_num, ), dtype=input_ids.dtype, device="cpu")
            for i in range(local_batch_size):
                local_input_ids[local_b_start_loc[i]: local_b_start_loc[i]+local_b_seq_len[i]] = \
                    input_ids[b_start_loc[i]+global_start: b_start_loc[i]+b_seq_len[i]: self.sp_world_size] \
                    .contiguous().clone()
            
            max_token_on_one_worker = (input_len + self.sp_world_size - 1) // self.sp_world_size
            max_token_num = max_token_on_one_worker * local_batch_size
            local_max_len_in_batch = max_token_on_one_worker

            input_kvargs = {
                "batch_size": local_batch_size,
                "total_token_num": local_total_token_num,
                "max_token_num": max_token_num,
                "max_len_in_batch": local_max_len_in_batch,
                "input_ids": local_input_ids,
                "b_req_idx": None,
                "b_start_loc": local_b_start_loc,
                "b_seq_len": local_b_seq_len,
                "first_token_global_idx": local_first_token_global_idx,
                "sp_master_rank": None,
                "logical_sp_peer_ranks": logical_sp_peer_ranks,
                "logical_sp_rank": logical_sp_rank,
                "newkv_alloc_sp_rank": None,
                "peer_sp_master_rank_list": None,
                "peer_query_buffer_range_list": None,
                "peer_batch_size": None,
                "peer_max_len_in_batch": None,
                "peer_b_req_idx": None,
                "peer_b_seq_len": None,
                "multimodal_params": None,
                "is_prefill": True,
                "need_context_migration": input_param.need_context_migration,
                "kv_cache_index_begin": kv_cache_index_begin,
                "kv_cache_index_end": kv_cache_index_end,
            }
            worker_rets.append(self.workers[total_rank_id].forward.remote(input_kvargs, True))

        # Gather output data from workers
        predict_ids = None
        prefill_time_usages = []
        for total_rank_id in range(self.total_world_size):
            sp_rank_id = total_rank_id // self.tp_world_size
            tp_rank_id = total_rank_id % self.tp_world_size

            output_kvargs = ray.get(worker_rets[total_rank_id])
            prefill_time_usages.append(output_kvargs["time_cost"])
            global_to_local_req_idx_table[total_rank_id] = output_kvargs["b_req_idx"].clone()

            if output_kvargs["local_predict_ids"] is not None and \
                (input_len-1) % self.sp_world_size == global_to_logical_sp_rank_table[total_rank_id]:
                output_kvargs["local_predict_ids"] = output_kvargs["local_predict_ids"].detach().clone().cpu()
                predict_ids = output_kvargs["local_predict_ids"]
            
            global_to_local_req_idx_table[total_rank_id] = output_kvargs["b_req_idx"].clone()

        predict_ids_list.append(predict_ids)  # predict_ids: [batch_size]
        prefill_time_usage = np.min(prefill_time_usages)
        
        # "Random shuffle" - Randomly perform decoding stage migration
        if self.sp_world_size != 1:
            for i in range(input_param.num_decoding_stage_migration):
                print(f"Migration #{i}")
                while True:
                    src_sp_rank = random.randint(0, self.sp_world_size-1)
                    dst_sp_rank = random.randint(0, self.sp_world_size-1)
                    if src_sp_rank == dst_sp_rank:
                        continue
                    migrate_len = torch.tensor([
                        random.randint(
                            0,
                            min(
                                global_to_local_seq_len_table[src_sp_rank*self.tp_world_size+0][i],
                                input_len - global_to_local_seq_len_table[dst_sp_rank*self.tp_world_size+0][i]
                            )
                        )
                        for i in range(batch_size)
                    ], dtype=torch.int32, device="cpu")
                    print(f"Migrating {torch.sum(migrate_len)} tokens from {src_sp_rank} to {dst_sp_rank}")
                    request_ids = torch.arange(0, batch_size, dtype=torch.int32, device="cpu")
                    src_worker_rets = []
                    dst_worker_rets = []
                    for tp_rank_id in range(self.tp_world_size):
                        src_total_rank = src_sp_rank * self.tp_world_size + tp_rank_id
                        dst_total_rank = dst_sp_rank * self.tp_world_size + tp_rank_id
                        src_worker_rets.append(
                            self.workers[src_total_rank].migrate.remote({
                                "is_receiver": False,
                                "b_req_idx": request_ids,
                                "b_migration_len": migrate_len,
                                "b_seq_len": global_to_local_seq_len_table[src_total_rank],
                                "peer_sp_rank": dst_sp_rank
                            })
                        )
                        dst_worker_rets.append(
                            self.workers[dst_total_rank].migrate.remote({
                                "is_receiver": True,
                                "b_req_idx": request_ids,
                                "b_migration_len": migrate_len,
                                "b_seq_len": global_to_local_seq_len_table[dst_total_rank],
                                "peer_sp_rank": src_sp_rank
                            })
                        )
                    for tp_rank_id in range(self.tp_world_size):
                        src_total_rank = src_sp_rank * self.tp_world_size + tp_rank_id
                        dst_total_rank = dst_sp_rank * self.tp_world_size + tp_rank_id
                        ray.get(src_worker_rets[tp_rank_id])
                        ray.get(dst_worker_rets[tp_rank_id])
                        global_to_local_seq_len_table[src_total_rank] -= migrate_len
                        global_to_local_seq_len_table[dst_total_rank] += migrate_len

                    break

        # Multi-master decoding phase
        first_token_global_idx = b_seq_len - 1
        max_mini_batch_size = math.ceil(batch_size / num_sp_master)
        mini_batch_range_list = []
        mini_batch_size_list = []
        for i in range(num_sp_master):
            start = i * max_mini_batch_size
            end = min((i + 1) * max_mini_batch_size, batch_size)
            mini_batch_range_list.append((start, end))
            mini_batch_size_list.append(end - start)
            if end == batch_size:
                num_sp_master = i + 1
                break

        for iter_i in range(output_len-1):
            # Generate input data for workers
            first_token_global_idx += 1
            worker_rets = []
            for total_rank_id in range(self.total_world_size):
                sp_rank_id = total_rank_id // self.tp_world_size
                tp_rank_id = total_rank_id % self.tp_world_size

                logical_sp_peer_ranks = [(self.sp_world_size - i - 1) * self.tp_world_size + tp_rank_id for i in range(self.sp_world_size)]
                logical_sp_rank = logical_sp_peer_ranks.index(total_rank_id)

                is_master_rank = (sp_rank_id < num_sp_master)
                mini_batch_size = mini_batch_size_list[sp_rank_id] if is_master_rank else 0
                mini_batch_start, mini_batch_end = mini_batch_range_list[sp_rank_id] if is_master_rank else (batch_size, batch_size)

                local_batch_size = mini_batch_size if is_master_rank else 0
                local_total_token_num = local_batch_size if is_master_rank else None

                local_input_ids = predict_ids_list[-1][mini_batch_start:mini_batch_end, ...].reshape(-1).clone() if is_master_rank else None
                local_b_req_idx = global_to_local_req_idx_table[total_rank_id][mini_batch_start:mini_batch_end, ...].clone() if is_master_rank else None

                if is_master_rank:
                    global_to_local_seq_len_table[total_rank_id][mini_batch_start:mini_batch_end, ...] += 1
                local_b_seq_len = global_to_local_seq_len_table[total_rank_id][mini_batch_start:mini_batch_end, ...].clone() if is_master_rank else None
                local_first_token_global_idx = first_token_global_idx[mini_batch_start:mini_batch_end, ...].clone() if is_master_rank else None
                local_max_len_in_batch = input_len+iter_i+1
                
                sp_master_rank = total_rank_id if is_master_rank else None
                newkv_alloc_sp_rank = sp_master_rank if is_master_rank else None

                peer_sp_master_rank_list = [sp_i * self.tp_world_size + tp_rank_id for sp_i in range(num_sp_master) if sp_i != sp_rank_id]
                if len(peer_sp_master_rank_list) == 0:
                    peer_sp_master_rank_list = None
                peer_query_buffer_range_list = [
                    (
                        mini_batch_range_list[sp_i][0] - (mini_batch_end - mini_batch_start if sp_i > sp_rank_id else 0),
                        mini_batch_range_list[sp_i][1] - (mini_batch_end - mini_batch_start if sp_i > sp_rank_id else 0)
                    )
                        for sp_i in range(num_sp_master) if sp_i != sp_rank_id
                ]
                peer_batch_size = batch_size - mini_batch_size
                peer_max_len_in_batch = local_max_len_in_batch
                peer_b_req_idx = torch.cat((global_to_local_req_idx_table[total_rank_id][:mini_batch_start, ...].clone(), global_to_local_req_idx_table[total_rank_id][mini_batch_end:, ...].clone()))
                peer_b_seq_len = torch.cat((global_to_local_seq_len_table[total_rank_id][:mini_batch_start, ...].clone(), global_to_local_seq_len_table[total_rank_id][mini_batch_end:, ...].clone()))

                input_kvargs = {
                    "batch_size": local_batch_size,
                    "total_token_num": local_total_token_num,
                    "max_token_num": None,
                    "max_len_in_batch": local_max_len_in_batch,
                    "input_ids": local_input_ids,
                    "b_req_idx": local_b_req_idx,
                    "b_start_loc": None,
                    "b_seq_len": local_b_seq_len,
                    "first_token_global_idx": local_first_token_global_idx,
                    "sp_master_rank": sp_master_rank,
                    "logical_sp_peer_ranks": logical_sp_peer_ranks,
                    "logical_sp_rank": logical_sp_rank,
                    "newkv_alloc_sp_rank": newkv_alloc_sp_rank,
                    "peer_sp_master_rank_list": peer_sp_master_rank_list,
                    "peer_query_buffer_range_list": peer_query_buffer_range_list,
                    "peer_batch_size": peer_batch_size,
                    "peer_max_len_in_batch": peer_max_len_in_batch,
                    "peer_b_req_idx": peer_b_req_idx,
                    "peer_b_seq_len": peer_b_seq_len,
                    "need_context_migration": False,
                    "kv_cache_index_begin": None,
                    "kv_cache_index_end": None,
                    "multimodal_params": None,
                    "is_prefill": False
                }
                worker_rets.append(
                    self.workers[total_rank_id].forward.remote(
                        input_kvargs,
                        False
                    )
                )
            
            # Gather output data from workers
            predict_ids = torch.full((batch_size, 1), fill_value=-1, dtype=torch.int32, device="cpu")
            cur_step_decoding_time_usages = []
            for total_rank_id in range(self.total_world_size):
                sp_rank_id = total_rank_id // self.tp_world_size
                tp_rank_id = total_rank_id % self.tp_world_size
                
                output_kvargs = ray.get(worker_rets[total_rank_id])
                if output_kvargs["local_predict_ids"] is not None:
                    output_kvargs["local_predict_ids"] = output_kvargs["local_predict_ids"].detach().clone().cpu()
                    mini_batch_start, mini_batch_end = mini_batch_range_list[sp_rank_id]
                    predict_ids[mini_batch_start:mini_batch_end, ...] = output_kvargs["local_predict_ids"].detach().clone().cpu()
                cur_step_decoding_time_usages = output_kvargs["time_cost"]
            predict_ids_list.append(predict_ids)
            decoding_time_usages[iter_i] = np.min(cur_step_decoding_time_usages)

        # current predict_ids: [output_len, batch_size, 1]
        predict_ids = torch.stack(predict_ids_list, dim=0).squeeze(-1).transpose(0, 1) # [batch_size, output_len]
        predict_texts = [self.tokenizer.decode(predict_ids[i, :].tolist()) for i in range(batch_size)]
        
        return input_ids, predict_ids, predict_texts, prefill_time_usage, decoding_time_usages
    