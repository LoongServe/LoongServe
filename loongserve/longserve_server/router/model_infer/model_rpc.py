import asyncio
import numpy as np
import rpyc
import torch
import time
from transformers.configuration_utils import PretrainedConfig
from loongserve.longserve_server.router.model_infer.infer_batch import InferBatch
# from rpyc.utils.classic import obtain

from loongserve.models.llama.longserve_model import LongServeLlamaModel
from loongserve.utils.infer_utils import set_random_seed
from loongserve.utils.infer_utils import calculate_time, mark_start, mark_end
from .pre_process import prepare_decode_inputs, prepare_prefill_inputs
from .post_process import sample
from .infer_batch import requests_mapping
from .infer_batch import InferReq
from loongserve.utils.log_utils import init_logger

import os
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"
import ray

class ModelRpcServer: #(rpyc.Service):
    def __init__(self) -> None:
        pass
    
    def get_ip(self):
        import socket
        return socket.gethostbyname(socket.gethostname())
    
    def exposed_init_model(self, kvargs):
        self.total_world_size = kvargs["total_world_size"]
        self.total_world_size = kvargs["total_world_size"]

        self.total_rank = kvargs["total_rank"]
        self.tp_rank = kvargs["tp_rank"]
        self.tp_world_size = kvargs["tp_world_size"]
        self.tp_group = None
        self.sp_rank = kvargs["sp_rank"]
        self.sp_world_size = kvargs["sp_world_size"]
        self.sp_rnccl_unique_id = kvargs["sp_rnccl_unique_id"]
        assert self.total_rank == self.sp_rank * self.tp_world_size + self.tp_rank

        self.weight_dir = kvargs["weight_dir"]
        self.max_total_token_num = kvargs["max_total_token_num"]
        self.load_way = kvargs["load_way"]
        self.mode = kvargs["mode"]
        self.max_req_num = kvargs.get("max_req_num", 1000)
        self.max_seq_length = kvargs.get("max_seq_length", 1024 * 5)

        self.user_defined_max_position_embeddings = kvargs.get("user_defined_max_position_embeddings", self.max_seq_length)

        self.max_mig_len = kvargs["max_mig_len"]

        self.cache = {}
        self.logger = init_logger(__name__)

        # NCCL performance tuning
        import os
        os.environ["NCCL_SOCKET_NTHREADS"] = "4"
        os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
        os.environ["NCCL_SOCKET_IFNAME"] = "ib0,ib1,ib2,ib3"
        # # debug
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        import torch
        import torch.distributed as dist
        import rnccl
        torch.set_num_threads(16)   # Speed up weight loading
        dist.init_process_group('nccl', init_method=f'tcp://{kvargs["host"]}:{kvargs["nccl_port"]}', rank=self.total_rank, world_size=self.total_world_size)
        print(f"[sp_rank = {self.sp_rank}/{self.sp_world_size}, tp_rank = {self.tp_rank}/{self.tp_world_size}, total_rank = {self.total_rank}/{self.total_world_size}] available devices: {torch.cuda.device_count()}, current device: { torch.cuda.current_device()}, device name = {ray.get_runtime_context().get_node_id()}, gpu ids = {ray.get_gpu_ids()}, max_mig_len = {self.max_mig_len}", flush=True)

        for i in range(self.sp_world_size):
            tp_rank_list = [i * self.tp_world_size + j for j in range(0, self.tp_world_size)]
            tp_group = dist.new_group(tp_rank_list, backend="nccl")
            if i == self.sp_rank:
                assert self.total_rank in tp_rank_list
                self.tp_group = tp_group
        
        # Force PyTorch creates NCCL communicators before the first P2P call
        dummy_tensor = torch.ones((1,), dtype=torch.float32, device="cuda")
        dist.all_reduce(dummy_tensor, op=dist.ReduceOp.SUM, group=self.tp_group, async_op=False)
        assert dummy_tensor.item() == self.tp_world_size
        dummy_tensor = None

        model_kvargs = {
            # LongServe related parameters
            "total_rank": self.total_rank,
            "total_world_size": self.total_world_size,
            "tp_rank": self.tp_rank,
            "tp_world_size": self.tp_world_size,
            "tp_group": self.tp_group,
            "sp_rank": self.sp_rank,
            "sp_world_size": self.sp_world_size,
            "sp_rnccl_unique_id": self.sp_rnccl_unique_id,

            # common parameters
            "world_size": self.tp_world_size,
            "weight_dir": self.weight_dir,
            "max_total_token_num": self.max_total_token_num,
            "load_way": self.load_way,
            "mode": self.mode,
            "max_req_num": self.max_req_num,
            "max_seq_length": self.max_seq_length,

            "user_defined_max_position_embeddings": self.user_defined_max_position_embeddings,
        }

        model_cfg, _ = PretrainedConfig.get_config_dict(
            self.weight_dir
        )

        try:
            self.model_type = model_cfg["model_type"]
            if self.model_type == "llama":
                self.model = LongServeLlamaModel(model_kvargs)
            else:
                raise Exception(f"can not support {self.model_type} now")
        except Exception as e:
            self.logger.error(f"load model error: {str(e)} {e} {type(e)}")
            import traceback
            traceback.print_exc()
            raise e
        
        set_random_seed(2147483647)

        print(f"[{self.sp_rank, self.tp_rank}] Number of k/v cache slots: {self.model.mem_manager.can_use_mem_size}", flush=True)
        print(f"[{self.sp_rank, self.tp_rank}] Number of request slots: {self.model.req_manager.can_use_req_size}", flush=True)
        return
    
    def exposed_add_batch(self, batch_id, reqs, dtype="fp16"):
        import torch
        if dtype == "fp16":
            dtype = torch.float16
        else:
            assert False, "error dtype"
        batch_data = InferBatch.init_batch(batch_id, reqs, dtype, torch.cuda.current_device(), self.model.req_manager, self.model.vocab_size)
        self.cache[batch_id] = batch_data
    
    def exposed_prefill_batch(self, batch_id, reqs, global_input_kvargs: tuple):
        self.exposed_add_batch(batch_id, reqs, "fp16")
        return self.forward(batch_id, global_input_kvargs, is_prefill=True)

    def exposed_decode_batch(self, batch_id, global_input_kvargs: tuple, cuda_sync_afterwards: bool = False):
        return self.forward(batch_id, global_input_kvargs, is_prefill=False, cuda_sync_afterwards = cuda_sync_afterwards)

    def exposed_filter_batch(self, batch_id, req_id_list, finished_req_id_list):
        batch: InferBatch = self.cache.pop(batch_id)
        filter_batch = batch.filter(req_id_list, finished_req_id_list)
        del batch
        self.cache[batch_id] = filter_batch
        return

    def exposed_pause_reqs(self, batch_id, req_list):
        batch1: InferBatch = self.cache.pop(batch_id)
        batch2 = batch1.pause_reqs(req_list)
        self.cache[batch_id] = batch2
        del batch1
        return

    def exposed_merge_batch(self, batch_id1, batch_id2):
        batch1 = self.cache.pop(batch_id1)
        batch2 = self.cache.pop(batch_id2)
        m_batch = InferBatch.merge(batch1, batch2)
        del batch1
        del batch2
        self.cache[batch_id1] = m_batch
        return

    def exposed_remove_batch(self, batch_id):
        batch: InferBatch = self.cache.pop(batch_id)
        batch.free_self()
        del batch
        return
    
    @torch.inference_mode()
    def exposed_migrate_batch(self, batch_id, is_receiver, request_ids, b_migration_len, peer_sp_rank):
        batch: InferBatch = self.cache.pop(batch_id)

        assert len(request_ids) == len(b_migration_len)
        while len(request_ids) > 0:
            
            idle_buf_size = self.max_mig_len
            nopad_b_req_idx = []
            nopad_b_migration_len = []
            nopad_b_seq_len = []
            while len(request_ids) > 0 and idle_buf_size > 0:
                request_id: int = request_ids.pop()
                migration_len: int = b_migration_len.pop()

                req : InferReq = requests_mapping[request_id]
                cur_mig_len = min(idle_buf_size, migration_len)

                idle_buf_size -= cur_mig_len
                migration_len -= cur_mig_len

                nopad_b_req_idx.append(req.req_idx)
                nopad_b_migration_len.append(cur_mig_len)
                nopad_b_seq_len.append(req.prompt_len)
                
                req.prompt_len += (cur_mig_len if is_receiver else -cur_mig_len)
                req.cur_kv_len += (cur_mig_len if is_receiver else -cur_mig_len)

                if idle_buf_size == 0:
                    if migration_len > 0:
                        request_ids.append(request_id)
                        b_migration_len.append(migration_len)
                    break
            
            nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device='cuda')
            nopad_b_migration_len = torch.tensor(nopad_b_migration_len, dtype=torch.int32, device='cuda')
            nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device='cuda')

            self.model.decoding_stage_migration(is_receiver, nopad_b_req_idx, nopad_b_migration_len, nopad_b_seq_len, peer_sp_rank)

        self.cache[batch_id] = batch
    
    @torch.inference_mode()
    def forward(self, batch_id, global_input_kvargs, is_prefill, cuda_sync_afterwards: bool = False):
        output_dict = []
        batch: InferBatch = self.cache.pop(batch_id)
        if is_prefill:
            kwargs, run_reqs, not_run_reqs = prepare_prefill_inputs(batch, global_input_kvargs, self)
            peer_run_reqs = []
        else:
            kwargs, run_reqs, peer_run_reqs, not_run_reqs = prepare_decode_inputs(batch, global_input_kvargs, self)
        
        if len(run_reqs) + len(peer_run_reqs) >= 1:
            logits = self.model.forward(**kwargs)
            if cuda_sync_afterwards:
                torch.cuda.synchronize()
            
            if self.tp_rank > 0:
                for req_obj in run_reqs:
                    req_obj.cur_kv_len = req_obj.prompt_len
                self.cache[batch.batch_id] = batch
                return

            if logits is None:
                dummy_list = [None for _ in range(len(run_reqs))]
                next_token_ids, next_token_logprobs = dummy_list, dummy_list
            else:
                next_token_ids, next_token_probs = sample(logits, run_reqs)
                next_token_ids = next_token_ids.detach().cpu().numpy()
                next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

            if is_prefill:
                should_decode: set = global_input_kvargs[4]
            for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
                # prefill and decode is same
                req_obj.cur_kv_len = req_obj.prompt_len
                if next_token_id is not None and (not is_prefill or req_obj.r_id in should_decode):
                    req_obj.out_token_id_count[next_token_id] += 1 # FIXME: fix it
                    next_token_id = int(next_token_id)
                    metadata = {
                        'id': next_token_id,
                        'logprob': float(next_token_logprob) if next_token_logprob is not None else None,
                    }
                    output_dict.append((req_obj.r_id, req_obj.cur_kv_len, next_token_id, metadata)) # req_id, cur_kv_len, token_id, metadata
                elif is_prefill:
                    output_dict.append((req_obj.r_id, req_obj.cur_kv_len, None, None)) # req_id, cur_kv_len, token_id, metadata

        self.cache[batch.batch_id] = batch
        if self.tp_rank > 0:
            return
        return output_dict

