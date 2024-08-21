import os, sys, socket
import ray

import torch
import torch.distributed as dist

from loongserve.models.llama.longserve_model import LongServeLlamaModel

@ray.remote(
    num_cpus=16,
    num_gpus=1
)
class LongServeWorker:
    def __init__(self):
        pass

    def get_ip(self):
        return socket.gethostbyname(socket.gethostname())

    @torch.inference_mode()
    def init_model(self, nccl_host: str, nccl_port: int, model_kvargs):
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        # NCCL performance tuning
        torch.set_grad_enabled(False)
        torch.set_num_threads(16)   # Speed up weight loading
        os.environ["NCCL_SOCKET_NTHREADS"] = "4"
        os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
        os.environ["NCCL_CHECKS_DISABLE"] = "1"
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        os.environ["NCCL_SOCKET_IFNAME"] = "ib0,ib1,ib2,ib3"
        
        # os.environ["NCCL_MAX_NCHANNELS"] = "4"
        
        self.tp_rank_id = model_kvargs["tp_rank"]
        self.tp_world_size = model_kvargs["tp_world_size"]
        self.sp_rank_id = model_kvargs["sp_rank"]
        self.sp_world_size = model_kvargs["sp_world_size"]
        self.rank_id = model_kvargs["total_rank"]
        self.world_size = model_kvargs["total_world_size"]

        print(f"{self.sp_rank_id, self.tp_rank_id} Worker initializing with sp_rank = {self.sp_rank_id}/{self.sp_world_size}, tp_rank = {self.tp_rank_id}/{self.tp_world_size}, total_rank = {self.rank_id}/{self.world_size}")
        
        dist.init_process_group('nccl', init_method=f'tcp://{nccl_host}:{nccl_port}', rank=self.rank_id, world_size=self.world_size)

        for i in range(self.sp_world_size):
            tp_rank_list = [i * self.tp_world_size + j for j in range(0, self.tp_world_size)]
            tp_group = dist.new_group(tp_rank_list, backend="nccl")
            if i == self.sp_rank_id:
                assert self.rank_id in tp_rank_list
                model_kvargs["tp_group"] = tp_group
        
        # Force PyTorch creates NCCL communicators before the first P2P call
        dummy_tensor = torch.ones((1,), dtype=torch.float32, device="cuda")
        dist.all_reduce(dummy_tensor, op=dist.ReduceOp.SUM, group=model_kvargs["tp_group"], async_op=False)
        assert dummy_tensor.item() == self.tp_world_size
        dummy_tensor = None

        dist.barrier(device_ids=[torch.cuda.current_device()])
        torch.cuda.empty_cache()

        self.model_part = LongServeLlamaModel(model_kvargs)

        print(f"{self.sp_rank_id, self.tp_rank_id} Worker initialized with sp_rank = {self.sp_rank_id}/{self.sp_world_size}, tp_rank = {self.tp_rank_id}/{self.tp_world_size}, total_rank = {self.rank_id}/{self.world_size}")
        print(f"{self.sp_rank_id, self.tp_rank_id} Number of k/v cache slots: {self.model_part.mem_manager.can_use_mem_size}")
        print(f"{self.sp_rank_id, self.tp_rank_id} Number of request slots: {self.model_part.req_manager.can_use_req_size}")

    @torch.inference_mode()
    def forward(self, input_kvargs: dict, clear_mem: bool):
        for key, value in input_kvargs.items():
            if isinstance(value, torch.Tensor):
                input_kvargs[key] = value.detach().contiguous().clone().to(torch.cuda.current_device())
                
        if input_kvargs["is_prefill"]:
            input_kvargs["b_req_idx"] = self.model_part.req_manager.alloc(input_kvargs["batch_size"]).int().clone()
        if clear_mem:
            self.model_part.mem_manager.free_all()
            self.model_part.req_manager.free_all()

        dist.barrier()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
    
        local_logics = self.model_part.forward(**input_kvargs)

        end_event.record()
        torch.cuda.synchronize()
        time_cost = start_event.elapsed_time(end_event)

        local_predict_ids = None
        if not input_kvargs["is_prefill"] and input_kvargs["sp_master_rank"] == self.rank_id:
            local_prob_out = torch.softmax(local_logics, dim=-1)
            local_predict_ids = torch.argmax(local_prob_out, dim=1, keepdim=True).detach().contiguous().clone().cpu()
        elif input_kvargs["is_prefill"]:
            # Here we return all predict ids during the prefill phase. It is
            # the controller's responsibility to decide which one to use.
            local_prob_out = torch.softmax(local_logics, dim=-1)
            local_predict_ids = torch.argmax(local_prob_out, dim=1, keepdim=True).detach().contiguous().clone().cpu()

        output_kvargs = {
            "b_req_idx": input_kvargs["b_req_idx"],
            "local_predict_ids": local_predict_ids,
            "time_cost": time_cost,
        }
        return output_kvargs
        
    @torch.inference_mode()
    def migrate(self, input_kvargs: dict):
        for key, value in input_kvargs.items():
            if isinstance(value, torch.Tensor):
                input_kvargs[key] = value.detach().contiguous().clone().to(torch.cuda.current_device())

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        self.model_part.decoding_stage_migration(**input_kvargs)

        end_event.record()
        torch.cuda.synchronize()
        time_cost = start_event.elapsed_time(end_event)
