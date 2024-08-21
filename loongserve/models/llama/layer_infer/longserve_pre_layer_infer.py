import torch
import torch.distributed as dist

from loongserve.models.llama.layer_weights.longserve_pre_and_post_layer_weight import LongServeLlamaPreAndPostLayerWeight
from loongserve.models.llama.longserve_infer_struct import LongServeLlamaInferStateInfo
from loongserve.utils.infer_utils import mark_cost_time

class LongServeLlamaPreLayerInfer:
    """
    """
    def __init__(self, total_rank, total_world_size, network_config, mode,
                 tp_rank, tp_world_size, tp_group,
                 sp_rank, sp_world_size):
        self.total_rank_ = total_rank
        self.total_world_size_ = total_world_size
        self.network_config_ = network_config
        self.mode = mode
        # LongServe need to set
        self.tp_rank_ = tp_rank
        self.tp_world_size_ = tp_world_size
        self.tp_group_ = tp_group
        
        self.sp_rank_ = sp_rank
        self.sp_world_size_ = sp_world_size
        assert self.total_world_size_ == self.tp_world_size_ * self.sp_world_size_
        assert self.total_rank_ == self.sp_rank_ * self.tp_world_size_ + self.tp_rank_

        self.eps_ = 1e-5
        tp_vocab_size_ = network_config["vocab_size"] // self.tp_world_size_
        self.vob_start_id_ = tp_vocab_size_ * self.tp_rank_
        self.vob_end_id_ = tp_vocab_size_ * (self.tp_rank_ + 1)
        return
    
    @mark_cost_time("pre context forward")
    def context_forward(self, input_ids, infer_state: LongServeLlamaInferStateInfo, layer_weight: LongServeLlamaPreAndPostLayerWeight):
        input_mask = torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_)
        tmp_input_ids = (input_ids - self.vob_start_id_)
        tmp_input_ids[input_mask] = 0
        input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        if self.tp_world_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, group=self.tp_group_, async_op=False)
        return input_embdings

    def token_forward(self, mastering_input_ids, infer_state: LongServeLlamaInferStateInfo, layer_weight: LongServeLlamaPreAndPostLayerWeight):
        input_mask = torch.logical_or(self.vob_start_id_ > mastering_input_ids, mastering_input_ids >= self.vob_end_id_)
        tmp_input_ids = (mastering_input_ids - self.vob_start_id_)
        tmp_input_ids[input_mask] = 0
        input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        if self.tp_world_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, group=self.tp_group_, async_op=False)
        return input_embdings