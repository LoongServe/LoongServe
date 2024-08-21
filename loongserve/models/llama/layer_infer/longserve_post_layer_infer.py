import torch
import torch.distributed as dist

from loongserve.models.llama.layer_weights.longserve_pre_and_post_layer_weight import LongServeLlamaPreAndPostLayerWeight
from einops import rearrange
from loongserve.models.llama.longserve_infer_struct import LongServeLlamaInferStateInfo
from loongserve.models.llama.triton_kernel.rmsnorm import rmsnorm_forward

class LongServeLlamaPostLayerInfer:
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

        self.eps_ = network_config["rms_norm_eps"]
        self.vocab_size_ = network_config["vocab_size"]
        self.embed_dim_ = network_config["n_embed"]
        return
    
    def _norm(self, input, infer_state, layer_weight:LongServeLlamaPreAndPostLayerWeight) -> torch.Tensor:
        return rmsnorm_forward(input, layer_weight.final_norm_weight_, eps=self.eps_)
    
    def _slice_get_last_input(self, input_embdings, infer_state: LongServeLlamaInferStateInfo):
        if infer_state.is_prefill and not infer_state.return_all_prompt_logprobs:
            batch_size = infer_state.batch_size
            last_input = torch.empty((batch_size, self.embed_dim_), device=input_embdings.device, dtype=torch.float16)
            last_index = infer_state.q_b_start_loc + infer_state.q_b_seq_len - 1
            last_input[:, :] = input_embdings[last_index, :]
            return last_input, batch_size

        if infer_state.is_prefill and infer_state.return_all_prompt_logprobs:
            total_tokens = infer_state.total_token_num
            return input_embdings, total_tokens
        
        if not infer_state.is_prefill:
            batch_size = infer_state.batch_size
            # TODO Why do we need to slice here?
            return input_embdings[-batch_size:, :], batch_size
        
        assert False, "Error State"
    
    def soft_max(self, data):
        return torch.softmax(data.permute(1, 0).float(), dim=-1)
    
    def context_forward(self, input_embdings, infer_state: LongServeLlamaInferStateInfo, layer_weight: LongServeLlamaPreAndPostLayerWeight, return_logics=False):
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        input_embdings = None
        last_input = self._norm(last_input, infer_state, layer_weight)
        last_input = rearrange(last_input, "batch embed_dim -> embed_dim batch").contiguous().reshape(-1, token_num)
        logic_batch = torch.mm(layer_weight.lm_head_weight_, last_input)

        last_input = None
        if self.tp_world_size_ == 1:
            gather_data = logic_batch
        else:
            gather_data = torch.empty((self.vocab_size_, token_num), device=logic_batch.device, dtype=torch.float16)
            split_size = self.vocab_size_ // self.tp_world_size_
            dist.all_gather([gather_data[i * split_size: (i + 1) * split_size, :]
                            for i in range(self.tp_world_size_)], logic_batch, group=self.tp_group_, async_op=False)
        logic_batch = None

        if not return_logics:
            prob_out = self.soft_max(gather_data)
            gather_data = None
            return prob_out
        else:
            ans_logics = gather_data.permute(1, 0).float()
            gather_data = None
            return ans_logics
    
    def token_forward(
        self,
        input_embdings,   # [num_mastering_inputs, embed_dim]
        infer_state: LongServeLlamaInferStateInfo,
        layer_weight: LongServeLlamaPreAndPostLayerWeight,
        return_logics=False
    ) -> torch.Tensor:
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        input_embdings = None
        last_input = self._norm(last_input, infer_state, layer_weight)
        last_input = rearrange(last_input, "batch embed_dim -> embed_dim batch").contiguous().reshape(-1, token_num)
        logic_batch = torch.mm(layer_weight.lm_head_weight_, last_input)    # [(tp_)vocab_size, num_mastering_inputs]

        last_input = None
        if self.tp_world_size_ == 1:
            gather_data = logic_batch
        else:
            gather_data = torch.empty((self.vocab_size_, token_num), device=logic_batch.device, dtype=torch.float16)
            split_size = self.vocab_size_ // self.tp_world_size_
            dist.all_gather([gather_data[i * split_size: (i + 1) * split_size, :]
                            for i in range(self.tp_world_size_)], logic_batch, group=self.tp_group_, async_op=False)
        logic_batch = None

        if not return_logics:
            prob_out = self.soft_max(gather_data)
            gather_data = None
            return prob_out
        else:
            ans_logics = gather_data.permute(1, 0).float()
            gather_data = None
            return ans_logics
        