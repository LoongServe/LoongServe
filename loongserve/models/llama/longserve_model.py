import os
import json
import torch
import time
import triton
import rnccl
from typing import final
from torch.profiler import profile, record_function, ProfilerActivity

from loongserve.common.basemodel.layer_weights.hf_load_utils import load_hf_weights
from loongserve.models.llama.layer_infer.longserve_pre_layer_infer import LongServeLlamaPreLayerInfer
from loongserve.models.llama.layer_infer.longserve_post_layer_infer import LongServeLlamaPostLayerInfer
from loongserve.models.llama.layer_infer.longserve_transformer_layer_infer import LongServeTransformerLayerInfer
from loongserve.models.llama.layer_weights.longserve_pre_and_post_layer_weight import LongServeLlamaPreAndPostLayerWeight
from loongserve.models.llama.layer_weights.longserve_transformer_layer_weight import LongServeLlamaTransformerLayerWeight
from loongserve.models.llama.longserve_infer_struct import LongServeLlamaInferStateInfo, LongServeLlamaKVMigrationMeta
from loongserve.common.longserve_mem_manager import LongServeMemoryManager
from loongserve.common.longserve_req_manager import LongServeReqManager
from loongserve.common.infer_utils import init_req_to_token_indexes
from loongserve.common.build_utils import repair_config
from loongserve.common.basemodel.triton_kernel.copy_kv_index_to_req import copy_kv_index_to_req

from loongserve.utils.log_utils import init_logger

torch.backends.cudnn.enabled = True

logger = init_logger(__name__)

class LongServeLlamaModel:
    # weight class
    pre_and_post_weight_class = LongServeLlamaPreAndPostLayerWeight
    transformer_weight_class = LongServeLlamaTransformerLayerWeight

    # infer class
    pre_layer_infer_class = LongServeLlamaPreLayerInfer
    post_layer_infer_class = LongServeLlamaPostLayerInfer
    transformer_layer_infer_class = LongServeTransformerLayerInfer

    # infer state class
    infer_state_class = LongServeLlamaInferStateInfo

    def __init__(self, kvargs):
        # parallelism related parameters
        self.total_world_size_ = kvargs["total_world_size"]
        self.total_rank_ = kvargs["total_rank"]
        self.tp_world_size_ = kvargs["tp_world_size"]
        self.tp_rank_ = kvargs["tp_rank"]
        self.tp_group_ = kvargs["tp_group"]
        self.sp_world_size_ = kvargs["sp_world_size"]
        self.sp_rank_ = kvargs["sp_rank"]
        self.sp_rnccl_unique_id_ = kvargs["sp_rnccl_unique_id"]
        
        # other parameters
        self.weight_dir_ = kvargs["weight_dir"]
        self.max_total_token_num = kvargs["max_total_token_num"]
        self.load_way = kvargs.get("load_way", "HF")
        self.mode = kvargs.get("mode", [])
        self.weight_dict = kvargs.get("weight_dict", None)
        self.max_req_num = kvargs.get("max_req_num", 1000)
        self.max_seq_length = kvargs.get("max_seq_length", 1024 * 5)
        self.return_all_prompt_logprobs = kvargs.get("return_all_prompt_logprobs", False)
        self.last_task_end_time = 0

        # long sequence related parameters
        self.user_defined_max_position_embeddings = kvargs.get("user_defined_max_position_embeddings", 0)

        self._init_config()
        self._verify_must()
        self._verify_params()
        self._init_weights()
        self._init_mem_manager()
        self._init_req_manager()
        self._init_rnccl_comm()
        self._init_infer_layer()
        self._init_some_value()
        self._init_custom()
        return
    
    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), 'r') as json_file:
            self.config = json.load(json_file)
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        self._reset_num_key_value_heads()
        return
    
    def _reset_num_key_value_heads(self):
        if "num_key_value_heads" not in self.config:
            self.config["num_key_value_heads"] = self.config["num_attention_heads"]
        return
    
    @final
    def _verify_must(self):
        assert self.load_way in ["HF", "DS"], "llama only supports HF and DS format to load Now!"
        assert self.config["num_key_value_heads"] % self.tp_world_size_ == 0
        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0
        return
    
    def _verify_params(self):
        assert self.load_way == "HF", "only support HF format weights"
        assert self.config["num_key_value_heads"] % self.tp_world_size_ == 0
        return

    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(self.total_rank_, self.total_world_size_, self.tp_rank_, self.tp_world_size_, self.sp_rank_, self.sp_world_size_, torch.float16, network_config=self.config, mode=self.mode)
        self.trans_layers_weight = [
            self.transformer_weight_class(i, self.total_rank_, self.total_world_size_, self.tp_rank_, self.tp_world_size_, self.sp_rank_, self.sp_world_size_, torch.float16, network_config=self.config, mode=self.mode)
            for i in range(self.config["n_layer"])
        ]
        load_hf_weights(
            "fp16",
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=self.weight_dict)
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]
        return 
    
    def _init_mem_manager(self):
        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0
        self.mem_manager = LongServeMemoryManager(self.max_total_token_num, 
                                         dtype=torch.float16,
                                         head_num=self.config["num_key_value_heads"] // self.tp_world_size_,
                                         head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                         layer_num=self.config["num_hidden_layers"],
                                         num_idle_slots=self.sp_world_size_)
        return
    
    def _init_req_manager(self):
        self.req_manager = LongServeReqManager(self.max_req_num, 
                                      self.max_seq_length,
                                      self.mem_manager)
        return 
    
    def _init_infer_layer(self):
        self.pre_infer = self.pre_layer_infer_class(total_rank=self.total_rank_, total_world_size=self.total_world_size_, tp_rank=self.tp_rank_, tp_world_size=self.tp_world_size_, tp_group=self.tp_group_, sp_rank=self.sp_rank_, sp_world_size=self.sp_world_size_, network_config=self.config, mode=self.mode)
        self.post_infer = self.post_layer_infer_class(total_rank=self.total_rank_, total_world_size=self.total_world_size_, tp_rank=self.tp_rank_, tp_world_size=self.tp_world_size_, tp_group=self.tp_group_, sp_rank=self.sp_rank_, sp_world_size=self.sp_world_size_, network_config=self.config, mode=self.mode)
        self.layers_infer = [
            self.transformer_layer_infer_class(
                i,
                total_rank=self.total_rank_,
                total_world_size=self.total_world_size_,
                tp_rank=self.tp_rank_,
                tp_world_size=self.tp_world_size_,
                tp_group=self.tp_group_,
                sp_rank=self.sp_rank_,
                sp_world_size=self.sp_world_size_,
                sp_comm=self.sp_comm,
                network_config=self.config,
                mode=self.mode) for i in range(
                self.config["n_layer"])]
        return
    
    def _init_some_value(self):
        self.head_dim_ = self.config["n_embed"] // self.config["num_attention_heads"]
        self.tp_q_head_num_ = self.config["num_attention_heads"] // self.tp_world_size_
        self.tp_k_head_num_ = self.config["num_key_value_heads"] // self.tp_world_size_
        self.tp_v_head_num_ = self.tp_k_head_num_
        self.layers_num = self.config["n_layer"]
        self.vocab_size = self.config["vocab_size"]
        return

    def _init_rnccl_comm(self):
        self.sp_comm = rnccl.RNCCLComm(self.sp_rnccl_unique_id_, self.sp_world_size_, self.sp_rank_)
    
    def _init_custom(self):
        """
        模型特殊的一些初始化
        """
        if self.config.get("use_rope_yarn", False):
            self._init_to_get_yarn_rotary()
        elif self.config.get("use_dynamic_ntk", False):
            self._init_to_get_dynamic_ntk_rotary()
        else:
            self._init_to_get_rotary()
        return
    
    def _init_to_get_yarn_rotary(self):
        from .yarn_rotary_utils import find_correction_range, linear_ramp_mask, get_mscale
        dim = self.head_dim_
        max_position_embeddings = self.config.get("max_position_embeddings", 2048)
        max_position_embeddings = max(max_position_embeddings, self.user_defined_max_position_embeddings)
        base = self.config.get("rope_theta", 10000.0)
        scale = self.config.get("rope_scaling", {}).get("factor", 1.0)
        original_max_position_embeddings = self.config.get("original_max_position_embeddings", 2048)
        extrapolation_factor = 1.0
        attn_factor = 1.0
        beta_fast = 32.0
        beta_slow = 1.0

        pos_freqs = base ** (torch.arange(0, dim, 2).float().cuda() / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scale * pos_freqs)

        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings)
        inv_freq_mask = (1 - linear_ramp_mask(low, high, dim // 2).float().cuda()) * extrapolation_factor # Get n-d rotational scaling corrected for extrapolation
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        mscale = float(get_mscale(scale) * attn_factor) # Get n-d magnitude scaling corrected for interpolation

        # Build here to make `torch.jit.trace` work.
        max_seq_len_cached = max_position_embeddings
        t = torch.arange(max_seq_len_cached, device="cuda", dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos().to(torch.float16).cuda() * mscale
        self._sin_cached = emb.sin().to(torch.float16).cuda() * mscale

        return
    
    def _init_to_get_dynamic_ntk_rotary(self):
        max_position_embeddings = self.config.get("max_position_embeddings", 2048)
        max_position_embeddings = max(max_position_embeddings, self.user_defined_max_position_embeddings)
        base = self.config.get("rope_theta", 10000.0)
        scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)
        max_seq_len = 32 * max_position_embeddings # 64k
        self._cos_cached = torch.zeros((max_seq_len, self.head_dim_ // 2), dtype=torch.float16, device="cuda")
        self._sin_cached = torch.zeros((max_seq_len, self.head_dim_ // 2), dtype=torch.float16, device="cuda")
        
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_))
        t = torch.arange(max_position_embeddings, device="cpu", dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self._cos_cached[0:max_position_embeddings, :] = torch.cos(freqs).to(torch.float16).cuda()
        self._sin_cached[0:max_position_embeddings, :] = torch.sin(freqs).to(torch.float16).cuda()

        for seq_loc_index in range(max_position_embeddings, max_seq_len, 1):
            new_base = base * ((scaling_factor * (seq_loc_index + 1) / max_position_embeddings) -(scaling_factor - 1)) ** (self.head_dim_ / (self.head_dim_ - 2))
            inv_freq = 1.0 / (new_base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_))
            t = torch.tensor([seq_loc_index,], device="cpu", dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached[seq_loc_index:seq_loc_index + 1, :] = torch.cos(freqs).to(torch.float16).cuda()
            self._sin_cached[seq_loc_index:seq_loc_index + 1, :] = torch.sin(freqs).to(torch.float16).cuda()
        return

    def _init_to_get_rotary(self, default_base=10000):
        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)

        base = self.config.get("rope_theta", float(default_base))
        # NOTE (intlsy): Set base to 600k can enhance the extrapolation ability of rotary position encoding
        # ref: https://arxiv.org/abs/2310.05209

        if "max_sequence_length" in self.config:
            max_seq_len = self.config["max_sequence_length"]
        else:
            max_position_embeddings = self.config.get(
                "max_position_embeddings",
                2048 if base <= 10000.0 + 1e-5 else 16384
            )
            max_position_embeddings = max(max_position_embeddings, self.user_defined_max_position_embeddings)
            max_seq_len = max_position_embeddings * rope_scaling_factor

        # NTK
        try:
            ntk_alpha = float(os.environ.get("LOONGSERVE_NTK_ALPHA", 1))
            assert ntk_alpha >= 1
            if ntk_alpha > 1:
                logger.info(f"Note: NTK enabled, alpha set to {ntk_alpha}")
            max_seq_len *= ntk_alpha
            base = base * (ntk_alpha ** (self.head_dim_ / (self.head_dim_-2))) #Base change formula
        except:
            pass

        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_))
        t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float16).cuda()
        self._sin_cached = torch.sin(freqs).to(torch.float16).cuda()
        return

    @torch.inference_mode()
    def forward(
            self,
            # The following parameters are analogous to the parameters in InferStateInfo
            # Refer to comments in InferStateInfo for more details
            # NOTE:
            # - For decoding stage, all input arguments should be the same
            #   across all SP workers, no matter whether a request's master is the
            #   current worker or not, input_ids should be passed to it
            # - For decoding stage, newkv_alloc_sp_rank
            batch_size: int,
            total_token_num: int,
            max_token_num: int,
            max_len_in_batch: int,
            input_ids: torch.Tensor,
            b_req_idx: torch.Tensor,
            b_start_loc: torch.Tensor,
            b_seq_len: torch.Tensor,
            
            # The following fields are newly added in LongServe, describing how
            # sequence parallelism (SP) is performed in the current batch
            # Please refer to comments in LongServeLlamaInferStateInfo for more details
            first_token_global_idx: torch.Tensor,
            sp_master_rank: int,
            logical_sp_peer_ranks: list[int],
            logical_sp_rank: int,
            newkv_alloc_sp_rank: torch.Tensor,
            peer_sp_master_rank_list: list[int],
            peer_query_buffer_range_list: list[tuple[int, int]],
            peer_batch_size: int,
            peer_max_len_in_batch: int,
            peer_b_req_idx: torch.Tensor,
            peer_b_seq_len: torch.Tensor,
            need_context_migration,
            kv_cache_index_begin,
            kv_cache_index_end,
            multimodal_params=None,
            is_prefill=True
        ):
        if is_prefill:
            res = self._prefill(batch_size, total_token_num, max_token_num, max_len_in_batch, input_ids, b_req_idx, b_start_loc, b_seq_len, first_token_global_idx, logical_sp_peer_ranks, logical_sp_rank, need_context_migration, kv_cache_index_begin, kv_cache_index_end, multimodal_params)
        else:
            res = self._decode(batch_size, total_token_num, max_len_in_batch, input_ids, b_req_idx, b_seq_len, first_token_global_idx, sp_master_rank, logical_sp_peer_ranks, logical_sp_rank, peer_sp_master_rank_list, peer_query_buffer_range_list, peer_batch_size, peer_max_len_in_batch, peer_b_req_idx, peer_b_seq_len, multimodal_params)
        return res

    
    def _prefill(
            self,
            batch_size,             # (to del) Number of requests in the current batch
            total_token_num,        # (to del) Number of local tokens in the current batch
            max_token_num,          # Maximum of total_token_num among all SP workers
            max_len_in_batch,       # (to del) Maximum of b_seq_len

            input_ids,
            b_req_idx,              # Request id of each request
            b_start_loc,            # The index of the first token (among input_ids) of request #i 
            b_seq_len,              # The number of local tokens of request #i
            first_token_global_idx, # For request #i, the global index of the first local token

            logical_sp_peer_ranks,  # SP ranks of all logical SP peers, which form a "subring" in RingAttention
            logical_sp_rank,        # Index of the current SP worker in logical_sp_peer_ranks

            need_context_migration, # Whether context migration (storing local token's K/V cache elsewhere) is needed
            kv_cache_index_begin,   # The index of the first token that I should store in K/V cache
            kv_cache_index_end,     # The index of the last token+1 that I should store in K/V cache

            multimodal_params
        ):
        infer_state = self.infer_state_class()
        
        infer_state.is_prefill = True
        infer_state.return_all_prompt_logprobs = self.return_all_prompt_logprobs

        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_token_num = max_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        
        assert input_ids.shape[0] == total_token_num
        assert b_req_idx.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0] == batch_size

        infer_state.b_req_idx = b_req_idx
        
        infer_state.q_b_start_loc = b_start_loc
        infer_state.q_b_seq_len = b_seq_len
        infer_state.q_first_token_global_idx = first_token_global_idx
        
        infer_state.kv_b_start_loc = torch.empty_like(b_start_loc)
        infer_state.kv_b_seq_len = torch.empty_like(b_seq_len)
        infer_state.kv_first_token_global_idx = torch.empty_like(first_token_global_idx)

        infer_state.peer_kv_b_start_loc = torch.empty_like(b_start_loc)
        infer_state.peer_kv_b_seq_len = torch.empty_like(b_seq_len)
        infer_state.peer_kv_first_token_global_idx = torch.empty_like(first_token_global_idx)

        infer_state.multimodal_params = multimodal_params

        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager

        if need_context_migration:
            assert kv_cache_index_begin.shape[0] == kv_cache_index_end.shape[0] == batch_size
            infer_state.need_context_migration = need_context_migration
            infer_state.kv_cache_index_begin = kv_cache_index_begin
            infer_state.kv_cache_index_end = kv_cache_index_end
            infer_state.cur_kv_cache_index = torch.empty((infer_state.batch_size,), dtype=torch.int32, device="cuda")
            infer_state.cached_kv_migration_meta = [
                LongServeLlamaKVMigrationMeta()
                for _ in range(len(logical_sp_peer_ranks))
            ]

        alloc_mem = self.mem_manager.alloc_contiguous(total_token_num) if not need_context_migration else None
        if alloc_mem is not None:
            infer_state.mem_is_contiguous = True
            infer_state.mem_index = alloc_mem[0]
            infer_state.mem_start = alloc_mem[1]
            infer_state.mem_end = alloc_mem[2]
        else:    
            infer_state.mem_is_contiguous = False
            alloc_token_num = total_token_num if not need_context_migration else torch.sum(kv_cache_index_end - kv_cache_index_begin)
            alloc_mem = self.mem_manager.alloc(alloc_token_num)
            infer_state.mem_index = alloc_mem
            del infer_state.mem_start
            del infer_state.mem_end

        buffer_size = max_token_num

        infer_state.kv_buffer = torch.empty((buffer_size, self.tp_k_head_num_+self.tp_v_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
        
        infer_state.peer_kv_buffer = torch.empty_like(infer_state.kv_buffer)
        # TODO (intlsy): buffer_size or total_token_num?
        infer_state.local_m = torch.empty((buffer_size, self.tp_q_head_num_), dtype=torch.float32, device="cuda")
        infer_state.local_l = torch.empty_like(infer_state.local_m)
        infer_state.logical_sp_peer_ranks = logical_sp_peer_ranks
        infer_state.logical_sp_rank = logical_sp_rank

        del infer_state.sp_master_rank
        del infer_state.BLOCK_SEQ
        del infer_state.global_mid_o
        del infer_state.global_mid_o_logexpsum
        del infer_state.global_merge_len
        del infer_state.global_out_logexpsum
        del infer_state.peer_sp_master_rank_list
        del infer_state.peer_query_buffer_range_list
        del infer_state.peer_batch_size
        del infer_state.peer_max_len_in_batch
        del infer_state.peer_b_req_idx
        del infer_state.peer_b_seq_len
        del infer_state.peer_local_mid_o
        del infer_state.peer_local_mid_o_logexpsum
        del infer_state.peer_query_buffer
        
        if not need_context_migration:
            init_req_to_token_indexes(self.req_manager.req_to_token_indexs, b_req_idx, b_seq_len,
                                infer_state.mem_index)

        infer_state.init_some_extra_state(self)
        predict_logics = self._context_forward(input_ids, infer_state)
        return predict_logics
    
    def _decode(
            self,
            batch_size,                     # Batch size
            total_token_num,                # (to del) = batch size
            max_len_in_batch,               # (to del) max{b_seq_len}

            input_ids,
            b_req_idx,                      # Request id of each request
            b_seq_len,                      # Number of local tokens (including the new one) of each request
            first_token_global_idx,         # For request #i, the global index of the new (decoding) token
            sp_master_rank,                 # Either self.total_rank_ or None, indicating whether the current SP worker is a master
            logical_sp_peer_ranks,
            logical_sp_rank,

            peer_sp_master_rank_list,
            peer_query_buffer_range_list,
            peer_batch_size,
            peer_max_len_in_batch,
            peer_b_req_idx,
            peer_b_seq_len,
            multimodal_params
        ):
        if sp_master_rank == self.total_rank_:
            assert b_req_idx.shape[0] == b_seq_len.shape[0] == batch_size == total_token_num == input_ids.shape[0], f"{b_req_idx}, {b_seq_len}, {batch_size}, {total_token_num}, {input_ids}"
            assert sp_master_rank in logical_sp_peer_ranks
        if peer_sp_master_rank_list is not None:
            assert len(peer_sp_master_rank_list) == len(peer_query_buffer_range_list), f"{peer_sp_master_rank_list}, {peer_query_buffer_range_list}"
            assert peer_batch_size == peer_b_req_idx.shape[0] == peer_b_seq_len.shape[0] == peer_query_buffer_range_list[-1][1], f"{peer_batch_size}, {peer_b_req_idx}, {peer_b_seq_len}, {peer_query_buffer_range_list}"
        
        infer_state = self.infer_state_class()

        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.b_req_idx = b_req_idx

        infer_state.q_b_seq_len = b_seq_len
        infer_state.q_first_token_global_idx = first_token_global_idx

        infer_state.max_len_in_batch = max_len_in_batch

        infer_state.is_prefill = False
        
        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager
        
        if sp_master_rank == self.total_rank_:
            assert infer_state.total_token_num > 0
            alloc_mem = self.mem_manager.alloc_contiguous(total_token_num)
            if alloc_mem is not None:
                infer_state.mem_is_contiguous = True
                infer_state.mem_index = alloc_mem[0]
                infer_state.mem_start = alloc_mem[1]
                infer_state.mem_end = alloc_mem[2]
                copy_kv_index_to_req(self.req_manager.req_to_token_indexs, b_req_idx, b_seq_len, infer_state.mem_index)
            else:
                infer_state.mem_is_contiguous = False
                alloc_mem = self.mem_manager.alloc(total_token_num)
                infer_state.mem_index = alloc_mem
                infer_state.kv_buffer = torch.empty((total_token_num, self.tp_k_head_num_+self.tp_v_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
                copy_kv_index_to_req(self.req_manager.req_to_token_indexs, b_req_idx, b_seq_len, infer_state.mem_index)
        
        infer_state.multimodal_params = multimodal_params

        infer_state.sp_master_rank = sp_master_rank
        infer_state.logical_sp_peer_ranks = logical_sp_peer_ranks
        infer_state.logical_sp_rank = logical_sp_rank

        infer_state.BLOCK_SEQ = min(
            triton.next_power_of_2(max(max_len_in_batch, peer_max_len_in_batch, 16)),
            256
        )

        if sp_master_rank == self.total_rank_:
            import math
            num_seq_block = (max_len_in_batch + infer_state.BLOCK_SEQ - 1) // infer_state.BLOCK_SEQ
            infer_state.num_seq_block = num_seq_block
            infer_state.global_mid_o = torch.empty((len(logical_sp_peer_ranks) - 1 + num_seq_block, batch_size, self.tp_q_head_num_, self.head_dim_), dtype=torch.float32, device="cuda")
            infer_state.global_mid_o_logexpsum = torch.empty((len(logical_sp_peer_ranks) - 1 + num_seq_block, batch_size, self.tp_q_head_num_), dtype=torch.float32, device="cuda")
            infer_state.global_merge_len = ((len(logical_sp_peer_ranks) - 1) + (b_seq_len + (infer_state.BLOCK_SEQ - 1)) // infer_state.BLOCK_SEQ).view((batch_size,)).contiguous().clone()
            infer_state.global_out_logexpsum = torch.empty((batch_size, self.tp_q_head_num_), dtype=torch.float32, device="cuda")
        
        if peer_sp_master_rank_list is not None:
            infer_state.peer_query_buffer = torch.empty((peer_batch_size, self.tp_q_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")

            infer_state.peer_sp_master_rank_list = peer_sp_master_rank_list
            infer_state.peer_query_buffer_range_list = peer_query_buffer_range_list

            infer_state.peer_batch_size = peer_batch_size
            infer_state.peer_max_len_in_batch = peer_max_len_in_batch
            if max_len_in_batch is not None:
                infer_state.peer_max_len_in_batch = max(infer_state.peer_max_len_in_batch, max_len_in_batch)
            infer_state.peer_b_req_idx = peer_b_req_idx
            infer_state.peer_b_seq_len = peer_b_seq_len
            num_seq_block = peer_max_len_in_batch // infer_state.BLOCK_SEQ + 1
            infer_state.peer_local_mid_o = torch.empty((num_seq_block, peer_batch_size, self.tp_q_head_num_, self.head_dim_), dtype=torch.float32, device="cuda")
            infer_state.peer_local_mid_o_logexpsum = torch.empty((num_seq_block, peer_batch_size, self.tp_q_head_num_), dtype=torch.float32, device="cuda")
        
        infer_state.init_some_extra_state(self)
        predict_logics = self._token_forward(input_ids, infer_state)

        return predict_logics
    
    @final
    def _context_forward(self, input_ids, infer_state: LongServeLlamaInferStateInfo):
        assert input_ids.device.type == "cuda"
        input_embs = self.pre_infer.context_forward(input_ids, infer_state, self.pre_post_weight)
        for i in range(self.layers_num):
            input_embs = self.layers_infer[i].context_forward(input_embs, infer_state, self.trans_layers_weight[i])
        predict_logics = self.post_infer.context_forward(input_embs, infer_state, self.pre_post_weight, return_logics=True)
        return predict_logics

    @final
    def _token_forward(self, input_ids, infer_state: LongServeLlamaInferStateInfo):
        assert input_ids is None or input_ids.device.type == "cuda"
        if infer_state.sp_master_rank == self.total_rank_:
            input_embs = self.pre_infer.token_forward(input_ids, infer_state, self.pre_post_weight)
            for i in range(self.layers_num):
                input_embs = self.layers_infer[i].token_forward(input_embs, infer_state, self.trans_layers_weight[i])
            predic_logics = self.post_infer.token_forward(input_embs, infer_state, self.pre_post_weight, return_logics=True)
            return predic_logics
        else:
            for i in range(self.layers_num):
                self.layers_infer[i].token_forward(None, infer_state, self.trans_layers_weight[i])
            return None
    
    @torch.inference_mode()
    def decoding_stage_migration(
        self,
        is_receiver: bool,              # Whether the current SP worker is the receiver
        b_req_idx: torch.Tensor,        # [batch_size], ids of requests to migrate
        b_migration_len: torch.Tensor,  # [batch_size], number of tokens to migrate for each request
        b_seq_len: torch.Tensor,        # [batch_size], number of local tokens of each request BEFORE MIGRATION
        peer_sp_rank: int,              # The SP rank of the peer (the oppposite receiver/sender, depends on `is_receiver`) worker
    ):
        """
        Perform decoding stage K/V cache migration

        During the decoding stage, we may need to move the K/V cache of some tokens
        from one SP worker to another one. This is called "decoding_stage_migration".

        The following fields are modified:
        - mem_manager.mem_state
        - mem_manager.can_use_mem_size
        - mem_manager.kv_buffer
        - req_manager.req_to_token_indexes

        Note that this function doesn't alter req_manager.req_state, i.e. if all
        tokens of a request are migrated, the request is not removed from req_manager.req_state.
        This design is due to the fact that workers never care about req_manager.req_state.

        PAY ATTENTION This kernel assumes `mem_state` of all allocated tokens to
        be 1, hence does not support beam search / prefix sharing
        """
        from loongserve.models.llama.triton_kernel.decoding_stage_migration import decoding_stage_migration_sender_kernel, decoding_stage_migration_receiver_kernel
        import torch.distributed as dist
        b_migration_len_sum = torch.sum(b_migration_len)
        num_layers = self.layers_num
        if is_receiver:
            recv_buf = torch.empty((num_layers, b_migration_len_sum, self.tp_k_head_num_+self.tp_v_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
            self.sp_comm.nccl_recv(recv_buf, peer_sp_rank)
            alloc_mem = self.mem_manager.alloc(b_migration_len_sum)
            self.sp_comm.let_default_stream_wait()
            decoding_stage_migration_receiver_kernel(
                recv_buf,
                self.req_manager.req_to_token_indexs,
                self.mem_manager.kv_buffer,
                alloc_mem,
                b_req_idx,
                b_migration_len,
                b_seq_len
            )
        else:
            send_buf = torch.empty((num_layers, b_migration_len_sum, self.tp_k_head_num_+self.tp_v_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
            decoding_stage_migration_sender_kernel(
                send_buf,
                self.req_manager.req_to_token_indexs,
                self.mem_manager.mem_state,
                self.mem_manager.kv_buffer,
                b_req_idx,
                b_migration_len,
                b_seq_len
            )
            self.sp_comm.wait_for_default_stream()
            self.sp_comm.nccl_send(send_buf, peer_sp_rank)
            self.mem_manager.can_use_mem_size += b_migration_len_sum
            self.sp_comm.let_default_stream_wait()