import torch
import numpy as np
from loongserve.common.longserve_req_manager import LongServeReqManager
from loongserve.common.longserve_mem_manager import LongServeMemoryManager

class LongServeLlamaKVMigrationMeta:
    # A structure for storing previous-calculated kv_range_begin, kv_range_end,
    # new_kv_cache_len, new_kv_cache_len_sum to reduce overhead
    # Only useful when need_context_migration is True
    def __init__(self):
        self.is_valid = False
        self.kv_range_begin = None
        self.kv_range_end = None
        self.new_kv_cache_len = None
        self.new_kv_cache_len_sum = None
        self.new_kv_cache_len_sum_item = 0
    
class LongServeLlamaInferStateInfo:
    def __init__(self):
        super().__init__()

        # Similar to the other LLM inference framework, we store all input tokens
        # in a single tensor with a shape of [total_token_num, embd_dim].
        self.batch_size = None      # Batch size
        self.total_token_num = None # The total number of tokens in the batch
        self.max_token_num = None  # The maximum number of tokens in the batch
        self.b_req_idx = None       # The request index of each request in the batch
                                    # This will be fed to ReqManager to retrieve the
                                    # indices of K/V slots for every decoding-stage requests

        self.q_b_start_loc = None   # The length of each request in the batch.
                                    # For context stage it is the number of tokens
                                    # in the prompt, while for decoding stage it is
                                    # the number of previous tokens + 1
        self.q_b_seq_len = None
        self.q_first_token_global_idx = None

        self.kv_b_start_loc = None
        self.kv_b_seq_len = None
        self.kv_first_token_global_idx = None

        self.peer_kv_b_start_loc = None
        self.peer_kv_b_seq_len = None
        self.peer_kv_first_token_global_idx = None
        self.max_len_in_batch = None    # The maximum length of requests in the batch
                                        # Equal to max(b_seq_len)
        self.is_prefill = None      # Whether the batch is for prefill stage
        
        self.mem_manager: LongServeMemoryManager = None
        self.req_manager: LongServeReqManager = None
        
        # The following fields define the newly allocated K/V cache
        self.mem_is_contiguous = None   # Whether we've allocated a contiguous memory
                                        # region for the new K/V space
        self.mem_index = None   # Indices of the newly allocated K/V cache in the huge token-level K/V cache array
        self.mem_start = None   # The start index of the new K/V space.
                                # Only valid when mem_is_contiguous is True
        self.mem_end = None     # The end index of the new K/V space.
                                # Only valid when mem_is_contiguous is True
                                
        # The following fields describe where we should temporarily store the newly
        # calculated K/V values
        # Since torch.mm() only accept continuous memory layout, we need a buffer
        # to store the newly calculated K/V values
        self.kv_buffer = None

        # Cosine and sine position embeddings for RoPE
        self.position_cos = None
        self.position_sin = None
        
        self.local_m = None             # A buffer need by the ring attention kernel
        self.local_l = None             # A buffer need by the ring attention kernel
        self.peer_kv_buffer = None
        
        self.sp_master_rank = None      # The SP (Sequence Parallel) rank of
                                        # the master worker of each request
                                        # only valid when is_prefill == False
                                        
        self.logical_sp_peer_ranks = None

        self.logical_sp_rank = None
        
        self.BLOCK_SEQ = None
        
        # Various buffers
        self.num_seq_block = None
        self.global_mid_o = None
        self.global_mid_o_logexpsum = None
        self.global_merge_len = None
        self.global_out_logexpsum = None

        self.peer_sp_master_rank_list = None
        self.peer_query_buffer_range_list = None

        self.peer_batch_size = None
        self.peer_max_len_in_batch = None
        self.peer_b_req_idx = None
        self.peer_b_seq_len = None
        self.peer_local_mid_o = None
        self.peer_local_mid_o_logexpsum = None
                                        
        self.peer_query_buffer = None   # A buffer for storing the Q values from other peers
                                        # Only valid when is_prefill == False
        
        self.need_context_migration = False
        self.kv_cache_index_begin = None
        self.kv_cache_index_end = None
        self.cur_kv_cache_index: torch.Tensor = None
        self.cached_kv_migration_meta: list[LongServeLlamaKVMigrationMeta] = None    # Some cached metadata (indexing, etc.) for the K/V cache migration
                                                # Refer to _context_attention_kernel in longerve_transformer_layer_infer.py for details
            
    def init_some_extra_state(self, model):
        if self.is_prefill:
            num_logical_sp_peer_ranks = len(self.logical_sp_peer_ranks)
            position_ids = \
                torch.concat([
                    torch.arange(
                        first_token_global_idx,
                        first_token_global_idx + seq_len * num_logical_sp_peer_ranks,
                        num_logical_sp_peer_ranks,
                        device="cuda"
                    )
                    for first_token_global_idx, seq_len in zip(self.q_first_token_global_idx, self.q_b_seq_len)
                ], axis=0)
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
            position_ids = None
        else:
            if self.sp_master_rank == model.total_rank_:
                position_ids = self.q_first_token_global_idx
                self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(self.q_b_seq_len.shape[0], -1)
                self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(self.q_b_seq_len.shape[0], -1)
        return
