import torch
import torch.distributed as dist
from typing import Tuple, List, Dict, Optional
from functools import partial
import triton
import rnccl
import longserve_cuda_kernels

from loongserve.utils.infer_utils import mark_cost_time
from loongserve.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv
from loongserve.models.llama.triton_kernel.ring_context_flashattention_nopad import context_attention_fwd
from loongserve.models.llama.triton_kernel.longserve_flash_decoding import longserve_token_decode_attention_flash_decoding
from loongserve.models.llama.triton_kernel.longserve_flash_decoding_stage1 import longserve_flash_decode_stage1
from loongserve.models.llama.triton_kernel.longserve_flash_decoding_stage2 import longserve_flash_decode_stage2
from loongserve.models.llama.triton_kernel.copy_and_register_kv import pre_copy_and_register_kv, destindex_copy_and_register_kv
from loongserve.models.llama.longserve_infer_struct import LongServeLlamaInferStateInfo, LongServeLlamaKVMigrationMeta
from loongserve.models.llama.layer_weights.longserve_transformer_layer_weight import LongServeLlamaTransformerLayerWeight

class LongServeTransformerLayerInfer:
    """
    """
    def __init__(self, layer_num, total_rank, total_world_size, network_config, mode,
                 tp_rank, tp_world_size, tp_group,
                 sp_rank, sp_world_size, sp_comm: rnccl.RNCCLComm):
        super().__init__()
        self.layer_num_ = layer_num
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
        self.sp_comm_ = sp_comm
        assert self.total_world_size_ == self.tp_world_size_ * self.sp_world_size_
        assert self.total_rank_ == self.sp_rank_ * self.tp_world_size_ + self.tp_rank_
        # need to set by subclass
        self.eps_ = network_config["rms_norm_eps"]
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.tp_world_size_
        self.tp_k_head_num_ = network_config["num_key_value_heads"] // self.tp_world_size_
        self.tp_v_head_num_ = network_config["num_key_value_heads"] // self.tp_world_size_
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.head_dim_ = network_config["hidden_size"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["hidden_size"]
        self._bind_func()
        return

    def _bind_func(self):
        self._bind_norm()
        self._bind_attention()
        return

    def _bind_norm(self):
        return
    
    def _bind_attention(self):
        self._is_context_attention_async = "_context_attention_async" in self.mode
        self._context_attention_kernel = partial(LongServeTransformerLayerInfer._context_attention_kernel, self)

        if "_token_decode_attention_overlapped" in self.mode:
            self._token_attention_kernel = partial(LongServeTransformerLayerInfer._token_decode_attention_overlapped, self)
        else:
            self._token_attention_kernel = None
        self._copy_kv_to_mem_cache = partial(LongServeTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
    
    def _att_norm(self, input, infer_state:LongServeLlamaInferStateInfo, layer_weight:LongServeLlamaTransformerLayerWeight)->torch.Tensor:
        return longserve_cuda_kernels.rms_norm(input, layer_weight.att_norm_weight_, self.eps_)
    
    def _ffn_norm(self, input, infer_state:LongServeLlamaInferStateInfo, layer_weight:LongServeLlamaTransformerLayerWeight)->torch.Tensor:
        return longserve_cuda_kernels.rms_norm(input, layer_weight.ffn_norm_weight_, self.eps_)
    
    def _pre_cache_kv(self, infer_state:LongServeLlamaInferStateInfo, layer_weight)->torch.Tensor:
        """
        Get buffer for newly calculated K/V
        """
        if infer_state.mem_is_contiguous:
            cache_kv = infer_state.mem_manager.kv_buffer[self.layer_num_][infer_state.mem_start:infer_state.mem_end, :, :]
        else:
            cache_kv = infer_state.kv_buffer
        return cache_kv

    def _get_qkv(self, input, cache_kv, infer_state:LongServeLlamaInferStateInfo, layer_weight:LongServeLlamaTransformerLayerWeight)->Tuple[torch.Tensor, torch.Tensor]:
        q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_weight_)
        torch.mm(input.view(-1, self.embed_dim_), layer_weight.kv_weight_,
                    out=cache_kv.view(-1, (self.tp_k_head_num_+self.tp_v_head_num_) * self.head_dim_)[:infer_state.total_token_num, ...])
        longserve_cuda_kernels.rotary_emb(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:infer_state.total_token_num, 0:self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin
        )
        return q, cache_kv
    
    def _post_cache_kv(self, cache_kv, infer_state:LongServeLlamaInferStateInfo, layer_weight):
        mem_manager = infer_state.mem_manager
        if not infer_state.mem_is_contiguous and not infer_state.need_context_migration:
            self._copy_kv_to_mem_cache(cache_kv, infer_state.mem_index, mem_manager)
            return
    
    def _context_attention_kernel(self, q, kv, infer_state:LongServeLlamaInferStateInfo, layer_weight, out=None)->torch.Tensor:
        if infer_state.mem_is_contiguous:
            kv = infer_state.mem_manager.kv_buffer[self.layer_num_][infer_state.mem_start:infer_state.mem_start+infer_state.max_token_num, ...]
        
        if infer_state.need_context_migration:
            infer_state.cur_kv_cache_index.zero_()
            mem_index = infer_state.mem_index
            num_used_mem_index = 0
        
        peer_kv_buffer = infer_state.peer_kv_buffer

        kv_b_start_loc = infer_state.q_b_start_loc
        kv_b_seq_len = infer_state.q_b_seq_len
        kv_first_token_global_idx = infer_state.q_first_token_global_idx

        peer_kv_b_start_loc = infer_state.peer_kv_b_start_loc
        peer_kv_b_seq_len = infer_state.peer_kv_b_seq_len
        peer_kv_first_token_global_idx = infer_state.peer_kv_first_token_global_idx

        o_tensor = torch.empty_like(q) if out is None else out
        o_tensor.zero_()
        m = infer_state.local_m.fill_(float("-inf"))
        l = infer_state.local_l.zero_()
        
        num_logical_sp_peers = len(infer_state.logical_sp_peer_ranks)
        for i in range(num_logical_sp_peers):
            if i == 1:
                kv_b_start_loc = infer_state.kv_b_start_loc
                kv_b_seq_len = infer_state.kv_b_seq_len
                kv_first_token_global_idx = infer_state.kv_first_token_global_idx
                kv = infer_state.kv_buffer
            
            if i > 0:
                kv, peer_kv_buffer = peer_kv_buffer, kv
                kv_b_start_loc, peer_kv_b_start_loc = peer_kv_b_start_loc, kv_b_start_loc
                kv_b_seq_len, peer_kv_b_seq_len = peer_kv_b_seq_len, kv_b_seq_len
                kv_first_token_global_idx, peer_kv_first_token_global_idx = peer_kv_first_token_global_idx, kv_first_token_global_idx

            # Let the compute stream sync with the comm stream
            self.sp_comm_.wait_for_default_stream()
            self.sp_comm_.let_default_stream_wait()

            context_attention_fwd(
                                q.view(-1, self.tp_q_head_num_, self.head_dim_),
                                kv.view(-1, self.tp_k_head_num_+self.tp_v_head_num_, self.head_dim_)[:, :self.tp_k_head_num_, :],
                                kv.view(-1, self.tp_k_head_num_+self.tp_v_head_num_, self.head_dim_)[:, self.tp_k_head_num_:, :],
                                o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                                infer_state.q_b_start_loc,
                                infer_state.q_b_seq_len,
                                infer_state.q_first_token_global_idx,
                                kv_b_start_loc,
                                kv_b_seq_len,
                                kv_first_token_global_idx,
                                len(infer_state.logical_sp_peer_ranks),
                                infer_state.max_len_in_batch,
                                m,
                                l)

            if i != num_logical_sp_peers - 1:
                logical_next_peer_rank = infer_state.logical_sp_peer_ranks[(infer_state.logical_sp_rank + 1) % num_logical_sp_peers]
                isend_list = [
                    (kv, logical_next_peer_rank),
                    (kv_b_start_loc, logical_next_peer_rank),
                    (kv_b_seq_len, logical_next_peer_rank),
                    (kv_first_token_global_idx, logical_next_peer_rank)
                ]

                logical_prev_peer_rank = infer_state.logical_sp_peer_ranks[(infer_state.logical_sp_rank - 1 + num_logical_sp_peers) % num_logical_sp_peers]
                irecv_list = [
                    (peer_kv_buffer, logical_prev_peer_rank),
                    (peer_kv_b_start_loc, logical_prev_peer_rank),
                    (peer_kv_b_seq_len, logical_prev_peer_rank),
                    (peer_kv_first_token_global_idx, logical_prev_peer_rank),
                ]
                self._sp_batch_isend_irecv(isend_list, irecv_list, sync_before_launch=not self._is_context_attention_async)

            if infer_state.need_context_migration:
                meta = infer_state.cached_kv_migration_meta[i]
                is_first_layer = self.layer_num_ == 0
                if is_first_layer:
                    # We are at the first layer, need to calculate metas
                    kv_range_begin, kv_range_end, new_kv_cache_len = pre_copy_and_register_kv(
                        infer_state.kv_cache_index_begin,
                        infer_state.kv_cache_index_end,
                        kv_first_token_global_idx,
                        num_logical_sp_peers
                    )
                    new_kv_cache_len_sum = torch.cumsum(new_kv_cache_len, dim=0, dtype=torch.int64)
                    meta.is_valid = True
                    meta.kv_range_begin = kv_range_begin
                    meta.kv_range_end = kv_range_end
                    meta.new_kv_cache_len = new_kv_cache_len
                    meta.new_kv_cache_len_sum = new_kv_cache_len_sum
                    meta.new_kv_cache_len_sum_item = new_kv_cache_len_sum[-1].item()

                destindex_copy_and_register_kv(
                    infer_state.batch_size,
                    meta.new_kv_cache_len,
                    meta.kv_range_begin,
                    meta.kv_range_end,
                    meta.new_kv_cache_len_sum,
                    kv,
                    kv_b_start_loc,
                    infer_state.cur_kv_cache_index,
                    mem_index, 
                    infer_state.mem_manager.kv_buffer[self.layer_num_],
                    infer_state.req_manager.req_to_token_indexs,
                    infer_state.b_req_idx,
                    num_used_mem_index,
                    infer_state.max_len_in_batch+1,  # max_len_in_batch is max{q_b_seqlen}, since we use striped attention, max{kv_range_end-kv_range_begin} <= max{kv_b_seqlen} <= max{q_b_seqlen}+1 holds
                    should_register_kv=is_first_layer
                )
                num_used_mem_index += meta.new_kv_cache_len_sum_item

                if is_first_layer:
                    # We need to update cur_kv_cache_index only when we are at the first layer
                    infer_state.cur_kv_cache_index += meta.new_kv_cache_len

        return o_tensor
    
    def _token_decode_attention_overlapped(
        self,
        q: torch.Tensor,
        infer_state: LongServeLlamaInferStateInfo,
        layer_weight: LongServeLlamaTransformerLayerWeight,
        out=None
    ):
        is_master_rank = (infer_state.sp_master_rank == self.total_rank_)
        is_peer_rank = (infer_state.peer_sp_master_rank_list is not None)
        num_sp_peers = len(infer_state.logical_sp_peer_ranks)

        if is_master_rank:
            self.sp_comm_.wait_for_default_stream()

        # act as the peer worker, receive query from master workers (step 1)
        if is_peer_rank:
            peer_gather_list_list = [
                [
                    infer_state.peer_query_buffer[buffer_begin:buffer_end, ...]
                        for buffer_begin, buffer_end in infer_state.peer_query_buffer_range_list
                ]
            ]
        
        # act as the master worker, send query to peer workers (step 1)
        if is_master_rank:
            q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        
        self.sp_comm_.nccl_group_start()
        # act as the peer worker, receive query from master workers (step 2)
        if is_peer_rank:
            self._selective_gather(
                tensor_list=[None],
                gather_list_list=peer_gather_list_list,
                dst_rank=self.total_rank_,
                selective_peer_ranks=infer_state.peer_sp_master_rank_list)
        # act as the master worker, send query to peer workers (step 2)
        if is_master_rank:
            self._selective_broadcast(
                q,
                src_rank=infer_state.sp_master_rank,
                selective_peer_ranks=infer_state.logical_sp_peer_ranks)
        self.sp_comm_.nccl_group_end()

        # act as the master worker, compute local flash-decoding stage-1
        if is_master_rank:
            cache_kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
            
            self.sp_comm_.let_default_stream_wait()
            longserve_flash_decode_stage1(q,
                                            cache_kv[:, :self.tp_k_head_num_, :],
                                            cache_kv[:, self.tp_k_head_num_:, :],
                                            infer_state.req_manager.req_to_token_indexs,
                                            infer_state.b_req_idx,
                                            infer_state.q_b_seq_len,
                                            infer_state.max_len_in_batch,
                                            infer_state.global_mid_o[num_sp_peers-1:],
                                            infer_state.global_mid_o_logexpsum[num_sp_peers-1:],
                                            infer_state.BLOCK_SEQ)

        # act as the peer worker, compute local flash-decoding
        if is_peer_rank:
            cache_kv = infer_state.mem_manager.kv_buffer[self.layer_num_]

            if getattr(infer_state, 'peer_local_o', None) is None:
                infer_state.peer_local_o = None
                infer_state.peer_local_out_logexpsum = None


            self.sp_comm_.let_default_stream_wait()
            infer_state.peer_local_o, infer_state.peer_local_out_logexpsum = longserve_token_decode_attention_flash_decoding(
                infer_state.peer_query_buffer,
                infer_state.req_manager.req_to_token_indexs,
                infer_state.peer_b_req_idx,
                infer_state.peer_b_seq_len,
                infer_state.peer_local_mid_o,
                infer_state.peer_local_mid_o_logexpsum,
                infer_state.peer_batch_size,
                infer_state.peer_max_len_in_batch,
                self.tp_q_head_num_,
                self.head_dim_,
                cache_kv[:, :self.tp_k_head_num_, :],
                cache_kv[:, self.tp_k_head_num_:, :],
                infer_state.BLOCK_SEQ,
                infer_state.peer_local_o,
                infer_state.peer_local_out_logexpsum)

        # act as the master worker, receive global flash-decoding results from peer workers (step 1)
        if is_master_rank:
            if getattr(infer_state, 'local_o_list', None) is None:
                infer_state.local_o_list = [
                    infer_state.global_mid_o[i, ...]
                    for i in range(num_sp_peers-1)
                ]
                infer_state.local_out_logexpsum_list = [
                    infer_state.global_mid_o_logexpsum[i, ...]
                    for i in range(num_sp_peers-1)
                ]

        # act as the peer worker, send local flash-decoding results to master workers (step 1)
        if is_peer_rank:
            if getattr(infer_state, 'peer_scatter_list_list', None) is None:
                infer_state.peer_scatter_list_list = [
                    [
                        infer_state.peer_local_o[buffer_begin:buffer_end, ...]
                            for buffer_begin, buffer_end in infer_state.peer_query_buffer_range_list
                    ],
                    [
                        infer_state.peer_local_out_logexpsum[buffer_begin:buffer_end, ...]
                            for buffer_begin, buffer_end in infer_state.peer_query_buffer_range_list
                    ]
                ]
            self.sp_comm_.wait_for_default_stream() # Wait for peer-side flash decoding to finish
        
        self.sp_comm_.nccl_group_start()
        # act as the master worker, receive global flash-decoding results from peer workers (step 2)
        if is_master_rank:
            self._selective_gather(
                tensor_list=[None, None],
                gather_list_list=[infer_state.local_o_list, infer_state.local_out_logexpsum_list],
                dst_rank=infer_state.sp_master_rank,
                selective_peer_ranks=[rank_id for rank_id in infer_state.logical_sp_peer_ranks if rank_id != self.total_rank_])
        # act as the peer worker, send local flash-decoding results to master workers (step 2)
        if is_peer_rank:   
            self._selective_scatter(
                tensor_list=[None, None],
                scatter_list_list=infer_state.peer_scatter_list_list,
                src_rank=self.total_rank_,
                selective_peer_ranks=infer_state.peer_sp_master_rank_list)
            o_tensor = None
        self.sp_comm_.nccl_group_end()

        # act as the master worker, merge global and local flash-decoding results
        if is_master_rank:
            o_tensor = torch.empty_like(q) if out is None else out

            self.sp_comm_.let_default_stream_wait()
            longserve_flash_decode_stage2(
                infer_state.global_mid_o,
                infer_state.global_mid_o_logexpsum, 
                infer_state.global_merge_len, 
                o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_), 
                infer_state.global_out_logexpsum,
                block_seq=1
            )
        else:
            self.sp_comm_.let_default_stream_wait()
        return o_tensor

    def _get_o(self, input, infer_state:LongServeLlamaInferStateInfo, layer_weight:LongServeLlamaTransformerLayerWeight)->torch.Tensor:
        o_tensor = torch.mm(input.view(-1, self.tp_o_head_num_ * self.head_dim_), layer_weight.o_weight_)
        return o_tensor

    def _copy_kv_to_mem_cache_normal(self, kv_buffer, mem_index, mem_manager):
        destindex_copy_kv(kv_buffer, mem_index, mem_manager.kv_buffer[self.layer_num_])
        return

    def _context_attention(self, input_embding, infer_state: LongServeLlamaInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        cache_kv = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_kv  = self._get_qkv(input1, cache_kv, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, group=self.tp_group_, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))
        torch.cuda.synchronize()
        return

    def _context_ffn(self, input_embdings, infer_state: LongServeLlamaInferStateInfo, layer_weight):
        ffn_out = longserve_cuda_kernels.ffn_block(
            input_embdings,
            layer_weight.ffn_norm_weight_,
            self.eps_,
            layer_weight.gate_up_proj,
            layer_weight.down_proj,
        )
        if self.tp_world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=self.tp_group_, async_op=False)
        input_embdings.add_(ffn_out)
        torch.cuda.synchronize()
        return

    # this impl dont to use @mark_cost_time
    def _token_attention(self, input_embding, infer_state: LongServeLlamaInferStateInfo, layer_weight):
        if infer_state.sp_master_rank == self.total_rank_:
            input1 = self._att_norm(input_embding, infer_state, layer_weight)
            cache_kv = self._pre_cache_kv(infer_state, layer_weight)
            q, cache_kv = self._get_qkv(input1, cache_kv, infer_state, layer_weight)
            input1 = None
            self._post_cache_kv(cache_kv, infer_state, layer_weight)
        
            o = self._token_attention_kernel(q, infer_state, layer_weight)
            q = None

            o = self._get_o(o, infer_state, layer_weight)
            if self.tp_world_size_ > 1:
                dist.all_reduce(o, op=dist.ReduceOp.SUM, group=self.tp_group_, async_op=False)
            input_embding.add_(o.view(-1, self.embed_dim_))
        else:
            o = self._token_attention_kernel(None, infer_state, layer_weight)
        return

    # this impl dont to use @mark_cost_time
    def _token_ffn(self, mastering_input_embdings, infer_state: LongServeLlamaInferStateInfo, layer_weight: LongServeLlamaTransformerLayerWeight):
        ffn_out = longserve_cuda_kernels.ffn_block(
            mastering_input_embdings,
            layer_weight.ffn_norm_weight_,
            self.eps_,
            layer_weight.gate_up_proj,
            layer_weight.down_proj,
        )
        if self.tp_world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=self.tp_group_, async_op=False)
        mastering_input_embdings.add_(ffn_out)
        return
    
    def context_forward(self, input_embdings, infer_state: LongServeLlamaInferStateInfo, layer_weight):
        self._context_attention(input_embdings,
                                      infer_state,
                                      layer_weight=layer_weight)
        self._context_ffn(input_embdings, infer_state, layer_weight)
        return input_embdings

    def token_forward(self, input_embdings, infer_state: LongServeLlamaInferStateInfo, layer_weight):
        self._token_attention(input_embdings,
                                    infer_state,
                                    layer_weight=layer_weight)
        if infer_state.sp_master_rank == self.total_rank_:
            self._token_ffn(input_embdings, infer_state, layer_weight)
        return input_embdings

    def _sp_batch_isend_irecv(self, isend_list:List[Tuple[torch.Tensor, int]], irecv_list:List[Tuple[torch.Tensor, int]], sync_before_launch):
        self.sp_comm_.nccl_group_start()
        for (irecv_tensor, irecv_rank) in irecv_list:
            self.sp_comm_.nccl_recv(irecv_tensor, irecv_rank//self.tp_world_size_)
        for (isend_tensor, isend_rank) in isend_list:
            self.sp_comm_.nccl_send(isend_tensor, isend_rank//self.tp_world_size_)
        if sync_before_launch:
            self.sp_comm_.wait_for_default_stream()
        self.sp_comm_.nccl_group_end()
    
    def _selective_broadcast(self, tensor: torch.Tensor, src_rank: int, selective_peer_ranks: List[int]):
        self.sp_comm_.nccl_group_start()
        if src_rank == self.total_rank_:
            for peer_rank in selective_peer_ranks:
                if peer_rank != self.total_rank_:
                    self.sp_comm_.nccl_send(tensor, peer_rank//self.tp_world_size_)
        else:
            self.sp_comm_.nccl_recv(tensor, src_rank//self.tp_world_size_)
        self.sp_comm_.nccl_group_end()

    def _selective_gather(self, tensor_list: List[torch.Tensor], gather_list_list: List[List[torch.Tensor]], dst_rank: int, selective_peer_ranks: List[int]):
        assert len(tensor_list) == len(gather_list_list)
        if dst_rank == self.total_rank_:
            for tensor, gather_list in zip(tensor_list, gather_list_list):
                for i, peer_rank in enumerate(selective_peer_ranks):
                    if peer_rank != self.total_rank_:
                        self.sp_comm_.nccl_recv(gather_list[i], peer_rank//self.tp_world_size_)
                    else:
                        gather_list[i].copy_(tensor, non_blocking=True)
        else:
            for tensor, gather_list in zip(tensor_list, gather_list_list):
                assert gather_list is None
                self.sp_comm_.nccl_send(tensor, dst_rank//self.tp_world_size_)
    
    def _selective_scatter(self, tensor_list: List[torch.Tensor], scatter_list_list: List[List[torch.Tensor]], src_rank: int, selective_peer_ranks: List[int]):
        assert len(tensor_list) == len(scatter_list_list)
        if src_rank == self.total_rank_:
            for tensor, scatter_list in zip(tensor_list, scatter_list_list):
                for i, peer_rank in enumerate(selective_peer_ranks):
                    if peer_rank != self.total_rank_:
                        self.sp_comm_.nccl_send(scatter_list[i], peer_rank//self.tp_world_size_)
                    else:
                        tensor.copy_(scatter_list[i], non_blocking=True)
        else:
            for tensor, gather_list in zip(tensor_list, scatter_list_list):
                assert gather_list is None
                self.sp_comm_.nccl_recv(tensor, src_rank//self.tp_world_size_)
