import torch

def longserve_token_decode_attention_flash_decoding(q, req_to_token_indexs, b_req_idx, b_seq_len, mid_o, mid_o_logexpsum, batch_size, max_len_in_batch, q_head_num, head_dim, cache_k, cache_v, BLOCK_SEQ, out=None, o_logexpsum=None):
    calcu_shape1 = (-1, q_head_num, head_dim)

    from .longserve_flash_decoding_stage1 import longserve_flash_decode_stage1
    from .longserve_flash_decoding_stage2 import longserve_flash_decode_stage2

    o_tensor = torch.empty_like(q, dtype=torch.float32) if out is None else out
    out_logexpsum = torch.empty([batch_size, q_head_num], dtype=torch.float32, device="cuda") if o_logexpsum is None else o_logexpsum

    longserve_flash_decode_stage1(q.view(calcu_shape1),
                                cache_k,
                                cache_v,
                                req_to_token_indexs,
                                b_req_idx,
                                b_seq_len,
                                max_len_in_batch,
                                mid_o,
                                mid_o_logexpsum,
                                BLOCK_SEQ)
    longserve_flash_decode_stage2(mid_o,
                        mid_o_logexpsum, 
                        b_seq_len, 
                        o_tensor.view(calcu_shape1), 
                        out_logexpsum,
                        BLOCK_SEQ)
    return o_tensor, out_logexpsum
