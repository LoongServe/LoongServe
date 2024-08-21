import math
import random

import torch
import triton
import triton.language as tl

TESLA = ('Tesla' in torch.cuda.get_device_name(0)) if torch.cuda.device_count() > 0 else False
RTX4090 = ('4090' in torch.cuda.get_device_name(0)) if torch.cuda.device_count() > 0 else False

if triton.__version__ >= "2.1.0":
    @triton.jit
    def _fwd_kernel(
        Q,      # [num_q_tokens, num_q_heads, head_dim]
        K,      # [num_kv_tokens, num_kv_heads, head_dim]
        V,      # [num_kv_tokens, num_kv_heads, head_dim]
        sm_scale,
        q_b_start_loc, q_b_seqlen, q_first_token_global_idx,  # [batch_size], define the start location and length of each sequence in the batch in Q
        kv_b_start_loc, kv_b_seqlen, kv_first_token_global_idx,
        logical_sp_peers_num: tl.constexpr,
        Out,    # [num_q_tokens, num_q_heads, head_dim]
        m, l,   # [num_q_tokens, num_q_heads]
        stride_qbs: tl.constexpr, stride_qh: tl.constexpr, stride_qd: tl.constexpr,
        stride_kbs: tl.constexpr, stride_kh: tl.constexpr, stride_kd: tl.constexpr,
        stride_vbs: tl.constexpr, stride_vh: tl.constexpr, stride_vd: tl.constexpr,
        stride_obs: tl.constexpr, stride_oh: tl.constexpr, stride_od: tl.constexpr,
        stride_mbs: tl.constexpr, stride_mh: tl.constexpr,
        stride_lbs: tl.constexpr, stride_lh: tl.constexpr,
        kv_group_num: tl.constexpr,
        BLOCK_M: tl.constexpr, DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr
    ):
        """
        A kernel for calculating the attention between local Q and remote KV, called
        during the context stage.
        """
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)
        blockIdz = tl.program_id(2)
        
        cur_kv_head = cur_head // kv_group_num

        cur_q_seq_len = tl.load(q_b_seqlen + cur_batch)
        if blockIdz*BLOCK_M >= cur_q_seq_len:
            return
        
        cur_kv_seq_len = tl.load(kv_b_seqlen + cur_batch)
        cur_q_start_index = tl.load(q_b_start_loc + cur_batch)
        cur_kv_start_index = tl.load(kv_b_start_loc + cur_batch)
        cur_q_first_token_global_idx = tl.load(q_first_token_global_idx + cur_batch)
        cur_kv_first_token_global_idx = tl.load(kv_first_token_global_idx + cur_batch)

        # offset pointers for (q/kv_start_index, head)
        Q += cur_q_start_index*stride_qbs + cur_head*stride_qh
        K += cur_kv_start_index*stride_kbs + cur_kv_head*stride_kh
        V += cur_kv_start_index*stride_vbs + cur_kv_head*stride_vh
        Out += cur_q_start_index*stride_obs + cur_head*stride_oh
        m += cur_q_start_index*stride_mbs + cur_head*stride_mh
        l += cur_q_start_index*stride_lbs + cur_head*stride_lh

        # initialize offsets
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, DMODEL)
        offs_m = blockIdz*BLOCK_M + tl.arange(0, BLOCK_M)
        multed_offs_m = offs_m * logical_sp_peers_num
        multed_offs_n = offs_n * logical_sp_peers_num

        # initialize pointers to q, k, v
        q_ptrs = Q + offs_m[:, None] * stride_qbs + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=offs_m[:, None] < cur_q_seq_len, other=0.0, cache_modifier=".cg") # [BLOCK_M, DMODEL]
        k_ptrs = K + offs_n[None, :] * stride_kbs + offs_d[:, None] * stride_kd
        v_ptrs = V + offs_n[:, None] * stride_vbs + offs_d[None, :] * stride_vd

        m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, DMODEL], dtype=tl.float32)

        loop_range = tl.minimum(
            tl.cdiv(
                cur_q_first_token_global_idx + ((blockIdz+1)*BLOCK_M-1)*logical_sp_peers_num+1 - cur_kv_first_token_global_idx,
                logical_sp_peers_num
            ),
            cur_kv_seq_len
        ) 
        loop_range = tl.maximum(loop_range, 0)

        # Explanaition of the following code:
        # In our kernel, each thread block is responsible for processing an attention
        # score matrix of shape [BLOCK_M, cur_kv_seq_len]. Within the thread block
        # we further divide the matrix into smaller blocks of shape [BLOCK_M, BLOCK_N],
        # which we call "tile".
        #
        # During RingAttention + StripedAttention, the attention mask looks like this:
        # 
        #   X O O O O O
        #   X X O O O O
        #   X X X O O O
        #   X X X X O O
        #   X X X X X O
        #   X X X X X X
        #
        # Or this:
        #
        #   O O O O O O
        #   X O O O O O
        #   X X O O O O
        #   X X X O O O
        #   X X X X O O
        #   X X X X X O
        #
        # From the example above, we could observe that, only the last few tiles
        # from each row are masked. Therefore, we could skip the computation
        # of the attention mask of the non-masked tiles. This speeds up the kernel.
        loop1_end = tl.maximum(loop_range-BLOCK_N*tl.cdiv(BLOCK_M, BLOCK_N), 0)
        for start_n in range(0, loop1_end, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)

            # Compute qk
            k = tl.load(k_ptrs + start_n*stride_kbs,
                        mask=(start_n + offs_n[None, :]) < cur_kv_seq_len, other=0.0, cache_modifier=".cg")   # [DMODEL, BLOCK_N]
            qk = tl.dot(q, k, out_dtype=tl.float32) # [BLOCK_M, BLOCK_N]
            k = None
            v = tl.load(v_ptrs + start_n*stride_vbs,
                        mask=(start_n + offs_n[:, None]) < cur_kv_seq_len, other=0.0, cache_modifier=".cg")  # [BLOCK_N, DMODEL]

            # NOTE The following condition is omitted since the block is not on the diagonal
            # ((cur_q_first_token_global_idx + multed_offs_m[:, None]) >= \
            #     (cur_kv_first_token_global_idx + start_n*logical_sp_peers_num + multed_offs_n[None, :]))
            # NOTE The following condition is omitted since start_n+BLOCK_N < loop1_end+BLOCK_N < loop_range <= cur_kv_seq_len
            # ((start_n + offs_n[None, :]) < cur_kv_seq_len)
            # NOTE The following condition is omitted since it does not affect the answer
            # offs_m[:, None] < cur_q_seq_len

            m_i_new = tl.maximum(m_i, tl.max(qk, 1)*sm_scale)
            alpha = tl.math.exp2(m_i - m_i_new)
            p = tl.math.exp2(qk*sm_scale - m_i_new[:, None])
            acc *= alpha[:, None]
            acc += tl.dot(p.to(tl.float16), v)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new

        # Deal with the last block. Only this block need to be masked
        for start_n in range(loop1_end, loop_range, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)

            # Compute qk
            k = tl.load(k_ptrs + start_n*stride_kbs,
                        mask=(start_n + offs_n[None, :]) < cur_kv_seq_len, other=0.0, cache_modifier=".cg")   # [DMODEL, BLOCK_N]
            qk = tl.dot(q, k, out_dtype=tl.float32) # [BLOCK_M, BLOCK_N]
            k = None
            v = tl.load(v_ptrs + start_n*stride_vbs,
                        mask=(start_n + offs_n[:, None]) < cur_kv_seq_len, other=0.0, cache_modifier=".cg")  # [BLOCK_N, DMODEL]

            # NOTE The following condition is omitted since it does not affect the answer
            # offs_m[:, None] < cur_q_seq_len
            qk = tl.where(
                ((cur_q_first_token_global_idx + multed_offs_m[:, None]) >= \
                    (cur_kv_first_token_global_idx + start_n*logical_sp_peers_num + multed_offs_n[None, :])) & \
                ((start_n + offs_n[None, :]) < cur_kv_seq_len),
                qk, float("-1e20")
            )

            m_i_new = tl.maximum(m_i, tl.max(qk, 1)*sm_scale)
            alpha = tl.math.exp2(m_i - m_i_new)
            p = tl.math.exp2(qk*sm_scale - m_i_new[:, None])
            acc *= alpha[:, None]
            acc += tl.dot(p.to(tl.float16), v)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new
        
        out_ptrs = Out + offs_m[:, None] * stride_obs + offs_d[None, :] * stride_od
        m_ptrs = m + offs_m * stride_mbs
        l_ptrs = l + offs_m * stride_lbs

        old_out = tl.load(out_ptrs, mask=offs_m[:, None] < cur_q_seq_len)           # [BLOCK_M, DMODEL]
        m_i_old = tl.load(m_ptrs, mask=offs_m < cur_q_seq_len, other=float("-inf")) # [BLOCK_M]
        l_i_old = tl.load(l_ptrs, mask=offs_m < cur_q_seq_len, other=0.)            # [BLOCK_M]

        m_i_new = tl.maximum(m_i, m_i_old)
        l_i_new = l_i_old*tl.math.exp2(m_i_old-m_i_new) + l_i*tl.math.exp2(m_i-m_i_new)
        out = (
            old_out * (l_i_old*tl.math.exp2(m_i_old-m_i_new))[:, None] + 
            acc.to(tl.float16) * tl.math.exp2(m_i-m_i_new)[:, None]
        ) / l_i_new[:, None]

        tl.store(out_ptrs, out, mask=offs_m[:, None] < cur_q_seq_len, cache_modifier=".cg")
        tl.store(m_ptrs, m_i_new, mask=offs_m < cur_q_seq_len, cache_modifier=".cg")
        tl.store(l_ptrs, l_i_new, mask=offs_m < cur_q_seq_len, cache_modifier=".cg")

    @torch.inference_mode()
    def context_attention_fwd(
        q: torch.Tensor,    # [(local)num_q_tokens, num_q_heads, head_dim]
        k: torch.Tensor,    # [(remote)num_kv_tokens, num_kv_heads, head_dim]
        v: torch.Tensor,    # [(remote)num_kv_tokens, num_kv_heads, head_dim]
        o: torch.Tensor,    # [(local)num_q_tokens, num_q_heads, head_dim]
        q_b_start_loc: torch.Tensor,  # [batch_size]
        q_b_seq_len: torch.Tensor,    # [batch_size]
        q_first_token_global_idx: torch.Tensor,   # [batch_size]
        kv_b_start_loc: torch.Tensor, # [batch_size]
        kv_b_seq_len: torch.Tensor,   # [batch_size]
        kv_first_token_global_idx: torch.Tensor,  # [batch_size]
        logical_sp_peers_num: int,
        max_q_b_seq_len: int, # int
        m: torch.Tensor,    # [(local)num_q_tokens, num_q_heads]
        l: torch.Tensor     # [(local)num_q_tokens, num_q_heads]
    ):
        """
        context_attention_fwd - Calculate the attention between local Q and local/remote KV
        
        This kernel is called during ring-attention stage in the context stage. In
        ring attention, SP workers repeatedly send their KV to the next worker, and
        calculate the attention between their local Q and the remote KV they receive.
        
        This kernel is designed for that purpose. It is intended to be called
        many times, fed with the same `q`, `o`, `q_b_start_loc`, `q_b_seq_len`,
        `q_first_token_global_idx`, `m`, `l` (those are LOCAL data) while the
        different `k`, `v`, `kv_b_start_loc`, `kv_b_seq_len`, `kv_first_token_global_idx`
        (those are REMOTE data) are fed in each time.
        
        Explanation of the arguments:
            - q: local Q
            - k/v: remote KV
            - o: output. Should be zero-initialized (in fact this is not necessary
                if m is -inf-initialized)
            - q_b_start_loc: the starting index of each request in `q`
            - q_b_seq_len: the length (number of local tokens) of each request in `q`
            - q_first_token_global_idx: q_first_token_global_idx[i] is the global
                index (i.e. among all SP workers, or say, among the original request)
                of the first local token in the i-th request in `q`
            - kv_b_start_loc / kv_b_seq_len / kv_first_token_global_idx: same as above
            - max_q_b_seq_len: equal to q_b_seq_len.max().item()
            - m/l: temporary buffers
        """
        
        BLOCK_M = 128 if not TESLA and not RTX4090 else 64
        BLOCK_N = 128 if not TESLA and not RTX4090 else 64
        
        # Here we reduce BLOCK_M and BLOCK_N, since that when max_q_b_seq_len is
        # small, large BLOCK_SIZE introduces unnecessary computation when computing
        # the attention score.
        # note: We restrict BLOCK_M >= 16 due to a limitation proposed by Triton
        if BLOCK_M//2 >= max(max_q_b_seq_len, 16):
            BLOCK_M = triton.next_power_of_2(max(max_q_b_seq_len, 16))
        if BLOCK_N//2 >= max(max_q_b_seq_len, 16):
            BLOCK_N = triton.next_power_of_2(max(max_q_b_seq_len, 16))
            
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}

        sm_scale = 1.0 / (Lq**0.5)
        # scale sm_scale by log_2(e) and use
        # 2^x instead of exp in the loop because CSE and LICM
        # don't work as expected with `exp` in the loop
        sm_scale *= 1.442695040888963
        batch_size = q_b_seq_len.shape[0]
        num_q_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        kv_group_num = num_q_heads // num_kv_heads
        
        grid = (batch_size, num_q_heads, triton.cdiv(max_q_b_seq_len, BLOCK_M))

        num_warps = 4 if Lk <= 64 else 8
        _fwd_kernel[grid](
            q, k, v, sm_scale,
            q_b_start_loc, q_b_seq_len, q_first_token_global_idx,
            kv_b_start_loc, kv_b_seq_len, kv_first_token_global_idx,
            logical_sp_peers_num,
            o,
            m, l,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            m.stride(0), m.stride(1),
            l.stride(0), l.stride(1),
            kv_group_num=kv_group_num,
            DMODEL=Lk,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=3,
        )


@torch.jit.script
def torch_att(
    xq: torch.Tensor,           # [num_tokens, num_q_heads, head_dim]
    xk: torch.Tensor,           # [num_tokens, num_kv_heads, head_dim]
    xv: torch.Tensor,           # [num_tokens, num_kv_heads, head_dim]
    input_lens: torch.Tensor    # [batch_size]
):
    batch_size = input_lens.shape[0]
    num_q_heads = xq.shape[1]
    num_kv_heads = xk.shape[1]
    head_dim = xq.shape[2]
    
    start_locs = torch.cumsum(input_lens, dim=0) - input_lens
    output = torch.empty_like(xq)
    
    for i in range(batch_size):
        cur_start_loc = start_locs[i]
        cur_seq_len = input_lens[i]
        
        q = xq[cur_start_loc: cur_start_loc + cur_seq_len]  # [cur_seq_len, num_q_heads, head_dim]
        k = xk[cur_start_loc: cur_start_loc + cur_seq_len]  # [cur_seq_len, num_kv_heads, head_dim]
        v = xv[cur_start_loc: cur_start_loc + cur_seq_len]  # [cur_seq_len, num_kv_heads, head_dim]
        
        k = torch.repeat_interleave(k, num_q_heads // num_kv_heads, dim=1)  # [cur_seq_len, num_q_heads, head_dim]
        v = torch.repeat_interleave(v, num_q_heads // num_kv_heads, dim=1)  # [cur_seq_len, num_q_heads, head_dim]
        
        attn_score = q.permute(1, 0, 2) @ k.permute(1, 2, 0)    # [num_q_heads, cur_seq_len, cur_seq_len]
        attn_score /= math.sqrt(head_dim)
        attn_score += torch.triu(torch.full_like(attn_score, -10000.0), diagonal=1)
        attn_score = attn_score.softmax(dim=-1)
        
        cur_output = attn_score @ v.permute(1, 0, 2)    # [num_q_heads, cur_seq_len, head_dim]
        output[cur_start_loc: cur_start_loc + cur_seq_len] = cur_output.permute(1, 0, 2)
        
    return output


@torch.inference_mode()
def test1():
    # Hyperparameters
    batch_size = 2
    num_q_heads = 40
    num_kv_heads = 4
    head_dim = 128

    num_nodes = 8
    max_input_len = 1024
    seed = 0

    # Set up environment
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float16)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    
    # Generate input data
    input_lens = torch.randint(low=1, high=max_input_len, size=(batch_size,), device="cuda")
    num_tokens = input_lens.sum().item()
    q = torch.normal(mean=0.1, std=0.2, size=(num_tokens, num_q_heads, head_dim), device="cuda")
    k = torch.normal(mean=0.4, std=0.2, size=(num_tokens, num_kv_heads, head_dim), device="cuda")
    v = torch.normal(mean=0.3, std=0.2, size=(num_tokens, num_kv_heads, head_dim), device="cuda")
    
    # Calculate std output
    std_output = torch_att(q, k, v, input_lens)
    
    # Generate input data for triton kernel
    def randomly_partition(summ: int, numel: int, lower_bound: int) -> torch.Tensor:
        """Randomly partition sum into numel parts, each in [lower_bound, summ]."""
        assert summ >= lower_bound * numel
        assert numel >= 1
        summ -= lower_bound * numel
        numbers = [random.randint(0, summ) for _ in range(numel - 1)] + [0, summ]
        numbers = sorted(numbers)
        numbers = torch.tensor(numbers, dtype=torch.int32, device="cpu")
        numbers = numbers[1:] - numbers[:-1] + lower_bound
        return numbers
    
    # batch_idx2kv_tokens_on_node[i][j] is the number of kv tokens on node j for batch i
    # batch_idx2prev_node_kv_tokens[i][j] is the number of kv tokens on nodes before node j for batch i
    batch_idx2kv_tokens_on_node = torch.stack([
        randomly_partition(input_lens[i].item(), num_nodes, 1)
        for i in range(batch_size)
    ])
    batch_idx2prev_node_kv_tokens = torch.cumsum(batch_idx2kv_tokens_on_node, dim=1) - batch_idx2kv_tokens_on_node
    
    q_b_seqlen = input_lens
    q_b_start_loc = torch.cumsum(q_b_seqlen, dim=0) - q_b_seqlen
    q_first_token_global_idx = torch.zeros_like(q_b_start_loc)
    
    output = torch.zeros_like(q)
    m = torch.full((num_tokens, num_q_heads), fill_value=-float("inf"), dtype=torch.float32, device="cuda")
    l = torch.zeros_like(m)
    for i in range(num_nodes):
        cur_k = torch.empty((0, num_kv_heads, head_dim), dtype=torch.float16, device="cuda")
        cur_v = torch.empty_like(cur_k)
        kv_b_start_loc = torch.empty_like(q_b_start_loc)
        kv_b_seqlen = torch.empty_like(q_b_start_loc)
        kv_first_token_global_idx = torch.empty_like(q_b_start_loc)
        for batch_idx in range(batch_size):
            kv_b_start_loc[batch_idx] = cur_k.shape[0]
            kv_b_seqlen[batch_idx] = batch_idx2kv_tokens_on_node[batch_idx][i]
            kv_first_token_global_idx[batch_idx] = batch_idx2prev_node_kv_tokens[batch_idx][i]
            token_range_start = q_b_start_loc[batch_idx] + batch_idx2prev_node_kv_tokens[batch_idx][i]
            token_range_end = token_range_start + kv_b_seqlen[batch_idx]
            cur_k = torch.cat([
                cur_k,
                k[token_range_start: token_range_end]
            ], dim = 0)
            cur_v = torch.cat([
                cur_v,
                v[token_range_start: token_range_end]
            ], dim = 0)
        
        context_attention_fwd(
            q, cur_k, cur_v, output,
            q_b_start_loc, q_b_seqlen, q_first_token_global_idx,
            kv_b_start_loc, kv_b_seqlen, kv_first_token_global_idx,
            q_b_seqlen.max().item(),
            m, l
        )
    
    def check_allclose(ans, std):
        if ans.shape != std.shape:
            print("Shape mismatch!")
            print("ans.shape:", ans.shape)
            print("std.shape:", std.shape)
            return False
        abs_err = torch.abs(ans - std)
        rel_err = abs_err / (torch.max(torch.abs(std), torch.abs(ans)) + 1e-2)
        max_abs_err = torch.max(abs_err)
        max_rel_err = torch.max(rel_err)
        if not (abs_err < 1e-2 + (1e-2) * torch.max(torch.abs(std), torch.abs(ans))).all():
            print("ans:", ans)
            print("std:", std)
            print(f"max abs err: {max_abs_err} at pos {torch.argmax(abs_err)} where ans={ans.flatten()[torch.argmax(abs_err)]} and std={std.flatten()[torch.argmax(abs_err)]}")
            print(f"max rel err: {max_rel_err} at pos {torch.argmax(rel_err)} where ans={ans.flatten()[torch.argmax(rel_err)]} and std={std.flatten()[torch.argmax(rel_err)]}")
            return False
        return True

    assert check_allclose(output, std_output)

if __name__ == "__main__":
    test1()
    print("Test 1 pass!")
    