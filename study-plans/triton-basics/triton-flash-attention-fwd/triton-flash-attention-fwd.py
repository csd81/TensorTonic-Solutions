import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, out_ptr,
    N_Q, N_KV, D, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load this program's Q row-block once; it stays in registers all kernel.
    q_ptrs = Q_ptr + offs_m[:, None] * D + offs_d[None, :]
    q_mask = (offs_m[:, None] < N_Q) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Online-softmax state, one entry per row of the Q block.
    m_i = tl.full((BLOCK_M,), -1e30, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    o_i = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    # Stream K and V in column-blocks; the scores never leave registers.
    for start_n in range(0, N_KV, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = (offs_n[:, None] < N_KV) & (offs_d[None, :] < D)

        k = tl.load(K_ptr + offs_n[:, None] * D + offs_d[None, :],
                    mask=kv_mask, other=0.0)
        v = tl.load(V_ptr + offs_n[:, None] * D + offs_d[None, :],
                    mask=kv_mask, other=0.0)

        # Scaled scores via a 2-D matmul: s[m, n] = (q[m] . k[n]) * scale.
        # Scale goes in before the running-max update, not after softmax.
        # IEEE precision is required: the default TF32 path crashes ptxas/
        # the compiler on pre-Ampere GPUs such as the T4 (sm_75).
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        # Padded K columns must get zero weight, not exp(0) = 1.
        s = tl.where(offs_n[None, :] < N_KV, s, -1e30)

        # Online recurrence: re-anchor the old state to the new running max,
        # then fold in this block's unnormalized weights and V rows.
        m_local = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        o_i = alpha[:, None] * o_i + tl.dot(p, v, input_precision="ieee")
        m_i = m_new

    o_i = o_i / l_i[:, None]

    out_ptrs = out_ptr + offs_m[:, None] * D + offs_d[None, :]
    out_mask = (offs_m[:, None] < N_Q) & (offs_d[None, :] < D)
    tl.store(out_ptrs, o_i, mask=out_mask)


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the FlashAttention forward kernel: out = softmax(Q K^T / sqrt(D)) V.

    Q is (N_Q, D); K and V are (N_KV, D). The two lengths may differ
    (cross-attention); out is (N_Q, D).
    """
    N_Q, D = Q.shape
    N_KV = K.shape[0]
    scale = 1.0 / (D ** 0.5)
    BLOCK_M = 16
    BLOCK_N = 16
    # tl.dot needs at least 16 along each operand dimension, so pad the head
    # dim to a power of two >= max(D, 16); the d-tail mask zeroes the padding.
    BLOCK_DMODEL = 16
    while BLOCK_DMODEL < D:
        BLOCK_DMODEL *= 2
    grid = ((N_Q + BLOCK_M - 1) // BLOCK_M,)
    flash_attention_fwd_kernel[grid](
        Q, K, V, out,
        N_Q, N_KV, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
    )