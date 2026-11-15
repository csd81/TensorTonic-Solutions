import math
import torch
import torch.nn.functional as F


def glm_forward(
    input_ids, embed_W, blocks, final_norm_g, lm_head_W, cos, sin,
    num_attention_heads, num_kv_heads, head_dim, rope_dim,
    n_routed_experts, num_experts_per_tok, n_group, topk_group,
    routed_scaling_factor, norm_topk_prob, first_k_dense_replace,
    eps_rms=1e-5,
):
    def _rms(t, g):
        return t * g * torch.rsqrt(torch.mean(t ** 2, dim=-1, keepdim=True) + eps_rms)

    def _rope(t):
        # t shape: (n_heads, n_tokens, head_dim)
        out = t.clone()
        rp = t[..., :rope_dim]
        half = rope_dim // 2
        rotated_half = torch.cat([-rp[..., half:], rp[..., :half]], dim=-1)
        cos_full = torch.cat([cos, cos], dim=-1)
        sin_full = torch.cat([sin, sin], dim=-1)
        out[..., :rope_dim] = rp * cos_full + rotated_half * sin_full
        return out

    def _swiglu(x, Wg, Wu, Wd):
        g = F.linear(x, Wg)
        u = F.linear(x, Wu)
        return F.linear(F.silu(g) * u, Wd)

    n = input_ids.shape[0]
    x = embed_W[input_ids]
    q_mult = num_attention_heads // num_kv_heads

    for li, lw in enumerate(blocks):
        # 1. Pre-Attention RMSNorm & Projections
        xn = _rms(x, lw["attn_norm_g"])
        Q = F.linear(xn, lw["W_q"], lw["b_q"]).reshape(n, num_attention_heads, head_dim).transpose(0, 1)
        K = F.linear(xn, lw["W_k"], lw["b_k"]).reshape(n, num_kv_heads, head_dim).transpose(0, 1)
        V = F.linear(xn, lw["W_v"], lw["b_v"]).reshape(n, num_kv_heads, head_dim).transpose(0, 1)

        # 2. QK Norm & Partial RoPE
        Q = _rope(_rms(Q, lw["q_norm_g"]))
        K = _rope(_rms(K, lw["k_norm_g"]))

        # 3. Grouped-Query Attention (GQA) with Causal Mask
        K_e = K.repeat_interleave(q_mult, dim=0)
        V_e = V.repeat_interleave(q_mult, dim=0)
        sc = (Q @ K_e.transpose(-2, -1)) / math.sqrt(head_dim)
        
        causal_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
        sc = sc.masked_fill(causal_mask.unsqueeze(0), -1e30)
        w = torch.softmax(sc, dim=-1)
        
        attn = (w @ V_e).transpose(0, 1).reshape(n, num_attention_heads * head_dim)
        x = x + F.linear(attn, lw["W_o"])

        # 4. Pre-MLP RMSNorm & Dense / MoE Block
        xn2 = _rms(x, lw["mlp_norm_g"])
        if li < first_k_dense_replace:
            mlp_out = _swiglu(xn2, lw["W_gate"], lw["W_up"], lw["W_down"])
        else:
            # Group Router
            scores = torch.sigmoid(F.linear(xn2, lw["gate_W"]))
            biased = scores + lw["e_score_correction_bias"]
            per_group = n_routed_experts // n_group
            bg = biased.view(n, n_group, per_group)
            
            gscore = bg.topk(min(2, per_group), dim=-1).values.sum(dim=-1)
            _, gi = gscore.topk(topk_group, dim=-1)
            
            mask = torch.zeros(n, n_group, dtype=torch.bool, device=x.device)
            mask.scatter_(1, gi, True)
            mask = mask.unsqueeze(-1).expand(-1, -1, per_group).reshape(n, n_routed_experts)
            
            biased_m = biased.masked_fill(~mask, -1e30)
            _, ti = biased_m.topk(num_experts_per_tok, dim=-1)
            
            sel = scores.gather(1, ti)
            if norm_topk_prob:
                sel = sel / (sel.sum(dim=-1, keepdim=True) + 1e-20)
            sel = sel * routed_scaling_factor

            # Routed Experts + Shared Expert
            moe_out = torch.zeros_like(xn2)
            experts = lw["experts"]
            for t in range(n):
                for k in range(num_experts_per_tok):
                    eid = int(ti[t, k].item())
                    wi = sel[t, k]
                    exp = experts[eid]
                    moe_out[t] += wi * _swiglu(xn2[t], exp["W_gate"], exp["W_up"], exp["W_down"])
            
            shared = _swiglu(xn2, lw["shared_W_gate"], lw["shared_W_up"], lw["shared_W_down"])
            mlp_out = moe_out + shared

        x = x + mlp_out

    # 5. Final RMSNorm & LM Head (Note: lm_head_W is not transposed here)
    x = _rms(x, final_norm_g)
    return x @ lm_head_W
