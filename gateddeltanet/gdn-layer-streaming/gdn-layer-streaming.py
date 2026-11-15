import torch
import torch.nn.functional as F
from typing import Optional

def gated_deltanet_layer(
    hidden_states: torch.Tensor,
    params: dict,
    num_heads: int,
    cache: Optional[tuple] = None,
    eps: float = 1e-5
) -> tuple:
    B, L, d = hidden_states.shape

    # 1. Unpack incoming cache if present
    conv_cache, state = cache if cache is not None else (None, None)

    # 2. Run the supplied input path once (keep its default eps=1e-6 for L2 norm)
    q, k, v, log_decay, write_strength, new_conv_cache = gdn_mixer_inputs(
        hidden_states,
        params['w_q'],
        params['w_k'],
        params['w_v'],
        params['conv_q'],
        params['conv_k'],
        params['conv_v'],
        params['w_alpha'],
        params['a_log'],
        params['dt_bias'],
        params['w_beta'],
        params['b_beta'],
        num_heads,
        conv_cache
    )

    # 3. Initialize recurrent state if cache was absent
    d_k = q.shape[-1]
    d_v = v.shape[-1]
    if state is None:
        state = q.new_zeros(B, num_heads, d_v, d_k)

    # 4. Run the supplied recurrence helper once with scaled queries
    outputs, new_state = gated_delta_recurrence(
        q * (d_k ** -0.5),
        k,
        v,
        log_decay,
        write_strength,
        state
    )

    # 5. Per-head RMS normalization over d_v with shared weight
    o_norm = outputs * torch.rsqrt(outputs.square().mean(dim=-1, keepdim=True) + eps) * params['norm_weight']

    # 6. Output gate: SiLU([x_t W_g]_h)
    gate = F.silu(hidden_states @ params['w_gate']).reshape(B, L, num_heads, d_v)

    # 7. Merge heads and project back to model width d
    merged = (o_norm * gate).reshape(B, L, num_heads * d_v)
    y = merged @ params['w_out']

    return y, (new_conv_cache, new_state)
