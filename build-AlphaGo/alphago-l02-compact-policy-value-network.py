import torch
import torch.nn.functional as F

def compact_policy_value_network(
    x: torch.Tensor,
    trunk_weight: torch.Tensor,
    trunk_scale: torch.Tensor,
    trunk_shift: torch.Tensor,
    residual_parameters: list[dict],
    head_parameters: dict,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Assembles and runs a compact dual-head policy-value network.
    Returns: float32 tensors (policy_logits, values), shaped (B, N*N+1) and (B,).
    """
    # Ensure all tensors are float32 and match input device
    device = x.device
    x = x.to(dtype=torch.float32)
    
    def norm_and_scale(Z, scale, shift):
        # Population variance and mean over batch and spatial dimensions
        mean = Z.mean(dim=(0, 2, 3), keepdim=True)
        var = Z.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        
        # Reshape scale and shift vectors for broadcasting
        gamma = scale.to(device=device, dtype=torch.float32).view(1, -1, 1, 1)
        delta = shift.to(device=device, dtype=torch.float32).view(1, -1, 1, 1)
        
        return gamma * (Z - mean) / torch.sqrt(var + epsilon) + delta

    # --- Trunk ---
    w_trunk = trunk_weight.to(device=device, dtype=torch.float32)
    h = F.conv2d(x, w_trunk, padding=1)
    h = norm_and_scale(h, trunk_scale, trunk_shift)
    h = F.relu(h)

    # --- Residual Blocks ---
    for block in residual_parameters:
        identity = h
        
        # First layer of residual block
        w1 = block["conv1_weight"].to(device=device, dtype=torch.float32)
        h = F.conv2d(h, w1, padding=1)
        h = norm_and_scale(h, block["scale1"], block["shift1"])
        h = F.relu(h)
        
        # Second layer of residual block
        w2 = block["conv2_weight"].to(device=device, dtype=torch.float32)
        h = F.conv2d(h, w2, padding=1)
        h = norm_and_scale(h, block["scale2"], block["shift2"])
        
        # Skip connection + activation
        h = F.relu(identity + h)

    # --- Policy Head ---
    p_conv_w = head_parameters["policy_conv_weight"].to(device=device, dtype=torch.float32)
    p_scale = head_parameters["policy_scale"]
    p_shift = head_parameters["policy_shift"]
    p_lin_w = head_parameters["policy_linear_weight"].to(device=device, dtype=torch.float32)
    p_lin_b = head_parameters["policy_linear_bias"].to(device=device, dtype=torch.float32)

    hp = F.conv2d(h, p_conv_w, padding=0)
    hp = norm_and_scale(hp, p_scale, p_shift)
    hp = F.relu(hp)
    hp_flat = hp.flatten(start_dim=1)
    policy_logits = F.linear(hp_flat, p_lin_w, p_lin_b)

    # --- Value Head ---
    v_conv_w = head_parameters["value_conv_weight"].to(device=device, dtype=torch.float32)
    v_scale = head_parameters["value_scale"]
    v_shift = head_parameters["value_shift"]
    v_hid_w = head_parameters["value_hidden_weight"].to(device=device, dtype=torch.float32)
    v_hid_b = head_parameters["value_hidden_bias"].to(device=device, dtype=torch.float32)
    v_out_w = head_parameters["value_output_weight"].to(device=device, dtype=torch.float32)
    v_out_b = head_parameters["value_output_bias"].to(device=device, dtype=torch.float32)

    hv = F.conv2d(h, v_conv_w, padding=0)
    hv = norm_and_scale(hv, v_scale, v_shift)
    hv = F.relu(hv)
    hv_flat = hv.flatten(start_dim=1)
    hv_hidden = F.relu(F.linear(hv_flat, v_hid_w, v_hid_b))
    value = torch.tanh(F.linear(hv_hidden, v_out_w, v_out_b)).squeeze(-1)

    return policy_logits, value
