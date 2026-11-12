import torch
import torch.nn.functional as F

def policy_value_heads(
    features, 
    policy_conv_weight, 
    policy_scale, 
    policy_shift, 
    policy_linear_weight, 
    policy_linear_bias, 
    value_conv_weight, 
    value_scale, 
    value_shift, 
    value_hidden_weight, 
    value_hidden_bias, 
    value_output_weight, 
    value_output_bias, 
    epsilon
):
    """
    Computes the policy and value heads using a purely functional API.
    """
    # 1. Cast input features to standard float32 on the target device
    source = features.to(dtype=torch.float32, device=features.device)

    # Helper function for manual Batch Normalization over dimensions (0, 2, 3)
    def batch_norm(v, scale, shift):
        v = v.to(dtype=torch.float32, device=features.device)
        mean = v.mean(dim=(0, 2, 3), keepdim=True)
        variance = (v - mean).square().mean(dim=(0, 2, 3), keepdim=True)
        normalized = (v - mean) * torch.rsqrt(variance + epsilon)
        
        s = scale.to(dtype=torch.float32, device=features.device).view(1, -1, 1, 1)
        b = shift.to(dtype=torch.float32, device=features.device).view(1, -1, 1, 1)
        return normalized * s + b

    # ----------------------------------------------------------------------
    # Policy Head
    # ----------------------------------------------------------------------
    policy = F.conv2d(source, policy_conv_weight.to(dtype=torch.float32, device=features.device))
    policy = batch_norm(policy, policy_scale, policy_shift)
    policy = F.relu(policy).flatten(start_dim=1)
    
    logits = F.linear(
        policy, 
        policy_linear_weight.to(dtype=torch.float32, device=features.device), 
        policy_linear_bias.to(dtype=torch.float32, device=features.device) if policy_linear_bias is not None else None
    )

    # ----------------------------------------------------------------------
    # Value Head
    # ----------------------------------------------------------------------
    value = F.conv2d(source, value_conv_weight.to(dtype=torch.float32, device=features.device))
    value = batch_norm(value, value_scale, value_shift)
    value = F.relu(value).flatten(start_dim=1)
    
    value = F.relu(F.linear(
        value, 
        value_hidden_weight.to(dtype=torch.float32, device=features.device), 
        value_hidden_bias.to(dtype=torch.float32, device=features.device) if value_hidden_bias is not None else None
    ))
    
    value = F.linear(
        value, 
        value_output_weight.to(dtype=torch.float32, device=features.device), 
        value_output_bias.to(dtype=torch.float32, device=features.device) if value_output_bias is not None else None
    ).squeeze(-1)

    return logits, torch.tanh(value)
