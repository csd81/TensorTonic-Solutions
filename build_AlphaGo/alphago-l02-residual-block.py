import torch
import torch.nn.functional as F

def alpha_go_residual_block(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    conv2_weight: torch.Tensor,
    scale2: torch.Tensor,
    shift2: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Returns: the float32 residual output, with the same shape and device as x.
    """
    device = x.device
    x_f32 = x.to(dtype=torch.float32, device=device)
    w1 = conv1_weight.to(dtype=torch.float32, device=device)
    w2 = conv2_weight.to(dtype=torch.float32, device=device)
    gamma1 = scale1.to(dtype=torch.float32, device=device).view(1, -1, 1, 1)
    beta1 = shift1.to(dtype=torch.float32, device=device).view(1, -1, 1, 1)
    gamma2 = scale2.to(dtype=torch.float32, device=device).view(1, -1, 1, 1)
    beta2 = shift2.to(dtype=torch.float32, device=device).view(1, -1, 1, 1)
    
    # Helper for manual batch normalization
    def custom_bn(tensor, gamma, beta):
        mean = tensor.mean(dim=(0, 2, 3), keepdim=True)
        var = (tensor - mean).square().mean(dim=(0, 2, 3), keepdim=True)
        normalized = (tensor - mean) * torch.rsqrt(var + epsilon)
        return normalized * gamma + beta

    # First Layer: Conv3x3 -> BN -> ReLU
    out1 = F.conv2d(x_f32, w1, bias=None, stride=1, padding=1)
    bn1 = custom_bn(out1, gamma1, beta1)
    h = F.relu(bn1)
    
    # Second Layer: Conv3x3 -> BN -> Add Shortcut -> ReLU
    out2 = F.conv2d(h, w2, bias=None, stride=1, padding=1)
    bn2 = custom_bn(out2, gamma2, beta2)
    
    return F.relu(x_f32 + bn2)
