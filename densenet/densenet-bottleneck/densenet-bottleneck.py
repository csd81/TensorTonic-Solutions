import torch
import torch.nn.functional as F

def bottleneck_layer(x: torch.Tensor, bn1_gamma: torch.Tensor, bn1_beta: torch.Tensor,
                     bn1_mean: torch.Tensor, bn1_var: torch.Tensor,
                     conv1_weight: torch.Tensor, bn2_gamma: torch.Tensor,
                     bn2_beta: torch.Tensor, bn2_mean: torch.Tensor,
                     bn2_var: torch.Tensor, conv2_weight: torch.Tensor,
                     eps: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 output of the DenseNet-B bottleneck transformation.
    """
    # Ensure all inputs are float64
    x = x.to(torch.float64)
    bn1_gamma, bn1_beta = bn1_gamma.to(torch.float64), bn1_beta.to(torch.float64)
    bn1_mean, bn1_var = bn1_mean.to(torch.float64), bn1_var.to(torch.float64)
    conv1_weight = conv1_weight.to(torch.float64)
    
    bn2_gamma, bn2_beta = bn2_gamma.to(torch.float64), bn2_beta.to(torch.float64)
    bn2_mean, bn2_var = bn2_mean.to(torch.float64), bn2_var.to(torch.float64)
    conv2_weight = conv2_weight.to(torch.float64)

    # Stage 1: BN1 -> ReLU -> Conv 1x1 (padding=0, stride=1, bias=None)
    x_norm1 = F.batch_norm(
        x, running_mean=bn1_mean, running_var=bn1_var,
        weight=bn1_gamma, bias=bn1_beta, training=False, eps=eps
    )
    h = F.conv2d(F.relu(x_norm1), weight=conv1_weight, bias=None, stride=1, padding=0)

    # Stage 2: BN2 -> ReLU -> Conv 3x3 (padding=1, stride=1, bias=None)
    h_norm2 = F.batch_norm(
        h, running_mean=bn2_mean, running_var=bn2_var,
        weight=bn2_gamma, bias=bn2_beta, training=False, eps=eps
    )
    y = F.conv2d(F.relu(h_norm2), weight=conv2_weight, bias=None, stride=1, padding=1)

    return y.to(torch.float64)
