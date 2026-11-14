import torch
import torch.nn.functional as F

def composite_layer(x: torch.Tensor, 
                    bn_gamma: torch.Tensor, 
                    bn_beta: torch.Tensor, 
                    bn_mean: torch.Tensor, 
                    bn_var: torch.Tensor, 
                    conv_weight: torch.Tensor, 
                    eps: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 output of the DenseNet composite transformation.
    Order: Evaluation-mode Batch Normalization -> ReLU -> Conv3x3 (bias-free, stride 1, pad 1)
    """
    # Ensure inputs are float64
    x = x.to(torch.float64)
    bn_gamma = bn_gamma.to(torch.float64)
    bn_beta = bn_beta.to(torch.float64)
    bn_mean = bn_mean.to(torch.float64)
    bn_var = bn_var.to(torch.float64)
    conv_weight = conv_weight.to(torch.float64)

    # 1. Per-channel Batch Normalization using running stats (evaluation mode)
    # Reshape gamma, beta, mean, var for broadcasting across (N, C, H, W)
    # bn_* shape is (C,) -> reshape to (1, C, 1, 1)
    view_shape = (1, x.shape[1], 1, 1)
    gamma = bn_gamma.view(view_shape)
    beta = bn_beta.view(view_shape)
    mean = bn_mean.view(view_shape)
    var = bn_var.view(view_shape)

    x_norm = gamma * (x - mean) / torch.sqrt(var + eps) + beta

    # 2. ReLU activation
    x_rel = F.relu(x_norm)

    # 3. Bias-free 3x3 convolution with stride=1, padding=1
    y = F.conv2d(x_rel, weight=conv_weight, bias=None, stride=1, padding=1)

    return y.to(torch.float64)
