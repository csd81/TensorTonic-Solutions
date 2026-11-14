import torch
import torch.nn.functional as F

def transition_layer(x: torch.Tensor, bn_gamma: torch.Tensor, bn_beta: torch.Tensor,
                     bn_mean: torch.Tensor, bn_var: torch.Tensor,
                     conv_weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 output of the DenseNet transition layer.
    """
    # Ensure all inputs are float64 tensors on the same device as x
    dtype = torch.float64
    device = x.device
    
    x = x.to(dtype=dtype)
    bn_gamma = bn_gamma.to(device=device, dtype=dtype)
    bn_beta = bn_beta.to(device=device, dtype=dtype)
    bn_mean = bn_mean.to(device=device, dtype=dtype)
    bn_var = bn_var.to(device=device, dtype=dtype)
    conv_weight = conv_weight.to(device=device, dtype=dtype)

    # 1. Evaluation-mode Batch Normalization
    out = F.batch_norm(x, running_mean=bn_mean, running_var=bn_var, 
                       weight=bn_gamma, bias=bn_beta, training=False, eps=eps)
    
    # 2. ReLU activation
    out = F.relu(out)
    
    # 3. 1x1 Convolution (bias-free)
    out = F.conv2d(out, weight=conv_weight, bias=None)
    
    # 4. 2x2 Average Pooling with stride 2
    y = F.avg_pool2d(out, kernel_size=2, stride=2)
    
    return y
