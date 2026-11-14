import torch
import torch.nn.functional as F

def dense_block(x: torch.Tensor, layers: list, growth_rate: int,
                eps: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 concatenation of the block input and all layer outputs.
    """
    features = [x]

    for layer in layers:
        # Concatenate all preceding feature maps along the channel axis (dim=1)
        concat_features = torch.cat(features, dim=1)

        # Convert supplied parameters to float64 tensors matching x's device
        bn_mean = torch.as_tensor(layer["bn_mean"], dtype=torch.float64, device=x.device)
        bn_var = torch.as_tensor(layer["bn_var"], dtype=torch.float64, device=x.device)
        bn_gamma = torch.as_tensor(layer["bn_gamma"], dtype=torch.float64, device=x.device)
        bn_beta = torch.as_tensor(layer["bn_beta"], dtype=torch.float64, device=x.device)
        conv_weight = torch.as_tensor(layer["conv_weight"], dtype=torch.float64, device=x.device)

        # Composite function H_l: BatchNorm -> ReLU -> Conv3x3 (with padding=1 to preserve H, W)
        out = F.batch_norm(
            concat_features,
            running_mean=bn_mean,
            running_var=bn_var,
            weight=bn_gamma,
            bias=bn_beta,
            training=False,
            eps=eps
        )
        out = F.relu(out)
        out = F.conv2d(out, weight=conv_weight, padding=1)

        # Store the new feature map for subsequent layers and the final output
        features.append(out)

    return torch.cat(features, dim=1)
