import torch
import torch.nn.functional as F

def densenet_forward(x: torch.Tensor, weights: dict, growth_rate: int,
                     eps: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 class logits from the complete DenseNet forward pass.
    """
    def to_t(val):
        if isinstance(val, torch.Tensor):
            return val.to(dtype=torch.float64)
        return torch.tensor(val, dtype=torch.float64)

    out = to_t(x)

    # 1. 3x3 Stem Convolution (padding=1 preserves H, W)
    stem_w = to_t(weights["stem_conv"])
    out = F.conv2d(out, stem_w, bias=None, stride=1, padding=1)

    blocks = weights["blocks"]
    transitions = weights["transitions"]

    # 2. Dense Blocks and Transition Layers
    for block_idx, block in enumerate(blocks):
        for layer in block:
            bn_out = F.batch_norm(
                out,
                running_mean=to_t(layer["bn_mean"]),
                running_var=to_t(layer["bn_var"]),
                weight=to_t(layer["bn_gamma"]),
                bias=to_t(layer["bn_beta"]),
                training=False,
                eps=eps,
            )
            relu_out = F.relu(bn_out)
            new_features = F.conv2d(
                relu_out,
                to_t(layer["conv_weight"]),
                bias=None,
                stride=1,
                padding=1,
            )
            out = torch.cat([out, new_features], dim=1)

        # Apply transition between adjacent dense blocks
        if block_idx < len(transitions):
            trans = transitions[block_idx]
            bn_out = F.batch_norm(
                out,
                running_mean=to_t(trans["bn_mean"]),
                running_var=to_t(trans["bn_var"]),
                weight=to_t(trans["bn_gamma"]),
                bias=to_t(trans["bn_beta"]),
                training=False,
                eps=eps,
            )
            relu_out = F.relu(bn_out)
            conv_out = F.conv2d(
                relu_out,
                to_t(trans["conv_weight"]),
                bias=None,
                stride=1,
                padding=0,
            )
            out = F.avg_pool2d(conv_out, kernel_size=2, stride=2)

    # 3. Final Batch Normalization and ReLU
    out = F.batch_norm(
        out,
        running_mean=to_t(weights["final_bn_mean"]),
        running_var=to_t(weights["final_bn_var"]),
        weight=to_t(weights["final_bn_gamma"]),
        bias=to_t(weights["final_bn_beta"]),
        training=False,
        eps=eps,
    )
    out = F.relu(out)

    # 4. Global Average Pooling -> shape (N, C)
    pooled = out.mean(dim=(2, 3))

    # 5. Linear Classifier -> shape (N, C_classes)
    logits = F.linear(
        pooled,
        to_t(weights["fc_weight"]),
        to_t(weights["fc_bias"]),
    )

    return logits.to(dtype=torch.float64)
