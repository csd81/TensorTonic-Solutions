import torch
import torch.nn.functional as F

def unet_skip(x: list, weights: dict) -> dict:
    """
    Implements a single-level U-Net forward pass with channel-wise skip connections.
    """
    # Convert input and weights to float64 tensors
    x_t = torch.tensor(x, dtype=torch.float64)
    w_tensors = {}
    for layer, params in weights.items():
        w_tensors[layer + "_w"] = torch.tensor(params["weight"], dtype=torch.float64)
        w_tensors[layer + "_b"] = torch.tensor(params["bias"], dtype=torch.float64)
        
    def conv2d_relu(input_tensor, layer_name, use_relu=True):
        w = w_tensors[layer_name + "_w"]
        b = w_tensors[layer_name + "_b"]
        # Match padding based on kernel size (K=3 -> padding=1, K=1 -> padding=0)
        padding = 1 if w.shape[2] == 3 else 0
        out = F.conv2d(input_tensor, w, b, stride=1, padding=padding)
        return F.relu(out) if use_relu else out

    # 1. Encoder: enc1 -> ReLU -> enc2 -> ReLU
    e1 = conv2d_relu(x_t, "enc1", use_relu=True)
    encoder = conv2d_relu(e1, "enc2", use_relu=True)
    
    # 2. Max Pooling (2x2 window, stride 2)
    pooled = F.max_pool2d(encoder, kernel_size=2, stride=2)
    
    # 3. Bottleneck: bridge1 -> ReLU -> bridge2 -> ReLU
    b1 = conv2d_relu(pooled, "bridge1", use_relu=True)
    bottleneck = conv2d_relu(b1, "bridge2", use_relu=True)
    
    # 4. Upsample: 2x2 Transposed Convolution (stride 2)
    up_w = w_tensors["up_w"]
    up_b = w_tensors["up_b"]
    U = F.conv_transpose2d(bottleneck, up_w, up_b, stride=2, padding=0, output_padding=0)
    
    # 5. Skip Connection: Concatenate U (upsampled) first and encoder features second along channel dim (dim=1)
    J = torch.cat([U, encoder], dim=1)
    
    # 6. Decoder: dec1 -> ReLU -> dec2 -> ReLU
    d1 = conv2d_relu(J, "dec1", use_relu=True)
    decoder = conv2d_relu(d1, "dec2", use_relu=True)
    
    # 7. Output Layer: out (1x1 conv, raw logits without activation)
    output = conv2d_relu(decoder, "out", use_relu=False)
    
    return {
        "encoder": encoder.tolist(),
        "bottleneck": bottleneck.tolist(),
        "decoder": decoder.tolist(),
        "output": output.tolist()
    }
