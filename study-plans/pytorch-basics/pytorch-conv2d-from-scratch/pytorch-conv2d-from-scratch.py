import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor of shape (N, out_channels, H-k+1, W-k+1).
        """
        k = self.kernel_size
        patches = x.unfold(2, k, 1).unfold(3, k, 1)
        out = torch.einsum("ncijuv,ocuv->noij", patches, self.weight)
        out = out + self.bias.view(1, -1, 1, 1)
        return out.to(torch.float32)
