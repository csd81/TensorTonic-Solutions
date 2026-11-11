import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=channels,
            eps=1e-5,
            affine=True
        )
        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.bn2 = nn.BatchNorm2d(
            num_features=channels,
            eps=1e-5,
            affine=True
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor with the same shape as x.
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.relu(x + out)
        return out
