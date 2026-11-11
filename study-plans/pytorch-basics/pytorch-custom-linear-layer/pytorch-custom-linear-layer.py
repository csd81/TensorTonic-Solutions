import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Initialize and register weight parameter with shape (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        
        # Initialize and register bias parameter with shape (out_features,)
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor of shape (batch, out_features).
        """
        # Compute Y = X * W^T + b
        # torch.matmul handles the matrix multiplication, automatically transposing W if we pass self.weight.t()
        # or we can use x @ self.weight.t()
        return torch.matmul(x, self.weight.t()) + self.bias
