import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        # Store the dropout probability.
        # Ensure it's treated as a float and bound within [0.0, 1.0]
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor with the same shape as x.
        """
        # 1. Evaluation mode or p = 0: pass the input directly through
        if not self.training or self.p == 0.0:
            return x
        
        # 2. Complete dropout (p = 1): return a tensor of zeros matching the input shape
        if self.p == 1.0:
            return torch.zeros_like(x)
        
        # 3. Training mode: Create a binary mask where elements are dropped with probability p.
        # torch.rand_like generates values uniformly between 0 and 1.
        # Values >= p are kept (1), values < p are dropped (0).
        mask = (torch.rand_like(x) >= self.p).float()
        
        # 4. Scale the surviving values by 1 / (1 - p) to maintain expected value consistency
        return (x * mask) / (1.0 - self.p)
