import torch
import torch.nn as nn

class RNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_ih = nn.Parameter(torch.empty(hidden_size, input_size, dtype=torch.float32))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size, dtype=torch.float32))
        self.b_ih = nn.Parameter(torch.empty(hidden_size, dtype=torch.float32))
        self.b_hh = nn.Parameter(torch.empty(hidden_size, dtype=torch.float32))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 hidden-state tensor of shape (batch, hidden_size).
        """
        x_proj = x @ self.W_ih.T + self.b_ih
        h_proj = h_prev @ self.W_hh.T + self.b_hh
        h_t = torch.tanh(x_proj + h_proj)
        return h_t
