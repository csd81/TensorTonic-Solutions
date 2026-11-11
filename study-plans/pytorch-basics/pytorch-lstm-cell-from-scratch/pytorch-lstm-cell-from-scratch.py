import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        
        # Input gate parameters
        self.W_ii = nn.Parameter(torch.empty(hidden_size, input_size, dtype=torch.float32))
        self.W_hi = nn.Parameter(torch.empty(hidden_size, hidden_size, dtype=torch.float32))
        self.b_ii = nn.Parameter(torch.empty(hidden_size, dtype=torch.float32))
        self.b_hi = nn.Parameter(torch.empty(hidden_size, dtype=torch.float32))
        
        # Forget gate parameters
        self.W_if = nn.Parameter(torch.empty(hidden_size, input_size, dtype=torch.float32))
        self.W_hf = nn.Parameter(torch.empty(hidden_size, hidden_size, dtype=torch.float32))
        self.b_if = nn.Parameter(torch.empty(hidden_size, dtype=torch.float32))
        self.b_hf = nn.Parameter(torch.empty(hidden_size, dtype=torch.float32))
        
        # Candidate cell state parameters
        self.W_ig = nn.Parameter(torch.empty(hidden_size, input_size, dtype=torch.float32))
        self.W_hg = nn.Parameter(torch.empty(hidden_size, hidden_size, dtype=torch.float32))
        self.b_ig = nn.Parameter(torch.empty(hidden_size, dtype=torch.float32))
        self.b_hg = nn.Parameter(torch.empty(hidden_size, dtype=torch.float32))
        
        # Output gate parameters
        self.W_io = nn.Parameter(torch.empty(hidden_size, input_size, dtype=torch.float32))
        self.W_ho = nn.Parameter(torch.empty(hidden_size, hidden_size, dtype=torch.float32))
        self.b_io = nn.Parameter(torch.empty(hidden_size, dtype=torch.float32))
        self.b_ho = nn.Parameter(torch.empty(hidden_size, dtype=torch.float32))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (hidden state, cell state), both float32 tensors of shape (batch, hidden_size).
        """
        # Compute the 4 gates
        i_t = torch.sigmoid(x @ self.W_ii.T + self.b_ii + h_prev @ self.W_hi.T + self.b_hi)
        f_t = torch.sigmoid(x @ self.W_if.T + self.b_if + h_prev @ self.W_hf.T + self.b_hf)
        g_t = torch.tanh(x @ self.W_ig.T + self.b_ig + h_prev @ self.W_hg.T + self.b_hg)
        o_t = torch.sigmoid(x @ self.W_io.T + self.b_io + h_prev @ self.W_ho.T + self.b_ho)
        
        # Update cell state and hidden state
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t
