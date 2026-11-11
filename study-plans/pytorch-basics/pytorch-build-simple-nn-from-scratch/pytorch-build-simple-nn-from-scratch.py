import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        # Define the first linear layer mapping from input to hidden space
        self.fc1 = nn.Linear(in_features, hidden_size)
        # Define the ReLU activation function
        self.relu = nn.ReLU()
        # Define the second linear layer mapping from hidden to output space
        self.fc2 = nn.Linear(hidden_size, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor of shape (batch, out_features).
        """
        # Pass input through the first linear layer
        out = self.fc1(x)
        # Apply the ReLU activation function
        out = self.relu(out)
        # Pass through the final linear layer (no activation at the end)
        out = self.fc2(out)
        return out
