import torch

class TransformPipeline:
    def __init__(self, mean: list, std: list):
        # Convert lists to float32 tensors and reshape to (C, 1, 1) for broadcasting
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Returns a normalized float32 tensor with shape (C, H, W).
        """
        # 1. Convert uint8 tensor to float32 and scale to [0.0, 1.0]
        x = image.to(torch.float32) / 255.0
        
        # 2. Permute from channel-last (H, W, C) to channel-first (C, H, W)
        x = x.permute(2, 0, 1)
        
        # 3. Apply channel-wise normalization via broadcasting
        return (x - self.mean) / self.std
