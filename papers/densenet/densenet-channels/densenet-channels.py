import math
import torch

def densenet_channel_counts(
    stem_channels: int, growth_rate: int, block_layers: list, compression: float
) -> torch.Tensor:
    """Returns the int64 channel count at every DenseNet stage."""
    channels = stem_channels
    counts = [channels]  # Include the initial stem channels
    num_blocks = len(block_layers)
    
    for i, n_layers in enumerate(block_layers):
        # Add growth contributed by the dense block
        channels += n_layers * growth_rate
        counts.append(channels)
        
        # Apply compression between blocks, except after the final block boundary
        if i < num_blocks - 1:
            channels = math.floor(compression * channels)
            counts.append(channels)
            
    return torch.tensor(counts, dtype=torch.int64)
