import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def create_balanced_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int) -> DataLoader:
    """
    Returns a DataLoader with inverse-frequency sampling and replacement.
    """
    # 1. Count occurrences of each class label
    # bincount works perfectly for non-negative int64 class indices
    class_counts = torch.bincount(labels)
    
    # 2. Compute sample weights: w_i = 1 / n_{y_i}
    # We index class_counts with labels to map each class's count to its corresponding samples
    sample_weights = 1.0 / class_counts[labels]
    
    # 3. Create the WeightedRandomSampler
    # It draws N samples (len(labels)) with replacement as requested
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
    
    # 4. Wrap features and labels into a TensorDataset
    dataset = TensorDataset(features, labels)
    
    # 5. Build and return the DataLoader
    # Note: When using a sampler, 'shuffle' must be set to None or omitted
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False  # Do not drop the final partial batch
    )
    
    return loader
