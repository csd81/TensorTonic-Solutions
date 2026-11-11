import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, data: list, label_col: int):
        # Convert the entire 2D list to a float32 tensor upfront for fast indexing
        tensor_data = torch.tensor(data, dtype=torch.float32)
        
        # Extract the label column and keep shape (N, 1)
        self.labels = tensor_data[:, label_col : label_col + 1]
        
        # Select all remaining columns in their original order
        feature_cols = [i for i in range(tensor_data.shape[1]) if i != label_col]
        self.features = tensor_data[:, feature_cols]

    def __len__(self) -> int:
        """
        Returns the number of rows.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (features, label) as float32 tensors of shapes (D,) and (1,).
        """
        return self.features[idx], self.labels[idx]
