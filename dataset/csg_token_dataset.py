
import torch
from torch.utils.data import Dataset
import numpy as np

class CSGTokenDataset(Dataset):
    def __init__(self, npy_path):
        self.data = np.load(npy_path, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_array = self.data[idx]
        return torch.tensor(token_array, dtype=torch.long)
