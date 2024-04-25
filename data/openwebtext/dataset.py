import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

class OpenWebTextDataset(Dataset):
    def __init__(self, filename, chunk_size):
        self.filename = filename
        self.chunk_size = chunk_size
        
        # Get the total length of the dataset
        with open(filename, 'rb') as file:
            file.seek(0, 2)  # Move the file pointer to the end
            total_bytes = file.tell()
        
        self.length = (total_bytes // 8) // chunk_size - 1

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        offset = idx * self.chunk_size * 8  # Shift by the full chunk size in bytes for each idx
        
        # Open the memmap with the specific offset and length
        input_data = np.memmap(self.filename, dtype=np.int64, mode='r', offset=offset, shape=(self.chunk_size,)).copy()
        target_data = np.memmap(self.filename, dtype=np.int64, mode='r', offset=offset + 8, shape=(self.chunk_size,)).copy()
        
        input_tensor = torch.from_numpy(input_data)
        target_tensor = torch.from_numpy(target_data)
        
        return input_tensor, target_tensor