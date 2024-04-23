import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

class OpenWebTextDataset(Dataset):
    def __init__(self, filename, chunk_size):
        self.filename = filename
        self.chunk_size = chunk_size
        
        with open(filename, 'rb') as file:
            file.seek(0, 2)  # Move the file pointer to the end
            self.length = file.tell() // 8  # Divide by 8 (int64 size) to get the number of elements
        
        self.length -= chunk_size + 1

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        offset = idx * 8  # Calculate the byte offset for the desired index
        
        input_data = np.memmap(self.filename, dtype=np.int64, mode='r', offset=offset, shape=(self.chunk_size,)).copy()
        target_data = np.memmap(self.filename, dtype=np.int64, mode='r', offset=offset+8, shape=(self.chunk_size,)).copy()
        
        input_tensor = torch.from_numpy(input_data)
        target_tensor = torch.from_numpy(target_data)
        
        return input_tensor, target_tensor