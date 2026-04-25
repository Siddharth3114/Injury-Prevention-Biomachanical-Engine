import torch
from torch.utils.data import Dataset
import numpy as np
import os

class PoseDataset(Dataset):
    def __init__(self, npy_dir, transform=None):
        """
        Args:
            npy_dir (string): Directory with all the .npy pose files. 
                              Or a single .npy file path.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.npy_dir = npy_dir
        self.transform = transform
        
        if os.path.isfile(npy_dir) and npy_dir.endswith('.npy'):
            self.file_list = [npy_dir]
        elif os.path.isdir(npy_dir):
            self.file_list = [os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.endswith('.npy')]
        else:
            self.file_list = []
            
        if not self.file_list:
            print(f"Warning: No .npy files found in {npy_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        npy_path = self.file_list[idx]
        # Shape is expected to be (C, T, V, M)
        data = np.load(npy_path)
        
        # Convert to torch tensor
        tensor_data = torch.from_numpy(data).float()
        
        if self.transform:
            tensor_data = self.transform(tensor_data)
            
        # Returns (C, T, V, M). 
        # The DataLoader will automatically add the Batch (N) dimension to make it (N, C, T, V, M)
        return tensor_data
