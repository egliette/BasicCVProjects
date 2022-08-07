import torch
import numpy as np
from utils.utils import to_device
from torch.utils.data import Dataset

class ShelterOutcomeDataset(Dataset):
    def __init__(self, X, Y, embedded_cols_names):
        self.X_cat = X.loc[:, embedded_cols_names].copy().values.astype(np.int64)
        self.X_num = X.drop(columns=embedded_cols_names).copy().values.astype(np.float32)
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X_cat[index], self.X_num[index], self.Y[index]

class DeviceDataloader():
    # Wrap a dataloader to move data to a device
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        # Yield a batch of data after moving it to device
        for batch in self.dataloader:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dataloader)
