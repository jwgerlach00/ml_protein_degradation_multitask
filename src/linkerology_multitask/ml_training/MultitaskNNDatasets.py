import torch
import numpy as np
from typing import Tuple


class MultitaskNNDataset(torch.utils.data.Dataset):
    def __init__(self, X:np.ndarray, Y:np.ndarray, device:torch.device) -> None:
        super(MultitaskNNDataset, self).__init__()
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f'X and Y should have the same number of entries. Instead X has {X.shape[0]} entries and Y has {Y.shape[0]} entries.')
        self.X = X.copy()
        self.Y = Y.copy() # Y is a matrix w/ shape (num_entries, num_targets)
        self.device = device

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index:int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.X[index, :], device=self.device).float(),
            torch.tensor(self.Y[index, :], device=self.device).float()
        )


class MultitaskNNDatasetMultitask(MultitaskNNDataset):
    def __init__(self, X:np.ndarray, Y:np.ndarray, device:torch.device) -> None:
        super(MultitaskNNDatasetMultitask, self).__init__(X, Y, device)
    
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.X[index, :], device=self.device).float(),
            torch.tensor(self.Y[index, :], device=self.device).long() # modify to return long rather than float
        )


class ComponentMultitaskNNDataset(MultitaskNNDataset):
    def __init__(self, X:np.ndarray, Y:np.ndarray, device:torch.device) -> None:
        X = X.reshape((X.shape[0], -1)) # from shape (num_entries, 3, bit_vector_length) to shape (num_entries, 3*bit_vector_length) \
            # 3 for each componet: target_warhead, linker, e3_warhead
        super(ComponentMultitaskNNDataset, self).__init__(X, Y, device)
