import torch
import numpy as np
import pandas as pd
import torch_geometric
from torch_geometric.data import Data
import copy

class Dataset(torch_geometric.data.Dataset):
    def __init__(self, data_file):
        # TODO: support a list of index
        super().__init__()
        self.data_file = data_file
        data = torch.load(data_file)
        self.weight = data.sample_weight.clip(min=0, max=1.)
        self.weight = self.weight.abs() * (data.y*1e3 + 1)
        self._data = data

    def len(self) -> int:
        self.size = len(self._data)
        return len(self._data)

    def get(self, idx: int) -> Data:
        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])
        
        data = self._data[idx].clone()
        data.weight = self.weight[idx]
        data.u = torch.where(torch.isnan(data.u), torch.full_like(data.u, 0), data.u) # clean up Nan
        # data.u *= 1e-10 
        # data.x[:2] *= 1e-10
        # data.x = data.x.type(torch.float)
        # data.edge_index = data.edge_index.type(torch.long)
        # data.edge_attr = data.edge_attr * 0.
        self._data_list[idx] = data
        return copy.copy(self._data_list[idx])
