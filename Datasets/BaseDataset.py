import pandas as pd
from abc import ABC,abstractmethod
import uproot as ur
import os
from collections import namedtuple
import torch_geometric
import torch

GraphDataFormat = namedtuple("GraphDataFormat", ["truth_label", "node_features", "edge_features", "global_features", "weight_original", "weight_train", "misc_features"])

class BaseDataset(ABC, torch_geometric.data.Dataset):
    def __init__(self, ntuple_path_list, path_save_graphs=None):
        self.ntuple_path_list = ntuple_path_list
        self.graph_list = []
        if (path_save_graphs is not None) or (not os.path.exists(path_save_graphs)):
            self.load()
            if path_save_graphs is not None:
                torch.save(self.graph_list, path_save_graphs)
        else:
            self.graph_list = torch.load(path_save_graphs)

    def load(self):
        self.tree_name, self.branch_names = self.get_tree_name(), self.get_branch_names()
        for ntuple_path in self.ntuple_path_list:
            tree = ur.open(ntuple_path)[self.tree_name]
            data: pd.DataFrame = tree.arrays(self.branch_names, library="pd")
            for i in range(len(data)):
                graph = self.generate_graph(data.iloc[i])
                graph = torch_geometric.data.Data.from_dict(graph._asdict())
                if graph is not None:
                    self.graph_list.append(graph)

    def len(self):
        return len(self.graph_list)
    
    def __len__(self):
        return self.len()

    def get(self, idx)->torch_geometric.data.Data:
        return self.graph_list[idx]
    
    def __getitem__(self, idx)->torch_geometric.data.Data:
        return self.get(idx)

    @abstractmethod
    def get_tree_name(self)->str:
        ...

    def get_branch_names(self)->list[str]:
        return None

    @abstractmethod
    def generate_graph_data(self, data: pd.Series)->GraphDataFormat:
        """
        Generates graph data based on the given input data.

        Args:
            data (pd.Series): The input data to generate graph data from.

        Returns:
            GraphDataFormat: The generated graph data in the specified format.
        """
        ...