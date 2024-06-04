import pandas as pd
from abc import ABC,abstractmethod
import uproot as ur
import os
from collections import namedtuple
import torch_geometric
import torch

GraphDataFormat = namedtuple("GraphDataFormat", ["truth_label", "node_features", "edge_index", "edge_features", "global_features", "weight_original", "weight_train", "misc_features"])

class BaseDataset(ABC):
    """
    Base class for creating custom datasets in PyTorch Geometric.

    Args:
        ntuple_path_list (list[str]): List of paths to the input ntuple files.
        graph_path (str, optional): Path to save the preprocessed graph data. If provided,
            the graph data will be loaded from this path if it exists, otherwise it will be generated
            and saved to this path. Defaults to None.
    """

    def __init__(self, data_path_list, graph_path=None):
        self.data_path_list = data_path_list
        self.graph_list = []
        if (graph_path is None) or (not os.path.exists(graph_path)):
            self.load_graphs_into_graph_list()
            if graph_path is not None:
                torch.save(torch_geometric.data.Batch.from_data_list(self.graph_list), graph_path)
        else:
            self.graph_list = torch.load(graph_path)

    @abstractmethod
    def load_graphs_into_graph_list(self):
        """
        Loads the graph data into self.graph_list
        """
        ...

    def len(self):
        return len(self.graph_list)
    
    def __len__(self):
        return self.len()

    def get(self, idx)->torch_geometric.data.Data:
        return self.graph_list[idx]
    
    def __getitem__(self, idx)->torch_geometric.data.Data:
        return self.get(idx)
