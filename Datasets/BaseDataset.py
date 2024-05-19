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

    def __init__(self, ntuple_path_list, graph_path=None):
        self.ntuple_path_list = ntuple_path_list
        self.graph_list = []
        if (graph_path is not None) or (not os.path.exists(graph_path)):
            self.load()
            if graph_path is not None:
                torch.save(self.graph_list, graph_path)
        else:
            self.graph_list = torch.load(graph_path)

    def load(self):
        """
        Loads the input data and generates the graph data.

        This method is called during initialization to load the input data from the ntuple files
        and generate the graph data for each entry.

        Raises:
            FileNotFoundError: If the ntuple file specified in `ntuple_path_list` does not exist.
        """
        self.tree_name, self.branch_names = self.get_tree_name(), self.get_branch_names()
        for ntuple_path in self.ntuple_path_list:
            tree = ur.open(ntuple_path)[self.tree_name]
            data: pd.DataFrame = tree.arrays(self.branch_names, library="pd")
            for i in range(len(data)):
                graph = self.generate_graph_data(data.iloc[i])
                if graph is not None:
                    graph = torch_geometric.data.Data.from_dict(graph._asdict())
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