import torch
from typing import Tuple
import pandas as pd
import numpy as np
import torch_geometric
from Datasets.BaseDataset import BaseDataset, GraphDataFormat

class MuonEnergyDataset(BaseDataset):
    def __init__(self, data_path_list, graph_path=None):
        super().__init__(data_path_list, graph_path)

    def load_graphs_into_graph_list(self):
        for data_path in self.data_path_list:
            self.graph_list += self.get_graph_list_from_path(data_path)

    @classmethod
    def get_graph_list_from_path(cls, data_path:str)->list:
        data = torch.load(data_path)
        graph_list = []
        for graph in data.to_data_list():
            graph_list.append(
                cls.define_graph(graph)
            )
        return graph_list
    
    @classmethod
    def define_graph(cls, graph:torch_geometric.data.Data)->torch_geometric.data.Data:
        # first_hit_tr = graph.hits_tr[0]
        # tr = graph.hits_tr - first_hit_tr
        tr = graph.hits_tr
        node_features = torch.cat([graph.nhits.view(-1,1), tr], dim=1)
        truth_label = torch.log10(graph.lep_energy).view(1,1)
        t = tr[:,0]
        pos = tr[:,1:]

        # # edge index
        # num_nodes = len(node_features)
        # # Create all possible combinations of node indices
        # row = torch.arange(num_nodes).repeat(num_nodes)
        # col = torch.arange(num_nodes).repeat_interleave(num_nodes)
        # edge_index = torch.stack([row, col], dim=0)
        return torch_geometric.data.Data(
            node_features = node_features,
            truth_label = truth_label,
            pos = pos,
            t = t,
            # edge_index = edge_index,
            weight_train = torch.tensor(1.0, dtype=torch.float32).view(1,1),
        )
