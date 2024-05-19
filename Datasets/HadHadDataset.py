from typing import Tuple
import pandas as pd
import numpy as np
from Datasets.BaseDataset import BaseDataset, GraphDataFormat

class HadHadGGFHighDataset(BaseDataset):
    def __init__(self, ntuple_path_list: list[str], fold_id: int, total_folds: int=3, path_save_graphs: int=None):
        super().__init__(ntuple_path_list, path_save_graphs)
        self.fold_id = fold_id
        self.total_folds = total_folds

        self.nodes = ["bbtt_HH", "bbtt_HH_vis", "bbtt_H_bb", "bbtt_mmc", "met_NOSYS", "bbtt_H_vis_tautau", "bbtt_mmc_nu1", "bbtt_mmc_nu2", "bbtt_Tau1", "bbtt_Jet_b1", "bbtt_Jet_b2"]
        self.node_feature_format = ["{particle}_eta", "{particle}_phi", "{particle}_pt_NOSYS", "{particle}_E", "{particle}_m"]
        self.edge_feature_format = ["dR_{p0}{p1}", "dPhi_{p0}{p1}", "M_{p0}{p1}",]
        self.glob_features = ["num_jets", "T1", "mTtau1", "spher_bbtt", "cent_bbtt", "met_NOSYS_sumet"]

    def get_tree_name(self):
        return "tree_2tag_OS_LL_GGFSR_350mHH"

    def generate_graph_data(self, data: pd.Series) -> GraphDataFormat:
        """
        Generates graph data in the specified format.
        """
        if data['eventNumber']%self.total_folds!=self.fold_id:
            return None
        truth_label = data["truth_label"]
        node_features = self.generate_node_features(data)
        edge_index, edge_features = self.generate_edge_index_and_features(data)
        global_features = data[self.glob_features].values
        weight_original, weight_train = self.generate_original_and_train_weights(data)
        misc_features = None
        return GraphDataFormat(truth_label, node_features, edge_index, edge_features, global_features, weight_original, weight_train, misc_features)
        

    def generate_node_features(self, data: pd.Series) -> np.ndarray:
        """
        Generate node features based on the given data.
        Returns:
            np.ndarray: The generated node features with shape (num_nodes, -1).
        """
        node_feature_names = []
        for particle_name in self.nodes:
            node_feature_names.extend([ft_format.format(particle=particle_name) for ft_format in self.node_feature_format])
        node_features = data[node_feature_names].values.reshape(len(self.nodes), len(self.node_feature_format)) 

        # misc features
        bjet_weights = data[["bbtt_Jet_b1_pcbt_GN2v01", "bbtt_Jet_b2_pcbt_GN2v01"]].values
        node_weights = np.array([1, 0.5, 1, 1, 1, 0.5, 0.5, 0.5, 1, bjet_weights[0], bjet_weights[1]]).reshape(-1, 1)
        node_id = np.arange(len(self.nodes)).reshape(-1, 1)

        node_features = np.concatenate([node_features, node_weights, node_id], axis=1)
        return node_features
        

    def generate_edge_index_and_features(self, data: pd.Series)->Tuple[np.ndarray, np.ndarray]:
        """
        Generate edge features based on the given data. Full connected graph. Each edge contains self.edge_feature_format features.
        Retruns:
            edge_index: shape (2, num_edges), each column is a edge, each element is the index of the node
            edge_features: shape (num_edges, len(self.edge_feature_format)), each row is a edge, each element is the feature of the edge
        """
        num_nodes = len(self.nodes)
        edge_index = []
        edge_features = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i==j:
                    continue
                edge_index.append([i, j])
                edge_features.append([ft_format.format(p0=self.nodes[i], p1=self.nodes[j]) for ft_format in self.edge_feature_format])
        edge_features = data[edge_features].values.reshape(-1, len(self.edge_feature_format))
        return np.array(edge_index), edge_features
        

    def generate_original_and_train_weights(self, data: pd.Series) -> Tuple[float, float]:
        """
        Generates the original weight and train weight based on the given data.

        Returns:
            Tuple[float, float]: A tuple containing the original weight and train weight.
        """
        origional_weight = data["weight_NOSYS"]
        train_weight = origional_weight.clip(0, 1)
        train_weight = train_weight if data["truth_label"] == 0 else train_weight * 800
        return origional_weight, train_weight