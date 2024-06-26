import torch_geometric
import pandas as pd
import torch
import time
import json
import numpy as np
import pandas as pd
import numpy as np
import uproot as ur
import torch_geometric.utils as tgu
from torch_geometric.data import Data
import torch_geometric.utils as tgu
import os
import argparse

parser = argparse.ArgumentParser(description="generate .pt file from data.root")
parser.add_argument('-i', '--input', type=str, help="input file name.", default='/lustre/collider/mocen/project/bbtautau/analysis/Htautau-recon/atlas_nu2flows/data/4top_dilepton/fourtop_2LSS_100w_filtered.h5')
args = parser.parse_args()

inpath = args.input


def process_gnn_data(entry_start, entry_stop, outpath):
    """
    Get graph from existed root files
    """
    with pd.HDFStore(args.input, "r") as f:
        jets = f['/recon/jets']
        leptons = f['/recon/leptons']
        MET = f['/recon/MET']
        neutrino = f['/truth/neutrinos']
        misc = f['/recon/misc']

    all_graph = []

    entry_stop = min(entry_stop, len(misc)) if entry_stop>0 else len(misc)
    for ientry in range(entry_start, entry_stop):
        graph = Data()
        jet_feature = jets.loc[ientry][['px','py','pz','log_energy']].values
        jet_mask = jets.loc[ientry]['mask'].astype(np.bool_)
        jet_feature = jet_feature[jet_mask]

        lepton_feature = leptons.loc[ientry][['px','py','pz']].values
        lepton_feature = np.concatenate([lepton_feature, np.log(np.linalg.norm(lepton_feature, axis=1, keepdims=True))], axis=1)
        lepton_mask = leptons.loc[ientry]['mask'].astype(np.bool_)
        lepton_feature = lepton_feature[lepton_mask]

        MET_feature = MET.loc[ientry][['px','py']].values.reshape(1,2)
        MET_feature = np.concatenate([MET_feature, np.zeros([len(MET_feature), 1]), np.log(np.linalg.norm(MET_feature, axis=1, keepdims=True))], axis=1)
        
        node_features = torch.from_numpy(
            np.concatenate(
                [
                    jet_feature,
                    lepton_feature,
                    MET_feature
                ]
            )
        ).type(torch.float32)

        global_features = torch.from_numpy(
            misc.loc[ientry][['njets', 'nbjets']].values
        ).type(torch.float32).view(1, 2)

        truth_label = torch.from_numpy(
            neutrino.loc[ientry][['px','py','pz','log_energy']].values.reshape(1, -1)
        ).type(torch.float32)

        weight_train = torch.tensor(1.0, dtype=torch.float32).view(1,1)

        graph = Data(
            node_features=node_features,
            truth_label=truth_label,
            global_features=global_features,
            weight_train=weight_train
        )

        all_graph += [graph] if graph!=None else []

    torch.save(torch_geometric.data.Batch.from_data_list(all_graph), outpath)


t_start = time.time()
break_points = [0, int(1e5), 120001, -1]
data_type_iter = iter(['train', 'validation', 'test'])
for i in range(len(break_points)-1):
    data_type = next(data_type_iter)
    os.makedirs(data_type, exist_ok=True)
    outpath = f'{data_type}/{data_type}.pt'
    process_gnn_data(break_points[i], break_points[i+1], outpath)
    print(f"From {break_points[i]} to {break_points[i+1]} finished.")
print(time.time()-t_start)

