import uproot as ur
import glob
import torch_geometric as tg
# from torch_cluster import knn_graph
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os
import math
from configs import get_graph_setting
from collections import namedtuple
ParticleProperty = namedtuple("ParticleProperty", ["Importance", "Loc"])


parser = argparse.ArgumentParser(description='Analysis Framework for HH->Multilepton-> 3 leptons')
parser.add_argument('-i', '--input', type=str, default='./data.root', help='input root file')
parser.add_argument('-o', '--output', type=str, default='./', help='output pt file')
parser.add_argument('-c', '--channel', default='hadhad', choices=['hadhad', 'lephadSLT'], help='sample channel')
parser.add_argument('-sr', '--signal_region', default='ggF_high', choices=['ggF_high', 'ggF_low', 'vbf'], help='signal region')
args = parser.parse_args()


filename = args.input
target_filename = args.output

graph_config = get_graph_setting(args.channel, args.signal_region)
num_nodes = len(graph_config.particles)
num_edges = len(graph_config.edges[0])

# Obtain extra node features: weight & loc
# Obtain index of bjet, so we can append PCBT as their weight
# TODO: very ugly. Find a better way for this
extra_node_feature = []
lead_sublead_bjet_loc = [-1, -1]
i = 0
for particle_name, property in graph_config.particles.items():
    extra_node_feature.append([property.Importance] + property.Loc)
    if particle_name=='bbtt_Jet_b1':
        lead_sublead_bjet_loc[0] = i
    elif particle_name=='bbtt_Jet_b2':
        lead_sublead_bjet_loc[1] = i
    i += 1
extra_node_feature = torch.tensor(extra_node_feature, dtype=torch.float)
bjets_importance_index = [lead_sublead_bjet_loc, 0]


# generate datalist for a root file
def generate_pt_list(root_file_name: str):
    pt_list = []
    sample_name_list = []
    try:
        f = ur.open(root_file_name)
        data = f[graph_config.tree_name].arrays(library='pd')#.groupby('entry').first()
    except:
        print(f"Error: file {root_file_name} cannot be opened", flush=True)
        data = pd.DataFrame([])
    if len(data)!=0:   
        y = 0
        if "hhttbb" in root_file_name:
            y = 1
        for ievent in range(len(data)):
            cur_event = data.loc[ievent]
            EvtNum = torch.tensor(int(cur_event['eventNumber']), dtype=torch.int)
            bdtScore = torch.tensor(cur_event['BDT_score'], dtype=torch.float)

            # sample weight
            sample_weight = torch.tensor(cur_event['weight_NOSYS'], dtype=torch.float)
            # global feature
            ft_glob = torch.tensor(cur_event[graph_config.glob_feature]).view(1,-1).float()

            # Get PCBT score
            pcbts = torch.tensor(cur_event[["bbtt_Jet_b1_pcbt_GN2v01", "bbtt_Jet_b2_pcbt_GN2v01"]], dtype=torch.float)
            extra_node_feature[bjets_importance_index] = pcbts / 4. # importance of bjets: pcbt=3 -> importance=0.75; pcbt=6->1.5
            # node feature
            ft_nodes = torch.tensor(cur_event[graph_config.node_feature], dtype=torch.float).view(num_nodes, -1)
            ft_nodes = torch.cat([ft_nodes, extra_node_feature], dim=-1)

            # edge feature
            ft_edges = torch.tensor(data.loc[ievent, graph_config.edge_feature]).view(num_edges, -1).float()

            pt = tg.data.Data(
                x=ft_nodes, edge_index=graph_config.edges, u=ft_glob, y=y, edge_attr=ft_edges, sample_weight=sample_weight,
                EvtNum=EvtNum, bdtScore=bdtScore
                )
            pt_list.append(pt)


    return pt_list, sample_name_list



# store sample name
pt_list, sample_name_list = generate_pt_list(filename)

if len(pt_list)!=0:
    torch.save(tg.data.Batch.from_data_list(pt_list), target_filename)
print(f'from {filename} to {target_filename} done.')
