import torch
import torch_geometric
import numpy as np
import glob
import os
from configs import out_path_format

class data_source:
    def __init__(self, data_file_list, nfold=3) -> None:
        self.data_file_list = data_file_list 
        self.nfold = nfold

    def get_data_list(self, ):
        if  not hasattr(self, 'data_list'):
            self.data_list = [[] for i in range(self.nfold)]
            for file in self.data_file_list:
                data = torch.load(file)
                evt_num = data.EvtNum
                self.data_list = [ self.data_list[ifold] + data[(evt_num%self.nfold==ifold)] for ifold in range(self.nfold)]
        return self.data_list
        
    @staticmethod
    def save_data_lists(data_list, dest_file_name):
        torch.save(torch_geometric.data.Batch.from_data_list(data_list), dest_file_name)

# merge following data
channels = [
    'hadhad',
    # 'lephadSLT'
]
SRs = [
    'ggF_high', 
    'ggF_low', 
    'vbf'
]
nfold = 3

for c in channels:
    for sr in SRs:
        print(f"Mergin channel {c} SR {sr}...")
        graph_path = os.path.dirname(out_path_format.format(channel=c, sr=sr, sample_name=""))
        source_graphs = glob.glob(graph_path + '/*.pt')
        dest_data_format = graph_path + '/../dataset{ifold}.pt'

        # get source graph
        source = data_source(source_graphs, nfold=nfold)
        data_list = source.get_data_list()

        # merge dest graph
        for i in range(nfold):
            data_source.save_data_lists(data_list[i], dest_file_name=dest_data_format.format(ifold=i))
            