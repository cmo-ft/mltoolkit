import torch
import torch_geometric
import numpy as np
import glob
import os

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

# save data in the following format
channel = "lephadSLT"
# channel = "hadhad"

source_dir = f"./{channel}_ntuples/"
data_suffix = "*.pt"
dest_data_format = f"../../dataset/{channel}/dataset%i.pt"

nfold = 3

data_files = glob.glob(source_dir + data_suffix)
source = data_source(data_files, nfold=nfold)
data_list = source.get_data_list()

for i in range(nfold):
    data_source.save_data_lists(data_list[i], dest_data_format%i)
    