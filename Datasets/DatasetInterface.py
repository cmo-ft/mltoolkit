from importlib import import_module
from collections import deque
import torch_geometric
import torch_geometric.loader

class DatasetInterface():
    def __init__(self, data_config):
        super().__init__()
        self.data_config = data_config
        self.load()
    
    def load(self):
        self.manipulate_folds()

        self.batch_size = self.data_config.get('batch_size')
        self.ntuple_path_list = self.data_config.get('ntuple_path_list')
        self.path_save_graphs = self.data_config.get('path_save_graphs')

        data_class_name = self.data_config.get('dataclass').split('.') # e.g. 'Datasets.HadHadDataset.HadHadDataset'
        self._dataclass = getattr(import_module(".".join(data_class_name[:-1])), data_class_name[-1])

        train_set = []
        for idx in self.idx_dict['train']:
            train_set += self._dataclass(self.ntuple_path_list, idx, self.total_folds, self.path_save_graphs).graph_list
        self.dataloader_dict = {
            'train': torch_geometric.loader.DataLoader(train_set, batch_size=self.batch_size, shuffle=True),
            'validation': torch_geometric.loader.DataLoader(self._dataclass(self.ntuple_path_list, self.idx_dict['validation'], self.total_folds, self.path_save_graphs).graph_list, batch_size=self.batch_size, shuffle=False),
            'test': torch_geometric.loader.DataLoader(self._dataclass(self.ntuple_path_list, self.idx_dict['test'], self.total_folds, self.path_save_graphs).graph_list, batch_size=self.batch_size, shuffle=False),
        }

    def manipulate_folds(self):
        self.fold_id, self.total_folds = self.data_config.get('fold_id'), self.data_config.get('total_folds')
        idx = deque(range(self.total_folds))
        idx.rotate(self.fold_id)
        self.idx_dict = {
            'train': list(idx)[:-2],
            'validation': list(idx)[-2],
            'test': list(idx)[-1],
        }

    def get_dataloader(self, key):
        return self.dataloader_dict[key]