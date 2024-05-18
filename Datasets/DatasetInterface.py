from importlib import import_module
from collections import deque
import torch_geometric
import torch_geometric.loader

class DatasetInterface():
    """
    A class representing a dataset interface.

    Args:
        data_config (dict): A dictionary containing the configuration for the dataset.

    Attributes:
        data_config (dict): The configuration for the dataset.
        batch_size (int): The batch size for the dataloader.
        ntuple_path_list (list): A list of paths to the ntuple files.
        path_save_graphs (str): The path to save the graphs.
        _dataclass (class): The data class for the dataset.
        dataloader_dict (dict): A dictionary containing the dataloaders for different sets.

    Methods:
        load(): Loads the dataset.
        manipulate_folds(): Manipulates the folds for training, validation, and testing.
        get_dataloader(key): Returns the dataloader for the specified key.

    """

    def __init__(self, data_config):
        super().__init__()
        self.data_config = data_config
        self.load()
    
    def load(self)->None:
        """
        Loads the dataset.

        This method sets the batch size, ntuple path list, path to save graphs,
        data class, and dataloaders for different sets.

        """
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

    def manipulate_folds(self)->None:
        """
        Manipulates the folds for training, validation, and testing.

        This method sets the fold ID and total number of folds, and creates
        a dictionary containing the indices for training, validation, and testing.

        """
        self.fold_id, self.total_folds = self.data_config.get('fold_id'), self.data_config.get('total_folds')
        idx = deque(range(self.total_folds))
        idx.rotate(self.fold_id)
        self.idx_dict = {
            'train': list(idx)[:-2],
            'validation': list(idx)[-2],
            'test': list(idx)[-1],
        }

    def get_dataloader(self, key:str)->torch_geometric.loader.DataLoader:
        """
        Returns the dataloader for the specified key.

        Args:
            key (str): The key for the dataloader.

        Returns:
            torch_geometric.loader.DataLoader: The dataloader for the specified key.

        """
        return self.dataloader_dict[key]