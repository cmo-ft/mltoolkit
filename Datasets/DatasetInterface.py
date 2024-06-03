import os
import uproot as ur
import pandas as pd
from importlib import import_module
from collections import deque
import torch_geometric
import torch_geometric.loader
import subprocess
import logging

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
        setup(): Loads the dataset.
        manipulate_folds(): Manipulates the folds for training, validation, and testing.
        get_dataloader(key): Returns the dataloader for the specified key.

    """

    def __init__(self, data_config):
        super().__init__()
        self.data_config = data_config
        self.setup()
    
    def setup(self)->None:
        """
        Loads the dataset.

        This method sets the batch size, ntuple path list, path to save graphs,
        data class, and dataloaders for different sets.

        """
        self.batch_size = self.data_config.get('batch_size')
        self.path_save_graphs = self.data_config.get('path_save_graphs')

        data_class_name = self.data_config.get('dataclass').split('.') # e.g. 'HadHadDataset.HadHadGGFHighDataset'
        print(".".join(['Datasets'] + data_class_name[:-1]))
        self._dataclass = getattr(import_module(".".join(['Datasets'] + data_class_name[:-1])), data_class_name[-1])
        self.dataloader_dict = {'train': None, 'validation': None, 'test': None}

    def get_dataloader(self, key:str)->torch_geometric.loader.DataLoader:
        """
        Returns the dataloader for the specified key.

        Args:
            key (str): The key for the dataloader.

        Returns:
            torch_geometric.loader.DataLoader: The dataloader for the specified key.

        """
        if key not in self.dataloader_dict or self.dataloader_dict[key] is None:
            paths = []
            for p in self.data_config.get(key):
                tmp_paths = subprocess.run( 'ls ' + p, shell=True, capture_output=True, text=True)
                paths += tmp_paths.stdout.split('\n')[:-1] # last element is empty string so remove it
            self.dataloader_dict[key] = torch_geometric.loader.DataLoader(self._dataclass(paths, f'{self.path_save_graphs}/{key}.pt').graph_list, batch_size=self.batch_size, shuffle=False)

        return self.dataloader_dict[key]
    