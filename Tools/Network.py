from importlib import import_module
from typing import Any
import torch
import torch.nn as nn


class NetworkWrapper():
    def __init__(self, network_config, pre_model_path=None) -> None:
        self.network_config = network_config
        self.pre_model_path = pre_model_path
        self.load()

    def load(self):
        # Load Network Structure
        self.network_name = self.network_config.get('Network')
        module_name = f'Tools.Networks.{self.network_name}'
        self.model = getattr(import_module(module_name), self.network_name)(self.network_config.get('model_setting'))

        # Initialize Network
        if self.pre_model_path is None:
            self.initialize_model(model=self.model)
        else:
            self.load_model(self.pre_model_path)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, model_path='./save/net.pt'):
        torch.save(self.model.state_dict(),model_path)

    def parameters(self):
        return self.model.parameters()
    
    def to(self, device):
        self.model.to(device)

    def __call__(self, *args: Any, **kwds: Any):
        self.model(*args, **kwds)

    @staticmethod
    def initialize_model(model):
        def weights_init_uniform(layer):
            # Initialize convolution layer
            if type(layer) == nn.Conv2d:
                nn.init.normal_(layer.weight, mean=0, std=0.5)
            # Initialize FC layer
            elif type(layer) == nn.Linear:
                nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
                if layer.bias != None:
                    nn.init.constant_(layer.bias, 0.)
        return model.apply(weights_init_uniform)
