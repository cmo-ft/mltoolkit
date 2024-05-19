from importlib import import_module
from typing import Any
import torch
import torch.nn as nn


class NetworkInterface():
    """
    A class representing a network interface for machine learning models.

    Args:
        network_config (dict): Configuration parameters for the network.
        pre_model_path (str, optional): Path to a pre-trained model. Defaults to None.
    """

    def __init__(self, network_config, pre_model_path=None) -> None:
        self.network_config = network_config
        self.pre_model_path = pre_model_path
        self.load()

    def load(self):
        """
        Load the network structure and initialize the network.
        """
        # Load Network Structure
        self.network_name = self.network_config.get('Network')
        module_name = f'Networks.{self.network_name}'
        self.model = getattr(import_module(module_name), self.network_name)(self.network_config.get('model_setting'))

        # Initialize Network
        if self.pre_model_path is None:
            self.model = self.initialize_model(model=self.model)
        else:
            self.load_model(self.pre_model_path)

    def load_model(self, model_path):
        """
        Load a pre-trained model from the given path.

        Args:
            model_path (str): Path to the pre-trained model.
        """
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, model_path='./save/net.pt'):
        """
        Save the current model to the specified path.

        Args:
            model_path (str, optional): Path to save the model. Defaults to './save/net.pt'.
        """
        torch.save(self.model.state_dict(), model_path)

    def parameters(self):
        """
        Get the parameters of the model.

        Returns:
            iterator: Iterator over the model's parameters.
        """
        return self.model.parameters()
    
    def to(self, device):
        """
        Move the model to the specified device.

        Args:
            device (str or torch.device): Device to move the model to.
        """
        self.model.to(device)
    
    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.model.eval()

    def train(self):
        """
        Set the model to training mode.
        """
        self.model.train()

    def __call__(self, *args: Any, **kwds: Any):
        """
        Call the model with the given arguments.

        Returns:
            Any: Result of calling the model.
        """
        return self.model(*args, **kwds)

    @staticmethod
    def initialize_model(model):
        """
        Initialize the model's weights.

        Args:
            model (nn.Module): The model to initialize.

        Returns:
            nn.Module: The initialized model.
        """
        def weights_init_uniform(layer):
            # Initialize convolution layer
            if type(layer) == nn.Conv2d:
                nn.init.normal_(layer.weight, mean=0, std=0.5)
            # Initialize FC layer
            elif type(layer) == nn.Linear:
                nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.)
        return model.apply(weights_init_uniform)
