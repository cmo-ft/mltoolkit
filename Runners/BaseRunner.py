from abc import ABC,abstractmethod
import os
import pandas as pd
import torch
import logging
import time
from Tools.Recorder import Recorder
from Metrics.MetricInterface import MetricInterface
from Datasets.DatasetInterface import DatasetInterface
from Networks.NetworkInterface import NetworkInterface

log = logging.getLogger(__name__)

class BaseRunner(ABC, Recorder):
    def __init__(self, config):
        self.config = config
        self.load()

    def load(self):
        self.dataset = DatasetInterface(self.config.get('data_config'))
        metric = MetricInterface(self.config.get('metric_config'))
        network = NetworkInterface(self.config.get('network_config'))
        super(Recorder, self).__init__(self.config, metric, network)

    def apply_model(self, data_loader, epoch=0, batch_type='test'):
        output_save, truth_label_save, weight_save = [], [], []
        output_save = []
        self.network.eval()
        with torch.no_grad():
            for batch_id, batch in enumerate(data_loader, 0):
                # Model output
                batch = batch.to(self.device)
                truth_label=batch.y
                output = self.network(batch)
                loss, metrics = self.metric(output=output, truth_label=truth_label, weight=batch.weight_train)
                self.end_of_batch(epoch=epoch, batch_type=batch_type, batch_id=batch_id, batch_weight=batch.weight_train.sum(), learning_rate=0, loss=loss.item(), metrics=metrics)

                output_save.append(output.detach().cpu())
                truth_label_save.append(truth_label.detach().cpu())
                weight_save.append(batch.weight.detach().cpu())
        return torch.cat(output_save), torch.cat(truth_label_save), torch.cat(weight_save)
    
    def train_model(self, data_loader, epoch):
        if not hasattr(self, "optimizer"):
            log.info("No optimizer found, create a new one.")
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.get('learning_rate'))

        self.network.train()
        for batch_id, batch in enumerate(data_loader, 0):
            # clear the gradients of all optimized variables
            self.optimizer.zero_grad()
            # Model output
            batch = batch.to(self.device)
            output = self.network(batch)
            truth_label=batch.y
            loss, metrics = self.metric(output=output, truth_label=truth_label, weight=batch.weight_train)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            self.optimizer.step()
            # record result
            self.end_of_batch(epoch=epoch, batch_type='train', batch_id=batch_id, batch_weight=batch.weight.sum(), learning_rate=self.optimizer.param_groups[0]['lr'], loss=loss.item(), metrics=metrics)

    @abstractmethod
    def execute(self):
        ...

    @abstractmethod
    def finish(self):
        ...
