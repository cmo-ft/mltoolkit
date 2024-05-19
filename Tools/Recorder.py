import os
import time
import logging
import torch
import numpy as np
import pandas as pd
from Metrics.MetricInterface import MetricInterface
from Datasets.DatasetInterface import DatasetInterface
from Networks.NetworkInterface import NetworkInterface

log = logging.getLogger(__name__)

class Recorder:
    def __init__(self, config: dict, metric: MetricInterface, dataset: DatasetInterface, network: NetworkInterface):
        self.time_start = time.time()
        self.setup(config, metric, dataset, network)

    
    def setup(self, config: dict, metric: MetricInterface, dataset: DatasetInterface, network: NetworkInterface):
        self.config = config
        self.metric = metric
        self.dataset = dataset
        self.network = network
        self.save_dir = self.config.get("save_dir")
        self.record_path = self.save_dir + '/record.csv'
        os.makedirs(self.save_dir + '/models/', exist_ok=True)

        pre_trained = self.config.get("pre_trained")
        if pre_trained:
            self.network.load_model(pre_trained)
            self.record = pd.read_csv(self.record_path)
            self.epoch_start = self.record.get('epoch')[-1] + 1
        else:
            self.record = pd.DataFrame(columns=['epoch', 'batch_type', 'batch_id', 'batch_weight', 'learning_rate', 'loss'] + self.metric.get_metric_keys())
            self.epoch_start = 0
        
        self.num_epochs = self.config.get('num_epochs')
        self.epoch_end = self.epoch_start+self.num_epochs
        self.cur_epoch = self.epoch_start
        self.best_epoch = self.epoch_start
    

    def begin_of_epoch(self):
        log.info(f"Epoch: {self.cur_epoch}/{self.epoch_end-1}.")


    def end_of_batch(self, epoch: int, batch_type: str, batch_id: int, batch_weight: float, learning_rate: float, loss: float, metrics: dict):
        self.record = self.record.append({'epoch': epoch, 'batch_type': batch_type, 'batch_id': batch_id, 'batch_weight': batch_weight, 'loss': loss, **metrics}, ignore_index=True)


    def end_of_epoch(self, output: torch.Tensor, truth_label: torch.Tensor, weight: torch.Tensor, test_epoch=False):
        self.record.to_csv(self.record_path, index=None)
        self.network.save_model(f'{self.save_dir}/model{self.cur_epoch}.pt')
        arr_to_save = torch.cat([output, truth_label, weight], dim=1).numpy()
        np.save(f"{self.save_dir}/valset_output.npy",arr=arr_to_save)

        self.plot_loss()
        self.plot_score(output=output)

        cur_loss, cur_metric = self.metric(output, truth_label, weight)
        cur_loss = cur_loss.item()
        if not test_epoch:
            if (self.best_epoch==self.cur_epoch) or (cur_metric > self.best_metric):
                self.best_epoch = self.cur_epoch
                self.best_metric = cur_metric
                self.network.save_model(f'{self.save_dir}/best_model.pt')
            log.info(f'Best epoch: {self.best_epoch} with metrics: {self.best_metric.metrics}.')
            
        log.info(f"Complete epoch {self.cur_epoch} in {(time.time()-self.time_start)/60.:.2f} min.")
        log.info(f"Mean loss: {cur_loss:.4f}. Metrics: {cur_metric.metrics}")

        self.cur_epoch += 1