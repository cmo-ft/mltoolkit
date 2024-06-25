import os
import time
import logging
import torch
import numpy as np
import pandas as pd
import torchinfo
from utils import utils
from Metrics.MetricInterface import MetricInterface
from Metrics.BaseMetric import BaseMetric
from Networks.NetworkInterface import NetworkInterface

log = logging.getLogger(__name__)

class Recorder:
    def __init__(self, config: dict, metric: MetricInterface, network: NetworkInterface):
        self.time_start = time.time()
        self.setup(config, metric, network)

    
    def setup(self, config: dict, metric: MetricInterface, network: NetworkInterface):
        self.config = config
        self.metric = metric
        self.network = network
        self.device = self.config.get('device')
        self.save_dir = self.config.get("save_dir")
        self.record_path = self.save_dir + '/record.csv'
        os.makedirs(self.save_dir + '/models/', exist_ok=True)

        network_info = torchinfo.summary(self.network.model).__repr__()
        log.info(f"Network info: {network_info}")
        pre_trained = self.config.get("pre_trained")
        if pre_trained:
            self.network.load_model(self.config.get("pre_model_path"))
            self.record = pd.read_csv(self.record_path)
            self.epoch_start = self.record['epoch'].iloc[-1] + 1
        else:
            self.record = pd.DataFrame(columns=['epoch', 'batch_type', 'batch_id', 'batch_weight', 'learning_rate', 'loss'] + self.metric.get_metric_keys())
            self.epoch_start = 0
        
        self.network.to(self.device)
        self.num_epochs = self.config.get('num_epochs')
        self.epoch_end = self.epoch_start+self.num_epochs
        self.cur_epoch = self.epoch_start
        self.best_epoch = self.epoch_start
        self.epoch_start_time = time.time()
    

    def begin_of_epoch(self):
        log.info(f"Epoch: {self.cur_epoch}/{self.epoch_end-1}.")
        self.epoch_start_time = time.time()


    def end_of_batch(self, epoch: int, batch_type: str, batch_id: int, batch_weight: float, learning_rate: float, loss: float, metrics: BaseMetric):
        self.record = self.record.append({'epoch': epoch, 'batch_type': batch_type, 'batch_id': batch_id, 'batch_weight': batch_weight, 'learning_rate': learning_rate, 'loss': loss, **(metrics.metrics)}, ignore_index=True)


    def end_of_epoch(self, output: torch.Tensor, truth_label: torch.Tensor, weight: torch.Tensor, test_epoch=False):
        print(self.record.loc[self.record['epoch']==self.cur_epoch])
        self.record.to_csv(self.record_path, index=None)
        arr_to_save = torch.cat([output, truth_label.view(len(weight), -1), weight.view(-1,1)], dim=1).numpy()

        cur_loss, cur_metric = self.metric(output, truth_label, weight)
        cur_loss = cur_loss.item()
        if not test_epoch:
            self.network.save_model(f'{self.save_dir}/models/model{self.cur_epoch}.pt')
            np.save(f"{self.save_dir}/valset_result.npy",arr=arr_to_save)

            if (self.best_epoch==self.cur_epoch) or (cur_metric > self.best_metric):
                self.best_epoch = self.cur_epoch
                self.best_metric = cur_metric
                self.network.save_model(f'{self.save_dir}/models/best_model.pt')
            log.info(f'Best epoch: {self.best_epoch} with metrics: {self.best_metric.metrics}.')
        else:
            np.save(f"{self.save_dir}/testset_result.npy",arr=arr_to_save)

        log.info(f"Complete epoch {self.cur_epoch} in {(time.time()-self.epoch_start_time)/60.:.2f} min.")
        log.info(f"Total time: {(time.time()-self.time_start)/60.:.2f} min.")
        log.info(f"Mean loss: {cur_loss:.4f}. Metrics: {cur_metric.metrics}")

        self.plot_loss()
        self.cur_epoch += 1
        return cur_loss, cur_metric
    
    def plot_loss(self):
        import copy
        result = copy.copy(self.record)
        batch_types = result['batch_type'].unique()
        result['loss'] = result['loss'] * result['batch_weight']
        plot = utils.PlotContainer(xlabel='Epoch', ylabel='Loss',figname=f'{self.save_dir}/loss_to_epoch.pdf')
        result = result.groupby(['batch_type', 'epoch']).sum()
        result['loss'] = result['loss'] / result['batch_weight']

        for btype in batch_types:
            cur_result = result.loc[btype]
            plot.ax.plot(cur_result.index, cur_result['loss'], linewidth=2, label=f"{btype} loss", marker='o')
        plot.apply_settings(if_legend=True)
        plot.savefig()