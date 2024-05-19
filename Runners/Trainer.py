import torch
import torchinfo
import logging
import time
import numpy as np
from torch_geometric import loader as tg_loader
import pandas as pd
from collections import namedtuple
from Runners.BaseRunner import BaseRunner
from utils import utils

log = logging.getLogger(__name__)

class Trainer(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self.reduce_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=8,
                verbose=False, threshold=0.1, threshold_mode='rel',
                cooldown=0, min_lr=1e-8, eps=1e-8)

    def execute(self):
        self.BestEpoch = namedtuple('BestEpoch', ['epoch', 'loss'])
        self.best_epoch = self.BestEpoch(0, float('inf'))

        epoch_start, epoch_end = self.init_epoch_id, self.init_epoch_id+self.num_epochs
        for epoch in range(epoch_start, epoch_end):
            self.begin_of_epoch()
            # train one epoch
            log.info(f"Training with learning rate: {self.optimizer.param_groups[0]['lr']}")
            start_time = time.time()
            self.train_model(data_loader=self.dataset.get('train'), epoch=epoch)
            log.info(f"Complete in {(time.time()-start_time)/60.:.2f} min.")
            
            # Validate one epoch
            log.info(f"Validating...")
            start_time = time.time()
            data_loader = tg_loader.DataLoader(self.data_sets.get('validation'), batch_size=self.batch_size, shuffle=False)
            output_save, truth_save, weight_save = self.apply_model(data_loader=data_loader, epoch=epoch, batch_type='validation')
            log.info(f"Complete in {(time.time()-start_time)/60.:.2f} min.")
            self.end_of_epoch(output=output_save, truth_label=truth_save, weight=weight_save, test_epoch=False)

        # Test model 
        # get best epoch
        epoch = self.best_epoch.epoch
        log.info(f"Testing model from epoch {epoch}...")
        start_time = time.time()
        self.network.load_model(f'{self.save_dir}/model.pt')
        data_loader = tg_loader.DataLoader(self.data_sets.get('test'), batch_size=self.batch_size, shuffle=False)
        output_save, truth_save, weight_save = self.apply_model(data_loader=data_loader, epoch=epoch, batch_type='test')
        self.end_of_epoch(output=output_save, truth_label=truth_save, weight=weight_save, test_epoch=True)

    def finish(self):
        self.save(record_path=f'{self.save_dir}/run_record.csv', model_path=f'{self.save_dir}/model.pt')