import torch
import logging
import time
from torch_geometric import loader as tg_loader
from Runners.BaseRunner import BaseRunner

log = logging.getLogger(__name__)

class Trainer(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.get('learning_rate'))
        self.reduce_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=8,
                verbose=False, threshold=0.1, threshold_mode='rel',
                cooldown=0, min_lr=1e-8, eps=1e-8)

    def execute(self):
        for epoch in range(self.epoch_start, self.epoch_end):
            self.begin_of_epoch()
            # train one epoch
            log.info(f"Training with learning rate: {self.optimizer.param_groups[0]['lr']}")
            start_time = time.time()
            self.train_model(data_loader=self.dataset.get_dataloader('train'), epoch=epoch)
            log.info(f"Complete in {(time.time()-start_time)/60.:.2f} min.")
            
            # Validate one epoch
            log.info(f"Validating...")
            start_time = time.time()
            data_loader = self.dataset.get_dataloader('validation')
            output_save, truth_save, weight_save = self.apply_model(data_loader=data_loader, epoch=epoch, batch_type='validation')
            log.info(f"Complete in {(time.time()-start_time)/60.:.2f} min.")
            self.end_of_epoch(output=output_save, truth_label=truth_save, weight=weight_save, test_epoch=False)

        # Test model 
        # get best epoch
        epoch = self.best_epoch.epoch
        log.info(f"Testing model from epoch {epoch}...")
        start_time = time.time()
        self.network.load_model(f'{self.save_dir}/best_model.pt')
        data_loader = self.dataset.get_dataloader('test')
        output_save, truth_save, weight_save = self.apply_model(data_loader=data_loader, epoch=epoch, batch_type='test')
        self.end_of_epoch(output=output_save, truth_label=truth_save, weight=weight_save, test_epoch=True)

    def finish(self):
        log.info("Finished training.")