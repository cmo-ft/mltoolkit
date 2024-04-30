import torch
import torchinfo
import logging
import time
import numpy as np
from torch_geometric import loader as tg_loader
import pandas as pd
from collections import namedtuple
from Runners.BaseRunner import BaseRunner
from Tools.Dataset import Dataset
from utils import utils

log = logging.getLogger(__name__)

class Trainer(BaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def load(self):
        super().load()
        torchinfo.summary(self.network_wrapper.model)

        # If pretrain, then load pretrained model
        self.pretrain = self.runner_config.get('pretrain')
        if self.pretrain:
            self.network_wrapper.load_model(self.runner_config.get('pre_model_path'))

        self.init_epoch_id = 0
        self.num_epochs = self.runner_config.get('num_epochs')
        self.lr = self.runner_config.get('learning_rate')
        self.optimizer = torch.optim.Adam(self.network_wrapper.parameters(), lr=self.lr)
        if self.pretrain:
            self.load_run_record(self.runner_config.get('pre_run_record'))
    
    def load_data(self):
        from collections import deque
        self.data_config = self.config.get('data_config')
        self.dataset_prefix = self.data_config.get('dataset_prefix')
        self.fold_id, self.fold_number = self.data_config.get('fold_id'), self.data_config.get('fold_number')

        # Get train / validation / test id
        idx = deque(range(self.fold_number))
        idx.rotate(self.fold_id)
        idx_dict = {
            # 'train': list(idx)[:-2],
            'train': list(idx)[-3],
            'validation': list(idx)[-2],
            'test': list(idx)[-1],
        }

        # Get data loader
        self.data_sets = {}
        for key, idx in idx_dict.items():
            # TODO: support a list of index
            self.data_sets[key] = Dataset(f"{self.dataset_prefix}{idx}.pt")

    def execute(self):
        self.BestEpoch = namedtuple('BestEpoch', ['epoch', 'loss'])
        self.best_epoch = self.BestEpoch(0, float('inf'))

        epoch_start, epoch_end = self.init_epoch_id, self.init_epoch_id+self.num_epochs
        for epoch in range(epoch_start, epoch_end):
            log.info(f"Epoch: {epoch}/{epoch_end-1}.")
            # train one epoch
            log.info(f"Training...")
            start_time = time.time()
            data_loader = tg_loader.DataLoader(self.data_sets.get('train'), batch_size=self.batch_size, shuffle=True)
            for batch_id, batch in enumerate(data_loader, 0):
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # Model output
                batch = batch.to(self.device)
                output = self.network_wrapper(batch)
                result = self.evaluator(output=output, truth_label=batch.y, weight=batch.weight)
                # backward pass: compute gradient of the loss with respect to model parameters
                result.loss.backward()
                self.optimizer.step()
                # record result
                self.do_record(epoch=epoch, batch_type='train', batch_id=batch_id, batch_weight=batch.weight.sum(), **result._asdict())
            log.info(f"Complete in {(time.time()-start_time)/60.:.2f} min.")
            
            # Validate one epoch
            log.info(f"Validating...")
            start_time = time.time()
            data_loader = tg_loader.DataLoader(self.data_sets.get('validation'), batch_size=self.batch_size, shuffle=False)
            output_save = self.apply_model(data_loader=data_loader, epoch=epoch, batch_type='validation')
            self.end_epoch(epoch=epoch, output=output_save)
            log.info(f"Complete in {(time.time()-start_time)/60.:.2f} min.")
            log.info(f"Total time: {(time.time()-self.time_start)/60.:.2f} min.")
            log.info('\n\n\n')

        # Test model 
        start_time = time.time()
        # get best epoch
        epoch = self.best_epoch.epoch
        log.info(f"Testing model in epoch {epoch}...")
        self.network_wrapper.load_model(f'{self.save_dir}/net_{epoch}.pt')
        data_loader = tg_loader.DataLoader(self.data_sets.get('test'), batch_size=self.batch_size, shuffle=False)
        self.apply_model(data_loader=data_loader, epoch=epoch, batch_type='test')
        # save score and record
        np.save(f"{self.save_dir}/testset_output.npy",arr=output_save.numpy())
        log.info(f"Complete in {(time.time()-start_time)/60.:.2f} min.")
        self.print_epoch_result(epoch=epoch)

    """
    Running at the end of each epoch
    """
    def end_epoch(self, epoch, output):
        output = output.numpy()
        np.save(f"{self.save_dir}/valset_output.npy",arr=output)
        self.save(record_path=f'{self.save_dir}/run_record.csv', model_path=f'{self.save_dir}/net_{epoch}.pt')
        self.plot_loss()
        self.plot_score(output=output)

        # Record best epoch
        cur_epoch_result = self.search_record(self.run_record, epoch=epoch, batch_type='validation')
        loss, weight = np.array(cur_epoch_result.get('loss')), np.array(cur_epoch_result.get('batch_weight'))
        cur_loss = (loss*weight).sum() / weight.sum()
        if cur_loss < self.best_epoch.loss:
            self.best_epoch = self.BestEpoch(epoch, cur_loss)
        self.print_epoch_result(epoch=epoch)
        log.info(f'Best epoch: {self.best_epoch.epoch} with loss {self.best_epoch.loss}')


    def save(self, record_path, model_path):
        record_df = pd.DataFrame(self.run_record)
        record_df.to_csv(record_path, index=None)
        self.network_wrapper.save_model(model_path)

    def finish(self):
        self.save(record_path=f'{self.save_dir}/run_record.csv', model_path=f'{self.save_dir}/net.pt')
    
    def plot_loss(self):
        result = pd.DataFrame(self.run_record)
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
    
    def plot_score(self, output):
        from scipy.special import softmax
        from sklearn.metrics import roc_curve
        weight = output[:,0]
        truth = output[:,1]
        score = softmax(output[:, 2:], axis=1)[:,1]
        signal_mask = (truth==1)

        # Plot scores
        plot = utils.PlotContainer(xlabel='Score', ylabel='A.U.', figname=f'{self.save_dir}/scores.pdf')
        bins = 50
        plot.ax.hist(score[signal_mask], bins=bins, weights=weight[signal_mask], label='Signal', color='red', histtype='step', linewidth=2)
        plot.ax.hist(score[~signal_mask], bins=bins, weights=weight[~signal_mask], label='Background', color='blue', histtype='step', linewidth=2)
        plot.apply_settings(if_legend=True)
        plot.savefig()

        # Plot ROC
        fpr, tpr, _ = roc_curve(truth, score, sample_weight=weight)
        roc_auc = np.trapz(tpr, fpr)
        log.info(f"auc: {roc_auc}.")
        plot = utils.PlotContainer(xlabel='False Positive Rate', ylabel='True Positive Rate', figname=f'{self.save_dir}/roc.pdf', xlim=[0,1], ylim=[0,1])
        plot.ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
        plot.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plot.ax.legend(loc='lower right')
        plot.apply_settings()
        plot.savefig()