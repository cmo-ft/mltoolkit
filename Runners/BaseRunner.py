from abc import ABC,abstractmethod
import os
import pandas as pd
import torch
import logging
import time
from Tools.Evaluator import Evaluator
from Tools.Network import NetworkWrapper

log = logging.getLogger(__name__)

class BaseRunner(ABC):
    def __init__(self, config):
        self.time_start = time.time()
        self.config = config
        self.load()
        os.makedirs(self.save_dir, exist_ok=True)

    @abstractmethod
    def load(self):
        self.runner_config = self.config.get('runner_config')
        self.batch_size = self.config.get('batch_size')
        self.device = self.config.get('device')
        self.save_dir = self.config.get("save_dir")
        self.load_model()
        self.load_evaluator()
        self.load_data()
        self.load_run_record()

    def load_model(self):
        self.network_wrapper = NetworkWrapper(self.config.get('network_config'), pre_model_path=None)
        self.network_wrapper.to(self.config.get('device'))

    def load_evaluator(self):
        self.evaluator = Evaluator(self.config.get('evaluator_config'))

    @abstractmethod
    def load_data(self):
        ...

    def load_run_record(self, run_record_path=None):
        if run_record_path:
            prev_record = pd.read_csv(run_record_path)
            self.run_record = prev_record.to_dict('list')
            self.init_epoch_id = self.run_record.get('epoch')[-1] + 1
        else:
            self.run_record = {
                'epoch': [],
                'batch_type': [], # train validation or test
                'batch_id': [],
                'batch_weight': [],
            }
            for key in self.evaluator.Critria._fields:
                self.run_record[key] = []

    def do_record(self, **kwds):
        for key, val in kwds.items():
            if type(val) == type(torch.tensor(0)):
                val = val.item()
            self.run_record[key].append(val)

    @staticmethod
    def search_record(record, epoch, batch_type):
        record = pd.DataFrame(record)
        return record.loc[(record['epoch']==epoch)&(record['batch_type']==batch_type)].to_dict('list')

    def print_epoch_result(self, epoch):
        epoch_result = pd.DataFrame(self.run_record)
        epoch_result = epoch_result.loc[epoch_result['epoch']==epoch]
        epoch_result = epoch_result.set_index('batch_type')
        for batch_type in epoch_result.index.unique():
            cur_result = epoch_result.loc[batch_type]
            cur_weight = cur_result['batch_weight']
            msg = f"Type: {batch_type}. Mean result: "
            for key in self.evaluator.Critria._fields:
                msg += f"{key}={(cur_result[key]*cur_weight).sum()/cur_weight.sum():.4f}\t"
            log.info(msg=msg)

    def apply_model(self, data_loader, epoch=0, batch_type='test'):
        output_save = []
        self.network_wrapper.eval()
        with torch.no_grad():
            for batch_id, batch in enumerate(data_loader, 0):
                # Model output
                batch = batch.to(self.device)
                truth_label=batch.y
                output = self.network_wrapper(batch)
                result = self.evaluator(output=output, truth_label=truth_label, weight=batch.weight)
                # record result
                self.do_record(epoch=epoch, batch_type=batch_type, batch_id=batch_id, batch_weight=batch.weight.sum(), **result._asdict())
                output_save.append(torch.cat([batch.weight.view(-1,1), truth_label.view(-1, 1), output ], 1).detach().cpu())
        return torch.cat(output_save).numpy()

    @abstractmethod
    def execute(self):
        ...

    @abstractmethod
    def finish(self):
        ...
