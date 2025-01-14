import logging
import time
from torch_geometric import loader as tg_loader
from Runners.BaseRunner import BaseRunner

log = logging.getLogger(__name__)

class Tester(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self.network.load_model(f'{self.save_dir}/models/best_model.pt', device=config.get('device'))

    def execute(self):
        log.info("Testing model...")
        start_time = time.time()
        data_loader = self.dataset.get_dataloader('test')
        output_save, truth_save, weight_save = self.apply_model(data_loader=data_loader, epoch=0, batch_type='test')
        log.info(f"Complete in {(time.time()-start_time)/60.:.2f} min.")
        self.end_of_epoch(output=output_save, truth_label=truth_save, weight=weight_save, test_epoch=True)

    def finish(self):
        log.info("Finished testing.")