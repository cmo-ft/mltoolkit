#!/lustre/collider/mocen/software/condaenv/hailing/bin/python
import os
import yaml
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_curve
import torch
from utils.utils import RatioPlotContainer


def read_yaml(yaml_file):
    with open(yaml_file, "r") as _info:
        yaml_info = yaml.safe_load(_info)  # probably a list of yaml dictionaries
    return yaml_info

def get_testset_id(fold_id, num_folds):
    from collections import deque
    # Get train / validation / test id
    idx = deque(range(num_folds))
    idx.rotate(fold_id)
    return idx[-1]


class Analyzer():
    def __init__(self, graph_path, ml_result_path, signal_sf):
        self.weight, self.truth, self.ml_score, self.bdt_score = None, None, None, None
        self.signal_sf = signal_sf
        self.append_results(graph_path, ml_result_path, signal_sf)

    def append_results(self, graph_path, ml_result_path, signal_sf):
        if signal_sf != self.signal_sf:
            log.warn(f'Signal scale factor not consistent. Old: {self.signal_sf}, new: {signal_sf}')
        # retrive graph and results
        graphs = torch.load(graph_path)
        ml_result = np.load(ml_result_path)

        # get score, label and weight
        cur_weight, cur_truth = graphs.sample_weight.numpy(), graphs.y.numpy()
        cur_ml_score = softmax(ml_result[:,2:], axis=1)[:,1]
        cur_bdt_score = graphs.bdtScore.numpy()

        # scale up signal weights
        cur_weight[cur_truth==1] *= signal_sf

        if self.weight is None:
            self.weight, self.truth, self.ml_score, self.bdt_score = cur_weight, cur_truth, cur_ml_score, cur_bdt_score 
        else:
            self.weight = np.concatenate([self.weight, cur_weight])
            self.truth = np.concatenate([self.truth, cur_truth])
            self.ml_score = np.concatenate([self.ml_score, cur_ml_score])
            self.bdt_score = np.concatenate([self.bdt_score, cur_bdt_score])
        
    def analyze(self, save_ratio_plot='./compare_score.pdf'):
        # rescale ml score to be [-1, 1]
        self.ml_score = (2 * self.ml_score - self.ml_score.max() - self.ml_score.min()) / (self.ml_score.max() - self.ml_score.min())

        # plot settings
        bins = np.linspace(-1, 1, 51)
        myplot = RatioPlotContainer(bins=bins, xlabel='score', xlim=[-1,1], ylabel='yields', figname=save_ratio_plot)
        x_values = (bins[1:] + bins[:-1])/2

        # load scores
        auc_bdt_gnn = [0, 0]
        signal_mask = (self.truth==1)
        for i, name, score, linestyle in zip(
                            list(range(2)),
                            ['BDT', 'GNN'], 
                            [self.bdt_score, self.ml_score],
                            ['dashed', None]
                            ):
            # get auc
            fpr, tpr, _ = roc_curve(self.truth, score, sample_weight=self.weight)
            auc_bdt_gnn[i] = np.trapz(tpr, fpr)
            # plot signal
            s, w = score[signal_mask], self.weight[signal_mask]
            hist, bins, _ = myplot.ax.hist(s, bins=bins, weights=w, label=fr'{name} Signal$\times$ {self.signal_sf:.0f}',  histtype='step', linewidth=2, color='red', linestyle=linestyle)
            myplot.insert_data(x_values=x_values, y_value=hist, ary_index=i, label='Signal', color='red')

            # plot background
            s, w = score[~signal_mask], self.weight[~signal_mask]
            hist, bins, _ = myplot.ax.hist(s, bins=bins, weights=w, label=f'{name} Background',  histtype='step', linewidth=2, color='blue', linestyle=linestyle)
            myplot.insert_data(x_values=x_values, y_value=hist, ary_index=i, label='Background', color='blue')

    
        myplot.draw_ratio(draw_error=False, ratio_ylabel='GNN/BDT')
        myplot.apply_settings()
        myplot.ax.legend(title=f"BDT AUC: {auc_bdt_gnn[0]:.3f}\nGNN AUC: {auc_bdt_gnn[1]:.3f}", 
                loc='upper center', fontsize=9, title_fontsize=9)
        myplot.ax_ratio.set_ylim([0, 5])
        myplot.savefig()


def compare_single_fold(config):
    # get graph path
    testset_id = get_testset_id(config.get('data_config').get('fold_id'), config.get('data_config').get('num_folds'))
    graph_path = config.get('data_config').get('dataset_prefix') + f"{testset_id}.pt"

    # get ml result path
    ml_result_path = config.get('save_dir') + '/testset_output.npy'
    analyzer = Analyzer(graph_path=graph_path, ml_result_path=ml_result_path, signal_sf=float(config.get('data_config').get('signal_scale_factor')))

    # do analyze
    analyzer.analyze(save_ratio_plot=config.get('save_dir') + "/compare_score.pdf")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compre ML and BDT score", add_help=False)
    parser.add_argument("--config_yaml", "-c", default='./config.yaml', help="YAML file for configuration")
    parser.add_argument("--log-level", "-l", type = str, default = "INFO" )

    args = parser.parse_args()

    import logging
    loglevel = logging.getLevelName(args.log_level.upper())
    logging.basicConfig(level = loglevel, format = ">>> [%(levelname)s] %(module).s%(name)s: %(message)s")
    log = logging.getLogger("Compare ML and BDT")

    config = read_yaml(args.config_yaml)

    compare_single_fold(config=config)

    log.info("Done")
    os._exit(0)