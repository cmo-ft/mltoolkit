# MLToolkit

A toolkit for Graph Neural Network (GNN) training.


## Getting started

- [GraphGenerator](GraphGenerator/): scripts to generate datasets (graphs) for GNN training. 
  - [configs.py](GraphGenerator/configs.py): design nodes, edges and features of your own graphs.

  - [merge_data.py](GraphGenerator/merge_data.py) merge all graphs and split them into `${nfold}` folds: `dataset0.pt, dataset1.pt ... dataset${nfold-1}.pt`. 

- [config](config/): configurations for training. 

- [bin](bin/): executables. [run_mltoolkit.py](bin/run_mltoolkit.py) is the startinng script for GNN training. [compare_ml_bdt.py](bin/compare_ml_bdt.py) is used to compare GNN and BDT results. Can be executed by `python ${MLTOOLKIT_PATH}/bin/run_mltoolkit.py -c your_config.yaml`
