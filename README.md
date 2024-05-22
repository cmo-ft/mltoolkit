# MLToolkit

The MLToolkit is a comprehensive machine learning library for developing and deploying machine learning models for bbtautau tasks. It is designed to be flexible, scalable, and easy to use.

## Directory Structure

- `bin`: This directory contains executable scripts for running the framework.

- `Datasets`:  This directory contains scripts for data preprocessing, including details on graph construction, features within each graph, and weight definitions. To create a custom dataset, derive a new class from `Datasets/BaseDataset.py` and specify it in the `data_config/dataclass` section of config.yaml. For reference, consult `Datasets/HadHadDataset.py`.

- `Metrics`: This directory contains scripts for calculating and evaluating different performance metrics of the machine learning models. To introduce a custom metric, create a new class derived from `Metrics/BaseMetric.py` and specify it in the `metric_config/metric` section of config.yaml. Refer to Metrics/ClassificationMetric.py for guidance.

- `Networks`: This directory contains the different neural network architectures that can be used in the machine learning tasks. To employ a new network structure, place it in `Networks/` and specify it in the `network_config/Network` section of `config.yaml`.

- `Runners`: This directory contains scripts that manage model training, testing, and related processes such as model selection, hyperparameter tuning, and cross-validation.

- `Tools`: This directory contains utility scripts that are used across the project. 

## Usage

1. **Setup the environment**: Before running the toolkit, you need to set up the environment:

    ```bash
    source setup.sh
    ```

2. **Run the toolkit**: The `run_mltoolkit.py` script is the main entry point to the toolkit. It takes a configuration file as an argument, which specifies various parameters for the machine learning task, such as the dataset to use, the model to train, and the hyperparameters for training. The configuration file is in YAML format. To run the toolkit with a specific configuration file (e.g., `config.yaml`), use the following command:

    ```bash
    python run_mltoolkit.py -c config.yaml
    ```

    Replace `config.yaml` with the path to your own configuration file. The `-c` flag is used to specify the configuration file.
