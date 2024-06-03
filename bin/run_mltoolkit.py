#!/lustre/collider/mocen/software/condaenv/hailing/bin/python
import os
import yaml
import random
import numpy as np
import torch
from importlib import import_module


def read_yaml(yaml_file):
    with open(yaml_file, "r") as _info:
        yaml_info = yaml.safe_load(_info)  # probably a list of yaml dictionaries
    return yaml_info


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results

    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MLToolkit", add_help=False)
    parser.add_argument("--config_yaml", "-c", default='./config.yaml', help="YAML file for configuration")
    parser.add_argument("--log-level", "-l", type = str, default = "INFO" )

    args = parser.parse_args()


    import logging
    loglevel = logging.getLevelName(args.log_level.upper())
    logging.basicConfig(level = loglevel, format = ">>> [%(levelname)s] %(module).s%(name)s: %(message)s")
    log = logging.getLogger("MLToolkit")

    config = read_yaml(args.config_yaml)
    config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(config.get('random_seed'))

    runner_type = config.get("runner_type")
    runner = None
    if runner_type=='Train':
        from Runners.Trainer import Trainer
        runner = Trainer(config)
    elif runner_type=='Test':
        from Runners.Tester import Tester
        runner = Tester(config)
    else:
        log.error(f'Unsupported runner type: {runner_type}')
        exit(0)
    
    runner.execute()
    runner.finish()
    log.info("Done")
    os._exit(0)
