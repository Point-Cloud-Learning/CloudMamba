import json
import os
import importlib

from Utils.Logger import get_root_logger
from Utils.Config import get_args, get_config, log_config_to_file
from Utils.Tool import create_experiment_dir, set_save_seed


def main():
    args = get_args()
    cfgs = get_config(args)

    cfgs.common.experiment_dir = os.path.join(cfgs.common.experiment_dir, cfgs.model.NAME, cfgs.dataset.NAME)
    create_experiment_dir(cfgs.common.experiment_dir)

    log_file = os.path.join(cfgs.common.experiment_dir, 'record.log')
    logger = get_root_logger(log_file=log_file, name=cfgs.common.log_name)

    seed_dir = os.path.join(cfgs.common.experiment_dir, 'seed.json')
    set_save_seed(seed_dir)

    log_config_to_file(cfgs, 'cfgs', logger=logger)

    runner = importlib.import_module(cfgs.model.MODULE)

    # run
    if args.train:
        getattr(runner, 'train_net')(cfgs)
    else:
        pass


if __name__ == '__main__':
    main()
