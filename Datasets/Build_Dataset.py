from Utils.Registry import Registry
from Utils.Misc import worker_init_fn
from torch.utils.data import DataLoader

datasets = Registry("Datasets")


def build_dataset(cfgs_dataset_type, default_args=None):
    """
        Build a dataset, defined by `dataset_name`.
        Args:
            cfgs_dataset_type :
            default_args :
        Returns:
            Dataset: a constructed dataset specified by dataset_name.
    """
    return datasets.build(cfgs_dataset_type, default_args=default_args)


def build_dataloader(cfgs_dataset_type):
    return DataLoader(build_dataset(cfgs_dataset_type),
                      batch_size=cfgs_dataset_type.batch_size,
                      shuffle=cfgs_dataset_type.mode == "train",
                      drop_last=cfgs_dataset_type.mode == "train",
                      num_workers=int(cfgs_dataset_type.num_workers),
                      worker_init_fn=worker_init_fn,
                      pin_memory=True)
