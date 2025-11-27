import torch
from timm.scheduler import CosineLRScheduler
from torch import optim

from Utils.Misc import build_lambda_sche, build_lambda_bnsche


def build_opti_sche(base_model, cfgs_optimizer, cfgs_scheduler):
    # optimizer
    if cfgs_optimizer.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    # print(name)
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]

        param_groups = add_weight_decay(base_model, weight_decay=cfgs_optimizer.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **cfgs_optimizer.kwargs)
    elif cfgs_optimizer.type == 'Adam':
        optimizer = optim.Adam(base_model.parameters(), **cfgs_optimizer.kwargs)
    elif cfgs_optimizer.type == 'SGD':
        optimizer = optim.SGD(base_model.parameters(), nesterov=True, **cfgs_optimizer.kwargs)
    else:
        raise NotImplementedError()

    # scheduler
    if cfgs_scheduler.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, cfgs_scheduler.kwargs)  # misc.py
    elif cfgs_scheduler.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=cfgs_scheduler.kwargs.epochs,
                                      lr_min=cfgs_scheduler.kwargs.lr_min,
                                      warmup_lr_init=cfgs_scheduler.kwargs.lr_min,
                                      warmup_t=cfgs_scheduler.kwargs.initial_epochs,
                                      cycle_limit=1,
                                      t_in_epochs=True)
    elif cfgs_scheduler.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **cfgs_scheduler.kwargs)
    elif cfgs_scheduler.type == 'function':
        scheduler = None
    else:
        raise NotImplementedError()

    if cfgs_optimizer.get('bnmscheduler') is not None:
        bnsche_config = cfgs_optimizer.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        scheduler = [scheduler, bnscheduler]

    return optimizer, scheduler
