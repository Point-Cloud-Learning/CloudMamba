import sys

import numpy as np
import torch
from pointnet2_ops import pointnet2_utils

from torch import nn
from tqdm import tqdm

from Augmentations.Build_Augmentation import build_augmentation
from Datasets import build_dataloader
from Models.Build_Model import build_model
from Optimizer.Build_Optimizer import build_opti_sche
from Utils import Misc
from Utils.Logger import get_logger, print_log
from Utils.Misc import summary_parameters


def train_net(cfgs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = get_logger(cfgs.common.log_name)

    # build dataset
    train_dataloader, test_dataloader = build_dataloader(cfgs.dataset.train), build_dataloader(cfgs.dataset.test)

    # build model
    base_model = build_model(cfgs.model).to(device)
    summary_parameters(base_model, logger=logger)
    base_model.zero_grad()

    # build optimizer and scheduler
    optimizer, scheduler = build_opti_sche(base_model, cfgs.optimizer, cfgs.scheduler)

    # augmentation setting
    transforms = build_augmentation(cfgs.augmentation)

    # record setting
    start_epoch = 0
    best_acc = 0.0
    best_mAcc = 0.0
    best_epoch = 0

    npoints = cfgs.common.num_point

    for epoch in range(start_epoch, cfgs.common.epoch + 1):

        num_iter = 0
        loss_function = nn.CrossEntropyLoss()

        # set model to training mode
        base_model.train()
        optimizer.zero_grad()

        # accumulating loss, correct sample number, and total sample number
        accu_loss, accu_num, sample_num = 0.0, 0, 0

        train_dataloader = tqdm(train_dataloader, file=sys.stdout)
        for step, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            points, label = data[0].to(device), data[1].to(device)

            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points.size(1) < point_all:
                point_all = points.size(1)

            fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)
            fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()

            points = transforms(points)

            ret = base_model(points)
            sample_num += points.shape[0]

            ret_cls = torch.max(ret, dim=1)[1]
            accu_num += torch.eq(ret_cls, label).sum()

            loss = loss_function(ret, label.long())
            loss.backward()
            accu_loss += loss.detach()

            if not torch.isfinite(loss):
                print("WARNING: non-finite loss, ending training ", loss)
                sys.exit(1)

            train_dataloader.desc = "[train epoch {}] loss: {:.4f}, acc: {:.4f}".format(epoch, accu_loss / (step + 1), accu_num / float(sample_num))

            if num_iter == cfgs.common.step_per_update:
                if cfgs.common.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), cfgs.common.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

        train_loss, train_acc = accu_loss / (step + 1), accu_num / sample_num

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        if epoch % cfgs.common.val_freq == 0 and epoch != 0:
            # Validate the current model
            val_loss, val_acc, val_mACC = val_net(base_model, test_dataloader, epoch, cfgs, device, logger=logger)

            print_log("epoch %d train_loss: %.4f, train_acc: %.4f, val_loss: %.4f, val_acc: %.4f, val_mACC: %.4f, lr: %f" % (
                epoch, train_loss, train_acc, val_loss, val_acc, val_mACC, optimizer.param_groups[0]['lr']), logger=logger)

            if val_acc >= best_acc:
                best_acc = val_acc
                best_epoch = epoch
                print_log("Saving ModelNet40...", logger=logger)
                # Save checkpoints
                torch.save(base_model.state_dict(), cfgs.common.experiment_dir + "/best_model.pth")
            if val_mACC >= best_mAcc:
                best_mAcc = val_mACC

    print_log(f"best_acc: %.4f, best_mAcc: %.4f, best_epoch: %d" % (best_acc, best_mAcc, best_epoch), logger=logger)
    print_log("End of training...", logger=logger)


def val_net(base_model, test_dataloader, epoch, cfgs, device, logger=None):
    npoints = cfgs.common.num_point
    num_category = cfgs.model.cls_dim

    # set model to eval mode
    base_model.eval()
    loss_function = nn.CrossEntropyLoss()

    accu_loss, accu_num, sample_num = 0.0, 0, 0

    # accumulate the number of samples for each category
    num_per_class = torch.as_tensor([0 for _ in range(num_category)], dtype=torch.float32).to(device)
    # accumulate the number of correctly classified samples for each category
    right_num_per_class = torch.as_tensor([0 for _ in range(num_category)], dtype=torch.float32).to(device)

    test_dataloader = tqdm(test_dataloader, file=sys.stdout)
    with torch.no_grad():
        for step, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = Misc.fps(points, npoints)

            logits = base_model(points)
            sample_num += points.shape[0]

            ret_cls = torch.max(logits, dim=1)[1]
            accu_num += torch.eq(ret_cls, label).sum()

            num_per_class += torch.bincount(label, minlength=num_category)
            right_num_per_class += torch.bincount(label, weights=label == ret_cls, minlength=num_category)

            loss = loss_function(logits, label.long())
            accu_loss += loss

            test_dataloader.desc = "[valid epoch {}] loss: {:.4f}, acc: {:.4f}".format(epoch, accu_loss / (step + 1), accu_num / float(sample_num))

        mAcc = right_num_per_class / num_per_class
        right_rate_per_class = '  '.join([str(x) for x in torch.round(mAcc, decimals=4).cpu().numpy()])
        num_per_class = '  '.join([str(x) for x in torch.round(num_per_class, decimals=4).cpu().numpy()])
        right_num_per_class = '  '.join([str(x) for x in torch.round(right_num_per_class, decimals=4).cpu().numpy()])

        print_log(f"epoch {epoch}\n  每个类的参与总数：{num_per_class}\n  每个类正确分类数：{right_num_per_class}\n  每个类的正确比例：{right_rate_per_class}", logger=logger)

    return accu_loss / (step + 1), accu_num / float(sample_num), torch.round(torch.mean(mAcc), decimals=4).cpu().numpy()
