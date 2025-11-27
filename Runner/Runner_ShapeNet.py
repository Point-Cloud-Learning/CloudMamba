import sys

import numpy as np
import torch
from tqdm import tqdm

from Datasets import build_dataloader
from Utils.Misc import summary_parameters
from Models.Build_Model import build_model
from Utils.Logger import get_logger, print_log
from Optimizer.Build_Optimizer import build_opti_sche
from Augmentations.Build_Augmentation import build_augmentation


def to_categorical(y, num_classes):
    return torch.eye(num_classes)[y.numpy(), ]


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
    best_epoch = 0.0
    best_class_avg_iou = 0.0
    best_instance_avg_iou = 0.0
    best_part_avg_accuracy = 0.0

    for epoch in range(start_epoch, cfgs.common.epoch):
        num_iter = 0
        loss_function = torch.nn.CrossEntropyLoss()

        # set model to training mode
        base_model.train()
        optimizer.zero_grad()

        # accumulating loss, accumulating correct label number, accumulating total label number
        accu_loss, accu_num, label_num = 0.0, 0, 0

        train_dataloader = tqdm(train_dataloader, file=sys.stdout)
        for step, (points, label, category) in enumerate(train_dataloader):
            num_iter += 1
            points, label = points.to(device), label.to(device)

            points = transforms(points)
            cat_prompt = to_categorical(category, cfgs.model.category).to(device)

            ret = base_model(points, cat_prompt).contiguous().view(-1, cfgs.model.cls_dim)

            ret_cls = torch.max(ret, dim=1)[1]
            label = label.view(-1, 1)[:, 0]

            label_num += ret.shape[0]
            accu_num += torch.eq(ret_cls, label).sum()

            loss = loss_function(ret, label.long())
            loss.backward()
            accu_loss += loss.detach()

            if not torch.isfinite(loss):
                print("WARNING: non-finite loss, ending training ", loss)
                sys.exit(1)

            train_dataloader.desc = "[train epoch {}] loss: {:.4f}, acc: {:.4f}".format(epoch, accu_loss / (step + 1), accu_num / float(label_num))

            if num_iter == cfgs.common.step_per_update:
                if cfgs.common.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), cfgs.common.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

        train_loss, train_acc = accu_loss / (step + 1), accu_num / label_num

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        if epoch % cfgs.common.val_freq == 0 and epoch != 0:
            # Validate the current model
            val_metrics = val_net(base_model, test_dataloader, epoch, cfgs, device, logger=logger)

            print_log(
                "epoch %d  train_loss: %.4f, train_acc: %.4f, val_loss: %.4f, val_acc: %.4f, instance_avg_mIOU: %.4f, class_avg_mIOU: %.4f, part_avg_accuracy: %.4f, lr: %f" % (
                    epoch, train_loss, train_acc, val_metrics["val_loss"], val_metrics["val_acc"], val_metrics["instance_avg_iou"], val_metrics["class_avg_iou"],
                    val_metrics["part_avg_accuracy"],
                    optimizer.param_groups[0]['lr']), logger=logger)

            if val_metrics["instance_avg_iou"] >= best_instance_avg_iou:
                best_instance_avg_iou = val_metrics["instance_avg_iou"]
                best_epoch = epoch
                print_log("Save ShapeNet...", logger=logger)
                torch.save(base_model.state_dict(), cfgs.common.experiment_dir + "/best_model.pth")
            if val_metrics["val_acc"] > best_acc:
                best_acc = val_metrics["val_acc"]
            if val_metrics["class_avg_iou"] > best_class_avg_iou:
                best_class_avg_iou = val_metrics["class_avg_iou"]
            if val_metrics["part_avg_accuracy"] > best_part_avg_accuracy:
                best_part_avg_accuracy = val_metrics["part_avg_accuracy"]

    print_log("best_instance_avg_iou: %.4f, best_acc: %.4f, best_class_avg_iou: %.4f, best_part_avg_accuracy: %.4f, best_epoch: %d" % (
        best_instance_avg_iou, best_acc, best_class_avg_iou, best_part_avg_accuracy, best_epoch), logger=logger)
    print_log("End of training...", logger=logger)


def val_net(base_model, test_dataloader, epoch, cfgs, device, logger=None):
    # set model to eval mode
    base_model.eval()
    loss_function = torch.nn.CrossEntropyLoss()

    # label parameters declaration
    par_label_to_cat = {}
    par_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                   'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
                   'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
                   'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
    for cat in par_classes.keys():
        for label in par_classes[cat]:
            par_label_to_cat[label] = cat

    accu_loss, accu_num, label_num = 0.0, 0, 0

    val_metrics = {}
    total_seen_part = [0 for _ in range(cfgs.model.cls_dim)]
    total_correct_part = [0 for _ in range(cfgs.model.cls_dim)]
    shape_ious = {cate: [] for cate in par_classes.keys()}

    test_dataloader = tqdm(test_dataloader, file=sys.stdout)
    with torch.no_grad():
        for step, (points, label, category) in enumerate(test_dataloader):
            points, label = points.to(device), label.to(device)
            cat_prompt = to_categorical(category, cfgs.model.category).to(device)

            logits = base_model(points, cat_prompt)

            # keep prediction results for each sample
            cur_pred_val_logit = np.array(logits.cpu())
            cur_pred_val = np.zeros((points.shape[0], cfgs.common.num_point)).astype(np.int64)

            # calculate loss on GPU
            logits = logits.contiguous().view(-1, cfgs.model.cls_dim)
            label = label.view(-1, 1)[:, 0]
            loss = loss_function(logits, label.long())
            accu_loss += loss.detach()

            # calculate classification results and keep the results to cur_pred_val
            label = label.view(points.shape[0], -1).cpu().numpy()
            for k in range(points.shape[0]):
                cate = par_label_to_cat[label[k, 0]]
                logit = cur_pred_val_logit[k, :, :]
                cur_pred_val[k, :] = np.argmax(logit[:, par_classes[cate]], 1) + par_classes[cate][0]

            # calculate various acc on CPU
            accu_num += np.sum(cur_pred_val == label)
            label_num += points.shape[0] * cfgs.common.num_point

            for j in range(cfgs.model.cls_dim):
                total_seen_part[j] += np.sum(label == j)
                total_correct_part[j] += np.sum((cur_pred_val == j) & (label == j))
                # total_iou_deno_part[j] += np.sum(((cur_pred_val == j) | (labels == j)))

            for k in range(points.shape[0]):
                segp = cur_pred_val[k, :]
                segl = label[k, :]
                cate = par_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(par_classes[cate]))]
                for n in par_classes[cate]:
                    if (np.sum(segl == n) == 0) and (np.sum(segp == n) == 0):  # part is not present, no prediction as well
                        part_ious[n - par_classes[cate][0]] = 1.0
                    else:
                        part_ious[n - par_classes[cate][0]] = np.sum((segl == n) & (segp == n)) / float(np.sum((segl == n) | (segp == n)))
                shape_ious[cate].append(np.mean(part_ious))

            test_dataloader.desc = "[valid epoch {}] loss: {:.4f}, acc: {:.4f}".format(epoch, accu_loss / (step + 1), accu_num / float(label_num))

        all_shape_ious = []
        for cate in shape_ious.keys():
            for iou in shape_ious[cate]:
                all_shape_ious.append(iou)
            shape_ious[cate] = np.mean(shape_ious[cate])

        for cate in sorted(shape_ious.keys()):
            print_log("eval mIoU of %s %.4f" % (cate + ' ' * (10 - len(cate)), shape_ious[cate]), logger=logger)

        mean_shape_ious = np.mean(list(shape_ious.values()))

        val_metrics["val_acc"] = accu_num / float(label_num)
        val_metrics["val_loss"] = accu_loss / (step + 1)

        # get firstly classification acc for each part and average the acc
        val_metrics["part_avg_accuracy"] = np.mean(np.array(total_correct_part) / np.array(total_seen_part, dtype=float))
        val_metrics["class_avg_iou"] = mean_shape_ious
        val_metrics["instance_avg_iou"] = np.mean(all_shape_ious)

        # val_acc: classification accuracy of all points, val_loss: classification loss of all points, class_avg_iou: average iou for all categories
        # part_avg_accuracy: average points classification accuracy for all parts, instance_avg_iou: average iou for all samples
        return val_metrics
