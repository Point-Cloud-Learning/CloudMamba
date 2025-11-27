import os.path
import sys
import torch
import numpy as np

from tqdm import tqdm
from Datasets.Build_Dataset import build_dataloader
from Utils.Misc import summary_parameters
from Models.Build_Model import build_model
from Utils.Logger import get_logger, print_log
from Optimizer.Build_Optimizer import build_opti_sche
from Augmentations import build_augmentation


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
    best_mIou = 0.0
    best_acc = 0.0
    best_mAcc = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, cfgs.common.epoch):
        num_iter = 0
        loss_function = torch.nn.CrossEntropyLoss()

        # set model to training mode
        base_model.train()
        optimizer.zero_grad()

        # record parameters
        loss_sum, total_correct, total_seen = 0.0, 0, 0

        train_dataloader = tqdm(train_dataloader, file=sys.stdout)
        for step, (points, label) in enumerate(train_dataloader):
            num_iter += 1
            points, label = points.to(device), label.to(device)

            points = transforms(points)

            logits = base_model(points).contiguous().view(-1, cfgs.model.cls_dim)

            ret_cls = torch.max(logits, dim=1)[1]
            seglabel = label.view(-1, 1)[:, 0]

            total_seen += logits.shape[0]
            total_correct += torch.eq(ret_cls, seglabel).sum()

            loss = loss_function(logits, seglabel.long())
            loss.backward()
            loss_sum += loss.detach()

            if not torch.isfinite(loss):
                print("WARNING: non-finite loss, ending training ", loss)
                sys.exit(1)

            train_dataloader.desc = "[train epoch {}] loss: {:.4f}, acc: {:.4f}".format(epoch, loss_sum / (step + 1), total_correct / float(total_seen))

            if num_iter == cfgs.common.step_per_update:
                if cfgs.common.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), cfgs.common.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

        train_loss, train_acc = loss_sum / (step + 1), total_correct / float(total_seen)

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        if epoch % cfgs.common.val_freq == 0:
            # Validate the current model
            val_metrics = val_net(base_model, test_dataloader, epoch, cfgs, loss_function, device, logger=logger)

            lr = optimizer.param_groups[0]['lr']
            print_log("epoch %d  train_loss: %.4f, train_acc: %.4f, val_loss: %.4f, val_acc: %.4f, mIoU: %.4f, avg_class_acc: %.4f, lr: %f" % (
                epoch, train_loss, train_acc, val_metrics["val_loss"], val_metrics["val_acc"], val_metrics["mIoU"], val_metrics["avg_class_acc"], lr), logger=logger)

            if val_metrics["mIoU"] >= best_mIou:
                best_mIou = val_metrics["mIoU"]
                best_epoch = epoch
                print_log("Save S3DIS...", logger=logger)
                torch.save(base_model.state_dict(), cfgs.common.experiment_dir + "/best_model.pth")
            if val_metrics["val_acc"] >= best_acc:
                best_acc = val_metrics["val_acc"]
            if val_metrics["avg_class_acc"] >= best_mAcc:
                best_mAcc = val_metrics["avg_class_acc"]

    print_log("best_mIou: %.4f, best_acc: %.4f, best_mAcc: %.4f, best_epoch: %d" % (best_mIou, best_acc, best_mAcc, best_epoch), logger=logger)
    print_log("End of training...", logger=logger)


def val_net(base_model, test_dataloader, epoch, cfgs, loss_function, device, logger=None):
    # set model to eval mode
    base_model.eval()

    # record parameters
    loss_sum, total_correct, total_seen = 0.0, 0, 0

    # label parameters declaration
    category = [x.rstrip() for x in open(os.path.join(cfgs.dataset.test.DATA_PATH, 'class_names.txt'))]
    category2label = {cls: i for i, cls in enumerate(category)}
    label2category = {i: cls for cls, i in category2label.items()}
    category2color = {"ceiling": [0, 255, 0], "floor": [0, 0, 255], "wall": [0, 255, 255], "beam": [255, 255, 0], "column": [255, 0, 255],
                      "sofa": [200, 100, 100], "window": [100, 100, 255], "door": [200, 200, 100], "table": [170, 120, 200], "chair": [255, 0, 0],
                      "bookcase": [10, 200, 100], "board": [200, 200, 200], "clutter": [50, 50, 50]}

    val_metrics = {}
    total_seen_category = [0 for _ in range(cfgs.model.cls_dim)]
    total_correct_category = [0 for _ in range(cfgs.model.cls_dim)]
    total_iou_deno_category = [0 for _ in range(cfgs.model.cls_dim)]
    label_weights = np.zeros(cfgs.model.cls_dim)

    test_dataloader = tqdm(test_dataloader, file=sys.stdout)
    with torch.no_grad():
        for step, (points, label) in enumerate(test_dataloader):
            points, label = points.to(device), label.to(device)
            logits = base_model(points).contiguous()

            logits_cpu = np.array(logits.contiguous().cpu())
            seglabel_cpu = np.array(label.cpu())

            logits = logits.contiguous().view(-1, cfgs.model.cls_dim)
            seglabel = label.view(-1, 1)[:, 0]
            loss = loss_function(logits, seglabel.long())
            loss_sum += loss.detach()

            logits_cpu = np.argmax(logits_cpu, axis=2)
            total_correct += np.sum((logits_cpu == seglabel_cpu))
            total_seen += logits.shape[0]
            tmp, _ = np.histogram(seglabel_cpu, range(cfgs.model.cls_dim + 1))
            label_weights += tmp

            for j in range(cfgs.model.cls_dim):
                total_seen_category[j] += np.sum((seglabel_cpu == j))
                total_correct_category[j] += np.sum((logits_cpu == j) & (seglabel_cpu == j))
                total_iou_deno_category[j] += np.sum(((logits_cpu == j) | (seglabel_cpu == j)))

            test_dataloader.desc = "[valid epoch {}] loss: {:.4f}, acc: {:.4f}".format(epoch, loss_sum / (step + 1), total_correct / float(total_seen))

        label_weights = label_weights.astype(np.float64) / np.sum(label_weights.astype(np.float64))
        Iou = np.array(total_correct_category) / (np.array(total_iou_deno_category, dtype=np.float64) + 1e-6)
        class_acc = np.array(total_correct_category) / (np.array(total_seen_category, dtype=np.float64) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        for k in range(cfgs.model.cls_dim):
            iou_per_class_str += "class %s weight: %.4f, IoU: %.4f, acc: %.4f \n" % (
            label2category[k] + ' ' * (14 - len(label2category[k])), label_weights[k], Iou[k], class_acc[k])
        print_log(iou_per_class_str, logger=logger)

        val_metrics["val_loss"] = loss_sum / (step + 1)
        val_metrics["val_acc"] = total_correct / float(total_seen)
        val_metrics["mIoU"] = np.mean(Iou)
        val_metrics["avg_class_acc"] = np.mean(class_acc)
        return val_metrics
