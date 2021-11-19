import math
import sys
import time
import torch
import transforms as T
import utils
from model import save_checkpoint, load_checkpoint
from dataloader import collate_fn, split_dataset, FLIRPseudoDataset, convert_subset
from torch.utils.data import DataLoader
import numpy as np
from train_teacher import evaluate


def self_training(model, labeled_dataset, unlabeled_dataset, optimizer, scheduler=None, batch_size=4, train_ratio=0.7, score_threshold=0.7, unlabeled_loss_weight=0.1, relabel_step=None,
                  device='cpu', max_epochs=100, print_freq=10, save_path=None, checkpoint=None):
    model.to(device)
    metric_logger = utils.MetricLogger(delimiter=" ")
    last_loss = 1e9

    cur_epoch = 0
    train_labeled_dataset, val_labeled_dataset = split_dataset(labeled_dataset, train_ratio)
    train_unlabeled_dataset, val_unlabeled_dataset = split_dataset(unlabeled_dataset, train_ratio)

    train_unlabeled_dataset = convert_subset(train_unlabeled_dataset)
    val_unlabeled_dataset = convert_subset(val_unlabeled_dataset)
 
    pseudo_train = FLIRPseudoDataset(model, train_unlabeled_dataset, batch_size=batch_size, device=device, score_threshold=score_threshold)
    pseudo_val = FLIRPseudoDataset(model, val_unlabeled_dataset, batch_size=batch_size, device=device, score_threshold=score_threshold)

    if checkpoint is not None:
        print("loading checkpoint:" + checkpoint)
        model, optimizer, scheduler, cur_epoch = load_checkpoint(model, optimizer, scheduler, device, checkpoint)

    for epoch in range(cur_epoch, max_epochs):
        print("epoch {} / {}".format(epoch + 1, max_epochs))
        train_loader = DataLoader(train_labeled_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
        train_one_epoch_self_training(model, optimizer, train_loader, 1, device, epoch, print_freq)
        train_loader = DataLoader(pseudo_train, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
        train_one_epoch_self_training(model, optimizer, train_loader, unlabeled_loss_weight, device, epoch, print_freq)

        vld_loader= DataLoader(val_labeled_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
        labeled_loss = evaluate(model, vld_loader, device, epoch, print_freq)

        vld_loader = DataLoader(pseudo_val, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
        unlabeled_loss = evaluate(model, vld_loader, device, epoch, print_freq)

        loss = labeled_loss + unlabeled_loss_weight * unlabeled_loss
        if save_path is not None:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, device, save_path)
            last_loss = loss
            print("save model, loss {}".format(loss))
        else:
            print("loss {}".format(loss))
            
        if scheduler is not None:
            scheduler.step()
        if relabel_step != None:
            if epoch % relabel_step == 0 and epoch != 0:
                pseudo_train = FLIRPseudoDataset(model, train_unlabeled_dataset, batch_size=batch_size, device=device, score_threshold=score_threshold)
                pseudo_val = FLIRPseudoDataset(model, val_unlabeled_dataset, batch_size=batch_size, device=device, score_threshold=score_threshold)
                

def train_one_epoch_self_training(model, optimizer, data_loader, weight, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values()) * weight
        # losses = loss_dict['loss_classifier'] + loss_dict['loss_box_reg']
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # losses_reduced = loss_dict_reduced['loss_classifier'] + loss_dict_reduced['loss_box_reg']

        loss_value = losses_reduced.item()

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if device == 'cuda':
            torch.cuda.empty_cache()
            del images
            del targets
            del losses_reduced
            del losses
            del loss_dict
            del loss_dict_reduced
