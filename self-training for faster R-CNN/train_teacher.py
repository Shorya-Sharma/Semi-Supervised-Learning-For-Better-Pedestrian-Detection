import math
import sys
import time
import torch
import transforms as T
import utils
from model import save_checkpoint, load_checkpoint
from dataloader import collate_fn, split_dataset
from torch.utils.data import DataLoader
import numpy as np


def train_teacher_model(model, labeled_dataset, optimizer, scheduler=None, train_ratio=0.7, batch_size=4, device='cpu', max_epochs=100, print_freq=10, save_path=None, checkpoint=None):
    model.to(device)
    metric_logger = utils.MetricLogger(delimiter=" ")
    last_loss = 1e9

    cur_epoch = 0
    if checkpoint is not None:
        print("loading checkpoint:" + checkpoint)
        model, optimizer, scheduler, cur_epoch = load_checkpoint(model, optimizer, scheduler, device, checkpoint)

    train_dataset, vld_dataset = split_dataset(labeled_dataset, train_ratio)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    vld_loader = DataLoader(vld_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    for epoch in range(cur_epoch, max_epochs):
        print("epoch {} / {}".format(epoch + 1, max_epochs))
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq)
        loss = evaluate(model, vld_loader, device, epoch, print_freq)
        
        if loss < last_loss and save_path != None:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, device, save_path)
            last_loss = loss
        if scheduler is not None:
            scheduler.step()


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())


        # loss in original paper
        # losses = loss_dict['loss_classifier'] + loss_dict['loss_box_reg']
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
        # break


def evaluate(model, data_loader, device, epoch, print_freq):  # test overfitting
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation'.format(epoch)
    sum_loss = []

    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, print_freq, header):
            # for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            # loss in origin paper
            # losses_reduced = loss_dict_reduced['loss_classifier'] + loss_dict_reduced['loss_box_reg']
            # losses = loss_dict['loss_classifier'] + loss_dict['loss_box_reg']
            if math.isfinite(losses.item()):
                sum_loss.append(losses.item())

            loss_value = losses_reduced.item()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

            if device == 'cuda':
                torch.cuda.empty_cache()
                del images
                del targets
                del losses_reduced
                del losses
                del loss_dict
                del loss_dict_reduced
            # break
    sum_loss = np.sum(sum_loss)
    return sum_loss
