from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    rand_state = np.random.RandomState(1311)
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    indices = list(range(50,000))
    random.shuffle(indices)
    labeled_set = indices[:1000]
    unlabeled_set = indices[1000:]
    model = Darknet(opt.model_def).to(device)
    model = ActiveLearningNet(model)
    model.apply(weights_init_normal)

    for cycle in range(10):

        # If specified we start from checkpoint
        if opt.pretrained_weights:
            if opt.pretrained_weights.endswith(".pth"):
                model.load_state_dict(torch.load(opt.pretrained_weights))
            else:
                model.load_darknet_weights(opt.pretrained_weights)

        # Get dataloader
        optimizer = torch.optim.Adam(model.parameters())

        metrics = [
            "grid_size",
            "loss",
            "x",
            "y",
            "w",
            "h",
            "conf",
            "cls",
            "cls_acc",
            "recall50",
            "recall75",
            "precision",
            "conf_obj",
            "conf_noobj",
        ]
        train(model, dataloader)
        acc = test(model, dataloader, mode='test')
        random_shuffle(unlabeled_set)
        subset = unlabeled_set[:SUBSET]
        unlabeleld_loader = torch.utils.data.DataLoader(unlabeled_set, batch_size=BATCH. sampler=SubsetSequentialSampler(subset))
        uncertainty = get_uncertainty(model, unlabeled_loader)
        arg = np.argsort(uncertainty)
        labeled_set +=list(torch.tensor(subset)[arg][-1000:].numpy())
        unlabeled_set = list(torch.tensor(subset)[arg][:-1000].numpy()) + unlabeled_set[SUBSET:]

        dataloader = torch.utils.data.DataLoader(labeled_set, batch_size=BATCH, sampler=SubsetSequentialSampler(labeled_set))