from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        # preprocess the thermal image which is gray picture
        # initially just generally broadcast the 1 channel in gray to 3 channel in RGB
        # such that it can be applied on the YOLOv3 model which was trained on RGB COCO dataset
        fimgs = torch.cat((input_imgs, input_imgs, input_imgs), dim=1)
        # timgs = torch.cat((fimgs, fimgs, fimgs), dim=1)
        input_imgs = Variable(fimgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")

    # set a list to save all the person detection numbers
    # count the person numbers in each image
    # each element is a ["file name", count]
    person_count_list = []

    # record all the bbox for person in this image
    # [ [file name, [[x1,y1,box_w,box_h], [x1,y1,box_w,box_h]...], [[file name, [box][box]......] ......]
    all_person_bbox_list = []

    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # count the person number in this image
        # [[file name, count], [file name, count], ......[file name, count]]
        person_count = 0

        # record all the bbox for person in this image
        # [[x1,y1,box_w,box_h], [x1,y1,box_w,box_h] ......]
        person_bbox_list = []

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                # cls_pred == 0 represent person
                if int(cls_pred) == 0:

                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                    # print(int(cls_pred))

                    box_w = x2 - x1
                    box_h = y2 - y1

                    person_count += 1

                    person_bbox = [x1, y1, box_w, box_h]
                    person_bbox_list.append(person_bbox)

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

        this_image_count = [filename, person_count]
        person_count_list.append(this_image_count)

        if person_count > 0:
            person_bbox_list.insert(0, filename)
            all_person_bbox_list.append(person_bbox_list)

    # write the person count for each image into person_count.txt
    with open("person_count.txt", "w") as OutFile:
        for item in person_count_list:
            OutFile.write(item[0] + "\t" + str(item[1]) + "\n")
        OutFile.close()

    # write the person boxes for each image into person_boxes.txt
    with open("person_boxes.txt", "w") as OutFile:
        for item in all_person_bbox_list:
            OutFile.write(item[0] + "\t")
            for i in range(len(item) - 1):
                OutFile.write(str(item[i + 1][0]) + "\t"
                              + str(item[i+1][1]) +"\t"
                              +str(item[i+1][2])+"\t"
                              +str(item[i+1][3])+"\t")
            OutFile.write("\n")
        OutFile.close()
