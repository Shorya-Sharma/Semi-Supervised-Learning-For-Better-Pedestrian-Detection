from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import torch
import sys
import json
import numpy as np


if __name__ == "__main__":
    path_target_json = sys.argv[1]
    # local "/Users/ranzhang/Downloads/FLIR_ADAS_1_3/val/thermal_annotations.json"

    target_labels = [0 for _ in range(1366)]
    predict_labels = [0 for _ in range(1366)]

    with open(path_target_json,"r") as rj:
        X = rj.read()
        new_dict = json.loads(X)
        for item in new_dict['annotations']:
            if item['category_id'] == 1:
                # target_labels[item['image_id']] += 1
                target_labels[item['image_id']] = 1
        rj.close()

    with open("person_count.txt", "r") as rf:
        Y = rf.read().splitlines()
        for i in range(len(Y)):
            ele = Y[i].split("\t")
            predict_labels[i] = 1 if int(ele[1]) else 0
            # predict_labels[i] = int(ele[1])

    f1_s = f1_score(target_labels, predict_labels, average='binary')
    # f1_s = f1_score(target_labels, predict_labels, average='micro')
    print("f1 score")
    print(f1_s)

    acc_s = accuracy_score(target_labels, predict_labels)
    print("accuracy")
    print(acc_s)

    recall = recall_score(target_labels, predict_labels, average='binary')
    print("recall score")
    print(recall)

    tmp = confusion_matrix(target_labels, predict_labels, labels=[0,1])
    intersection = np.diag(tmp)
    ground_truth_set = tmp.sum(axis=1)
    predict_set = tmp.sum(axis=0)
    union = ground_truth_set + predict_set - intersection
    IoU = intersection/union.astype(np.float32)
    print(IoU)
    print(np.mean(IoU))