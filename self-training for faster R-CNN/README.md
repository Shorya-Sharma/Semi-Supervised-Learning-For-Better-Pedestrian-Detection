# Transfer learning for pedestrian detection in thermal images

Introduction
--
The repository contains two baseline model for fine-tuning Mask RCNN and Faster RCNN.

Dependency
--
The system and pre-trained models are developed based on [Detectron2](https://github.com/facebookresearch/detectron2), which is an actively maintained open-source R-CNN system with lots of pre-trained models. It has a flexible training and evaluation configuration, giving users the flexibility to customize their own training loops and evaluators.

The scripts all include how to download the Detectron2.

Baseline models
--
1. project-fine-tuning-mask-rcnn.ipynb: 
  This is a fine-tuning script for Mask RCNN.

2. project-fine-tuning-faster-rcnn: 
  This is a fine-tuning script for Mask RCNN.

More descriptions of how to run are in the scripts.

Dataset and preparing
--
Download the FLIR thermal dataset: https://www.flir.com/oem/adas/adas-dataset-form/

The dataset has this structure:

├──FLIR_ADAS_1_3

|---├── train

|---|---├── Annotated_thermal_8_bit

|---|---|---├── images

|---|---├── RGB

|---|---|---├── images

|---|---├── thermal_8_bit

|---|---|---├── images

|---|---├── thermal_16_bit

|---|---|---├── images 

|---|---└── thermal_annotations.json

|---├── val

|---|---├── Annotated_thermal_8_bit

|---|---|---├── images

|---|---├── RGB

|---|---|---├── images

|---|---├── thermal_8_bit

|---|---|---├── images

|---|---├── thermal_16_bit

|---|---|---├── images

|---|---└── thermal_annotations.json

|---├── video

|---|---├── Annotated_thermal_8_bit

|---|---|---├── images

|---|---├── RGB

|---|---|---├── images

|---|---├── thermal_8_bit

|---|---|---├── images 

|---|---├── thermal_16_bit

|---|---|---├── images

|---|---└── thermal_annotations.json

|---└── ReadMe



We use the images in thermal_8_bit for training and testing. Run the train_vld_split() in the scripts will divide the train/thermal_annotations.json into train/train.json and train/vld.json. The test data and labels are saved in val/thermal_annotations.json. The train_vld_split() can run once (keep the the same splitting) or everytime (change the splitting everytime).

Before running the data, change the dataset_dir into the absolute path of FLIR_ADAS_1_3. That is, the folder "train" and "val" are saved in dataset_dir + "/train" and dataset_dir + "/val".

