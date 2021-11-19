# 11785-project-transfer-learning

This is the repo for the 11785 Fall 2020 final project: Semi-Supervised Learning For Better Pedestrian Detection


# Organization of the repo

We use git submodule to organize our work. The folders include:

11785-project: self-training for faster R-CNN (by Wanwen)

11785-project-jianchang: active learning for faster R-CNN (by Jianchang)

11785-project-active-learning-for-YOLO: active learning for YOLO and active learning loss (by Seo Young)

11785-project-ranzhang-yolo: self-training for YOLO (by Ran)

# Dataset
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

We use the images in thermal_8_bit for training and testing.
