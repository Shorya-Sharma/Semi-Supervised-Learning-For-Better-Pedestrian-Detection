# Semi-Supervised Learning For Better Pedestrian

This repository contains the code and results for the project titled **Semi-Supervised Learning For Better Pedestrian** that I accomplished during my research internship at IIT Roorkee under the guidance of Dr. Partha Pratim Roy


# Organization of the repo

We use git submodule to organize our work. The folders include:

self-training for faster R-CNN 

active learning for faster R-CNN 

active-learning-for-YOLO: active learning for YOLO and active learning loss

self-training for YOLO 

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
