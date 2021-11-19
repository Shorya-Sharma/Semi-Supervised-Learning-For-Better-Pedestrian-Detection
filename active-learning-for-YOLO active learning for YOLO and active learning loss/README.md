# Active Learning for Pedestrian Detection in Thermal Image
Active learning for pedestrian detection in thermal images based on PyTorch-YOLOv3 which is pre-trained on coco dataset.

## Installation
##### Clone and install requirements
    $ git clone 
    $ sudo pip3 install -r requirements.txt

##### Download Pre-Trained Weights
    $ cd weights/
    $ bash download_weights.sh

##### Dataset
    For the dataset, we download the dataset from https://www.flir.com/oem/adas/adas-dataset-agree/ on the google drive and use it from google Colab
    
## Inference On YOLOv3
    $ python3 detect.py --image_folder /content/drive/MyDrive/FLIR/FLIR_ADAS_1_3/val/thermal_8_bit
    (the image_folder should be your own dataset path)

## Active Learning
Active Learning algorithms allow users to select a subset of images so that adding them to the training dataset will result in the highest improvement in the model's accuracy. With active learning, users can prioritize which images to be annotated for large datasets because annotating the entire dataset is time-consuming and cost ineffficient. 

In order to incorporate active learning with object detection, we are using "Learning loss for active learning" algorithm [Learning Loss for Active Learning"](https://arxiv.org/pdf/1905.03677.pdf).
 
The code runs for 10 cycles and selects 1000 images per cycle, then trains the model on those selected images. 

## Running The Code On FLIR Dataset On YOLOv3 With Active Learning
# For running with random selection.
    $ python3 main.py --data_config /content/PyTorch-YOLOv3/config/flir.data --pretrained_weights weights/darknet53.conv.74
# For running with loss prediction active learning.
    $ python3 main.py --use_active_learning=True --data_config /content/PyTorch-YOLOv3/config/flir.data --pretrained_weights weights/darknet53.conv.74


## Credit
### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
