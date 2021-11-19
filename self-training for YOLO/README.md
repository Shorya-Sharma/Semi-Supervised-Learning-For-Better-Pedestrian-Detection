# Transfer learning for predestrain detection in thermometer image
Transfer learning for predestrain detection in thermometer image based on PyTorch-YOLOv3 which pre-trained with coco dataset.

## Installation
##### Clone and install requirements
    $ git clone https://github.com/RanZhang-CR/PyTorch-YOLOv3.git
    $ cd PyTorch-YOLOv3/
    $ sudo pip3 install -r requirements.txt

##### Download pretrained weights
    $ cd weights/
    $ bash download_weights.sh

##### Download FLIR
    we just clone the dataset from https://www.flir.com/oem/adas/adas-dataset-agree/ to our own google drive and use it from colab
    

## Inference

    $ python3 detect.py --image_folder /content/gdrive/My\ Drive/FLIR/FLIR_ADAS_1_3/val/thermal_8_bit
    (the path should be your own data set path)
    (we just clone the dataset from https://www.flir.com/oem/adas/adas-dataset-agree/ to our own google drive and use it from colab)



## Train
```
$ train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--gradient_accumulations GRADIENT_ACCUMULATIONS]
                [--model_def MODEL_DEF] [--data_config DATA_CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--compute_map COMPUTE_MAP]
                [--multiscale_training MULTISCALE_TRAINING]
```

#### Example (FLIR)
To train on COCO using a Darknet-53 backend pretrained on ImageNet run: 
```
$ python3 train.py --data_config /content/PyTorch-YOLOv3/config/flir.data  --pretrained_weights weights/darknet53.conv.74
```
(if you wish to run your own dataset, please adjust config/coco.data or flir.data to point to your own dataset)
(we just clone the dataset from https://www.flir.com/oem/adas/adas-dataset-agree/ to our own google drive and use it from colab)



## Train on Custom Dataset

#### Custom model
Run the commands below to create a custom model definition, replacing `<num-classes>` with the number of classes in your dataset.

```
$ cd config/                                # Navigate to config dir
$ bash create_custom_model.sh <num-classes> # Will create custom model 'yolov3-custom.cfg'
```

#### Classes
Add class names to `data/custom/classes.names`. This file should have one row per class name.

#### Image Folder
Move the images of your dataset to `data/custom/images/`.

#### Annotation Folder
Move your annotations to `data/custom/labels/`. The dataloader expects that the annotation file corresponding to the image `data/custom/images/train.jpg` has the path `data/custom/labels/train.txt`. Each row in the annotation file should define one bounding box, using the syntax `label_idx x_center y_center width height`. The coordinates should be scaled `[0, 1]`, and the `label_idx` should be zero-indexed and correspond to the row number of the class name in `data/custom/classes.names`.

#### Define Train and Validation Sets
In `data/custom/train.txt` and `data/custom/valid.txt`, add paths to images that will be used as train and validation data respectively.

#### Train
To train on the custom dataset run:

```
$ python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data
```

Add `--pretrained_weights weights/darknet53.conv.74` to train using a backend pretrained on ImageNet.


#### To realize self-train
Run the detection on unlabeled dataset, set threshold by "--conf_thres",
it will then generate images with pseudo label.
Then put those images under data/custom/images, their coresponding label file under data/custom/labels
Then run the training process,
then run detection, then training, and so on.


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
