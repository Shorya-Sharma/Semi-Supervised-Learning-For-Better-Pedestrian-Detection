import os, json, cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from torch.utils.data import Subset


def read_image(image_path):
    """
    read grayscale image and normalize it
    """
    # img = Image.open(img_path).convert("RGB")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255.
    img = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)
    return img
    

class FLIRDataset(Dataset):
    """
    FLIR Dataset initalization (with label)
    """

    def __init__(self, root, json_file_name, transforms=None):
        """
        root: root of folder (save the json file)
        json_file_name: json file which stores data and annotations
        """
        self.root = root
        print(root)
        json_file = os.path.join(root, json_file_name)
        print(json_file)
        with open(json_file) as f:
            json_data = json.load(f)
        imgs = json_data["images"]
        self.categories = json_data["categories"]
        self.info = json_data["info"]
        self.license = json_data["licenses"]

        # filter out not pedestrians annotation
        self.annotations = list(filter(lambda x:x["category_id"]==1, json_data["annotations"]))

        self.imgs = []
        all_labels = {}
        self.targets = []

        # filter out negative instances without pedestrians
        for i, img in enumerate(imgs):
            annotation = list(filter(lambda x:x["image_id"] == img["id"], self.annotations))
            if len(annotation) > 0:
                self.imgs.append(img)
                all_labels[img["id"]] = annotation
        
        for i, img in enumerate(self.imgs):
            img_id = img["id"]
            img_annotations = all_labels[img_id]
            num_objs = len(img_annotations)
            boxes = []
            area = []
            for i, anno in enumerate(img_annotations):
                bbox = anno["bbox"]
                boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                area.append(anno["area"])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            area = torch.as_tensor(area, dtype=torch.float32)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = torch.tensor([img_id])
            target["area"] = area
            target["iscrowd"] = iscrowd
            self.targets.append(target)
        
        self.transforms = transforms

      
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load images ad masks
        img = self.imgs[idx]
        target= self.targets[idx]
        img_id = img["id"]
        img_filename = img["file_name"]
        img_path = os.path.join(self.root, img_filename)
        img = read_image(img_path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def pseudo_labeling():
        pass

    
class FLIRPseudoDataset(Dataset):
    """
    dataset that contains pseudo labeling
    """
    def __init__(self, model, sub_dataset, batch_size, device, score_threshold, transforms=None):
        model.eval()
        model = model.to(device)
        dataset = sub_dataset.dataset
        indices = sub_dataset.indices

        self.root = dataset.root
        self.imgs = []
        labels = {}
        self.targets = []

        dataloader = DataLoader(sub_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        idx = 0
        with torch.no_grad():
            for i, (images, targets) in enumerate(dataloader):
                length = len(images)
                images = [img.to(device) for img in images]
                preds = model(images)
                # preds: a list of prediction
                # {boxes: tensor [N x 1], labels: tensor [N x 1], scores: tensor [N x 1]}
                for prediction in preds:
                    img = dataset.imgs[indices[idx]]
                    img_id = img["id"]
                    img_filename = img["file_name"]
                    annotations = []
                    index = prediction["scores"] > score_threshold
                    if sum(index.type(torch.int)) > 0:
                        target = {}
                        boxes = prediction["boxes"][index, :]
                        valid_box_index = boxes[:, 3] > boxes[:, 1]
                        boxes = boxes[valid_box_index, :]
                        valid_box_index = boxes[:, 2] > boxes[:, 0]
                        boxes = boxes[valid_box_index, :]
                        target["boxes"] = boxes.to('cpu').detach()
                        target["labels"] = prediction["labels"][index].to('cpu').detach()
                        target["image_id"] = torch.tensor([img_id])
                        target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                        target["iscrowd"] = torch.zeros((len(index),), dtype=torch.int64)
                        self.imgs.append(img)
                        self.targets.append(target)

                    idx += 1
                del images
                del preds
                torch.cuda.empty_cache()
        
        self.transforms = transforms

      
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load images ad masks
        img = self.imgs[idx]
        img_id = img["id"]
        img_filename = img["file_name"]
        img_path = os.path.join(self.root, img_filename)
        img = read_image(img_path)
        target = self.targets[idx]

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    

def split_dataset(dataset, ratio, seed=1):
    """
    randomly split one dataset to two dataset with ratio / 1 - ratio data
    """
    torch.manual_seed(seed)
    data_len = dataset.__len__()
    num_part = int(np.ceil(data_len * ratio))
    dataset_1, dataset_2 = random_split(dataset, [num_part, data_len - num_part])
    return dataset_1, dataset_2

def split_training_dataset(dataset, labeled_ratio, training_ratio):
    """
    split the dataset in train and vaidation set
    """
    labeled_dataset, unlabeled_dataset = split_dataset(dataset, labeled_ratio)
    labeled_train_dataset, labeled_val_dataset = split_dataset(labeled_dataset, training_ratio)
    unlabeled_train_dataset, unlabeled_val_dataset = split_dataset(unlabeled_dataset, training_ratio)
    return labeled_train_dataset, labeled_val_dataset, unlabeled_train_dataset, unlabeled_val_dataset
        
    
def get_dataloader(dataset, batch_size, is_train=False, labeled_ratio=1, training_ratio=0.7):
    if is_train:
        # split into labeled and unlabeled
        labeled_dataset, unlabeled_dataset = split_dataset(dataset, labeled_ratio)
        
        # split into train and validation
        labeled_train_dataset, labeled_val_dataset = split_dataset(labeled_dataset, training_ratio)
        unlabeled_train_dataset, unlabeled_val_dataset = split_dataset(unlabeled_dataset, training_ratio)

        # create dataloader
        labeled_train_dataloader = DataLoader(labeled_train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        unlabeled_train_dataloader = DataLoader(unlabeled_train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        labeled_val_dataloader = DataLoader(labeled_val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        unlabeled_val_dataloader = DataLoader(unlabeled_val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        return labeled_train_dataloader, unlabeled_train_dataloader, labeled_val_dataloader, unlabeled_val_dataloader
    else:
        dataloader = DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=False)
        return dataloader

def collate_fn(batch):
    """
    collate function
    """
    return tuple(zip(*batch))


def check_dataloading(dataset):
    """
    visualize the image and annotation
    """
    image, target = dataset.__getitem__(random.randint(0, dataset.__len__() - 1))
    image = image.detach().numpy()
    image = image[0, :, :]

    fig, ax = plt.subplots(1)
    box = target["boxes"].detach().numpy()
    ax.imshow(image, cmap='gray')
    for i in range(box.shape[0]):
      rect = patches.Rectangle((box[i, 0], box[i, 1]), box[i, 2] - box[i, 0], box[i, 3] - box[i, 1], linewidth=1, edgecolor='r', facecolor='none')
      ax.add_patch(rect)

    plt.show()

    
def convert_subset(sub_dataset):
    """
    convert a subset of a subset to a subset of normal dataset
    """
    dataset = sub_dataset.dataset.dataset
    indices = sub_dataset.indices
    upper_indices = sub_dataset.dataset.indices
    new_indices = [upper_indices[i] for i in indices]
    new_subset = Subset(dataset=dataset, indices=new_indices)
    return new_subset
