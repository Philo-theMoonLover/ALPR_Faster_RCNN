# The code base on: https://www.kaggle.com/code/pdochannel/object-detection-fasterrcnn-tutorial/notebook

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torchvision
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
import copy
import math
from PIL import Image
import cv2
import albumentations as A  # our data augmentation library

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
from collections import defaultdict, deque
import datetime
import time
from tqdm import tqdm  # progress bar
from torchvision.utils import draw_bounding_boxes

from pycocotools.coco import COCO

from albumentations.pytorch import ToTensorV2

import sys

"""## 3. Dataset:
#### 3.1 Create transform function:
"""

def get_transforms(train=False):
    if train:
        transform = A.Compose([
            A.Resize(600, 600),  # our input size can be 600px
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            A.Resize(600, 600),  # our input size can be 600px
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    return transform


class LicensePlateDataset(datasets.VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, transforms=None):
        # the 3 transform parameters are reuqired for datasets.VisionDataset
        super().__init__(root, transforms, transform, target_transform)
        self.split = split  # train, valid, test
        self.coco = COCO(os.path.join(root, split, "_annotations.coco.json"))  # annotatiosn stored here
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, self.split, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        target = copy.deepcopy(self._load_target(id))

        boxes = [t['bbox'] + [t['category_id']] for t in target]  # required annotation format for albumentations
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)

        image = transformed['image']
        boxes = transformed['bboxes']

        new_boxes = []  # convert from xywh to xyxy
        for box in boxes:
            xmin = box[0]
            xmax = xmin + box[2]
            ymin = box[1]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(new_boxes, dtype=torch.float32)

        targ = {}  # here is our transformed target
        targ['boxes'] = boxes
        targ['labels'] = torch.tensor([t['category_id'] for t in target], dtype=torch.int64)
        targ['image_id'] = torch.tensor([t['image_id'] for t in target])
        targ['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # we have a different area
        targ['iscrowd'] = torch.tensor([t['iscrowd'] for t in target], dtype=torch.int64)
        return image.div(255), targ  # scale images

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    return tuple(zip(*batch))


"""### 4.3 Training:
The following is a function that will train the model for one epoch.
Torchvision Object Detections models have a loss function built in, and it will calculate the loss automatically if you pass in the `inputs` and `targets`
"""

def train_one_epoch(model, optimizer, loader, device, epoch):
    model.to(device)
    model.train()

    #     lr_scheduler = None
    #     if epoch == 0:
    #         warmup_factor = 1.0 / 1000 # do lr warmup
    #         warmup_iters = min(1000, len(loader) - 1)

    #         lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor = warmup_factor, total_iters=warmup_iters)

    all_losses = []
    all_losses_dict = []

    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)  # the model computes the loss automatically if we pass in targets
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()

        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)

        if not math.isfinite(loss_value):
            print("Loss is {loss_value}, stopping trainig")  # train if loss becomes infinity
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    #         if lr_scheduler is not None:
    #             lr_scheduler.step() #

    all_losses_dict = pd.DataFrame(all_losses_dict)  # for printing
    print(
        "Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
            epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
            all_losses_dict['loss_classifier'].mean(),
            all_losses_dict['loss_box_reg'].mean(),
            all_losses_dict['loss_rpn_box_reg'].mean(),
            all_losses_dict['loss_objectness'].mean()
        ))


if __name__ == "__main__":
    # License Plates.v3-original-license-plates.coco
    dataset_path = 'D:/ALPR_Collections/Data/Plate_Dataset/Dataset'

    # load classes
    coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
    categories = coco.cats
    n_classes = len(categories.keys())

    """This code just gets a list of classes"""
    classes = [i[1]['name'] for i in categories.items()]

    """## 4. Training:"""
    train_dataset = LicensePlateDataset(root=dataset_path, transforms=get_transforms(True))
    """This is a sample image and its bounding boxes, this code does not get the model's output"""

    # Lets view a sample
    sample = train_dataset[2]
    img_int = torch.tensor(sample[0] * 255, dtype=torch.uint8)
    plt.imshow(draw_bounding_boxes(
        img_int, sample[1]['boxes'], [classes[i] for i in sample[1]['labels']], width=4
    ).permute(1, 2, 0))

    """### 4.1 Chose Model:
    Our model is FasterRCNN with a backbone of `MobileNetV3-Large`
    """

    # lets load the faster rcnn model
    model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # we need to change the head
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)

    """This is our collating function for the train dataloader, it allows us to create batches of data that can be easily pass into the model"""

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)

    """The following blocks ensures that the model can take in the data and that it will not crash during training"""

    images, targets = next(iter(train_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images, targets)  # just make sure this runs without error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    """### 4.2 Optimizer:
    Here, we define the optimizer. If you wish, you can also define the LR Scheduler, but it is not necessary for this notebook since our dataset is so small.
    > Note, there are a few bugs with the current way `lr_scheduler` is implemented. If you wish to use the scheduler, you will have to fix those bugs
    """

    # Now, and optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True, weight_decay=1e-4)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1) # lr scheduler

    """10 Epochs should be enough to train this model for a high accuracy"""
    num_epochs = 100

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
    #     lr_scheduler.step()

    # our learning rate was too low, due to a lr scheduler bug. For this task, we wont need a scheudul.er

    """## Trying on sample Images
    This is the inference code for the model. First, we set the model to evaluation mode and clear the GPU Cache.
    We also load a test dataset, so that we can use fresh images that the model hasn't seen.
    """

    # we will watch first epoich to ensure no errrors
    # while it is training, lets write code to see the models predictions. lets try again
    model.eval()
    torch.cuda.empty_cache()

    test_dataset = LicensePlateDataset(root=dataset_path, split="test", transforms=get_transforms(False))
    img, _ = test_dataset[7]
    img_int = torch.tensor(img * 255, dtype=torch.uint8)
    with torch.no_grad():
        prediction = model([img.to(device)])
        pred = prediction[0]

    # it did learn

    fig = plt.figure(figsize=(14, 10))
    plt.imshow(draw_bounding_boxes(img_int,
                                   pred['boxes'][pred['scores'] > 0.8],
                                   [classes[i] for i in pred['labels'][pred['scores'] > 0.8].tolist()], width=4
                                   ).permute(1, 2, 0))

    torch.save(model, 'LP_model_9616images_100e.pth')
