import pathlib
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from utils import FSDataMixin, ModelMixin
from torch.optim import AdamW
import torchvision
import clip


FEW_SHOT_DATA = pathlib.Path("data/coco_crops_few_shot")
MODELS = pathlib.Path("models")
DATA_MEANS, DATA_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] #imagenet numbers
# DATA_MEANS, DATA_STD = [0.48145466, 0.4578275, 0.40821073], [0.229, 0.224, 0.225] # clip numbers


train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8,1.0), ratio=(0.9,1.1)),
                # transforms.RandomHorizontalFlip(0.3),
                transforms.ToTensor(),
                transforms.Normalize(DATA_MEANS, DATA_STD),
                ])
clip.available_models()
_, preprocess = clip.load("ViT-B/16")

train_dataset = ImageFolder(FEW_SHOT_DATA/"train", transform = preprocess)
trf = transforms.ToPILImage()
NUM_IMAGES = 4
images = [train_dataset[idx][0] for idx in range(NUM_IMAGES)]
orig_images = [trf(train_dataset[idx][0]) for idx in range(NUM_IMAGES)]
preproc_images = [preprocess(img) for img in orig_images]
train_images = [train_transform(img) for img in orig_images]



img_grid = torchvision.utils.make_grid(torch.stack(images+ preproc_images+ train_images, dim=0), nrow=4, normalize=True, pad_value=0.5)
# img_grid = img_grid.permute(1, 2, 0)

img_grid.shape
plt.figure(figsize=(12,12))
plt.title("Augmentation examples on CIFAR10")
plt.imshow(trf(img_grid))
plt.axis('off')
plt.show()
plt.close()
