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

# TODO recompute means std
# TODO make viz
FEW_SHOT_DATA = pathlib.Path("data/coco_crops_few_shot")
MODELS = pathlib.Path("models")
DATA_MEANS, DATA_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] #imagenet numbers
DATA_MEANS, DATA_STD = [0.48145466, 0.4578275, 0.40821073], [0.229, 0.224, 0.225] # clip numbers


class FSLMixin:
    """ Mixin class for consistent data handling
    """
    def train_dataloader(self):
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8,1.0), ratio=(0.9,1.1)),
                transforms.RandomHorizontalFlip(0.3),
                transforms.ToTensor(),
                transforms.Normalize(DATA_MEANS, DATA_STD),
                ])
        train_dataset = ImageFolder(FEW_SHOT_DATA/"train", transform = train_transform)
        pl.seed_everything(42)
        # train_set, _ = torch.utils.data.random_split(train_dataset)
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True)


    def val_dataloader(self):
        test_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(DATA_MEANS, DATA_STD),
        ])

        test_dataset = ImageFolder(FEW_SHOT_DATA/"test", transform = test_transform)
        pl.seed_everything(42)
        # _, val_set = torch.utils.data.random_split(train_dataset, )
        return DataLoader(test_dataset, batch_size=self.hparams.batch_size, shuffle=True)


    def test_dataloader(self):
        test_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(DATA_MEANS, DATA_STD),
        ])

        test_dataset = ImageFolder(FEW_SHOT_DATA/"test", transform = test_transform)
        return DataLoader(test_dataset, batch_size=self.hparams.batch_size, shuffle=True)




class ModelMixin:
    """ Mixin class for training/evaluating/testing
    """
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = (y == preds.argmax(dim=-1)).float().mean()

        # perform logging
        self.log("train_loss", loss)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = (y == preds.argmax(dim=-1)).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)


    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).argmax(dim=-1)
        acc = (y == preds).float().mean()
        self.log("test_acc", acc)