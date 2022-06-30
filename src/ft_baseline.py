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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from utils import FSDataMixin, ModelMixin
from torch.optim import AdamW

FEW_SHOT_DATA = pathlib.Path("data/coco_crops_few_shot")
MODELS = pathlib.Path("models")
DATA_MEANS, DATA_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] #imagenet numbers
DATA_MEANS, DATA_STD = [0.48145466, 0.4578275, 0.40821073], [0.229, 0.224, 0.225] # clip numbers


class ResNetClassifier(ModelMixin, FSDataMixin, pl.LightningModule):
    def __init__(self, num_classes=8, shot_pct=.2, optimizer=AdamW, lr=1e-4, l2=1e-3, batch_size=32, fc_only=True):
        super().__init__()

        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()
        self.example_input_array = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

        # resnet backbone
        self.resnet_model = resnet50(pretrained=True)
        linear_size = list(self.resnet_model.children())[-1].in_features
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)

        if fc_only: # only tune FC layers ~ linear probing in CLIP paper
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):
        return self.hparams.optimizer(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.l2)




if __name__ == "__main__":
    model = ResNetClassifier(num_classes = 8, shot_pct=1.)

    logger = TensorBoardLogger("logs", name="resnet_baseline_simple")
    save_ckpt = ModelCheckpoint(dirpath="models", filename=f'resnet{model.hparams.shot_pct}'+'{epoch}', save_weights_only=True, mode="max", monitor="val_acc")
    stop_early = EarlyStopping(patience=5, mode="min", monitor="val_loss")

    trainer_args = {'log_every_n_steps': 3,
                    'max_epochs': 100,
                    'logger': logger,
                    'callbacks':[save_ckpt, stop_early]}

    trainer = pl.Trainer(**trainer_args)
    trainer.logger._default_hp_metric = None
    trainer.fit(model,)
    trainer.validate(model)
    trainer.test(model)
    print(save_ckpt.best_model_path)


    save_path = MODELS/'resnet_baseline_simple.ckpt'
    trainer.save_checkpoint(save_path)

