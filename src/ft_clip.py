""" Clip finetuning class """
import clip
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW

from utils import FSDataMixin, ModelMixin


class ClipClassifier(ModelMixin, FSDataMixin, pl.LightningModule):
    def __init__(
        self,
        clip_kind="ViT-B/32",
        num_classes=8,
        shot_pct=0.8,
        optimizer=AdamW,
        lr=1e-4,
        l2=1e-3,
        batch_size=32,
        fc_only=True,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()
        self.example_input_array = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

        # backbone
        self.clip_model, _ = clip.load(
            clip_kind
        )  # ignore preprocess as it should be handle by dataloader
        # self.linear_clf = nn.Linear(self.clip_model.visual.output_dim, num_classes)

        self.linear_clf = self.classifier = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        if fc_only:  # freeze encoding
            for child in self.clip_model.children():
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, X):
        I_f = self.clip_model.encode_image(X)
        return self.linear_clf(I_f)

    def configure_optimizers(self):
        return self.hparams.optimizer(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.l2
        )


if __name__ == "__main__":
    for shot in [0.2, 0.4, 0.6, 0.8, 1.0]:
        model = ClipClassifier(clip_kind="RN50", shot_pct=shot)
        logger = TensorBoardLogger("logs", name=f"{model.hparams.clip_kind}_baseline")
        stop_early = EarlyStopping(patience=5, mode="min", monitor="val_loss")
        save_ckpt = ModelCheckpoint(
            dirpath="models",
            filename=f"{model.hparams.clip_kind}ml{model.hparams.shot_pct}" + "{epoch}",
            save_top_k=1,
            save_weights_only=True,
            mode="max",
            monitor="val_acc",
        )

        trainer_args = {
            "log_every_n_steps": 3,
            "max_epochs": 50,
            "logger": logger,
            "callbacks": [save_ckpt, stop_early],
        }

        trainer = pl.Trainer(**trainer_args)
        trainer.logger._log_graph = True
        trainer.logger._default_hp_metric = None
        trainer.fit(model)
        trainer.validate(model)
        trainer.test(model)
        print(save_ckpt.best_model_path)
