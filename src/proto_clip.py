""" WIP proto-CLIP """
import random
from collections import defaultdict

import clip
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.optim import AdamW, lr_scheduler
from torch.utils import data
from torchvision.datasets import ImageFolder

from src.utils import FEW_SHOT_DATA

CHECKPOINT_PATH = "models"
shot_pct = 0.75
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# train/val/test/classes are the same


classes = torch.randperm(8)
train_classes, val_classes = classes[:6], classes[6:]

_, preprocess = clip.load("ViT-B/32")
train_transform = preprocess
ds = ImageFolder(FEW_SHOT_DATA / "train", preprocess)


class FewShotBatchSampler:
    def __init__(
        self,
        dataset_targets,
        N_way,
        K_shot,
        include_query=False,
        shuffle=True,
        shuffle_once=False,
    ):
        super().__init__()
        self.dataset_targets = dataset_targets
        self.N_way = N_way
        self.K_shot = K_shot
        self.shuffle = shuffle
        self.include_query = include_query
        if self.include_query:
            self.K_shot *= 2
        self.batch_size = self.N_way * self.K_shot

        self.classes = torch.unique(self.dataset_targets).tolist()
        self.num_classes = len(self.classes)
        self.indices_per_class = {}
        self.batches_per_class = {}
        for c in self.classes:
            self.indices_per_class[c] = torch.where(self.dataset_targets == c)[0]
            self.batches_per_class[c] = (
                self.indices_per_class[c].shape[0] // self.K_shot
            )

        # Create a list of classes from which we select the N classes per batch
        self.iterations = sum(self.batches_per_class.values()) // self.N_way
        self.class_list = [
            c for c in self.classes for _ in range(self.batches_per_class[c])
        ]
        if shuffle_once or self.shuffle:
            self.shuffle_data()
        else:
            # For testing, we iterate over classes instead of shuffling them
            sort_idxs = [
                i + p * self.num_classes
                for i, c in enumerate(self.classes)
                for p in range(self.batches_per_class[c])
            ]
            self.class_list = np.array(self.class_list)[np.argsort(sort_idxs)].tolist()

    def shuffle_data(self):
        # Shuffle the examples per class
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]

        random.shuffle(self.class_list)

    def __iter__(self):
        if self.shuffle:
            self.shuffle_data()

        start_index = defaultdict(int)
        for it in range(self.iterations):
            class_batch = self.class_list[
                it * self.N_way : (it + 1) * self.N_way
            ]  # Select N classes for the batch
            index_batch = []
            for (
                c
            ) in (
                class_batch
            ):  # For each class, select the next K examples and add them to the batch
                index_batch.extend(
                    self.indices_per_class[c][
                        start_index[c] : start_index[c] + self.K_shot
                    ]
                )
                start_index[c] += self.K_shot
            if self.include_query:
                index_batch = index_batch[::2] + index_batch[1::2]
            yield index_batch

    def __len__(self):
        return self.iterations


N_WAY = 4
K_SHOT = 4

train_data_loader = data.DataLoader(
    ds,
    batch_sampler=FewShotBatchSampler(
        torch.LongTensor(ds.targets),
        include_query=True,
        N_way=N_WAY,
        K_shot=K_SHOT,
        shuffle=True,
    ),
)


def split_batch(imgs, targets):
    support_imgs, query_imgs = imgs.chunk(2, dim=0)
    support_targets, query_targets = targets.chunk(2, dim=0)
    return support_imgs, query_imgs, support_targets, query_targets


class ProtoNet(pl.LightningModule):
    def __init__(self, proto_dim, lr):
        """Inputs.

        proto_dim - Dimensionality of prototype feature space
        lr - Learning rate of Adam optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        self.model, _ = clip.load("ViT-B/32")
        self.linear_clf = nn.Linear(
            self.model.visual.output_dim, self.hparams.proto_dim
        )
        for child in self.model.children():
            for param in child.parameters():
                param.requires_grad = False

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=[140, 180], gamma=0.1
        )
        return [optimizer], [scheduler]

    @staticmethod
    def calculate_prototypes(features, targets):
        # Given a stack of features vectors and labels, return class prototypes
        # features - shape [N, proto_dim], targets - shape [N]
        classes, _ = torch.unique(targets).sort()  # Determine which classes we have
        prototypes = []
        for c in classes:
            p = features[torch.where(targets == c)[0]].mean(
                dim=0
            )  # Average class feature vectors
            prototypes.append(p)
        prototypes = torch.stack(prototypes, dim=0)
        # Return the 'classes' tensor to know which prototype belongs to which class
        return prototypes, classes

    def classify_feats(self, prototypes, classes, feats, targets):
        # Classify new examples with prototypes and return classification error
        dist = torch.pow(prototypes[None, :] - feats[:, None], 2).sum(
            dim=2
        )  # Squared euclidean distance
        preds = F.log_softmax(-dist, dim=1)
        labels = (classes[None, :] == targets[:, None]).long().argmax(dim=-1)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        return preds, labels, acc

    def calculate_loss(self, batch, mode):
        # Determine training loss for a given support and query set
        imgs, targets = batch
        # features = self.model(imgs)  # Encode all images of support and query set
        # feats = []
        # for img in imgs:
        features = self.model.encode_image(imgs)
        features = self.linear_clf(features)
        # feats.append(features)
        # features= torch.concat(feats)
        support_feats, query_feats, support_targets, query_targets = split_batch(
            features, targets
        )
        prototypes, classes = ProtoNet.calculate_prototypes(
            support_feats, support_targets
        )
        preds, labels, acc = self.classify_feats(
            prototypes, classes, query_feats, query_targets
        )
        loss = F.cross_entropy(preds, labels)

        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.calculate_loss(batch, mode="val")


if __name__ == "__main__":
    trainer = pl.Trainer(
        gpus=1 if str(device) == "cuda:0" else 0,
        max_epochs=200,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
        ],
        progress_bar_refresh_rate=0,
    )
    trainer.logger._default_hp_metric = None
    model = ProtoNet(proto_dim=64, lr=2e-4)
    trainer.fit(model, train_data_loader, train_data_loader)
