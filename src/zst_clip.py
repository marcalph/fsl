import torch
import clip
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils import FEW_SHOT_DATA, FSDataMixin
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms

DATA_MEANS, DATA_STD = [0.48145466, 0.4578275, 0.40821073], [
  0.26862954, 0.26130258, 0.27577711
]  # clip numbers




class ClipZSTClassifier(pl.LightningModule):
    def __init__(self, batch_size=16):
        super().__init__()

        self.save_hyperparameters()
        self.clip_model, _ = clip.load("ViT-B/32")

    def test_dataloader(self):
        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(DATA_MEANS, DATA_STD),
            ]
        )
        test_dataset = ImageFolder(FEW_SHOT_DATA / "test", transform=test_transform)
        self.classes = test_dataset.classes
        return DataLoader(
            test_dataset, batch_size=self.hparams.batch_size, shuffle=False
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        text_desc = [f"This is a photo of a {label}" for label in self.classes]
        text_toks = clip.tokenize(text_desc)
        I_f = self.clip_model.encode_image(x)
        T_f = self.clip_model.encode_text(text_toks)
        T_f /= T_f.norm(dim=-1, keepdim=True)

        preds = (100.0 * I_f @ T_f.T).argmax(dim=-1)
        acc = (y == preds).float().mean()
        self.log("test_acc", acc)


if __name__ == "__main__":
    model = ClipZSTClassifier()

    logger = TensorBoardLogger("logs", name="ClipZST")

    trainer_args = {
        "log_every_n_steps": 3,
        "max_epochs": 100,
        "logger": logger,
    }
    trainer = pl.Trainer(**trainer_args)

    trainer.test(model)
