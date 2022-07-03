import clip
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils import DATA_MEANS, DATA_STD, FEW_SHOT_DATA


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
