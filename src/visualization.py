import seaborn as sns
import pandas as pd
import numpy as np
import pathlib
import pytorch_lightning as pl
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from ft_baseline import ResNetClassifier
from ft_clip import ClipClassifier

FEW_SHOT_DATA = pathlib.Path("data/coco_crops_few_shot")
test = ImageFolder(FEW_SHOT_DATA / "test")
DATA_MEANS, DATA_STD = [0.48145466, 0.4578275, 0.40821073], [
    0.26862954,
    0.26130258,
    0.27577711,
]
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEANS, DATA_STD),
    ]
)
test = ImageFolder(FEW_SHOT_DATA / "test", transform=test_transform)
testdl = DataLoader(test, batch_size=32, shuffle=False)

if __name__ == "__main__":
    trainer = pl.Trainer()
    vit_clips = sorted(pathlib.Path("models").rglob("ViT-B/32ml*.ckpt"))
    resnets = sorted(pathlib.Path("models").rglob("resnet*.ckpt"))
    x = np.linspace(0.2, 1, 5)
    vit_perfs = []
    resnet_perfs = []
    for mod in resnets:
        model = ResNetClassifier.load_from_checkpoint(mod)
        p = trainer.test(model, testdl)
        resnet_perfs.append(p[0]["test_acc"])
    for mod in vit_clips:
        model = ClipClassifier.load_from_checkpoint(mod)
        p = trainer.test(model, testdl)
        vit_perfs.append(p[0]["test_acc"])

    hm = dict(x=x, y=vit_perfs, z=resnet_perfs)
    data = pd.DataFrame(hm)
    fig, ax = plt.subplots()
    ax = sns.lineplot(x="x", y="y", data=data)
    plt.text(1 + 0.02, vit_perfs[-1] + 0.02, "ft_clip")
    ax = sns.lineplot(x="x", y="z", data=data)
    plt.text(1 + 0.02, resnet_perfs[-1] + 0.02, "ft_resnet")
    from zst_clip import ClipZSTClassifier

    model = ClipZSTClassifier()
    p = trainer.test(model)
    plt.scatter(0, p[0]["test_acc"], marker="o", s=100)
    plt.text(0 + 0.02, p[0]["test_acc"] + 0.02, "zs-clip")
    fig.savefig("reports/figures/FSL_ZSL_figure.png")
