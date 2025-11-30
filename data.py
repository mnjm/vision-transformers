import math
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2 as T
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

class HFDatasetWrapper(Dataset):

    def __init__(self, hf_dataset, transform=None):
        super().__init__()
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        label = item['label']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

supported_dataset = ["oxford-flowers102", "tiny-imagenet", 'cifar100']
def init_dataloaders(cfg):
    ds_cfg = cfg.dataset
    assert ds_cfg.name in supported_dataset, f"{ds_cfg.name} is not supported. Supported datasets: {supported_dataset}"

    cache_dir = Path("./dataset") / ds_cfg.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    transforms = []
    if getattr(ds_cfg.aug, 'hor_flip_aug', False):
        transforms.append(T.RandomHorizontalFlip())
    if getattr(ds_cfg.aug, "rand_augment", False):
        transforms.append(T.RandAugment())
    # prefer rand augment if both rand augment and auto augment is enabled
    elif getattr(ds_cfg.aug, "auto_augment", False):
        transforms.append(T.AutoAugment(T.AutoAugmentPolicy.IMAGENET))

    resize = [ T.Resize(ds_cfg.img_size) ]
    cast_scale = [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5]*ds_cfg.img_chls, std=[0.5]*ds_cfg.img_chls),
    ]
    train_transforms = T.Compose(resize + transforms + cast_scale)
    val_transforms = T.Compose(resize + cast_scale)
    if ds_cfg.name == "oxford-flowers102":
        train_ds = datasets.ImageFolder(cache_dir / "train", transform=train_transforms)
        val_ds = datasets.ImageFolder(cache_dir / "val", transform=val_transforms)
    elif ds_cfg.name == "cifar100":
        train_ds = datasets.CIFAR100(cache_dir / "train", train=True, download=True, transform=train_transforms)
        val_ds = datasets.CIFAR100(cache_dir / "val", train=False, download=True, transform=val_transforms)
    else:
        train_ds = load_dataset('Maysee/tiny-imagenet', split='train', cache_dir=cache_dir)
        val_ds = load_dataset('Maysee/tiny-imagenet', split='valid', cache_dir=cache_dir)
        train_ds = HFDatasetWrapper(train_ds, transform=train_transforms)
        val_ds = HFDatasetWrapper(val_ds, transform=val_transforms)

    # build dataloader
    kwargs = dict(cfg.dataloader)
    train_dataloader = DataLoader(train_ds, shuffle=True, **kwargs)
    kwargs['drop_last'] = False
    val_dataloader = DataLoader(val_ds, shuffle=False, **kwargs)

    return train_dataloader, val_dataloader

def show_batch(dataloader, N, nrow=None):
    batch = next(iter(dataloader))
    imgs, lbls = batch
    B = imgs.shape[0]
    assert B >= N, f"{N=} should be <= batch size {B}"
    imgs = imgs[:N].detach().cpu().permute(0, 2, 3, 1).numpy()
    lbls = lbls[:N].detach().cpu().numpy()
    # normalize image
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)

    nrow = nrow if nrow else math.ceil(math.sqrt(N))
    ncol = math.ceil(N / nrow)
    figsize_scale = 2.0
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * figsize_scale, ncol * figsize_scale))
    def on_key(event):
        if event.key == 'q':
            plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', on_key)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.set_axis_off()
        if i >= N:
            break
        if imgs[i].shape[-1] == 1:
            ax.imshow(imgs[i], cmap="gray")
        else:
            ax.imshow(imgs[i])
        ax.set_title(f"{int(lbls[i])}")

    plt.tight_layout()
    plt.show(block=True)
    plt.close(fig)