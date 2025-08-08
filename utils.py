import torch
import pytz
from datetime import datetime
from torchvision import datasets
from pathlib import Path
import torchvision.transforms.v2 as T
from datasets import load_dataset
from torch.utils.data import Dataset

def torch_get_device(device_type):
    if device_type == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available :(, `python train.py +device=auto`"
        device = torch.device("cuda")
    elif device_type == "auto":
        assert not torch.cuda.is_available(), "CUDA is available :), switch to cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            try:
                import torch_xla.core.xla_model as xm  # type: ignore
                device = xm.xla_device()
            except ImportError:
                device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device

def torch_set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def torch_compile_ckpt_fix(state_dict):
    # when torch.compiled a model, state_dict is updated with a prefix '_orig_mod.', renaming this
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict

def get_ist_time_now(fmt="%d-%m-%Y-%H%M%S"):
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(ist)
    return now_ist.strftime(fmt)

class HFDatasetWrapper(Dataset):

    def __init__(self, hf_dataset, transform=None):
        super().__init__()
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Ensure image has 3 channels and is in (H, W, C) format before transforming
        image = item['image'].convert('RGB')
        label = item['label']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

supported_dataset = ["oxford-flowers102", "tiny-imagenet"]
def get_dataset(cfg):
    assert cfg.name in supported_dataset, f"{cfg.name} is not supported. Supported datasets: {supported_dataset}"

    cache_dir = Path("./dataset") / cfg.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    resize_to = cfg.aug.center_crop if "center_crop" in cfg.aug else cfg.img_size
    transforms = [ T.Resize(resize_to) ]
    if "center_crop" in cfg.aug:
        transforms.append(T.CenterCrop(cfg.img_size))
    if cfg.aug.hor_flip_aug:
        transforms.append(T.RandomHorizontalFlip())
    transforms.extend([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=False),
        T.Normalize(mean=[0.5]*cfg.img_chls, std=[0.5]*cfg.img_chls),
    ])
    transforms = T.Compose(transforms)
    if cfg.name == "oxford-flowers102":
        train_ds = datasets.ImageFolder(cache_dir / "train", transform=transforms)
        val_ds = datasets.ImageFolder(cache_dir / "val", transform=transforms)
    else:
        train_ds = load_dataset('Maysee/tiny-imagenet', split='train', cache_dir=cache_dir)
        val_ds = load_dataset('Maysee/tiny-imagenet', split='valid', cache_dir=cache_dir)
        train_ds = HFDatasetWrapper(train_ds, transform=transforms)
        val_ds = HFDatasetWrapper(val_ds, transform=transforms)
    return train_ds, val_ds