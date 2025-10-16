import math
from datetime import datetime
from pathlib import Path
import pytz
import torch
import torchvision.transforms.v2 as T
from datasets import load_dataset
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from torchvision import datasets

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
    tz = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(tz)
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
        image = item['image'].convert('RGB')
        label = item['label']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

supported_dataset = ["oxford-flowers102", "tiny-imagenet", 'cifar100']
def get_dataset(cfg):
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
        T.ToDtype(torch.float32, scale=False),
        T.Normalize(mean=[0.5]*ds_cfg.img_chls, std=[0.5]*ds_cfg.img_chls),
    ]
    train_transforms = T.Compose(resize + transforms + cast_scale)
    val_transforms = T.Compose(resize + cast_scale)
    if ds_cfg.name == "oxford-flowers102":
        train_ds = datasets.ImageFolder(cache_dir / "train", transform=train_transforms)
        val_ds = datasets.ImageFolder(cache_dir / "val", transform=val_transforms)
    elif ds_cfg.name == "cifar100":
        train_ds = datasets.CIFAR100(cache_dir / "train", train=True, download=True, transform=train_transforms)
        val_ds = datasets.CIFAR100(cache_dir / "val", train=False, download=True, transform=train_transforms)
    else:
        train_ds = load_dataset('Maysee/tiny-imagenet', split='train', cache_dir=cache_dir)
        val_ds = load_dataset('Maysee/tiny-imagenet', split='valid', cache_dir=cache_dir)
        train_ds = HFDatasetWrapper(train_ds, transform=train_transforms)
        val_ds = HFDatasetWrapper(val_ds, transform=val_transforms)
    return train_ds, val_ds

def cosine_with_linear_warmup_lr_scheduler(optimizer, total_steps, warmup_pct, decay_step_pct, min_lr_pct):
    """
    Learning rate scheduler with linear warmup, cosine decay, then constant LR.
    1. Warmup: LR increases linearly from 0 to LR over (warmup_pct * total_steps)
    2. Decay: LR follows cosine decay from LR to (min_lr_pct * LR) of it over next (decay_step_pct * total_steps)
    3. Constant: LR stays at (min_lr_pct * LR) for remaining steps.
    """
    warmup_steps = int(warmup_pct * total_steps)
    decay_steps = warmup_steps + int(decay_step_pct * total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # warmup
            return float(current_step + 1) / float(max(1, warmup_steps))
        elif current_step > decay_steps:
            # constant
            return min_lr_pct
        else:
            # cosine
            progress = float(current_step - warmup_steps) / float(max(1, decay_steps - warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_pct + (1 - min_lr_pct) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)

@torch.no_grad()
def calc_accuracy(output, target, topk=(1,)):
    """ Computes the accuracy over the k top predictions for the specified values of k. """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

class AverageMetric:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0