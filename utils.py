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
        image = item['image'].convert('RGB')
        label = item['label']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

class MixupCutmixWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset, num_classes, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
        self.dataset = dataset
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img1, label1 = self.dataset[idx]
        if torch.rand(1).item() > self.prob:
            return img1, torch.nn.functional.one_hot(torch.tensor(label1), self.num_classes).float()
        idx2 = torch.randint(0, len(self.dataset), (1,)).item()
        img2, label2 = self.dataset[idx2]

        if torch.rand(1).item() < 0.5 and self.mixup_alpha > 0: # MixUp
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
            img = lam * img1 + (1 - lam) * img2
            label = lam * torch.nn.functional.one_hot(torch.tensor(label1), self.num_classes).float() \
                + (1 - lam) * torch.nn.functional.one_hot(torch.tensor(label2), self.num_classes).float()
        else: # CutMix
            lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample().item()
            _, H, W = img1.shape
            cx, cy = torch.randint(W, (1,)).item(), torch.randint(H, (1,)).item()
            cut_w, cut_h = int(W * (1 - lam) ** 0.5), int(H * (1 - lam) ** 0.5)
            x1, y1 = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
            x2, y2 = min(cx + cut_w // 2, W), min(cy + cut_h // 2, H)
            img = img1.clone()
            img[:, y1:y2, x1:x2] = img2[:, y1:y2, x1:x2]
            lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
            label = lam * torch.nn.functional.one_hot(torch.tensor(label1), self.num_classes).float() \
                + (1 - lam) * torch.nn.functional.one_hot(torch.tensor(label2), self.num_classes).float()
        return img, label

supported_dataset = ["oxford-flowers102", "tiny-imagenet"]
def get_dataset(cfg):
    ds_cfg = cfg.dataset
    assert ds_cfg.name in supported_dataset, f"{ds_cfg.name} is not supported. Supported datasets: {supported_dataset}"

    cache_dir = Path("./dataset") / ds_cfg.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    transforms = []
    if getattr(ds_cfg.aug, 'hor_flip_aug', False):
        transforms.append(T.RandomHorizontalFlip())
    if getattr(ds_cfg.aug, "auto_augment", False):
        transforms.append(T.AutoAugment(T.AutoAugmentPolicy.IMAGENET))
    if getattr(ds_cfg.aug, "rand_augment", False):
        transforms.append(T.RandAugment())

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
    else:
        train_ds = load_dataset('Maysee/tiny-imagenet', split='train', cache_dir=cache_dir)
        val_ds = load_dataset('Maysee/tiny-imagenet', split='valid', cache_dir=cache_dir)
        train_ds = HFDatasetWrapper(train_ds, transform=train_transforms)
        val_ds = HFDatasetWrapper(val_ds, transform=val_transforms)
    if getattr(ds_cfg.aug, "mixup_cutmix", False):
        train_ds = MixupCutmixWrapper(train_ds, num_classes=ds_cfg.dataset.n_class)
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