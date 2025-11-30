import math
import os
from datetime import datetime
from functools import wraps
from time import time
import pytz
import torch
import wandb
from dotenv import load_dotenv
from torch.optim.lr_scheduler import LambdaLR

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time()
        ret = func(*args, **kwargs)
        t = time() - t0
        return t, ret
    return wrapper

class WandBLogger:

    def __init__(self, project, run, config, tags, metrics, enable=False):
        self.enable = enable
        if self.enable:
            load_dotenv()
            wnb_key = os.getenv('WANDB_API_KEY')
            assert wnb_key is not None, "WANDB_API_KEY not loaded in env"
            wandb.login(key=wnb_key)
            wandb.init(project=project, name=run, config=config)
            self.tags = tags
            self.metrics = metrics
            wandb.define_metric("epoch")
            for tag in tags:
                for metric in metrics:
                    wandb.define_metric(f"{tag}/{metric}", step_metric="epoch")

    def log(self, tag, data):
        if self.enable:
            assert tag in self.tags, f"{tag=} not created"
            new_data = {'epoch': data['epoch']}
            for metric, val in data.items():
                if metric == "epoch":
                    continue
                assert metric in self.metrics, f"{metric=} not created"
                new_data[f"{tag}/{metric}"] = val
            wandb.log(new_data)