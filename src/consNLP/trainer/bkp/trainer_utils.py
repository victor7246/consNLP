import random
import os
import numpy as np
import torch

def set_seed(seed=666):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

try:
    #from torch.cuda import amp  # noqa: F401
    from apex import amp
    _has_apex = True
except ImportError:
    _has_apex = False

try:
    import pytorch_lightning as pl
    _torch_lightning_available = True
except ImportError:
    _torch_lightning_available = False

try:
    import torch_xla.core.xla_model as xm  # noqa: F401
    _torch_tpu_available = True
except ImportError:
    _torch_tpu_available = False

'''
try:
    import wandb
    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn("W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except (ImportError, AttributeError):
    _has_wandb = False
'''

try:
    import wandb
    _has_wandb = True
except (ImportError, AttributeError):
    _has_wandb = False

_torch_gpu_available = False
if torch.cuda.is_available():
    _torch_gpu_available = True
    
_num_gpus = torch.cuda.device_count()




