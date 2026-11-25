import os
import random

import numpy as np
import torch

__all__ = [
    "set_seed",
]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
