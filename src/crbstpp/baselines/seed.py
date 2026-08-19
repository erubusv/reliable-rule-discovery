from __future__ import annotations

import os
import random

import numpy as np


def validate_seed(seed: int) -> int:
    seed = int(seed)
    if not 0 <= seed < 2**32:
        raise ValueError("seed must lie in [0, 2**32)")
    return seed


def set_reproducible_seed(seed: int) -> int:
    """Set every RNG used by the baseline suite to one declared seed."""

    seed = validate_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed
