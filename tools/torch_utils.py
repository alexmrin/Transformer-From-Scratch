import os

import torch
import torch.distributed as dist

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def barrier() -> None:
    if is_distributed():
        dist.barrier()

def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK") or 0)

def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    else:
        return 1

def get_global_rank() -> int:
    return int(os.getenv("RANK") or dist.get_rank())

def seed_all(seed: int):
    import random
    import numpy as np

    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed {seed} is invalid. It must be on [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)