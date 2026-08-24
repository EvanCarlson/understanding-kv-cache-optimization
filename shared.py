import random
from enum import StrEnum

import torch

DEFAULT_SEED = 42


class Device(StrEnum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


def get_device() -> Device:
    if torch.cuda.is_available():
        return Device.CUDA

    if torch.backends.mps.is_available():
        return Device.MPS

    return Device.CPU


def seed_everything(*, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
