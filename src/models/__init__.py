"""Wave forecast model zoo."""

from .cnn3d import CNN3D
from .convlstm import ConvLSTM
from .persistence import Climatology, Persistence
from .unet3d import UNet3D

REGISTRY = {
    "unet3d": UNet3D,
    "cnn3d": CNN3D,
    "convlstm": ConvLSTM,
}


def build(name: str, in_channels: int, lookback: int = 48, **kwargs):
    """Instantiate a trainable model by name."""
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](in_channels=in_channels, lookback=lookback, **kwargs)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["UNet3D", "CNN3D", "ConvLSTM", "Persistence", "Climatology",
           "REGISTRY", "build", "count_parameters"]
