from __future__ import annotations
from typing import Any, Dict
import torch

from .dinov3 import build_custom_dinov3

def get_model(config: Dict[str, Any]) -> torch.nn.Module:
    model_cfg = config["model"]
    name = str(model_cfg["name"]).lower()
    if name == "dinov3":
        return build_custom_dinov3(config=config)
    else:
        raise ValueError(f"Unknown model: {name}")
