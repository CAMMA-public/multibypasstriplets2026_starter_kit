from typing import Optional, Dict, Any, Tuple, List
import torch
import os
import numpy as np
from tqdm import tqdm
import json
from torchmetrics.functional.classification import multilabel_average_precision

import logging
LOGGER = logging.getLogger(__name__)

def mAP(output: torch.Tensor, target: torch.Tensor) -> float:
    return multilabel_average_precision(output, target.long(), num_labels=output.shape[1]).item()

class AverageMeter(object):
    """Computes and stores the average and current value"""
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
        self.avg = self.sum / self.count


def get_class_weights(config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if config.dataset.name == "cholect50":
        cls_wt_i = torch.ones(6)
        cls_wt_v = torch.ones(10)
        cls_wt_t = torch.ones(15)
        cls_wt_ivt = torch.ones(100)
        
    elif config.dataset.name == "multibypasst40":
        cls_wt_i = [1.0] * config.model.num_tool_classes
        cls_wt_v = [1.0] * config.model.num_verb_classes
        cls_wt_t = [1.0] * config.model.num_target_classes
        cls_wt_ivt = [1.0] * config.model.num_triplet_classes

        cls_wt_i = torch.tensor(cls_wt_i)
        cls_wt_v = torch.tensor(cls_wt_v)
        cls_wt_t = torch.tensor(cls_wt_t)
        cls_wt_ivt = torch.tensor(cls_wt_ivt)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset.name}")
    
    return cls_wt_i, cls_wt_v, cls_wt_t, cls_wt_ivt

def compute_grad_total_norm(parameters, norm_type: float = 2.0) -> float:
    """Return total gradient norm across all parameters."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    if norm_type == float("inf"):
        return max(g.abs().max().item() for g in grads)
    total = 0.0
    for g in grads:
        param_norm = g.detach().float().norm(norm_type)
        total += float(param_norm) ** norm_type
    return total ** (1.0 / norm_type)

def backward_step_single_optim(
    loss: torch.Tensor,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    grad_clip_norm: float,
) -> Optional[float]:
    optim.zero_grad(set_to_none=True)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optim)

        for param in model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                scaler.update()
                return None

        grad_norm = compute_grad_total_norm(model.parameters())
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scaler.step(optim)
        scaler.update()
        return grad_norm if torch.isfinite(torch.tensor(grad_norm)) else None

    loss.backward()
    grad_norm = compute_grad_total_norm(model.parameters())
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    optim.step()
    return grad_norm

def get_progress_bar(data_loader, mode, desc=None):
    """Create a tqdm progress bar with consistent formatting."""
    return tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc=f"{mode.upper()}" if desc is None else desc,
        bar_format='{desc:<12} {percentage:3.0f}%|{bar:30}{r_bar}',
        ncols=200,
        leave=True,
        dynamic_ncols=False
    )

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def setup_logging(cfg: Dict[str, Any]) -> None:
    level_name = str(cfg.get("level", "info")).lower()
    level = _LEVELS.get(level_name, logging.INFO)
    fmt = cfg.get("format", "[%(asctime)s] %(levelname)s - %(message)s")
    datefmt = cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")

    handlers = []
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    handlers.append(stream_handler)

    log_dir = cfg.get("log_dir")
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, cfg.get("file_name", "train.log"))
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers)