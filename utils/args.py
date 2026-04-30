from __future__ import annotations

import argparse
from typing import Optional
import os
import random
import numpy as np
import torch


class ConfigNode(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_cli_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training arguments for MultiSAT2026 Starter Kit")

    parser.add_argument("--config", type=str, default="configs/models/dinov3.yaml")
    parser.add_argument("--output", type=str, default="logs/")
    parser.add_argument("--run", type=str, default="default")
    parser.add_argument("--expname", type=str, default="baseline")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=324)

    parser.add_argument("--dataset.name", type=str, default="multibypasst40")
    parser.add_argument("--dataset.setting", type=str, default="default0")
    parser.add_argument("--dataset.batch_size", type=int, default=4)
    parser.add_argument("--dataset.video_dir_prefix", type=str, default="videos")
    parser.add_argument("--dataset.video_path", type=str, default="MultiBypassT40/videos")
    parser.add_argument("--dataset.label_path", type=str, default="MultiBypassT40/labels")
    parser.add_argument("--dataset.aug_type", type=str, default="aug0")
    parser.add_argument("--dataset.test_fold", type=int, default=-1)
    parser.add_argument("--dataset.img_height", type=int, default=224)
    parser.add_argument("--dataset.img_width", type=int, default=224)
    parser.add_argument("--dataset.sampling_percentage", type=float, default=1.0)
    parser.add_argument("--dataset.clip_len", type=int, default=1)
    parser.add_argument("--dataset.clip_position", type=str, default="center")
    parser.add_argument("--dataset.clip_center_mode", type=str, default="symmetric")
    parser.add_argument("--dataset.clip_aggregation", type=str, default="mean")

    parser.add_argument("--optim.name", type=str, default="adamw")
    parser.add_argument("--optim.lr", type=float, default=0.0001)
    parser.add_argument("--optim.backbone_lr", type=float, default=0.00002)
    parser.add_argument("--optim.weight_decay", type=float, default=0.06)
    parser.add_argument("--optim.grad_clip_norm", type=float, default=1.2)

    parser.add_argument("--eval.method", type=str, default="torchmetrics", choices="torchmetrics")
    parser.add_argument("--eval.per_video", action="store_true", default=False)
    parser.add_argument("--eval.checkpoint_path", type=str, default="")
    parser.add_argument("--eval.split", type=str, default="test", choices=["val", "test", "hidden_test"])

    parser.add_argument("--lr_scheduler.name", type=str, default="cosine")

    parser.add_argument("--model.name", type=str, default="dinov3")
    parser.add_argument("--model.backbone_name", type=str, default="dinov3_vits16")
    parser.add_argument("--model.num_triplet_classes", type=int, default=85)
    parser.add_argument("--model.num_tool_classes", type=int, default=12)
    parser.add_argument("--model.num_verb_classes", type=int, default=13)
    parser.add_argument("--model.num_target_classes", type=int, default=15)
    parser.add_argument("--model.apply_fc", type=str, default="i,v,t,ivt")
    parser.add_argument("--model.apply_temporal", action="store_true", default=False)
    parser.add_argument("--model.fc_input_dim", type=int, default=384)
    parser.add_argument("--model.apply_class_weights", type=str, default="i,v,t")
    parser.add_argument("--model.val_ckpt_key", type=str, default="ivt")
    parser.add_argument("--model.val_ckpt_patience", type=int, default=5)
    parser.add_argument("--model.ignore_null_labels", action="store_true", default=False)

    parser.add_argument("--training.epochs", type=int, default=30)

    ns = parser.parse_args(argv)
    flat = vars(ns)

    nested: dict = {}
    for key, value in flat.items():
        if "." not in key:
            nested[key] = value
            continue
        parts = key.split(".")
        cur = nested
        for part in parts[:-1]:
            if part not in cur or not isinstance(cur[part], dict):
                cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = value

    # Convenience mirrors used in existing code paths.
    nested.setdefault("training", {})
    nested["training"].setdefault("run", nested.get("run"))
    nested["training"].setdefault("expname", nested.get("expname"))
    nested["training"].setdefault("device", "auto")

    nested.setdefault("dataset", {})
    nested["dataset"].setdefault("num_workers", 4)

    nested.setdefault("logging", {})
    nested["logging"].setdefault("log_dir", nested.get("output", "logs/"))

    nested.setdefault("resume", {})
    nested["resume"].setdefault("checkpoint_path", None)

    nested.setdefault("eval", {})
    nested["eval"].setdefault("checkpoint_path", "")
    nested["eval"].setdefault("split", "test")

    # nested.setdefault("model", {})
    # nested["model"].setdefault("ignore_null_labels", False)

    nested.setdefault("disable_autocast", False)
    nested.setdefault("save_predictions", False)

    nested.setdefault("lr_scheduler", {})
    nested["lr_scheduler"].setdefault("eta_min", 0.0)
    nested["lr_scheduler"].setdefault("step_size", 10)
    nested["lr_scheduler"].setdefault("gamma", 0.1)
    nested["lr_scheduler"].setdefault("milestones", [10, 20])

    def to_node(obj):
        if isinstance(obj, dict):
            node = ConfigNode()
            for k, v in obj.items():
                node[k] = to_node(v)
            return node
        return obj

    return to_node(nested)
