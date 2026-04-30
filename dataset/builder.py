from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
from torch.utils.data import Dataset
from .multibypasst40 import MultiByPassT40

def get_datasets(
    config: Dict[str, Any], 
    split: str = "train",
    aug = None
    ) -> Dataset:
    dataset_cfg = config["dataset"]
    name = str(dataset_cfg["name"]).lower()

    if name == "multibypasst40":
        return MultiByPassT40(config=config, 
                        split=split,
                        test_fold=dataset_cfg["test_fold"],
                        video_dir_prefix=dataset_cfg["video_dir_prefix"],
                        video_path=dataset_cfg["video_path"],
                        label_path=dataset_cfg["label_path"],
                        img_size=(dataset_cfg["img_height"], dataset_cfg["img_width"]),
                        aug=aug,
                )
    raise ValueError(f"Unknown dataset: {name}")

