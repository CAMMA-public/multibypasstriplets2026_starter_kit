from __future__ import annotations
from typing import Any, Dict, Tuple
import os
from utils.helpers import setup_logging

DEFAULT_CONFIG_PATH = "configs/default.yaml"

# def dataset_config_path(dataset_name: str) -> str:
#     root = "/home/ssharma/projects/repos/tripbench/datasets"
#     return os.path.join(root, dataset_name, "config.yaml")

def initialize_runtime(merged_config: Dict[str, Any]) -> None:
    # Prepare logging configuration with nested run/exp directories and filename
    logging_cfg = dict(merged_config["logging"])
    training_cfg = merged_config["training"]
    run = training_cfg.get("run")
    exp = training_cfg.get("expname") or logging_cfg.get("expname")

    # Determine base file name
    if not logging_cfg.get("file_name"):
        if run or exp:
            base = "train"
            if run and exp:
                base = f"{run}_{exp}"
            elif run:
                base = str(run)
            elif exp:
                base = str(exp)
        else:
            base = "train"
        logging_cfg["file_name"] = f"{base}.log"

    # Determine root log directory (from config or default "logs")
    log_root = logging_cfg.get("log_dir") or "logs"
    # Always nest under <log_root>/<run>/<exp> when the identifiers are available
    path_parts = [log_root]
    if run:
        path_parts.append(str(run))
    if exp:
        path_parts.append(str(exp))
    nested_log_dir = os.path.join(*path_parts)
    logging_cfg["log_dir"] = nested_log_dir
    # Also reflect this back into merged_config so the rest of the code sees it
    merged_config["logging"]["log_dir"] = nested_log_dir

    setup_logging(logging_cfg)
    # Handle GPU selection from config (training.gpus): accepts comma-list or integer count
    training_cfg = merged_config["training"]
    gpus = str(training_cfg.get("gpus", "")).strip()
    if gpus:
        # If it's a pure integer (count), map to 0..count-1
        if gpus.isdigit():
            count = int(gpus)
            if count > 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(count))
        else:
            # Assume comma-separated ids
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus


