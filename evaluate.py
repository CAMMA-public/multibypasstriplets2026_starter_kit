from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict
import torch
from dataset.builder import get_datasets
from engine import validate_one_epoch
from models.builder import get_model
from utils.args import parse_cli_args, set_all_seeds

logger = logging.getLogger(__name__)

def select_device(device_pref: str) -> torch.device:
    if device_pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_pref in {"cpu", "cuda"}:
        if device_pref == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_pref)
    return torch.device("cpu")


def _setup_logging(config: Dict[str, Any]) -> logging.Logger:
    run_name = config.get("training", {}).get("run") or config.get("run")
    expname = config.get("training", {}).get("expname") or config.get("expname")
    log_root = config.get("logging", {}).get("log_dir") or config.get("output") or "logs"

    path_parts = [log_root]
    if run_name:
        path_parts.append(str(run_name))
    if expname:
        path_parts.append(str(expname))
    log_dir = os.path.join(*path_parts)
    os.makedirs(log_dir, exist_ok=True)

    base = "eval"
    if run_name and expname:
        base = f"{run_name}_{expname}_eval"
    elif run_name:
        base = f"{run_name}_eval"
    elif expname:
        base = f"{expname}_eval"

    log_file = os.path.join(log_dir, f"{base}.log")
    file_logger = logging.getLogger("file_only_eval")
    file_logger.setLevel(logging.INFO)
    file_logger.propagate = False
    if not file_logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        file_logger.addHandler(file_handler)
    return file_logger


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> int:
    if not checkpoint_path:
        raise ValueError("Please provide --eval.checkpoint_path for evaluate.py")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if torch.__version__ > "2.6":
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    else:
        ckpt = torch.load(checkpoint_path, map_location=device)

    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict"))
    if state_dict is None:
        raise KeyError("Checkpoint missing both 'state_dict' and 'model_state_dict'")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Handle common DataParallel prefix mismatch.
        if all(k.startswith("module.") for k in state_dict.keys()):
            stripped = {k[len("module."):]: v for k, v in state_dict.items()}
            model.load_state_dict(stripped, strict=True)
        else:
            prefixed = {f"module.{k}": v for k, v in state_dict.items()}
            model.load_state_dict(prefixed, strict=True)

    return int(ckpt.get("epoch", 0))


def main() -> None:
    config = parse_cli_args()
    if "seed" in config and config["seed"] is not None:
        set_all_seeds(int(config["seed"]))

    file_logger = _setup_logging(config)
    file_logger.info("Config:\n%s", json.dumps(config, indent=2, sort_keys=True))
    print(json.dumps(config, indent=2, sort_keys=True))

    device_pref = str(config["training"].get("device", "auto"))
    device = select_device(device_pref)
    file_logger.info("Using device: %s", device)

    split = config.eval.split
    if split == "hidden_test" and config.dataset.setting != "challenge":
        raise ValueError("hidden_test split is only available for dataset.setting=challenge")

    dataset = get_datasets(config, split=split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=True,
    )
    file_logger.info("Eval split=%s size=%d", split, len(dataset))

    model = get_model(config).to(device)
    eval_epoch = _load_checkpoint(model, config.eval.checkpoint_path, device=device)
    file_logger.info("Loaded checkpoint from %s (epoch=%d)", config.eval.checkpoint_path, eval_epoch)

    validate_one_epoch(
            model=model,
            dataloader=dataloader,
            config=config,
            device=device,
            file_logger=file_logger,
            mode=split,
            epoch=eval_epoch,
        )

    file_logger.info("Evaluation finished successfully.")


if __name__ == "__main__":
    main()

