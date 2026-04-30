from __future__ import annotations

from typing import Dict, Optional
import os
import torch
import json
from dataset.builder import get_datasets
from engine import train_one_epoch, validate_one_epoch
from models.builder import get_model
from utils.args import set_all_seeds, parse_cli_args

import logging
logger = logging.getLogger(__name__)

def select_device(device_pref: str) -> torch.device:
    if device_pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_pref in {"cpu", "cuda"}:
        if device_pref == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_pref)
    return torch.device("cpu")

def build_main_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    config: Dict,
    epochs_total: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs_total),
            eta_min=config["lr_scheduler"]["eta_min"],
        )
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["lr_scheduler"]["step_size"],
            gamma=config["lr_scheduler"]["gamma"],
        )
    raise ValueError(f"Unknown lr scheduler: {scheduler_name}")

def main() -> None:
    config = parse_cli_args()

    # Seeds
    if "seed" in config and config["seed"] is not None:
        set_all_seeds(int(config["seed"]))

    # Logging & Device (simple path setup; no runtime initializer)
    if isinstance(config, dict):
        run_name = config.get("training", {}).get("run") or config.get("run")
        expname = config.get("training", {}).get("expname") or config.get("expname")
        log_root = config.get("logging", {}).get("log_dir") or config.get("output") or "logs"
    else:
        run_name = getattr(getattr(config, "training", None), "run", None) or getattr(config, "run", None)
        expname = getattr(getattr(config, "training", None), "expname", None) or getattr(config, "expname", None)
        log_root = getattr(config, "output", None) or "logs"

    path_parts = [log_root]
    if run_name:
        path_parts.append(str(run_name))
    if expname:
        path_parts.append(str(expname))
    log_dir = os.path.join(*path_parts)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        device_pref = str(config["training"]["device"])
    except Exception:
        device_pref = "auto"
    device = select_device(device_pref)

    # Logging
    os.makedirs(log_dir, exist_ok=True)

    base = "train"
    if run_name and expname:
        base = f"{run_name}_{expname}"
    elif run_name:
        base = str(run_name)
    elif expname:
        base = str(expname)
    log_file = os.path.join(log_dir, f"{base}.log")
    logger.info(f"Logging Experiment: {base} to file: {log_file}")

    # set up file-only logger (no console output)
    file_logger = logging.getLogger('file_only')
    file_logger.setLevel(logging.INFO)
    file_logger.propagate = False  # Don't propagate to root logger
    if not file_logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(fmt="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        file_logger.addHandler(file_handler)

    # add the config details to the log file
    config_for_log = config if isinstance(config, dict) else vars(config)
    file_logger.info("Config:\n%s", json.dumps(config_for_log, indent=2, sort_keys=True))

    # dump to print or terminal the config
    print(json.dumps(config_for_log, indent=2, sort_keys=True))

    # Dataset
    train_dataset = get_datasets(config, split="train")
    logger.info(f"Training dataset size: {len(train_dataset)}")
    

    drop_last = False
    train_loader = torch.utils.data.DataLoader(
                                    train_dataset, 
                                    batch_size=config["dataset"]["batch_size"], 
                                    shuffle=True, 
                                    num_workers=config["dataset"]["num_workers"],
                                    drop_last=drop_last,
                                    pin_memory=True)
    val_dataset = get_datasets(config, split="val")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    val_loader = torch.utils.data.DataLoader(
                                    val_dataset, 
                                    batch_size=config["dataset"]["batch_size"], 
                                    shuffle=False, 
                                    num_workers=config["dataset"]["num_workers"],
                                    pin_memory=True)

    test_dataset = get_datasets(config, split="test")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    test_loader = torch.utils.data.DataLoader(
                                    test_dataset, 
                                    batch_size=config["dataset"]["batch_size"], 
                                    shuffle=False, 
                                    num_workers=config["dataset"]["num_workers"],
                                    pin_memory=True)
    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info("Datasets and DataLoaders initialized successfully! 🚀")

    # Model
    model = get_model(config).to(device)

    # set model to DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # Separate parameters into backbone and non-backbone groups for different learning rates
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
    
    # Build parameter groups with different learning rates
    backbone_lr = config.optim.get("backbone_lr", config.optim.lr)
    param_groups = [
        {"params": other_params, "lr": config.optim.lr},
        {"params": backbone_params, "lr": backbone_lr},
    ]
    logger.info(f"Using lr={config.optim.lr} for non-backbone params, backbone_lr={backbone_lr} for backbone params")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_mb = total_params / 1e6
    logger.info(f"Total trainable parameters: {total_params} ({total_params_mb:.2f} MB)")

    logger.info("Trainable parameters in the model:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name} | {param.requires_grad}")
    logger.info("--------------------------------")
    
    # learnable_params = [p for p in model.parameters() if p.requires_grad]
    
    optim_name = config.optim.name
    if optim_name == "adam":
        optim = torch.optim.Adam(
            param_groups,
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    elif optim_name == "sgd":
        optim = torch.optim.SGD(
            param_groups,
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
            momentum=0.95
        )
    elif optim_name == "adamw":
        optim = torch.optim.AdamW(
            param_groups,
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")

    # lr scheduler with optional warmup
    epochs_total = int(config["training"]["epochs"])
    lr_scheduler = build_main_scheduler(
        optimizer=optim,
        scheduler_name=config["lr_scheduler"]["name"],
        config=config,
        epochs_total=epochs_total
    )
    
    best_metric = 0.0

    # resume training if checkpoint is provided
    resume_cfg = config["resume"]
    if resume_cfg.get("checkpoint_path") is not None:
        checkpoint = torch.load(resume_cfg["checkpoint_path"])
        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optim_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0
    
    best_epoch = 0
    logger.info(f"Starting Training model: {config.model.name} with backbone: {config.model.backbone_name} for {epochs_total} epochs 🚀")
    for epoch in range(start_epoch, epochs_total):
        train_one_epoch(
            model=model,
            dataloader=train_loader,
            optim=optim,
            sched=lr_scheduler,
            config=config,
            device=device,
            epoch=epoch,
            file_logger=file_logger,
        )

        # validation after every epoch
        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            config=config,
            device=device,
            file_logger=file_logger,
            mode="val",
            epoch=epoch,
        )

        # add checkpoint saving
        cur_val_metric = val_metrics["overall_mAP"][config.model.val_ckpt_key]
        if cur_val_metric > best_metric:
            best_metric = cur_val_metric
            best_epoch = epoch
            is_best = True
        else:
            is_best = False
        
        if config.model.val_ckpt_patience > 0:
            if epoch - best_epoch >= config.model.val_ckpt_patience:
                logger.info(f"Early stopping triggered at epoch {epoch} with best metric {best_metric} at epoch {best_epoch}")
                break

        if is_best:
            st1 = f"New best metric {round(best_metric, 3)} at epoch {best_epoch}! Saving checkpoint... 💾 "
            file_logger.info(st1)
            print(st1)
            
            best_checkpoint_file = os.path.join(ckpt_dir, f"{base}_best.pth")
            torch.save({
                "epoch": best_epoch,
                "state_dict": model.state_dict(),
                "optim": optim.state_dict() if not isinstance(optim, list) else [optim.state_dict() for optim in optim],
                "sched": lr_scheduler.state_dict() if not isinstance(lr_scheduler, list) else [scheduler.state_dict() for scheduler in lr_scheduler],
                "best_metric": {config.model.val_ckpt_key: best_metric},
            }, best_checkpoint_file)

    logger.info(f"Training completed! Best metric {best_metric} at epoch {best_epoch} 🎉")
    logger.info("Evaluating on test set... 🧪")
    best_checkpoint_file = os.path.join(ckpt_dir, f"{base}_best.pth")
    logger.info(f"Loading best checkpoint... from {best_checkpoint_file} 💾 ")

    if torch.__version__ > '2.6':
        best_checkpoint = torch.load(best_checkpoint_file, map_location='cpu', weights_only=False)
    else:
        best_checkpoint = torch.load(best_checkpoint_file, map_location='cpu')

    model.load_state_dict(best_checkpoint['state_dict'])
    best_epoch = best_checkpoint['epoch']

    logger.info(f"Evaluating on test set at epoch {best_epoch} with best metric {best_metric} 🧪")
    test_metrics = validate_one_epoch(
        model=model,
        dataloader=test_loader,
        config=config,
        device=device,
        file_logger=file_logger,
        mode="test",
        epoch=best_epoch,
    )
    logger.info(f"Test Eval completed! 🎉")

    # if config.dataset.setting == "challenge":
    #     hidden_test_dataset = get_datasets(config, split="hidden_test")
    #     logger.info(f"Hidden Test dataset size: {len(hidden_test_dataset)}")
    #     hidden_test_loader = torch.utils.data.DataLoader(
    #                                     hidden_test_dataset, 
    #                                     batch_size=config["dataset"]["batch_size"], 
    #                                     shuffle=False, 
    #                                     num_workers=config["dataset"]["num_workers"],
    #                                     pin_memory=True)

    #     logger.info(f"Evaluating on hidden test set at epoch {best_epoch} with best metric {best_metric} 🧪")
    #     hidden_test_metrics = validate_one_epoch(
    #         model=model,
    #         dataloader=hidden_test_loader,
    #         config=config,
    #         device=device,
    #         file_logger=file_logger,
    #         mode="hidden_test",
    #         epoch=best_epoch,
    #     )
    #     logger.info(f"Hidden Test Eval completed! 🎉")    

if __name__ == "__main__":
    main()
