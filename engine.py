from __future__ import annotations
from typing import Callable, Dict, Iterable, Any
import torch
import time
from utils.helpers import (AverageMeter, get_class_weights, mAP, get_progress_bar, 
                           backward_step_single_optim)
from utils.metric_collater import compute_triplet_metrics, format_overall_metrics_ascii
from utils.triplet_mappings import triplet_maps

# turn off warnings from torchmetrics
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
LOGGER = logging.getLogger(__name__)

def train_one_epoch(
    model: torch.nn.Module, 
    dataloader: Iterable, 
    optim: torch.optim.Optimizer, 
    sched: torch.optim.lr_scheduler._LRScheduler,
    config: Dict[str, Any],
    device: torch.device, 
    epoch: int,
    file_logger: logging.Logger,
    ) -> Dict[str, float]:
    
    # set up modules to train mode
    model.train()
    
    component_fcs = config.model.apply_fc.split(",")

    cls_wt_i, cls_wt_v, cls_wt_t, cls_wt_ivt = get_class_weights(config)
    cls_wt_i = cls_wt_i.to(device=device, dtype=torch.float32)
    cls_wt_v = cls_wt_v.to(device=device, dtype=torch.float32)
    cls_wt_t = cls_wt_t.to(device=device, dtype=torch.float32)
    cls_wt_ivt = cls_wt_ivt.to(device=device, dtype=torch.float32)

    criteria = {}
    fc_to_weight = {"i": cls_wt_i, "v": cls_wt_v, "t": cls_wt_t, "ivt": cls_wt_ivt}
    for fc in ["i", "v", "t", "ivt"]:
        criteria[fc] = torch.nn.BCEWithLogitsLoss(pos_weight=fc_to_weight[fc])

    # init trackers for training logs
    loss_trackers = {
        "loss": AverageMeter(),
        "loss_i": AverageMeter(),
        "loss_v": AverageMeter(),
        "loss_t": AverageMeter(),
        "loss_ivt": AverageMeter(),
    }
    batch_time = AverageMeter()
    grad_trackers = {
        "norm": AverageMeter(),
    }
    map_trackers = {
        "mAP_i": AverageMeter(),
        "mAP_v": AverageMeter(),
        "mAP_t": AverageMeter(),
        "mAP_ivt": AverageMeter(),
    }

    file_logger.info(f"Training on device: {device} 📈")
    scaler = None # Initialize gradient scaler for mixed precision training
    if not config.disable_autocast and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        file_logger.info("Using gradient scaler for mixed precision training")
    
    # train loop
    epoch_start_time = time.time()
    start_time = epoch_start_time
    with get_progress_bar(dataloader, f"train-ep[{epoch+1}/{config.training.epochs}]") as pbar:
        for batch_idx, batch in pbar:
            images = batch['image'].to(device)
            
            # flatten the time dimension
            b, t, c, h, w = images.shape
            images = images.view(b * t, c, h, w)

            # labels only correspond to the key frame (end or center)
            label_i = batch['label_i'].to(device)
            label_v = batch['label_v'].to(device)
            label_t = batch['label_t'].to(device)
            label_ivt = batch['label_ivt'].to(device)

            # forward pass
            if config.disable_autocast:
                outputs = model(images)
            else:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(images)
            
            if t > 1:
                if outputs['logit_ivt'].shape[0] == b * t:
                    outputs['logit_i'] = outputs['logit_i'].view(b, t, -1)[:, -1, :]
                    outputs['logit_v'] = outputs['logit_v'].view(b, t, -1)[:, -1, :]
                    outputs['logit_t'] = outputs['logit_t'].view(b, t, -1)[:, -1, :]
                    outputs['logit_ivt'] = outputs['logit_ivt'].view(b, t, -1)[:, -1, :]

            # compute loss
            loss_i, loss_v, loss_t, loss_ivt = 0.0, 0.0, 0.0, 0.0
            batch_size = outputs['logit_ivt'].shape[0]
            loss_i = criteria["i"](outputs['logit_i'], label_i.float())
            loss_trackers["loss_i"].update(loss_i.item(), n=batch_size)
            loss_v = criteria["v"](outputs['logit_v'], label_v.float())
            loss_trackers["loss_v"].update(loss_v.item(), n=batch_size)
            loss_t = criteria["t"](outputs['logit_t'], label_t.float())
            loss_trackers["loss_t"].update(loss_t.item(), n=batch_size)
            loss_ivt = criteria["ivt"](outputs['logit_ivt'], label_ivt.float())
            loss_trackers["loss_ivt"].update(loss_ivt.item(), n=batch_size)
            loss = loss_i + loss_v + loss_t + loss_ivt

            # if the loss is nan and kill the process
            if torch.isnan(loss):
                raise ValueError(f"Loss is nan at batch {batch_idx} in epoch {epoch}")

            # track total loss
            loss_trackers["loss"].update(loss.item(), n=batch_size)

            # backward pass (single optimizer path)
            grad_norm = backward_step_single_optim(loss, model, optim, scaler, config.optim.grad_clip_norm)
            if grad_norm is not None:
                grad_trackers["norm"].update(grad_norm, n=1)

            model.zero_grad()

            # collect loss values for training logs
            batch_time.update(time.time() - start_time, n=1)

            # compute simple mAP values for training logs
            if "i" in component_fcs:
                mAP_i = mAP(outputs['logit_i'], label_i)
                map_trackers["mAP_i"].update(mAP_i, n=1)
            if "v" in component_fcs:
                mAP_v = mAP(outputs['logit_v'], label_v)
                map_trackers["mAP_v"].update(mAP_v, n=1)
            if "t" in component_fcs:
                mAP_t = mAP(outputs['logit_t'], label_t)
                map_trackers["mAP_t"].update(mAP_t, n=1)
            if "ivt" in component_fcs:
                mAP_ivt = mAP(outputs['logit_ivt'], label_ivt)
                map_trackers["mAP_ivt"].update(mAP_ivt, n=1)

            # update progress bar with compact metrics
            loss_parts = [f"{loss_trackers[f'loss_{fc}'].avg:.3f}" 
                          for fc in component_fcs if f'loss_{fc}' in loss_trackers]
            map_parts = [f"{map_trackers[f'mAP_{fc}'].avg:.3f}" 
                         for fc in component_fcs if f'mAP_{fc}' in map_trackers]

            loss_detail = "/".join(loss_parts) if loss_parts else "n/a"
            map_detail = "/".join(map_parts) if map_parts else "n/a"

            log_str = (
                f"t:{batch_time.avg:.2f} "
                f"L:{loss_trackers['loss'].avg:.3f}({loss_detail}) "
                f"mAP:{map_detail} "
                f"G:{grad_trackers['norm'].avg:.1f}"
            )
            pbar.set_postfix_str(log_str, refresh=False)

            # update start time
            start_time = time.time()

    # Print final epoch summary before progress bar clears
    loss_summary_final = [f"{fc}:{loss_trackers[f'loss_{fc}'].avg:.3f}" 
                          for fc in component_fcs if f'loss_{fc}' in loss_trackers]
    map_summary_final = [f"{fc}:{map_trackers[f'mAP_{fc}'].avg:.3f}" 
                         for fc in component_fcs if f'mAP_{fc}' in map_trackers]
    loss_detail_final = ", ".join(loss_summary_final) if loss_summary_final else "n/a"
    map_detail_final = ", ".join(map_summary_final) if map_summary_final else "n/a"
    print(f"Ep [{epoch+1}/{config.training.epochs}] | L:{loss_trackers['loss'].avg:.3f}({loss_detail_final}) | mAP:{map_detail_final} | G:{grad_trackers['norm'].avg:.1f}")

    # update scheduler
    sched.step()

    epoch_time = time.time() - epoch_start_time
    
    # Build summary strings efficiently
    loss_summary = [f"{fc}:{loss_trackers[f'loss_{fc}'].avg:.3f}" 
                    for fc in component_fcs if f'loss_{fc}' in loss_trackers]
    map_summary = [f"{fc}:{map_trackers[f'mAP_{fc}'].avg:.3f}" 
                   for fc in component_fcs if f'mAP_{fc}' in map_trackers]
    
    loss_summary_str = ", ".join(loss_summary) if loss_summary else "n/a"
    map_summary_str = ", ".join(map_summary) if map_summary else "n/a"

    epoch_log = (
        f"Ep [{epoch+1}/{config.training.epochs}] | time:{epoch_time:.2f}s "
        f"[L] total:{loss_trackers['loss'].avg:.3f}, {loss_summary_str} "
        f"[mAP]{map_summary_str} "
        f"[Grad]norm:{grad_trackers['norm'].avg:.3f}"
    )
    file_logger.info(epoch_log)

def validate_one_epoch(
    model: torch.nn.Module, 
    dataloader: Iterable, 
    config: Dict[str, Any],
    device: torch.device, 
    file_logger: logging.Logger,
    mode: str = "val",
    epoch: int = 0,
    ) -> Dict[str, float]:

    save_predictions = config.save_predictions

    # set up modules to evaluation mode
    model.eval()

    targets_ivt = []
    preds_ivt = []
    video_list = []
    frame_keys_list = []

    # forward pass through all eval videos 
    with torch.no_grad():
        with get_progress_bar(dataloader, mode) as pbar:
            for _, batch in pbar:
                images = batch['image'].to(device)
                
                b, t, c, h, w = images.shape
                images = images.view(b * t, c, h, w)

                label_ivt = batch['label_ivt'].to(device)
                video_ids = batch['video_id']
                frame_keys = batch['frame_key']

                # forward pass
                if config.disable_autocast:
                    outputs = model(images)
                else:
                    with torch.amp.autocast(device_type=device.type):
                        outputs = model(images)

                pred_ivt = torch.sigmoid(outputs['logit_ivt']).detach().cpu()

                preds_ivt.append(pred_ivt)
                targets_ivt.append(label_ivt.detach().cpu())
                video_list.extend(video_ids)
                frame_keys_list.extend(frame_keys)

    preds_ivt = torch.cat(preds_ivt, dim=0)
    targets_ivt = torch.cat(targets_ivt, dim=0).long()

    results = compute_triplet_metrics(
        preds_ivt,
        targets_ivt,
        video_list,
        num_classes=config.model.num_triplet_classes,
        dataset_name=config.dataset.name,
        ignore_null_labels=config.model.ignore_null_labels,
        get_per_center=config.eval.get('per_center', False) # NOTE: NEW addded
    )
    print("\n" + format_overall_metrics_ascii(results, mode=mode, epoch=epoch) + "\n")
    
    if mode == 'test' or mode == 'hidden_test':
        dataset_name = config.dataset.name
        class_mappings = triplet_maps[dataset_name]
        
        # display 
        header_line = "=" * 110
        print(f"\n{header_line}")
        # add videowise average mAP results (single line summary)
        videowise_summary = " | ".join(
            f"{component.upper()}: {results['videowise_mAP'][component]:.3f}"
            for component in ['i', 'v', 't', 'ivt']
        )
        print(f"VIDEOWISE AVERAGE mAP RESULTS: [{mode}] at epoch [{epoch+1}/{config.training.epochs}] | {videowise_summary}")
        print(header_line)
        file_logger.info(header_line)
        file_logger.info(
            "VIDEOWISE AVERAGE mAP RESULTS: [%s] at epoch [%d/%d] | %s",
            mode,
            epoch + 1,
            config.training.epochs,
            videowise_summary,
        )
        file_logger.info(header_line)

        # add classwise average mAP results
        print(f"CLASSWISE mAP RESULTS: [{mode}] at epoch [{epoch+1}/{config.training.epochs}]")
        print(f"{header_line}")
        file_logger.info(header_line)
        file_logger.info("CLASSWISE mAP RESULTS: [%s] at epoch [%d/%d]", mode, epoch + 1, config.training.epochs)
        file_logger.info(header_line)
        
        for component in ['i', 'v', 't', 'ivt']:
            component_names = class_mappings[component]
            component_scores = results['overall_mAP'][component + '_per_class']
            sorted_items = sorted(component_names.items(), key=lambda x: int(x[0]))
            
            section_sep = "-" * 110
            print(f"\n{section_sep}")
            print(f"Component: {component.upper()}")
            print(f"{section_sep}")
            print(f"{'Class ID':<10} {'Class Name':<55} {'mAP':>10}")
            print(f"{section_sep}")

            file_logger.info(section_sep)
            file_logger.info("Component: %s", component.upper())
            file_logger.info(section_sep)
            file_logger.info("%-10s %-55s %10s", "Class ID", "Class Name", "mAP")
            file_logger.info(section_sep)
            
            for class_id, class_name in sorted_items:
                idx = int(class_id)
                if idx < len(component_scores):
                    score = component_scores[idx]
                    row_str = f"{class_id:<10} {class_name:<55} {score:>10.3f}"
                    print(row_str)
                    file_logger.info(row_str)
            
            # Print mean row for this component
            mean_score = results['overall_mAP'][component]
            print(section_sep)
            mean_row = f"{'---':<10} {'MEAN':<55} {mean_score:>10.3f}"
            print(mean_row)
            file_logger.info(section_sep)
            file_logger.info(mean_row)
        
        print(f"\n{header_line}\n")
        file_logger.info(header_line)
    return results