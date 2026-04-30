import torch
import numpy as np
import warnings
import re
from typing import List, Dict, Tuple
from torchmetrics.functional.classification import (multilabel_average_precision, multilabel_f1_score, multilabel_precision,
                                                   multilabel_recall, multilabel_accuracy) 
from utils.triplet_mappings import triplet_maps

COMPONENT_MAP = {'ivt': 0, 'i': 1, 'v': 2, 't': 3, 'iv': 4, 'it': 5, 'vt': 6}

def extract_component_data(
    input_tensor: torch.Tensor, 
    component: str, 
    dataset_name: str,
) -> torch.Tensor:
    """
    Extract component-specific data by aggregating triplet predictions.
    
    Args:
        input_tensor: Input tensor with triplet predictions/labels
        component: Component name ('i', 'v', 't', 'iv', 'it', 'ivt')
        dataset_name: Name of the dataset
        
    Returns:
        Aggregated tensor for the component
    """
    component_key = COMPONENT_MAP[component]
    component_maps = torch.tensor(triplet_maps[dataset_name]["component_maps"])[:, component_key]
        
    unique_indices = torch.unique(component_maps).sort()[0]
    
    component_data = []
    for idx in unique_indices:
        mask = (component_maps == idx)
        try:
            component_data.append(input_tensor[:, mask].max(dim=1)[0])
        except Exception as e:
            raise RuntimeError(
                f"extract_component_data failed for component='{component}', "
                f"dataset='{dataset_name}', idx={int(idx.item())}, "
                f"input_shape={tuple(input_tensor.shape)}"
            ) from e

    if component_data:
        return torch.stack(component_data, dim=1)
    return torch.empty(
        (input_tensor.shape[0], 0),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )

def _compute_component_map(
    preds: torch.Tensor,
    labels: torch.Tensor,
    component: str,
    dataset_name: str,
    num_classes: int,
    skip_filtering: bool = False
) -> torch.Tensor:
    """Helper function to compute mAP for a specific component."""

    if not skip_filtering:
        if component == 'ivt':
            comp_preds, comp_labels = preds, labels
        else:
            comp_preds = extract_component_data(preds, component, dataset_name)
            comp_labels = extract_component_data(labels, component, dataset_name)
    else:
        comp_preds = preds
        comp_labels = labels
    
    # Check for empty inputs
    if comp_preds.shape[0] == 0 or comp_labels.shape[0] == 0:
        nan_result = torch.tensor(float('nan'), device=comp_preds.device)
        return nan_result, torch.full((comp_preds.shape[1],), float('nan'), device=comp_preds.device)
    
    num_classes = comp_preds.shape[1]
    classwise_map = multilabel_average_precision(
        comp_preds, comp_labels, num_labels=num_classes, average='none'
    )
    
    has_positives = comp_labels.sum(dim=0) > 0  # shape: (num_classes,)
    empty_mask = ~has_positives
    if empty_mask.any():
        # Only set to NaN if the value is not already NaN and the class has no positives
        # This way we preserve NaN values that torchmetrics set for other reasons
        already_nan = torch.isnan(classwise_map)
        needs_nan = empty_mask & ~already_nan  # Empty classes that aren't already NaN
        if needs_nan.any():
            nan_tensor = torch.full_like(classwise_map, float('nan'))
            classwise_map = torch.where(needs_nan, nan_tensor, classwise_map)

    # Compute mean ignoring NaN values (empty classes)
    # Only compute mean if we have at least one valid (non-NaN) value
    valid_mask = ~torch.isnan(classwise_map)
    if valid_mask.any():
        mean_map = classwise_map[valid_mask].mean()
    else:
        # All classes are empty - return NaN
        mean_map = torch.tensor(float('nan'), device=classwise_map.device, dtype=classwise_map.dtype)
    
    return mean_map, classwise_map


def _compute_component_f1(
    preds: torch.Tensor,
    labels: torch.Tensor,
    component: str,
    dataset_name: str,
    threshold: float,
    skip_filtering: bool = False
) -> torch.Tensor:
    """Helper function to compute F1 for a specific component at a given threshold."""

    if not skip_filtering:
        if component == 'ivt':
            comp_preds, comp_labels = preds, labels
        else:
            comp_preds = extract_component_data(preds, component, dataset_name)
            comp_labels = extract_component_data(labels, component, dataset_name)
    else:
        comp_preds = preds
        comp_labels = labels

    # Check for empty inputs
    if comp_preds.shape[0] == 0 or comp_labels.shape[0] == 0:
        nan_result = torch.tensor(float('nan'), device=comp_preds.device)
        return nan_result, torch.full((comp_preds.shape[1],), float('nan'), device=comp_preds.device)

    # Per-class F1 for multilabel data at the selected threshold
    classwise_f1 = multilabel_f1_score(
        comp_preds,
        comp_labels.long(),
        num_labels=comp_preds.shape[1],
        threshold=threshold,
        average='none',
    )

    # Set classes with no positives to NaN for fair macro averaging
    has_positives = comp_labels.sum(dim=0) > 0
    empty_mask = ~has_positives
    if empty_mask.any():
        already_nan = torch.isnan(classwise_f1)
        needs_nan = empty_mask & ~already_nan
        if needs_nan.any():
            nan_tensor = torch.full_like(classwise_f1, float('nan'))
            classwise_f1 = torch.where(needs_nan, nan_tensor, classwise_f1)

    valid_mask = ~torch.isnan(classwise_f1)
    if valid_mask.any():
        mean_f1 = classwise_f1[valid_mask].mean()
    else:
        mean_f1 = torch.tensor(float('nan'), device=classwise_f1.device, dtype=classwise_f1.dtype)

    return mean_f1, classwise_f1


def _compute_component_f1_at_k(
    preds: torch.Tensor,
    labels: torch.Tensor,
    component: str,
    dataset_name: str,
    topk: int,
    skip_filtering: bool = False
) -> torch.Tensor:
    """Helper function to compute F1 for a specific component using top-k predictions per frame."""

    if not skip_filtering:
        if component == 'ivt':
            comp_preds, comp_labels = preds, labels
        else:
            comp_preds = extract_component_data(preds, component, dataset_name)
            comp_labels = extract_component_data(labels, component, dataset_name)
    else:
        comp_preds = preds
        comp_labels = labels

    if comp_preds.shape[0] == 0 or comp_labels.shape[0] == 0:
        nan_result = torch.tensor(float('nan'), device=comp_preds.device)
        return nan_result, torch.full((comp_preds.shape[1],), float('nan'), device=comp_preds.device)

    n_samples, n_classes = comp_preds.shape
    k = int(max(1, min(int(topk), int(n_classes))))

    # Convert scores to binary top-k predictions per sample
    topk_idx = torch.topk(comp_preds, k=k, dim=1, largest=True, sorted=False).indices
    pred_bin = torch.zeros_like(comp_preds)
    pred_bin.scatter_(1, topk_idx, 1.0)

    classwise_f1 = multilabel_f1_score(
        pred_bin,
        comp_labels.long(),
        num_labels=n_classes,
        threshold=0.5,
        average='none',
    )

    has_positives = comp_labels.sum(dim=0) > 0
    empty_mask = ~has_positives
    if empty_mask.any():
        already_nan = torch.isnan(classwise_f1)
        needs_nan = empty_mask & ~already_nan
        if needs_nan.any():
            nan_tensor = torch.full_like(classwise_f1, float('nan'))
            classwise_f1 = torch.where(needs_nan, nan_tensor, classwise_f1)

    valid_mask = ~torch.isnan(classwise_f1)
    if valid_mask.any():
        mean_f1 = classwise_f1[valid_mask].mean()
    else:
        mean_f1 = torch.tensor(float('nan'), device=classwise_f1.device, dtype=classwise_f1.dtype)

    return mean_f1, classwise_f1


def _compute_component_precision_recall_at_k(
    preds: torch.Tensor,
    labels: torch.Tensor,
    component: str,
    dataset_name: str,
    topk: int,
    skip_filtering: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute classwise/mean precision and recall at top-k predictions per sample."""

    if not skip_filtering:
        if component == 'ivt':
            comp_preds, comp_labels = preds, labels
        else:
            comp_preds = extract_component_data(preds, component, dataset_name)
            comp_labels = extract_component_data(labels, component, dataset_name)
    else:
        comp_preds = preds
        comp_labels = labels

    if comp_preds.shape[0] == 0 or comp_labels.shape[0] == 0:
        nan_result = torch.tensor(float('nan'), device=comp_preds.device)
        nan_vec = torch.full((comp_preds.shape[1],), float('nan'), device=comp_preds.device)
        return nan_result, nan_vec, nan_result, nan_vec

    n_samples, n_classes = comp_preds.shape
    k = int(max(1, min(int(topk), int(n_classes))))

    topk_idx = torch.topk(comp_preds, k=k, dim=1, largest=True, sorted=False).indices
    pred_bin = torch.zeros_like(comp_preds)
    pred_bin.scatter_(1, topk_idx, 1.0)

    classwise_precision = multilabel_precision(
        pred_bin,
        comp_labels.long(),
        num_labels=n_classes,
        threshold=0.5,
        average='none',
    )
    classwise_recall = multilabel_recall(
        pred_bin,
        comp_labels.long(),
        num_labels=n_classes,
        threshold=0.5,
        average='none',
    )

    has_positives = comp_labels.sum(dim=0) > 0
    empty_mask = ~has_positives
    if empty_mask.any():
        for metric in ['precision', 'recall']:
            arr = classwise_precision if metric == 'precision' else classwise_recall
            already_nan = torch.isnan(arr)
            needs_nan = empty_mask & ~already_nan
            if needs_nan.any():
                nan_tensor = torch.full_like(arr, float('nan'))
                arr = torch.where(needs_nan, nan_tensor, arr)
            if metric == 'precision':
                classwise_precision = arr
            else:
                classwise_recall = arr

    valid_p = ~torch.isnan(classwise_precision)
    mean_p = classwise_precision[valid_p].mean() if valid_p.any() else torch.tensor(float('nan'), device=classwise_precision.device, dtype=classwise_precision.dtype)

    valid_r = ~torch.isnan(classwise_recall)
    mean_r = classwise_recall[valid_r].mean() if valid_r.any() else torch.tensor(float('nan'), device=classwise_recall.device, dtype=classwise_recall.dtype)

    return mean_p, classwise_precision, mean_r, classwise_recall


def _compute_component_accuracy_at_k(
    preds: torch.Tensor,
    labels: torch.Tensor,
    component: str,
    dataset_name: str,
    topk: int,
    skip_filtering: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute classwise/mean multilabel accuracy at top-k predictions per sample."""

    if not skip_filtering:
        if component == 'ivt':
            comp_preds, comp_labels = preds, labels
        else:
            comp_preds = extract_component_data(preds, component, dataset_name)
            comp_labels = extract_component_data(labels, component, dataset_name)
    else:
        comp_preds = preds
        comp_labels = labels

    if comp_preds.shape[0] == 0 or comp_labels.shape[0] == 0:
        nan_result = torch.tensor(float('nan'), device=comp_preds.device)
        nan_vec = torch.full((comp_preds.shape[1],), float('nan'), device=comp_preds.device)
        return nan_result, nan_vec

    _, n_classes = comp_preds.shape
    k = int(max(1, min(int(topk), int(n_classes))))

    topk_idx = torch.topk(comp_preds, k=k, dim=1, largest=True, sorted=False).indices
    pred_bin = torch.zeros_like(comp_preds)
    pred_bin.scatter_(1, topk_idx, 1.0)

    classwise_acc = multilabel_accuracy(
        pred_bin,
        comp_labels.long(),
        num_labels=n_classes,
        threshold=0.5,
        average='none',
    )

    valid_a = ~torch.isnan(classwise_acc)
    mean_a = classwise_acc[valid_a].mean() if valid_a.any() else torch.tensor(float('nan'), device=classwise_acc.device, dtype=classwise_acc.dtype)

    return mean_a, classwise_acc


def _compute_component_hit_at_k(
    preds: torch.Tensor,
    labels: torch.Tensor,
    component: str,
    dataset_name: str,
    topk: int,
    skip_filtering: bool = False,
) -> torch.Tensor:
    """Compute sample-level top-k hit rate: any GT positive is inside top-k predictions."""

    if not skip_filtering:
        if component == 'ivt':
            comp_preds, comp_labels = preds, labels
        else:
            comp_preds = extract_component_data(preds, component, dataset_name)
            comp_labels = extract_component_data(labels, component, dataset_name)
    else:
        comp_preds = preds
        comp_labels = labels

    if comp_preds.shape[0] == 0 or comp_labels.shape[0] == 0:
        return torch.tensor(float('nan'), device=comp_preds.device)

    _, n_classes = comp_preds.shape
    k = int(max(1, min(int(topk), int(n_classes))))

    topk_idx = torch.topk(comp_preds, k=k, dim=1, largest=True, sorted=False).indices
    pred_bin = torch.zeros_like(comp_preds)
    pred_bin.scatter_(1, topk_idx, 1.0)

    gt_bin = comp_labels > 0
    valid_mask = gt_bin.sum(dim=1) > 0  # samples with at least one positive label
    if not valid_mask.any():
        return torch.tensor(float('nan'), device=comp_preds.device, dtype=comp_preds.dtype)

    hits = ((pred_bin > 0) & gt_bin).any(dim=1).float()
    return hits[valid_mask].mean()


def _extract_center(video_id: str) -> str:
    """
    Extract center id from video names like C1Vx, C2Vx, C3Vx, C4Vx.
    Returns '' when not matched.
    """
    m = re.match(r"^(C[1-4])V", str(video_id))
    return m.group(1) if m else ""


def _compute_metrics_for_video_subset(
    preds: torch.Tensor,
    labels: torch.Tensor,
    video_ids: List[str],
    selected_video_ids: List[str],
    num_classes: int,
    dataset_name: str,
    ignore_null_labels: bool,
    f1_thresholds: List[float],
    f1_topk_values: List[int] | None = None,
) -> Dict[str, float]:
    components = ['ivt', 'i', 'v', 't', 'iv', 'it']
    subset_results: Dict[str, Dict] = {}

    # Per-video metrics
    per_video_map = {}
    unique_video_ids = sorted(set(selected_video_ids))

    for video_id in unique_video_ids:
        video_mask = torch.tensor([vid == video_id for vid in video_ids])
        video_preds = preds[video_mask]
        video_labels = labels[video_mask]

        video_metrics = {}
        for component in components:
            component_map, classwise_map = _compute_component_map(
                video_preds, video_labels, component, dataset_name, num_classes
            )
            video_metrics[component] = round(component_map.detach().cpu().item(), 3)

            per_class_array = classwise_map.detach().cpu().numpy().round(3)
            per_class_array = np.where(np.signbit(per_class_array) & (per_class_array == 0.0), 0.0, per_class_array)
            video_metrics[component + '_per_class'] = per_class_array

        per_video_map[video_id] = video_metrics

    subset_results['videowise'] = per_video_map

    subset_results['videowise_mAP'] = {}
    videowise_metrics = {'i': [], 'v': [], 't': [], 'iv': [], 'it': [], 'ivt': []}
    for video_id in unique_video_ids:
        video_metrics = subset_results['videowise'][video_id]
        for component in components:
            videowise_metrics[component].append(video_metrics[component])

    for component in components:
        if len(videowise_metrics[component]) == 0:
            subset_results['videowise_mAP'][component] = float("nan")
        else:
            videowise_metrics[component] = np.nanmean(videowise_metrics[component], axis=0)
            subset_results['videowise_mAP'][component] = round(float(videowise_metrics[component]), 3)

    subset_results['overall_mAP'] = {}
    for component in components:
        classwise_mAPs = [per_video_map[video_id][component + '_per_class'] for video_id in unique_video_ids]
        if len(classwise_mAPs) == 0:
            continue

        classwise_mAPs = np.stack(classwise_mAPs, axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
            videowise_mAP = np.nanmean(classwise_mAPs, axis=0)
        effective_videowise_mAP = videowise_mAP
        if component == 'ivt' and ignore_null_labels:
            if dataset_name == 'cholect50':
                effective_videowise_mAP = videowise_mAP[:-6]
            elif dataset_name == 'multibypasst40':
                effective_videowise_mAP = videowise_mAP[:-10]

        subset_results['overall_mAP'][component + '_per_class'] = effective_videowise_mAP
        subset_results['overall_mAP'][component] = round(np.nanmean(effective_videowise_mAP), 3)

    # F1 at threshold(s)
    subset_results['videowise_F1'] = {}
    subset_results['overall_F1'] = {}
    for threshold in f1_thresholds:
        threshold_key = f"@{threshold}"
        per_video_f1 = {}

        for video_id in unique_video_ids:
            video_mask = torch.tensor([vid == video_id for vid in video_ids])
            video_preds = preds[video_mask]
            video_labels = labels[video_mask]

            video_f1_metrics = {}
            for component in components:
                component_f1, classwise_f1 = _compute_component_f1(
                    video_preds, video_labels, component, dataset_name, threshold
                )
                video_f1_metrics[component] = round(component_f1.detach().cpu().item(), 3)

                per_class_array = classwise_f1.detach().cpu().numpy().round(3)
                per_class_array = np.where(np.signbit(per_class_array) & (per_class_array == 0.0), 0.0, per_class_array)
                video_f1_metrics[component + '_per_class'] = per_class_array

            per_video_f1[video_id] = video_f1_metrics

        subset_results['videowise_F1'][threshold_key] = {}
        f1_videowise_metrics = {'i': [], 'v': [], 't': [], 'iv': [], 'it': [], 'ivt': []}
        for video_id in unique_video_ids:
            video_metrics = per_video_f1[video_id]
            for component in components:
                f1_videowise_metrics[component].append(video_metrics[component])

        for component in components:
            if len(f1_videowise_metrics[component]) == 0:
                subset_results['videowise_F1'][threshold_key][component] = float("nan")
            else:
                f1_videowise_metrics[component] = np.nanmean(f1_videowise_metrics[component], axis=0)
                subset_results['videowise_F1'][threshold_key][component] = round(float(f1_videowise_metrics[component]), 3)

        subset_results['overall_F1'][threshold_key] = {}
        for component in components:
            classwise_f1s = [per_video_f1[video_id][component + '_per_class'] for video_id in unique_video_ids]
            if len(classwise_f1s) == 0:
                continue
            classwise_f1s = np.stack(classwise_f1s, axis=0)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
                videowise_f1 = np.nanmean(classwise_f1s, axis=0)
            effective_videowise_f1 = videowise_f1

            if component == 'ivt' and ignore_null_labels:
                if dataset_name == 'cholect50':
                    effective_videowise_f1 = videowise_f1[:-6]
                elif dataset_name == 'multibypasst40':
                    effective_videowise_f1 = videowise_f1[:-10]

            subset_results['overall_F1'][threshold_key][component + '_per_class'] = effective_videowise_f1
            subset_results['overall_F1'][threshold_key][component] = round(np.nanmean(effective_videowise_f1), 3)

    # F1 at top-k(s): keeps existing F1 structure intact and adds complementary ranking-based F1
    topk_values = [5, 10, 20] if f1_topk_values is None else [int(k) for k in f1_topk_values]
    subset_results['videowise_F1_at_k'] = {}
    subset_results['overall_F1_at_k'] = {}
    for k in topk_values:
        k_key = f"@{k}"
        per_video_f1k = {}

        for video_id in unique_video_ids:
            video_mask = torch.tensor([vid == video_id for vid in video_ids])
            video_preds = preds[video_mask]
            video_labels = labels[video_mask]

            video_f1_metrics = {}
            for component in components:
                component_f1, classwise_f1 = _compute_component_f1_at_k(
                    video_preds, video_labels, component, dataset_name, topk=k
                )
                video_f1_metrics[component] = round(component_f1.detach().cpu().item(), 3)

                per_class_array = classwise_f1.detach().cpu().numpy().round(3)
                per_class_array = np.where(np.signbit(per_class_array) & (per_class_array == 0.0), 0.0, per_class_array)
                video_f1_metrics[component + '_per_class'] = per_class_array

            per_video_f1k[video_id] = video_f1_metrics

        subset_results['videowise_F1_at_k'][k_key] = {}
        f1k_videowise_metrics = {'i': [], 'v': [], 't': [], 'iv': [], 'it': [], 'ivt': []}
        for video_id in unique_video_ids:
            video_metrics = per_video_f1k[video_id]
            for component in components:
                f1k_videowise_metrics[component].append(video_metrics[component])

        for component in components:
            if len(f1k_videowise_metrics[component]) == 0:
                subset_results['videowise_F1_at_k'][k_key][component] = float("nan")
            else:
                f1k_videowise_metrics[component] = np.nanmean(f1k_videowise_metrics[component], axis=0)
                subset_results['videowise_F1_at_k'][k_key][component] = round(float(f1k_videowise_metrics[component]), 3)

        subset_results['overall_F1_at_k'][k_key] = {}
        for component in components:
            classwise_f1s = [per_video_f1k[video_id][component + '_per_class'] for video_id in unique_video_ids]
            if len(classwise_f1s) == 0:
                continue
            classwise_f1s = np.stack(classwise_f1s, axis=0)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
                videowise_f1 = np.nanmean(classwise_f1s, axis=0)
            effective_videowise_f1 = videowise_f1

            if component == 'ivt' and ignore_null_labels:
                if dataset_name == 'cholect50':
                    effective_videowise_f1 = videowise_f1[:-6]
                elif dataset_name == 'multibypasst40':
                    effective_videowise_f1 = videowise_f1[:-10]

            subset_results['overall_F1_at_k'][k_key][component + '_per_class'] = effective_videowise_f1
            subset_results['overall_F1_at_k'][k_key][component] = round(np.nanmean(effective_videowise_f1), 3)

    # Precision/Recall at top-k(s): additive metrics on top of existing mAP/F1 blocks
    subset_results['videowise_P_at_k'] = {}
    subset_results['overall_P_at_k'] = {}
    subset_results['videowise_R_at_k'] = {}
    subset_results['overall_R_at_k'] = {}
    subset_results['videowise_ACC_at_k'] = {}
    subset_results['overall_ACC_at_k'] = {}
    subset_results['videowise_HIT_at_k'] = {}
    subset_results['overall_HIT_at_k'] = {}

    for k in topk_values:
        k_key = f"@{k}"
        per_video_prk = {}
        per_video_acck = {}
        per_video_hitk = {}

        for video_id in unique_video_ids:
            video_mask = torch.tensor([vid == video_id for vid in video_ids])
            video_preds = preds[video_mask]
            video_labels = labels[video_mask]

            video_pr_metrics = {}
            for component in components:
                mean_p, classwise_p, mean_r, classwise_r = _compute_component_precision_recall_at_k(
                    video_preds, video_labels, component, dataset_name, topk=k
                )
                video_pr_metrics[f"{component}_p"] = round(mean_p.detach().cpu().item(), 3)
                video_pr_metrics[f"{component}_r"] = round(mean_r.detach().cpu().item(), 3)

                p_arr = classwise_p.detach().cpu().numpy().round(3)
                p_arr = np.where(np.signbit(p_arr) & (p_arr == 0.0), 0.0, p_arr)
                r_arr = classwise_r.detach().cpu().numpy().round(3)
                r_arr = np.where(np.signbit(r_arr) & (r_arr == 0.0), 0.0, r_arr)

                video_pr_metrics[f"{component}_p_per_class"] = p_arr
                video_pr_metrics[f"{component}_r_per_class"] = r_arr

            per_video_prk[video_id] = video_pr_metrics
            # accuracy@k
            video_acc_metrics = {}
            for component in components:
                mean_a, classwise_a = _compute_component_accuracy_at_k(
                    video_preds, video_labels, component, dataset_name, topk=k
                )
                video_acc_metrics[f"{component}_acc"] = round(mean_a.detach().cpu().item(), 3)

                a_arr = classwise_a.detach().cpu().numpy().round(3)
                a_arr = np.where(np.signbit(a_arr) & (a_arr == 0.0), 0.0, a_arr)
                video_acc_metrics[f"{component}_acc_per_class"] = a_arr

            per_video_acck[video_id] = video_acc_metrics
            # top-k hit (sample-level)
            video_hit_metrics = {}
            for component in components:
                mean_h = _compute_component_hit_at_k(
                    video_preds, video_labels, component, dataset_name, topk=k
                )
                video_hit_metrics[f"{component}_hit"] = round(mean_h.detach().cpu().item(), 3)
            per_video_hitk[video_id] = video_hit_metrics

        # videowise macro
        subset_results['videowise_P_at_k'][k_key] = {}
        subset_results['videowise_R_at_k'][k_key] = {}
        subset_results['videowise_ACC_at_k'][k_key] = {}
        subset_results['videowise_HIT_at_k'][k_key] = {}
        p_videowise = {'i': [], 'v': [], 't': [], 'iv': [], 'it': [], 'ivt': []}
        r_videowise = {'i': [], 'v': [], 't': [], 'iv': [], 'it': [], 'ivt': []}
        a_videowise = {'i': [], 'v': [], 't': [], 'iv': [], 'it': [], 'ivt': []}
        h_videowise = {'i': [], 'v': [], 't': [], 'iv': [], 'it': [], 'ivt': []}

        for video_id in unique_video_ids:
            vm = per_video_prk[video_id]
            am = per_video_acck[video_id]
            hm = per_video_hitk[video_id]
            for component in components:
                p_videowise[component].append(vm[f"{component}_p"])
                r_videowise[component].append(vm[f"{component}_r"])
                a_videowise[component].append(am[f"{component}_acc"])
                h_videowise[component].append(hm[f"{component}_hit"])

        for component in components:
            subset_results['videowise_P_at_k'][k_key][component] = (
                float("nan") if len(p_videowise[component]) == 0 else round(float(np.nanmean(p_videowise[component], axis=0)), 3)
            )
            subset_results['videowise_R_at_k'][k_key][component] = (
                float("nan") if len(r_videowise[component]) == 0 else round(float(np.nanmean(r_videowise[component], axis=0)), 3)
            )
            subset_results['videowise_ACC_at_k'][k_key][component] = (
                float("nan") if len(a_videowise[component]) == 0 else round(float(np.nanmean(a_videowise[component], axis=0)), 3)
            )
            subset_results['videowise_HIT_at_k'][k_key][component] = (
                float("nan") if len(h_videowise[component]) == 0 else round(float(np.nanmean(h_videowise[component], axis=0)), 3)
            )

        # overall per-class then macro
        subset_results['overall_P_at_k'][k_key] = {}
        subset_results['overall_R_at_k'][k_key] = {}
        subset_results['overall_ACC_at_k'][k_key] = {}
        subset_results['overall_HIT_at_k'][k_key] = {}
        for component in components:
            classwise_ps = [per_video_prk[video_id][f"{component}_p_per_class"] for video_id in unique_video_ids]
            classwise_rs = [per_video_prk[video_id][f"{component}_r_per_class"] for video_id in unique_video_ids]
            classwise_as = [per_video_acck[video_id][f"{component}_acc_per_class"] for video_id in unique_video_ids]
            if len(classwise_ps) == 0 or len(classwise_rs) == 0 or len(classwise_as) == 0:
                continue

            classwise_ps = np.stack(classwise_ps, axis=0)
            classwise_rs = np.stack(classwise_rs, axis=0)
            classwise_as = np.stack(classwise_as, axis=0)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
                videowise_p = np.nanmean(classwise_ps, axis=0)
                videowise_r = np.nanmean(classwise_rs, axis=0)
                videowise_a = np.nanmean(classwise_as, axis=0)

            effective_p = videowise_p
            effective_r = videowise_r
            effective_a = videowise_a
            if component == 'ivt' and ignore_null_labels:
                if dataset_name == 'cholect50':
                    effective_p = videowise_p[:-6]
                    effective_r = videowise_r[:-6]
                    effective_a = videowise_a[:-6]
                elif dataset_name == 'multibypasst40':
                    effective_p = videowise_p[:-10]
                    effective_r = videowise_r[:-10]
                    effective_a = videowise_a[:-10]

            subset_results['overall_P_at_k'][k_key][component + '_per_class'] = effective_p
            subset_results['overall_P_at_k'][k_key][component] = round(np.nanmean(effective_p), 3)
            subset_results['overall_R_at_k'][k_key][component + '_per_class'] = effective_r
            subset_results['overall_R_at_k'][k_key][component] = round(np.nanmean(effective_r), 3)
            subset_results['overall_ACC_at_k'][k_key][component + '_per_class'] = effective_a
            subset_results['overall_ACC_at_k'][k_key][component] = round(np.nanmean(effective_a), 3)
            hit_vals = [per_video_hitk[video_id][f"{component}_hit"] for video_id in unique_video_ids]
            subset_results['overall_HIT_at_k'][k_key][component] = (
                float("nan") if len(hit_vals) == 0 else round(float(np.nanmean(hit_vals)), 3)
            )

    return subset_results


def compute_triplet_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    video_ids: List[str],
    num_classes: int,
    dataset_name: str,
    ignore_null_labels: bool = False,
    f1_thresholds: List[float] = [0.5],
    f1_topk_values: List[int] = [5, 10, 20],
    get_per_center: bool = False,
    ) -> Dict[str, float]:
    """
    Compute mAP macro and F1 at per-video level and overall level for triplet components.
    
    Args:
        preds: Predictions tensor of shape (N, num_classes)
        labels: Labels tensor of shape (N, num_classes)
        video_ids: List of video IDs of length N
        num_classes: Number of triplet classes
        dataset_name: Name of dataset
        ignore_null_labels: Whether to ignore null labels
        f1_thresholds: List of thresholds for computing F1 scores
        get_per_center: Whether to compute metrics per center (C1, C2, C3, C4)
        Dictionary containing per-video and overall mAP (existing) and F1 scores
        for all components.
    """
    # assign config.eval.per_video = True to get_per_video variable below
     
    unique_video_ids = sorted(set(video_ids))
    # 1) Overall results using all videos
    overall_results = _compute_metrics_for_video_subset(
        preds=preds,
        labels=labels,
        video_ids=video_ids,
        selected_video_ids=unique_video_ids,
        num_classes=num_classes,
        dataset_name=dataset_name,
        ignore_null_labels=ignore_null_labels,
        f1_thresholds=f1_thresholds,
        f1_topk_values=f1_topk_values,
    )
    results = overall_results

    if not get_per_center:
        return results
    
    # 2) Per-center results (C1, C2, C3, C4)
    center_to_videos = {"C1": [], "C2": [], "C3": [], "C4": []}
    for vid in unique_video_ids:
        center = _extract_center(vid)
        if center in center_to_videos:
            center_to_videos[center].append(vid)

    per_center = {}
    for center, center_video_ids in center_to_videos.items():
        per_center[center] = _compute_metrics_for_video_subset(
            preds=preds,
            labels=labels,
            video_ids=video_ids,
            selected_video_ids=center_video_ids,
            num_classes=num_classes,
            dataset_name=dataset_name,
            ignore_null_labels=ignore_null_labels,
            f1_thresholds=f1_thresholds,
            f1_topk_values=f1_topk_values,
        )

    results['per_center'] = per_center
    results['per_center_video_ids'] = center_to_videos

    return results


def format_overall_metrics_ascii(
    results: Dict,
    mode: str = "val",
    epoch: int | None = None,
    f1_threshold: str = "@0.5",
    topk_values: List[int] = [5, 10, 20],
) -> str:
    """Format compact overall metrics table: mAP, F1@threshold, and F1@K (if present)."""
    components = ['i', 'v', 't', 'iv', 'it', 'ivt']

    overall_map = results.get('overall_mAP', {})
    overall_f1 = results.get('overall_F1', {})
    overall_f1_at_k = results.get('overall_F1_at_k', {})
    overall_p_at_k = results.get('overall_P_at_k', {})
    overall_r_at_k = results.get('overall_R_at_k', {})
    overall_acc_at_k = results.get('overall_ACC_at_k', {})
    overall_hit_at_k = results.get('overall_HIT_at_k', {})

    def _safe(metric_dict, key):
        if not isinstance(metric_dict, dict):
            return float('nan')
        val = metric_dict.get(key, float('nan'))
        try:
            return float(val)
        except Exception:
            return float('nan')

    sep = '=' * 102
    title = f"[{mode}] results"
    if epoch is not None:
        title += f" at ep [{epoch}]"

    lines = [sep, title, sep]
    lines.append(f"{'FINAL METRIC':<10} {'I':>8} {'V':>8} {'T':>8} {'IV':>8} {'IT':>8} {'IVT':>8}")
    lines.append('-' * 102)

    map_row = [_safe(overall_map, c) for c in components]
    lines.append(f"{'mAP':<10} " + " ".join(f"{v:>8.3f}" for v in map_row))

    f1_main = overall_f1.get(f1_threshold)
    if isinstance(f1_main, dict):
        row = [_safe(f1_main, c) for c in components]
        lines.append(f"{('F1@thr' + f1_threshold[1:]):<10} " + " ".join(f"{v:>8.3f}" for v in row))

    for k in topk_values:
        f1k = overall_f1_at_k.get(f'@{k}')
        if isinstance(f1k, dict):
            row = [_safe(f1k, c) for c in components]
            lines.append(f"{f'F1@{k}':<10} " + " ".join(f"{v:>8.3f}" for v in row))

        hk = overall_hit_at_k.get(f'@{k}')
        if isinstance(hk, dict):
            row = [_safe(hk, c) for c in components]
            lines.append(f"{f'Hit@{k}':<10} " + " ".join(f"{v:>8.3f}" for v in row))

        ak = overall_acc_at_k.get(f'@{k}')
        if isinstance(ak, dict):
            row = [_safe(ak, c) for c in components]
            lines.append(f"{f'Acc@{k}':<10} " + " ".join(f"{v:>8.3f}" for v in row))

        pk = overall_p_at_k.get(f'@{k}')
        if isinstance(pk, dict):
            row = [_safe(pk, c) for c in components]
            lines.append(f"{f'P@{k}':<10} " + " ".join(f"{v:>8.3f}" for v in row))

        rk = overall_r_at_k.get(f'@{k}')
        if isinstance(rk, dict):
            row = [_safe(rk, c) for c in components]
            lines.append(f"{f'R@{k}':<10} " + " ".join(f"{v:>8.3f}" for v in row))

    lines.append(sep)
    return "\n".join(lines)


def format_results_table(results: Dict, mode: str = "val") -> str:
    """
    Format triplet metrics results into a pretty table.
    
    Args:
        results: Dictionary containing 'videowise_mAP' and 'overall_mAP'
        
    Returns:
        Formatted string table
    """
    components = ['i', 'v', 't', 'iv', 'it', 'ivt']
    payload = results.get('overall', results)
    
    # Header
    output = []
    output.append("\n" + "=" * 85)
    output.append(f"TRIPLET METRICS RESULTS {mode.upper()} SET.")
    output.append("Collect classwise mAP for each video and then average across all videos to get overall mAP")
    output.append("=" * 85)
    
    # Column headers
    header = f"{'Video ID':<20} {'I':>8} {'V':>8} {'T':>8} {'IV':>8} {'IT':>8} {'IVT':>8}"
    output.append("\n" + header)
    output.append("-" * 80)
    
    # Per-video metrics
    if 'videowise' in payload:
        for video_id, metrics in sorted(payload['videowise'].items()):
            row = f"{video_id:<20}"
            for comp in components:
                value = metrics.get(comp, 0.0)
                row += f" {value:>8.3f}"
            output.append(row)
    
    # Separator before overall metrics
    output.append("-" * 80)
    
    # Overall metrics
    if 'overall_mAP' in payload:
        row = f"{'OVERALL mAP':<20}"
        for comp in components:
            value = payload['overall_mAP'].get(comp, 0.0)
            row += f" {value:>8.3f}"
        output.append(row)
    
    output.append("=" * 80 + "\n")
    
    return "\n".join(output)
