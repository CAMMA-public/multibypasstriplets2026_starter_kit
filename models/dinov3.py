from __future__ import annotations
from typing import Any, Dict, Optional
import logging
import os
import timm
import torch
from .temporal_layers import SimpleTransformerTemporalModel
LOGGER = logging.getLogger(__name__)

class CustomDinov3(torch.nn.Module):
    def __init__(self, config: Dict[str, Any], backbone: torch.nn.Module) -> None:
        super().__init__()
        model_cfg: Dict[str, Any] = config["model"]
        self.config = config
        self.clip_len = config.dataset.clip_len
        self.backbone = backbone

        # temporal modeling on the pooled features
        if config.model.apply_temporal:
            # used 8 in vits
            self.temporal_model = SimpleTransformerTemporalModel(d_model=config.model.fc_input_dim, 
                                                        nhead=2,
                                                        num_layers=1, 
                                                        dim_feedforward=2048, 
                                                        dropout=0.1, 
                                                        pe_choice=config.model.pe_choice, 
                                                        max_seq_len=config.dataset.clip_len, 
                                                        temporal_pool=config.model.temporal_feat_aggr)
        else:
            self.temporal_model = None

        # apply the fc layers either single or multiple
        apply_fc_options = model_cfg["apply_fc"].split(",")
        fc_input_dim = model_cfg["fc_input_dim"]
        self.fc_tool = None
        self.fc_verb = None
        self.fc_target = None
        self.fc_triplet = None
        
        for apply_fc in apply_fc_options:
            if apply_fc == "i":
                self.fc_tool = torch.nn.Linear(fc_input_dim, model_cfg["num_tool_classes"])
            elif apply_fc == "v":
                self.fc_verb = torch.nn.Linear(fc_input_dim, model_cfg["num_verb_classes"])
            elif apply_fc == "t":
                self.fc_target = torch.nn.Linear(fc_input_dim, model_cfg["num_target_classes"])
            elif apply_fc == "ivt":
                self.fc_triplet = torch.nn.Linear(fc_input_dim, model_cfg["num_triplet_classes"])
            else:
                raise ValueError(f"Invalid apply_fc option: {apply_fc}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x) #.pooler_output
        
        # reshape the features to the clip length
        features = features.view(features.shape[0] // self.clip_len, self.clip_len, -1)

        # apply temporal model if present else simple aggregation
        
        if self.temporal_model is not None:
            features = self.temporal_model(features)
        else:
            if self.config.dataset.clip_aggregation == "mean":
                features = features.mean(dim=1)
            elif self.config.dataset.clip_aggregation == "sum":
                features = features.sum(dim=1)
            else:
                raise ValueError(f"Invalid clip aggregation: {self.config.dataset.clip_aggregation}")

        outputs = {}
        if self.fc_tool is not None:
            outputs["logit_i"] = self.fc_tool(features)
        if self.fc_verb is not None:
            outputs["logit_v"] = self.fc_verb(features)
        if self.fc_target is not None:
            outputs["logit_t"] = self.fc_target(features)
        if self.fc_triplet is not None:
            outputs["logit_ivt"] = self.fc_triplet(features)
        return outputs

def build_custom_dinov3(config: Dict[str, Any]) -> torch.nn.Module:
    # get the model and init 
    model_cfg = config["model"]
    checkpoint_path = model_cfg.get("checkpoint_path")
    pretrained_model_name_or_path = str(model_cfg["backbone_name"])

    REPO_DIR = "../dinov3"
    if "vits16" in pretrained_model_name_or_path:
        wt_name = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    elif "vitb16" in pretrained_model_name_or_path:
        wt_name = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    elif "vitl16" in pretrained_model_name_or_path:
        wt_name = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    else:
        raise ValueError(f"Invalid backbone name: {pretrained_model_name_or_path}")

    wts_file = os.path.join("weights", wt_name)
    backbone = torch.hub.load(REPO_DIR, pretrained_model_name_or_path, source='local', weights=wts_file)

    if checkpoint_path is not None:
        # checkpoint path should be provided in the config

        if torch.__version__ > '2.6':
            best_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        else:
            best_checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'state_dict' in best_checkpoint:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k,v in best_checkpoint['state_dict'].items():
                if k.startswith('module.backbone.'):
                    name = k[7:]
                    new_state_dict[name] = v
            # backbone.load_state_dict(best_checkpoint['state_dict'])
            backbone.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded eval checkpoint from {checkpoint_path}")
        else:
            backbone.load_state_dict(best_checkpoint)
            print(f"Loaded eval checkpoint from {checkpoint_path}")
    
    # get the model
    model = CustomDinov3(config=config, backbone=backbone)

    freeze_backbone = model_cfg.get("freeze_backbone", "no")
    if freeze_backbone == "all":
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
    elif freeze_backbone.isdigit():
        freeze_up_to = int(freeze_backbone)
        for name, param in model.backbone.named_parameters():
            if  config.model.backbone_name == "dinov3_vitb16" or config.model.backbone_name == "dinov3_vitl16":
                is_early_block = any(f"blocks.{i}." in name for i in range(freeze_up_to + 1))
            else:
                is_early_block = any(f"backbone.blocks.{i}." in name for i in range(freeze_up_to + 1))

            if is_early_block:
                param.requires_grad = False
            else:
                param.requires_grad = True

    return model

