#!/bin/bash

runname=${1:-baselines}
expname=${2:-eval_dinov3_vits16_x1}

python evaluate.py \
    --config configs/models/dinov3.yaml \
    --output logs/ \
    --run "${runname}" \
    --expname "${expname}" \
    --gpu 0 \
    --dataset.name multibypasst40 \
    --dataset.setting challenge \
    --dataset.test_fold 1 \
    --dataset.batch_size 4 \
    --dataset.video_dir_prefix data \
    --dataset.video_path MultiBypassT40/videos \
    --dataset.label_path MultiBypassT40/label_files_challenge \
    --dataset.img_height 224 \
    --dataset.img_width 224 \
    --dataset.sampling_percentage 1.0 \
    --dataset.clip_len 4 \
    --dataset.clip_position end \
    --dataset.clip_center_mode symmetric \
    --dataset.clip_aggregation mean \
    --optim.name adamw \
    --optim.lr 0.0001 \
    --optim.backbone_lr 0.00001 \
    --optim.weight_decay 0.02 \
    --optim.grad_clip_norm 1.0 \
    --lr_scheduler.name cosine \
    --model.name dinov3 \
    --model.backbone_name dinov3_vits16 \
    --model.num_triplet_classes 85 \
    --model.num_tool_classes 12 \
    --model.num_verb_classes 13 \
    --model.num_target_classes 15 \
    --model.apply_fc i,v,t,ivt \
    --model.fc_input_dim 384 \
    --model.apply_class_weights i,v,t \
    --model.val_ckpt_key ivt \
    --model.val_ckpt_patience 5 \
    --eval.split test \
    --eval.checkpoint_path logs/baselines/train_dinov3_vits16_x1/checkpoints/baselines_train_dinov3_vits16_x1_best.pth \
    --training.epochs 30 \
