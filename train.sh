#!/bin/bash

# Cyclegan + ffe + pixel loss -11
CUDA_VISIBLE_DEVICES=3 python train_cyc_mbf_light.py \
    --dataroot /ssd01/wanghuijiao/CGFE/WHU-VIS-NIR/combined_faces \
    --name WHU-VIS-NIR_cyclegan_ffe_pixel_loss \
    --model cycle_gan \
    --loss_flag pixel \
    --dataset_mode aligned \
    --display_freq 400 \
    --print_freq 1000 \
    --save_epoch_freq 5 \
    --phase train \
    --n_epochs 100 \
    --n_epochs_decay 80 \
    --lr 0.0002 \
    --netG MBFResnet_6blocks \
    --ffe_model mobilefacenet \
    --ffe_pretrained_weights '/ssd01/wanghuijiao/CG/checkpoint/model_mobilefacenet.pth' 



