#!/bin/bash

# python -m visdom.server

# python train_mbfnet_softmax.py --dataroot ./datasets/selfmadevis2nir --name selfmadevis2nir_cyclegan --model cycle_gan --serial_batches

# python train_cyc_mbf_light.py --dataroot /hdd01/wanghuijiao/HD06/datasets/selfmadevis2nir --name selfmadevis2nir_cyclegan --model cycle_gan --serial_batches --loss_flag fea

# CUDA_VISIBLE_DEVICES=0 python train_cyc_mbf_light.py --dataroot /hdd01/wanghuijiao/CG/datasets/selfmadevis2nir --name selfmadevis2nir_cyclegan --model cycle_gan --loss_flag cyclegan --continue_train

# # MobileFaceNet
# CUDA_VISIBLE_DEVICES=7 python train_cyc_mbf_light.py \
#     --dataroot /ssd01/wanghuijiao/CGFE/whj_combine \
#     --name whj_FFEcyclegan_1 \
#     --model cycle_gan \
#     --loss_flag cyclegan \
#     --dataset_mode aligned \
#     --save_epoch_freq 20 \
#     --n_epochs 50 \
#     --n_epochs_decay 50 \
#     --ffe_model mobilefacenet \
#     --ffe_pretrained_weights /ssd01/wanghuijiao/CG/checkpoint/model_mobilefacenet.pth \
#     --display_freq 40 \
#     --netG MBFResnet_6blocks

# CycleGAN
# CUDA_VISIBLE_DEVICES=7 python train_cyc_mbf_light.py \
#     --dataroot /ssd01/wanghuijiao/CGFE/whj_combine \
#     --name whj_Cyclegan_1 \
#     --model cycle_gan \
#     --loss_flag cyclegan \
#     --dataset_mode aligned \
#     --save_epoch_freq 20 \
#     --n_epochs 50 \
#     --n_epochs_decay 50 \
#     --display_freq 40 \
#     --netG resnet_9blocks

# FFE -5
CUDA_VISIBLE_DEVICES=1 python train_cyc_mbf_light.py \
    --dataroot /ssd01/wanghuijiao/CGFE/WHU-VIS-NIR/combined_faces \
    --name WHU-VIS-NIR_ffe \
    --model cycle_gan \
    --loss_flag cyclegan \
    --dataset_mode aligned \
    --display_freq 400 \
    --print_freq 1000 \
    --save_epoch_freq 5 \
    --phase train \
    --n_epochs 100 \
    --n_epochs_decay 80 \
    --lr 0.00005 \
    --netG MBFResnet_6blocks \
    --ffe_model mobilefacenet \
    --ffe_pretrained_weights '/ssd01/wanghuijiao/CG/checkpoint/model_mobilefacenet.pth' \
    --continue_train \
    --epoch_count 13


# CycleGAN -1 -4
CUDA_VISIBLE_DEVICES=0 python train_cyc_mbf_light.py \
    --dataroot /ssd01/wanghuijiao/CGFE/WHU-VIS-NIR/combined_faces \
    --name WHU-VIS-NIR_cyclegan \
    --gpu_ids 0 \
    --model cycle_gan \
    --loss_flag cyclegan \
    --dataset_mode aligned \
    --netG resnet_9blocks \
    --display_freq 400 \
    --print_freq 1000 \
    --save_epoch_freq 5 \
    --phase train \
    --n_epochs 100 \
    --n_epochs_decay 80 \
    --lr 0.0002 \
    --ffe_model ''


# CycleGAN + pixel loss -8
CUDA_VISIBLE_DEVICES=2 python train_cyc_mbf_light.py \
    --dataroot /ssd01/wanghuijiao/CGFE/WHU-VIS-NIR/combined_faces \
    --name WHU-VIS-NIR_cyclegan_pixel_loss-8 \
    --model cycle_gan \
    --loss_flag pixel \
    --dataset_mode aligned \
    --display_freq 400 \
    --print_freq 1000 \
    --save_epoch_freq 5 \
    --phase train \
    --n_epochs 50 \
    --n_epochs_decay 0 \
    --lr 0.0002 \
    --netG resnet_9blocks \
    --ffe_model '' 


# Cyclegan + pixel_huber loss -9
CUDA_VISIBLE_DEVICES=4 python train_cyc_mbf_light.py \
    --dataroot /ssd01/wanghuijiao/CGFE/WHU-VIS-NIR/combined_faces \
    --name WHU-VIS-NIR_cyclegan_pixel_huber_loss \
    --model cycle_gan \
    --loss_flag pixel_huber \
    --dataset_mode aligned \
    --display_freq 400 \
    --print_freq 1000 \
    --save_epoch_freq 5 \
    --phase train \
    --n_epochs 100 \
    --n_epochs_decay 80 \
    --lr 0.0002 \
    --netG resnet_9blocks \
    --ffe_model '' \
    --lambda_A 100 \
    --lambda_B 100


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



