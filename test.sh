#!/bin/bash
# python test.py --dataroot /hdd01/wanghuijiao/CGFE/faces-combine/ --dataset_mode aligned --name selfmadevis2nir_cyclegan --model cycle_gan --no_dropout --num_test 5000
# python test.py --dataroot /hdd01/wanghuijiao/CGFE/oulu_casia --dataset_mode unaligned --name selfmadevis2nir_cyclegan1v1 --model cycle_gan --no_dropout --num_test 5000 --epoch latest

# python concate_r_imgs.py --stage test


# test cyclegan
python test.py \
    --dataroot /ssd01/wanghuijiao/CGFE/WHU-VIS-NIR/combined_faces \
    --dataset_mode aligned \
    --name 1-4-WHU-VIS-NIR_cyclegan \
    --model cycle_gan \
    --no_dropout \
    --num_test 5000 \
    --epoch 15 \
    --netG resnet_9blocks \
    --phase test \
    --results_dir ./results/


# test cyclegan + FFE
python test.py \
    --dataroot /ssd01/wanghuijiao/CGFE/WHU-VIS-NIR/combined_faces \
    --dataset_mode aligned \
    --name 5-WHU-VIS-NIR_ffe \
    --model cycle_gan \
    --no_dropout \
    --num_test 5000 \
    --epoch 175 \
    --netG MBFResnet_6blocks \
    --phase test \
    --results_dir ./results/

# test cyclegan + pixel loss -8
python test.py \
    --dataroot /ssd01/wanghuijiao/CGFE/WHU-VIS-NIR/combined_faces \
    --dataset_mode aligned \
    --name WHU-VIS-NIR_cyclegan_pixel_loss-8 \
    --model cycle_gan \
    --no_dropout \
    --num_test 5000 \
    --epoch 5 \
    --netG resnet_9blocks \
    --phase test \
    --results_dir ./results/ 


# test 9-WHU-VIS-NIR_cyclegan_pixel_huber_loss
python test.py \
    --dataroot /ssd01/wanghuijiao/CGFE/WHU-VIS-NIR/combined_faces \
    --dataset_mode aligned \
    --name 9-WHU-VIS-NIR_cyclegan_pixel_huber_loss \
    --model cycle_gan \
    --no_dropout \
    --num_test 5000 \
    --epoch 65 \
    --netG resnet_9blocks \
    --phase test \
    --results_dir ./results/


# test cyclegan + FFE + pixel loss
python test.py \
    --dataroot /ssd01/wanghuijiao/CGFE/WHU-VIS-NIR/combined_faces \
    --dataset_mode aligned \
    --name 11-WHU-VIS-NIR_cyclegan_ffe_pixel_loss \
    --model cycle_gan \
    --no_dropout \
    --num_test 5000 \
    --epoch 50 \
    --netG MBFResnet_6blocks \
    --phase test \
    --results_dir ./results/


# CycleGAN + pixel loss -8
CUDA_VISIBLE_DEVICES=2 python train_cyc_mbf_light.py \
    --dataroot /ssd01/wanghuijiao/CGFE/WHU-VIS-NIR/combined_faces \
    --name WHU-VIS-NIR_cyclegan_pixel_loss \
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
    --netG resnet_9blocks \
    --ffe_model ''

# # Cyclegan + pixel_huber loss -9
# CUDA_VISIBLE_DEVICES=4 python train_cyc_mbf_light.py \
#     --dataroot /ssd01/wanghuijiao/CGFE/WHU-VIS-NIR/combined_faces \
#     --name WHU-VIS-NIR_cyclegan_pixel_huber_loss \
#     --model cycle_gan \
#     --loss_flag pixel_huber \
#     --dataset_mode aligned \
#     --display_freq 400 \
#     --print_freq 1000 \
#     --save_epoch_freq 5 \
#     --phase train \
#     --n_epochs 100 \
#     --n_epochs_decay 80 \
#     --lr 0.0002 \
#     --netG resnet_9blocks \
#     --ffe_model '' \
#     --lambda_A 100 \
#     --lambda_B 100
