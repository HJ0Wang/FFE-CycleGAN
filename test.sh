#!/bin/bash

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

