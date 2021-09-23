#!/bin/bash

python combine_V_and_N.py \
    --fold_V /ssd01/wanghuijiao/CGFE/WHU-VIS-NIR/aligned_faces/VIS \
    --fold_N /ssd01/wanghuijiao/CGFE/WHU-VIS-NIR/aligned_faces/NIR \
    --fold_VN /ssd01/wanghuijiao/CGFE/WHU-VIS-NIR/combined_faces \
    --num_imgs 10000000 
