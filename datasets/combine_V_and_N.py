import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool


def image_write(path_A, path_B, path_AB):
    im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_AB = np.concatenate([im_A, im_B], 1)
    cv2.imwrite(path_AB, im_AB)


parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_V', dest='fold_V', help='input directory for VIS image ', type=str, default='../dataset/50kshoes_edges')
parser.add_argument('--fold_N', dest='fold_N', help='input directory for NIR image', type=str, default='../dataset/50kshoes_jpg')
parser.add_argument('--fold_VN', dest='fold_VN', help='output directory', type=str, default='../dataset/test_AB')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument('--use_VN', dest='use_VN', help='if true: (1(1_V, 1(1_N) to (1(1_VN)', action='store_true')
parser.add_argument('--no_multiprocessing', dest='no_multiprocessing', help='If used, chooses single CPU execution instead of parallel execution', action='store_true',default=False)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

splits = os.listdir(args.fold_V)

if not args.no_multiprocessing:
    pool=Pool()

for sp in splits:
    img_fold_V = os.path.join(args.fold_V, sp)
    img_fold_N = os.path.join(args.fold_N, sp)
    img_list = os.listdir(img_fold_V)
    if args.use_VN:
        img_list = [img_path for img_path in img_list if '_A.' in img_path]

    num_imgs = min(args.num_imgs, len(img_list))
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    img_fold_VN = os.path.join(args.fold_VN, sp)
    if not os.path.isdir(img_fold_VN):
        os.makedirs(img_fold_VN)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_V, name_A)
        if args.use_VN:
            name_B = name_A.replace('_V.', '_N.')
        else:
            name_B = name_A
        path_B = os.path.join(img_fold_N, name_B)
        if os.path.isfile(path_A) and os.path.isfile(path_B):
            name_AB = name_A
            if args.use_VN:
                name_AB = name_AB.replace('_V.', '.')  # remove _A
            path_AB = os.path.join(img_fold_VN, name_AB)
            if not args.no_multiprocessing:
                pool.apply_async(image_write, args=(path_A, path_B, path_AB))
            else:
                im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_AB = np.concatenate([im_A, im_B], 1)
                cv2.imwrite(path_AB, im_AB)

if not args.no_multiprocessing:
    pool.close()
    pool.join()
