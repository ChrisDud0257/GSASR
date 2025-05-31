import os
import cv2
from PIL import Image
import numpy as np
import argparse
import random

from basicsr.utils.matlab_functions import imresize
from basicsr.utils import imfromfile

def modcrop(img, scale):
    h,w,_ = img.shape
    mod_h = h % scale
    mod_w = w % scale
    return img[:h - mod_h, :w - mod_w, :]

def main(args):
    for dataset in args.gt:

        save_path_dataset = os.path.join(f"{args.save_path}_GT{args.gt_size}")
        os.makedirs(save_path_dataset, exist_ok=True)

        for scale in args.scale_list:
            save_path_dataset_scale = os.path.join(save_path_dataset, 'x'+f'{scale}')
            
            save_path_GT = os.path.join(save_path_dataset_scale, "GT")
            save_path_LR = os.path.join(save_path_dataset_scale, "LR")
            save_path_bicubic = os.path.join(save_path_dataset_scale, "Bicubic")

            os.makedirs(save_path_GT, exist_ok = True)
            os.makedirs(save_path_LR, exist_ok = True)
            os.makedirs(save_path_bicubic, exist_ok = True)
            for img in os.listdir(dataset):
                img_path = os.path.join(dataset, img)
                print(f"{img_path}")
                img_name, _ = os.path.splitext(img)
                image = imfromfile(path=img_path, float32=True)

                h, w, _ = image.shape
                top = random.randint(0, h - args.gt_size)
                left = random.randint(0, w - args.gt_size)

                top = 80
                left = 80

                image_GT = image[top:top+args.gt_size, left:left+args.gt_size, :]
                assert image_GT.shape[0] == image_GT.shape[1] == args.gt_size, f"image_GT shape is {image_GT.shape}"
                image_LR = imresize(image_GT, scale=float(1.0/scale))
                image_bicubic = imresize(image_LR, scale = float(scale))
                image_LR = np.clip((image_LR * 255.0).round(), 0, 255).astype(np.uint8)
                image_bicubic = np.clip((image_bicubic * 255.0).round(), 0, 255).astype(np.uint8)
                image_GT = np.clip((image_GT * 255.0).round(), 0, 255).astype(np.uint8)

                cv2.imwrite(os.path.join(save_path_GT, f'{img_name}.png'), image_GT)
                cv2.imwrite(os.path.join(save_path_LR, f'{img_name}.png'), image_LR)
                cv2.imwrite(os.path.join(save_path_bicubic, f'{img_name}_bicubicx{scale}.png'), image_bicubic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gt', type=str, default='/home/chendu/data2_hdd10t/chendu/dataset/DIV2K/DIV2K_valid_HR_mod24', help='Path to gt (Ground-Truth)')
    parser.add_argument('--gt', nargs='+', default=[
                                                    # '/home/notebook/data/group/chendu/dataset/basicsr/BSDS100/GT/original',
                                                    # '/home/notebook/data/group/chendu/dataset/basicsr/Set5/GT/original',
                                                    # '/home/notebook/data/group/chendu/dataset/basicsr/Set14/GT/original',
                                                    # '/home/notebook/data/group/chendu/dataset/LSDIR/validation/val1/HR/val',
                                                    # '/home/notebook/data/group/chendu/dataset/basicsr/Urban100/GT/original',
                                                    # '/home/notebook/data/group/chendu/dataset/basicsr/Manga109/GT/original',
                                                    # '/home/notebook/data/group/chendu/dataset/basicsr/General100/GT/original',
                                                    '/home/notebook/data/group/chendu/dataset/basicsr/DIV2K100/GT/original'
                                                    ],
                        help='Path to gt (Ground-Truth)')

    parser.add_argument('--scale_list', type=list, default=[2,3,4,6,8,12])
    parser.add_argument('--save_path', type = str, default = '/home/notebook/data/group/chendu/dataset/TimeTest/DIV2K100')
    parser.add_argument('--gt_size', type = int, default = 720)
    args = parser.parse_args()
    main(args)