import os
import cv2
from PIL import Image
import numpy as np
import argparse
import math

from basicsr.utils.matlab_functions import imresize
from basicsr.utils import imfromfile

def modcrop(img, h_gt_new, w_gt_new):
    return img[:h_gt_new, :w_gt_new, :]

def main(args):
    for dataset in args.gt:
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(dataset)))

        save_path_dataset = os.path.join(args.save_path, dataset_name)
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

                h_gt, w_gt, _ = image.shape

                h_lr = math.floor(h_gt / scale)
                w_lr = math.floor(w_gt / scale)

                h_gt_new = math.floor(h_lr * scale)
                w_gt_new = math.floor(w_lr * scale)

                image_GT_new = modcrop(image, h_gt_new, w_gt_new)
                image_LR = imresize(image, scale=float(1.0/scale))
                image_LR_new = modcrop(image_LR, h_lr, w_lr)
                image_bicubic = imresize(image_LR, scale = float(scale))
                image_bicubic_new = modcrop(image_bicubic, h_gt_new, w_gt_new)
                image_LR_new = np.clip((image_LR_new * 255.0).round(), 0, 255).astype(np.uint8)
                image_bicubic_new = np.clip((image_bicubic_new * 255.0).round(), 0, 255).astype(np.uint8)
                image_GT_new = np.clip((image_GT_new * 255.0).round(), 0, 255).astype(np.uint8)

                cv2.imwrite(os.path.join(save_path_GT, f'{img_name}.png'), image_GT_new)
                cv2.imwrite(os.path.join(save_path_LR, f'{img_name}.png'), image_LR_new)
                cv2.imwrite(os.path.join(save_path_bicubic, f'{img_name}_bicubicx{scale}.png'), image_bicubic_new)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gt', type=str, default='/home/chendu/data2_hdd10t/chendu/dataset/DIV2K/DIV2K_valid_HR_mod24', help='Path to gt (Ground-Truth)')
    parser.add_argument('--gt', nargs='+', default=[
                                                    # '/home/notebook/data/group/chendu/dataset/basicsr/BSDS100/GT/original',
                                                    # '/home/notebook/data/group/chendu/dataset/basicsr/Set5/GT/original',
                                                    # '/home/notebook/data/group/chendu/dataset/basicsr/Set14/GT/original',
                                                    '/home/notebook/data/group/chendu/dataset/LSDIR/validation/val1/HR/val',
                                                    # '/home/notebook/data/group/chendu/dataset/basicsr/Urban100/GT/original',
                                                    # '/home/notebook/data/group/chendu/dataset/basicsr/Manga109/GT/original',
                                                    # '/home/notebook/data/group/chendu/dataset/basicsr/General100/GT/original',
                                                    ],
                        help='Path to gt (Ground-Truth)')

    parser.add_argument('--scale_list', type=list, default=[2.5, 3.5])
    parser.add_argument('--save_path', type = str, default = '/home/notebook/data/group/chendu/dataset/AnyScaleTestBicubic')
    args = parser.parse_args()
    main(args)
