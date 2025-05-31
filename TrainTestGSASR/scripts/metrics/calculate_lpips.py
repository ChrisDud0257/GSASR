import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize
import argparse
import os
from basicsr.utils import img2tensor

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


def main(args):

    loss_fn_vgg = lpips.LPIPS(net='alex').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []
    img_list = sorted(glob.glob(osp.join(args.gt, '*')))

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    
    if args.scale <= 8:
        crop_border = int(args.scale)
    else:
        crop_border = 8
    
    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_restored = cv2.imread(osp.join(args.restored, basename + args.suffix + ext), cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.

        if crop_border != 0:
            img_gt = img_gt[crop_border:-crop_border, crop_border:-crop_border, ...]
            img_restored = img_restored[crop_border:-crop_border, crop_border:-crop_border, ...]


        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())

        print(f'{i+1:3d}: {basename:25}. \tLPIPS: {lpips_val.item():.6f}.')
        lpips_all.append(lpips_val.item())

    print(f'Average: LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type = str, default='/home/notebook/data/sharedgroup/RG_YLab/aigc_share_group_data/chendu/dataset/AnyScaleTestBicubic/DIV2K100/x4/GT',
                        help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', type = str, default='/home/notebook/code/personal/S9053766/chendu/FinalUpload/GSASR/figures/QuickInferenceExpResults/Benchmark/SwinIR/x4',
                        help='Path to restored images')
    parser.add_argument('--scale', type=float, default=4)
    parser.add_argument('--suffix', type=str, default='_SwinIR', help='Suffix for restored images')
    parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.')
    args = parser.parse_args()
    main(args)