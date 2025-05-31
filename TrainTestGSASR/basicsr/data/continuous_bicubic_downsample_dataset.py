from torch.utils import data as data
from torchvision.transforms.functional import normalize
import os
import scipy.io as io
import numpy as np
import random
import math
import torch
import torch.nn.functional as F
import cv2

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor, imfromfile
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.matlab_functions import imresize_new

@DATASET_REGISTRY.register()
class ContinuousBicubicDownsampleDataset(data.Dataset):
    def __init__(self, opt):
        super(ContinuousBicubicDownsampleDataset, self).__init__()
        self.opt = opt
        self.gt_folder = opt['all_gt_list']

        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        all_img_list = []
        for dataset in self.gt_folder:
            dataset_img_list = os.listdir(dataset)
            for img in dataset_img_list:
                img_path = os.path.join(dataset, img)
                all_img_list.append(img_path)

        self.all_img_list = all_img_list

        self.scale_list = opt['scale_list']
        self.lr_size = opt['lr_size']
        self.sample_size = opt['sample_size']

        self.scale_max = self.scale_list[1]
        self.gt_size_max = math.ceil(self.scale_max * self.lr_size)

        self.round_mode = opt.get('round_mode', 'ceil')

    def __getitem__(self, index):
        img_path = self.all_img_list[index]

        img_gt = imfromfile(path=img_path, float32=True)  #h*w*c, 0-1, ndarray

        h_img_gt, w_img_gt, _ = img_gt.shape

        if len(self.scale_list) == 2:
            scale = float(random.uniform(self.scale_list[0], self.scale_list[1]))
        else:
            scale = random.choice(self.scale_list)

        lr_size = torch.tensor([self.lr_size, self.lr_size])

        if self.round_mode == 'ceil':
            gt_size = torch.tensor([math.ceil(scale * lr_size[0]), math.ceil(scale * lr_size[1])])
        elif self.round_mode == 'round':
            gt_size = torch.tensor([round(scale * lr_size[0].item()), round(scale * lr_size[1].item())])

        start_h_crop_gt = random.randint(0, h_img_gt - gt_size[0])
        start_w_crop_gt = random.randint(0, w_img_gt - gt_size[1])

        crop_gt = img_gt[start_h_crop_gt:start_h_crop_gt + gt_size[0], start_w_crop_gt:start_w_crop_gt+gt_size[1], :]

        scale_modify_h = float(crop_gt.shape[0] / self.lr_size)
        scale_modify_w = float(crop_gt.shape[1] / self.lr_size)
        crop_lr = np.ascontiguousarray(imresize_new(img = crop_gt, scale_h = 1 / scale_modify_h, scale_w = 1 / scale_modify_w, antialiasing=True))

        scale_modify = torch.tensor([scale_modify_h, scale_modify_w])

        img_gt, img_lq = augment([crop_gt, crop_lr], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        if self.sample_size > 0:
            sample_coords = np.random.randint(0,[gt_size[0], gt_size[1]], size=(self.sample_size, 2))
            sample_coords = torch.tensor(sample_coords)
            # Fetching the colour of the pixels in each coordinates
            colour_values = [img_gt[:, coord[0], coord[1]] for coord in sample_coords]
            img_gt = torch.stack(colour_values, dim = 1)

        else:
            sample_coords = None
            pad_h = self.gt_size_max - gt_size[0]
            pad_w = self.gt_size_max - gt_size[1]
            #pad gt to the maximum size in order to do paraller training
            img_gt = F.pad(img_gt, (0, pad_w, 0, pad_h), 'constant', 0)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        if sample_coords is not None:
            return_d = {'gt':img_gt, 'lq':img_lq, 'sample_coords': sample_coords, 'scale': scale, 
                        'gt_size': gt_size, 'scale_modify': scale_modify}
        else:
            return_d = {'gt': img_gt, 'lq': img_lq, 'scale': scale,
                        'gt_size': gt_size, 'pad_h': pad_h, 'pad_w': pad_w, 'scale_modify': scale_modify}
        return return_d

    def __len__(self):
        return len(self.all_img_list)
