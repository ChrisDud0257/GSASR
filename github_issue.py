import os
import math
import random
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

def resize_fn(img, size):
    img = transforms.Resize(size, InterpolationMode.BICUBIC)(img)
    return img

def generate_lr_images(gt_root, lr_root, scale, img_exts={'.png', '.jpg', '.jpeg'}):
    os.makedirs(lr_root, exist_ok=True)
    img_names = [f for f in os.listdir(gt_root) if os.path.splitext(f)[1].lower() in img_exts]

    for name in img_names:
        img_path = os.path.join(gt_root, name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = TF.to_tensor(img)  # (C, H, W)

        h_hr = int(math.ceil(img_tensor.shape[1] / scale) * scale)
        w_hr = int(math.ceil(img_tensor.shape[2] / scale) * scale)
        img_tensor = TF.center_crop(img_tensor, (h_hr, w_hr))  # center crop instead of padding

        h_lr = int(h_hr / scale)
        w_lr = int(w_hr / scale)
        img_lr_tensor = resize_fn(img_tensor, (h_lr, w_lr))

        lr_img = TF.to_pil_image(img_lr_tensor.clamp_(0, 1))
        base, ext = os.path.splitext(name)
        out_name = f"{base}x{scale:.1f}{ext}"
        lr_img.save(os.path.join(lr_root, out_name))

        print(f"Saved: {out_name}")

gt_root = '/home/jijang/ssd_data/projects/ContinuousSR/data/1_Image_SR/test/Set5/HR'
lr_root = '/home/jijang/ssd_data/projects/ContinuousSR/data/1_Image_SR/test/Set5/LR_bicubic/X12'

scale = 12

generate_lr_images(gt_root, lr_root, scale)