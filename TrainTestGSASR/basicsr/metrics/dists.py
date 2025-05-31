import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize
import torch
import os
from collections import Counter
import argparse
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F
from DISTS_pytorch import DISTS


from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.utils import img2tensor

def prepare_image(image, resize=False):
    if resize and min(image.size)>256:
        image = transforms.functional.resize(image,256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)

def cropborder(imgs, border_size = 0):
    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [i[:, :, border_size:-border_size, border_size:-border_size] for i in imgs]
    if len(imgs) == 0:
        return imgs[0]
    else:
        return imgs

@METRIC_REGISTRY.register()
def calculate_dists(img, img2, crop_border, color_order = 'BGR', **kwargs):
    if color_order == 'BGR':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img = prepare_image(img)
    img2 = prepare_image(img2)
    img, img2 = cropborder([img, img2], border_size=crop_border)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # os.environ['http_proxy'] = 'http://nbproxy.mlp.oppo.local:8888'
    # os.environ['https_proxy'] = 'http://nbproxy.mlp.oppo.local:8888'
    model = DISTS().to(device)
    # del os.environ['http_proxy']
    # del os.environ['https_proxy']

    img = img.to(device)
    img2 = img2.to(device)

    score = model(img2, img).item()


    return score
