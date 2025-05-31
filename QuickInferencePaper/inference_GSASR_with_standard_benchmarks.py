import argparse
import cv2
import glob
import numpy as np
import os
import torch
import math
import torch.nn.functional as F
import time

from utils.edsrbaseline import EDSRNOUP
from utils.rdn import RDNNOUP
from utils.swinir import SwinIRNOUP
from utils.fea2gs import Fea2GS
from utils.gaussian_splatting import generate_2D_gaussian_splatting_step, generate_2D_gaussian_splatting_step_buffer
from utils.split_and_joint_image import split_and_joint_image

def preprocess(x, denominator):
    # pad input image to be a multiple of denominator
    _,c,h,w = x.shape
    if h % denominator > 0:
        pad_h = denominator - h % denominator
    else:
        pad_h = 0
    if w % denominator > 0:
        pad_w = denominator - w % denominator
    else:
        pad_w = 0
    x_new = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
    return x_new


def postprocess(x, gt_size_h, gt_size_w):
    x_new = x[:, :, :gt_size_h, :gt_size_w]
    return x_new


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.encoder_name == 'EDSR-Baseline':
        encoder = EDSRNOUP()
    elif args.encoder_name == 'RDN':
        encoder = RDNNOUP()
    elif args.encoder_name == 'SwinIR':
        encoder = SwinIRNOUP()
    else:
        raise ValueError(f"args.encoder_name-{args.encoder_name} must be EDSR-Baseline, RDN or SwinIR")
    
    decoder = Fea2GS()


    final_save_path = os.path.join(args.save_sr_path, args.encoder_name, f"x{args.scale}")

    os.makedirs(final_save_path, exist_ok=True)

    encoder.load_state_dict(torch.load(args.encoder_path)['params_ema'], strict=True)
    encoder.eval()
    encoder = encoder.to(device)

    decoder.load_state_dict(torch.load(args.decoder_path)['params_ema'], strict=True)
    decoder.eval()
    decoder = decoder.to(device)

    time_cost_list = []
    used_memory_list = []

    img_list = os.listdir(args.input_img_path)
    for img in img_list:
        img_path = os.path.join(args.input_img_path, img)

        imgname, ext = os.path.splitext(os.path.basename(img_path))
        print(f"Testing {os.path.basename(img_path)} with up-scaling factor {args.scale}")
        # read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)

        ### Since in real-world scenes, there is no GT, we use math.floor to ensure that the boundary pixels of SR image will not overflow
        gt_size = [math.floor(args.scale * img.shape[2]), math.floor(args.scale * img.shape[3])]

        if args.tile_process:
            assert args.tile_size % args.denominator == 0, f"args.tile_size-{args.tile_size} should be divisible by args.denominator-{args.denominator}"
            with torch.no_grad():
                output = split_and_joint_image(lq = img, scale_factor=args.scale,
                                                split_size=args.tile_size,
                                                overlap_size=args.tile_overlap,
                                                model_g=encoder,
                                                model_fea2gs=decoder,
                                                crop_size=args.crop_size,
                                                scale_modify = torch.tensor([args.scale, args.scale]),
                                                default_step_size = 1.2,
                                                cuda_rendering=True,
                                                mode = 'scale_modify',
                                                if_dmax = True,
                                                dmax_mode = 'fix',
                                                dmax = args.dmax)

        else:
            ### the LR image should be divisible by the denominator
            lq_pad = preprocess(img, args.denominator)
            gt_size_pad = torch.tensor([math.floor(args.scale * lq_pad.shape[2]), math.floor(args.scale * lq_pad.shape[3])])
            gt_size_pad = gt_size_pad.unsqueeze(0)

            ### Test the infernce time and GPU memory
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.time()

            with torch.no_grad():
                encoder_output = encoder(lq_pad)  # b,c,h,w
                scale_vector = torch.tensor(args.scale, dtype=torch.float32).unsqueeze(0).to(device)

                batch_gs_parameters = decoder(encoder_output, scale_vector)
                gs_parameters = batch_gs_parameters[0, :]
                b_output = generate_2D_gaussian_splatting_step(gs_parameters=gs_parameters,
                                    sr_size=gt_size_pad[0],
                                    scale = args.scale,
                                    sample_coords=None,
                                    scale_modify = torch.tensor([args.scale, args.scale]),
                                    default_step_size = 1.2,
                                    cuda_rendering=True,
                                    mode = 'scale_modify',
                                    if_dmax = True,
                                    dmax_mode = 'fix',
                                    dmax = args.dmax)
                output = b_output.unsqueeze(0)
            
            torch.cuda.synchronize()
            end = time.time()
            time_cost = end - start
            used_memory = torch.cuda.max_memory_allocated()

            print(f"Time cost is {time_cost*1000:.4f} ms, GPU memory is {used_memory /1024 /1024:.4f} MB")

            time_cost_list.append(time_cost)
            used_memory_list.append(used_memory)

        output = postprocess(output, gt_size[0], gt_size[1])

        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(final_save_path, f'{imgname}_{args.encoder_name}{ext}'), output)

    ### When computing the average computational cost, we remove the first two results since the time and GPU memory are unstable during the inference initialization phase
    if len(time_cost_list) > 0 and len(time_cost_list) > 0:
        print(f"Average time is {sum(time_cost_list[2:])/len(time_cost_list[2:])*1000:.4f} ms")
        print(f"Average GPU memory is {sum(used_memory_list[2:])/len(used_memory_list[2:]) / 1024 / 1024:.4f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--encoder_path',
        type=str,
        default='/home/notebook/code/personal/S9053766/chendu/FinalUpload/GSASR/pretrained_models/Paper/EDSR-Baseline-Train-with-DIV2K/net_g.pth')
    # parser.add_argument(
    #     '--encoder_path',
    #     type=str,
    #     default='/home/notebook/code/personal/S9053766/chendu/FinalUpload/GSASR/pretrained_models/Paper/RDN-Train-with-DIV2K/net_g.pth')
    # parser.add_argument(
    #     '--encoder_path',
    #     type=str,
    #     default='/home/notebook/code/personal/S9053766/chendu/FinalUpload/GSASR/pretrained_models/Paper/SwinIR-Train-with-DIV2K/net_g.pth')
    parser.add_argument(
        '--decoder_path',
        type=str,
        default='/home/notebook/code/personal/S9053766/chendu/FinalUpload/GSASR/pretrained_models/Paper/EDSR-Baseline-Train-with-DIV2K/net_fea2gs.pth')
    # parser.add_argument(
    #     '--decoder_path',
    #     type=str,
    #     default='/home/notebook/code/personal/S9053766/chendu/FinalUpload/GSASR/pretrained_models/Paper/RDN-Train-with-DIV2K/net_fea2gs.pth')
    # parser.add_argument(
    #     '--decoder_path',
    #     type=str,
    #     default='/home/notebook/code/personal/S9053766/chendu/FinalUpload/GSASR/pretrained_models/Paper/SwinIR-Train-with-DIV2K/net_fea2gs.pth')
    parser.add_argument('--input_img_path', type=str, default='/home/notebook/data/sharedgroup/RG_YLab/aigc_share_group_data/chendu/dataset/TimeTest/DIV2K100_GT720/x4/LR', help='input test image folder')
    parser.add_argument('--save_sr_path', type=str, default='/home/notebook/code/personal/S9053766/chendu/FinalUpload/GSASR/figures/QuickInferenceExpResults/Benchmark', help='output folder')
    parser.add_argument('--scale', type = float, default = 4)
    parser.add_argument('--suffix', type = str, default = 'GSASR')
    parser.add_argument('--encoder_name', type = str, default = 'EDSR-Baseline') # EDSR-Baseline, RDN or SwinIR
    ### In the following, denominator must be the least common miltiple of the encoder's window size (if there exists window attention) and the decoder's window size
    ### For EDSR-Baseline and RDN, please set denominator to 12, for SwinIR, please set denominator to 24
    parser.add_argument('--denominator', type = int, default = 12) 
    ### If the GPU memory is limited, please use tile_process to lower the inference GPU memory
    ### Note that, if using tile_process, the final reconstruction result would sacrifice some fidelity
    ### The smaller tile_size, the worse performance
    ### In most cases, if your GPU memory is limited, please only change tile_size, but we recommend not to change tile_overlap or crop_size
    ### For example, when testing on DIV2K/LSDIR/Urban100 (all of the datasets with full size images), to lower the memory, we set tile_size to 480
    ### However, when testing the computational cost on DIV2K with 720*720 GT images, we don't use tile_process. Since it will remarkably lower the GPU memory and acclerate the speed. For fair comparison, we just test with full size LR image as input.
    parser.add_argument('--tile_process', action='store_true')  
    parser.add_argument('--tile_size', type = int, default = 480) # tile_size must be divisible by denominator
    parser.add_argument('--tile_overlap', type = int, default = 8) # 2 * tile_overlap < tile_size
    parser.add_argument('--crop_size', type = int, default = 4) # 2* crop_size <= tile_overlap
    ### The following dmax means the rasterization ratio in our paper, 0 < dmax <= 1.0
    ### If the scaling factor is small (less than 3), or the input LR image size is larger than 200*200, you could set dmax to 0.05 to further accelerate the speed while not sacrificing too much performance.
    ### In you own implementation, we recommend that dmax >= 0.05, when we test computational cost with x2 scaling factor on DIV2K with 720*720 GT images, we set it to 0.05, while on other larger scaling factors, we set it to 0.1
    parser.add_argument('--dmax', type = float, default = 0.1)
    args = parser.parse_args()
    main(args)
