import argparse
import cv2
import glob
import numpy as np
import os
import torch
import math
import torch.nn.functional as F

from utils.edsrbaseline import EDSRNOUP
from utils.hatropeamp import HATNOUP_ROPE_AMP
from utils.rdn import RDNNOUP
from utils.swinir import SwinIRNOUP
from utils.fea2gsropeamp import Fea2GS_ROPE_AMP
from utils.gaussian_splatting import generate_2D_gaussian_splatting_step, generate_2D_gaussian_splatting_step_buffer
from utils.split_and_joint_image import split_and_joint_image
from huggingface_hub import hf_hub_download

def load_model(
    pretrained_model_name_or_path: str = "mutou0308/GSASR",
    model_name: str = "EDSR_DIV2K",
    device: str | torch.device = "cuda"
):
    enc_path = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=os.path.join(model_name, 'encoder.pth')
        )
    dec_path = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=os.path.join(model_name, 'decoder.pth')
        )

    enc_weight = torch.load(enc_path, weights_only=True)['params_ema']
    dec_weight = torch.load(dec_path, weights_only=True)['params_ema']

    if model_name in ['EDSR_DIV2K', 'EDSR_DF2K']:
        encoder = EDSRNOUP()
        decoder = Fea2GS_ROPE_AMP()
    elif model_name in ['RDN_DIV2K', 'RDN_DF2K']:
        encoder = RDNNOUP()
        decoder = Fea2GS_ROPE_AMP(num_crossattn_blocks = 2)
    elif model_name in ['SWIN_DIV2K', 'SWIN_DF2K']:
        encoder = SwinIRNOUP()
        decoder = Fea2GS_ROPE_AMP(num_crossattn_blocks=2, num_crossattn_layers=4, num_gs_seed=256, window_size=16)
    elif model_name in ['HATL_SA1B']:
        encoder = HATNOUP_ROPE_AMP()
        decoder = Fea2GS_ROPE_AMP(channel=192, num_crossattn_blocks=4, num_crossattn_layers=4, num_selfattn_blocks=8, num_selfattn_layers=6,
                                  num_gs_seed=256, window_size=16)
    else:
        raise ValueError(f"args.model-{model_name} must be in ['EDSR_DIV2K', 'EDSR_DF2K', 'RDN_DIV2K', 'RDN_DF2K', 'SWIN_DIV2K', 'SWIN_DF2K', 'HATL_SA1B']")

    encoder.load_state_dict(enc_weight, strict=True)
    decoder.load_state_dict(dec_weight, strict=True)
    encoder.eval()
    decoder.eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    return encoder, decoder


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
    encoder, decoder = load_model(model_name=args.model, device=device)

    imgname, ext = os.path.splitext(os.path.basename(args.input_img_path))
    print(f"Testing {os.path.basename(args.input_img_path)} with up-scaling factor {args.scale}")
    # read image
    img = cv2.imread(args.input_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = img.unsqueeze(0).to(device)

    ### Since in real-world scenes, there is no GT, we use math.floor to ensure that the boundary pixels of SR image will not overflow
    gt_size = [math.floor(args.scale * img.shape[2]), math.floor(args.scale * img.shape[3])]

    if args.AMP_test:
        with torch.autocast(device_type = 'cuda', dtype = torch.bfloat16):
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

    else:
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

    output = postprocess(output, gt_size[0], gt_size[1])

    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    os.makedirs(args.save_sr_path, exist_ok=True)
    cv2.imwrite(os.path.join(args.save_sr_path, f'{imgname}_{args.suffix}_{args.model}_{args.scale}{ext}'), output)
    print(f"Saved SR image to {os.path.join(args.save_sr_path, f'{imgname}_{args.suffix}_{args.model}_{args.scale}{ext}')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img_path', type=str, default='./assets/0873x4_cropped_120x120.png', help='input test image folder')
    parser.add_argument('--save_sr_path', type=str, default='./output', help='output folder')
    parser.add_argument('--scale', type = float, default = 20.0)
    parser.add_argument('--suffix', type = str, default = 'GSASR')
    parser.add_argument('--model', type = str, default = 'HATL_SA1B', choices=['EDSR_DIV2K', 'EDSR_DF2K', 'RDN_DIV2K', 'RDN_DF2K', 'SWIN_DIV2K', 'SWIN_DF2K', 'HATL_SA1B']) 
    ### In the following, denominator must be the least common miltiple of the encoder's window size (if there exists window attention) and the decoder's window size
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
    ### If you want to further lower the GPU memory and accelerate the inference speed, we recommend you to test with Automatic Mixed Precision (AMP).
    ### Please note that, AMP might sacrifice a little fidelity.
    ### During training phase, we train model with AMP, to ensure that it will also work well under inference stage.
    ### Please note that, we use AMP on NVIDIA A100 GPU with a mixture of bfloat16 and fp32 precision. However, some GPUs do not support for bfloat16 precision, such as NVIDIA V100.
    ### To make sure you could use AMP in your environment, we recommend you to check that whether your GPU support for bfloat16 precision.
    ### If your GPU does not support for bfloat16, please don't use AMP_test
    parser.add_argument('--AMP_test', action='store_true') 
    args = parser.parse_args()
    # set denominator 12 for EDSR and RDN, 16 for SwinIR and HATL
    args.denominator = 12 if args.model in ['EDSR_DIV2K', 'EDSR_DF2K', 'RDN_DIV2K', 'RDN_DF2K'] else 16
    main(args)
