import torch
import torch.nn.functional as F
import math
from basicsr.utils.gaussian_splatting import generate_2D_gaussian_splatting_step, generate_2D_gaussian_splatting_step_buffer


def split_and_joint_image(lq, scale_factor, split_size,
                                            overlap_size, model_g, model_fea2gs,
                                            scale_modify, crop_size = 2,
                                            default_step_size = 1.2, mode = 'scale_modify',
                                            cuda_rendering = True,
                                            if_dmax = False,
                                            dmax_mode = 'fix',
                                            dmax = 25):
    h_lq, w_lq = lq.shape[-2:]

    # assert h_lq > split_size, f'h_lq-{h_lq} should be larger than split_size-{split_size}, please do not use tile_process, or decrease the split_size'
    # assert w_lq > split_size, f'w_lq-{w_lq} should be larger than split_size-{split_size}, please do not use tile_process, or decrease the split_size'

    assert overlap_size > 0 and overlap_size < split_size // 2, f"overlap size is wrong"

    tile_nums_h = math.ceil((h_lq - overlap_size) / (split_size - overlap_size))
    tile_nums_w = math.ceil((w_lq - overlap_size) / (split_size - overlap_size))

    pad_h_lq = tile_nums_h * (split_size - overlap_size) + overlap_size - h_lq
    pad_w_lq = tile_nums_w * (split_size - overlap_size) + overlap_size - w_lq

    assert pad_h_lq < h_lq, f'pad_h_lq-{pad_h_lq} should be smaller than h_lq-{h_lq}, please decrease the split_size-{split_size}'
    assert pad_w_lq < w_lq, f'pad_w_lq-{pad_w_lq} should be smaller than w_lq-{w_lq}, please decrease the split_size-{split_size}'

    lq_pad = F.pad(input=lq, pad=(0, pad_w_lq, 0, pad_h_lq), mode='reflect')

    split_size_sr = math.ceil(split_size * scale_factor)
    sr_tile_list = []
    for h_num in range(tile_nums_h):
        for w_num in range(tile_nums_w):
            tile_lq_position_start_h = h_num * (split_size - overlap_size)
            tile_lq_position_start_w = w_num * (split_size - overlap_size)
            tile_lq_position_end_h = tile_lq_position_start_h + split_size
            tile_lq_position_end_w = tile_lq_position_start_w + split_size

            input_tile = lq_pad[:,:, tile_lq_position_start_h:tile_lq_position_end_h, tile_lq_position_start_w:tile_lq_position_end_w]

            model_g_output = model_g(input_tile)
            
            scale_vector = scale_modify[0].unsqueeze(0).to(model_g_output.device)
            batch_gs_parameters = model_fea2gs(model_g_output, scale_vector)

            gs_parameters = batch_gs_parameters[0, :]
            b_output = generate_2D_gaussian_splatting_step(sr_size=torch.tensor([split_size_sr, split_size_sr]), gs_parameters=gs_parameters,
                                                           scale=scale_factor, sample_coords=None,
                                                           scale_modify = scale_modify,
                                                           default_step_size = default_step_size, mode = mode,
                                                           cuda_rendering = cuda_rendering,
                                                           if_dmax = if_dmax,
                                                           dmax_mode = dmax_mode,
                                                           dmax = dmax)
            sr_tile_list.append(b_output.unsqueeze(0))

    tile_sr_h = sr_tile_list[0].shape[2]
    tile_sr_w = sr_tile_list[0].shape[3]

    assert tile_sr_w == split_size_sr and tile_sr_h == split_size_sr, \
        f'tile_sr_h-{tile_sr_w}, tile_sr_w-{tile_sr_w}, split_size_sr-{split_size_sr} is not the same'

    overlap_sr = math.ceil(overlap_size * scale_factor)

    sr_pad = torch.zeros(lq.shape[0], lq.shape[1],
                         (tile_nums_h - 1) * (split_size_sr - overlap_sr) + split_size_sr,
                         (tile_nums_w - 1) * (split_size_sr - overlap_sr) + split_size_sr,
                         device=lq.device)

    idx = 0
    
    if scale_factor != int(scale_factor):
        for h_num in range(tile_nums_h):
            for w_num in range(tile_nums_w):
                tile_sr_position_start_w = w_num * (split_size_sr - overlap_sr)
                tile_sr_position_end_w = tile_sr_position_start_w + split_size_sr
                tile_sr_position_start_h = h_num * (split_size_sr - overlap_sr)
                tile_sr_position_end_h = tile_sr_position_start_h + split_size_sr
                if h_num == 0 and w_num == 0:
                    sr_pad[:, :, tile_sr_position_start_h:tile_sr_position_end_h,
                    tile_sr_position_start_w:tile_sr_position_end_w] = sr_tile_list[idx]
                elif h_num == 0 and w_num !=0:
                    if w_num != tile_nums_w - 1:
                        sr_pad[:, :, tile_sr_position_start_h:tile_sr_position_end_h,
                        tile_sr_position_start_w+crop_size:tile_sr_position_end_w] = sr_tile_list[idx][:,:,:,crop_size:]
                    else:
                        sr_pad[:, :, tile_sr_position_start_h:tile_sr_position_end_h,
                        tile_sr_position_start_w+crop_size:sr_pad.shape[3]] = sr_tile_list[idx][:,:,:,crop_size:sr_pad.shape[3] - tile_sr_position_start_w]
                elif h_num != 0 and w_num ==0:
                    if h_num != tile_nums_h - 1:
                        sr_pad[:, :, tile_sr_position_start_h+crop_size:tile_sr_position_end_h,
                        tile_sr_position_start_w:tile_sr_position_end_w] = sr_tile_list[idx][:,:,crop_size:,:]
                    else:
                        sr_pad[:, :, tile_sr_position_start_h+crop_size:sr_pad.shape[2],
                        tile_sr_position_start_w:tile_sr_position_end_w] = sr_tile_list[idx][:,:,crop_size:sr_pad.shape[2] - tile_sr_position_start_h,:]
                else:
                    if w_num != tile_nums_w - 1 and h_num != tile_nums_h - 1:
                        sr_pad[:,:,tile_sr_position_start_h+crop_size:tile_sr_position_end_h,
                        tile_sr_position_start_w+crop_size:tile_sr_position_end_w] = sr_tile_list[idx][:,:,crop_size:,crop_size:]
                    elif w_num == tile_nums_w - 1 and h_num != tile_nums_h - 1:
                        sr_pad[:, :, tile_sr_position_start_h:tile_sr_position_end_h,
                        tile_sr_position_start_w+crop_size:sr_pad.shape[3]] = sr_tile_list[idx][:,:,:,crop_size:sr_pad.shape[3] - tile_sr_position_start_w]
                    elif w_num != tile_nums_w - 1 and h_num == tile_nums_h - 1:
                        sr_pad[:, :, tile_sr_position_start_h+crop_size:sr_pad.shape[2],
                        tile_sr_position_start_w:tile_sr_position_end_w] = sr_tile_list[idx][:,:,crop_size:sr_pad.shape[2] - tile_sr_position_start_h,:]
                    elif w_num == tile_nums_w - 1 and h_num == tile_nums_h - 1:
                        sr_pad[:,:,tile_sr_position_start_h+crop_size:sr_pad.shape[2],
                        tile_sr_position_start_w+crop_size:sr_pad.shape[3]] = sr_tile_list[idx][:,:,crop_size:sr_pad.shape[2] - tile_sr_position_start_h,crop_size:sr_pad.shape[3] - tile_sr_position_start_w]
                idx = idx + 1
    else:
        for h_num in range(tile_nums_h):
            for w_num in range(tile_nums_w):
                tile_sr_position_start_w = w_num * (split_size_sr - overlap_sr)
                tile_sr_position_end_w = tile_sr_position_start_w + split_size_sr
                tile_sr_position_start_h = h_num * (split_size_sr - overlap_sr)
                tile_sr_position_end_h = tile_sr_position_start_h + split_size_sr
                if h_num == 0 and w_num == 0:
                    sr_pad[:, :, tile_sr_position_start_h:tile_sr_position_end_h,
                    tile_sr_position_start_w:tile_sr_position_end_w] = sr_tile_list[idx]
                elif h_num == 0 and w_num !=0:
                    sr_pad[:, :, tile_sr_position_start_h:tile_sr_position_end_h,
                    tile_sr_position_start_w+crop_size:tile_sr_position_end_w] = sr_tile_list[idx][:,:,:,crop_size:]
                elif h_num != 0 and w_num ==0:
                    sr_pad[:, :, tile_sr_position_start_h+crop_size:tile_sr_position_end_h,
                    tile_sr_position_start_w:tile_sr_position_end_w] = sr_tile_list[idx][:,:,crop_size:,:]
                else:
                    sr_pad[:,:,tile_sr_position_start_h+crop_size:tile_sr_position_end_h,
                    tile_sr_position_start_w+crop_size:tile_sr_position_end_w] = sr_tile_list[idx][:,:,crop_size:,crop_size:]
                idx = idx + 1

    print(f"sr_pad shape is {sr_pad.shape}")

    # sr_final = sr_pad[:,:, 0:math.ceil(h_lq * scale_factor), 0: math.ceil(w_lq * scale_factor)]
    sr_final = sr_pad

    return sr_final


def split_and_joint_image_buffer(lq, scale_factor, split_size,
                                            overlap_size, model_g, model_fea2gs,
                                            scale_modify, crop_size = 2,
                                            default_step_size = 1.2, mode = 'scale_modify',
                                            cuda_rendering = True,
                                            if_dmax = False,
                                            dmax_mode = 'fix',
                                            dmax = 25,
                                            buffer_size = 4000000):
    h_lq, w_lq = lq.shape[-2:]

    assert overlap_size > 0 and overlap_size < split_size // 2, f"overlap size is wrong"

    tile_nums_h = math.ceil((h_lq - overlap_size) / (split_size - overlap_size))
    tile_nums_w = math.ceil((w_lq - overlap_size) / (split_size - overlap_size))

    pad_h_lq = tile_nums_h * (split_size - overlap_size) + overlap_size - h_lq
    pad_w_lq = tile_nums_w * (split_size - overlap_size) + overlap_size - w_lq

    assert pad_h_lq < h_lq, f'pad_h_lq-{pad_h_lq} should be smaller than h_lq-{h_lq}, please decrease the split_size-{split_size}'
    assert pad_w_lq < w_lq, f'pad_w_lq-{pad_w_lq} should be smaller than w_lq-{w_lq}, please decrease the split_size-{split_size}'

    lq_pad = F.pad(input=lq, pad=(0, pad_w_lq, 0, pad_h_lq), mode='reflect')

    split_size_sr = math.ceil(split_size * scale_factor)
    sr_tile_list = []
    for h_num in range(tile_nums_h):
        for w_num in range(tile_nums_w):
            tile_lq_position_start_h = h_num * (split_size - overlap_size)
            tile_lq_position_start_w = w_num * (split_size - overlap_size)
            tile_lq_position_end_h = tile_lq_position_start_h + split_size
            tile_lq_position_end_w = tile_lq_position_start_w + split_size

            input_tile = lq_pad[:,:, tile_lq_position_start_h:tile_lq_position_end_h, tile_lq_position_start_w:tile_lq_position_end_w]

            # option 1: if using the scale embedding
            model_g_output = model_g(input_tile)
            
            scale_vector = scale_modify[0].unsqueeze(0).to(model_g_output.device)
            batch_gs_parameters = model_fea2gs(model_g_output, scale_vector)

            # option 2: num_gs = hr_size * gs_repeat_factor
            # batch_gs_parameters = model_fea2gs(model_g(input_tile), scale_modify[0])

            gs_parameters = batch_gs_parameters[0, :]
            b_output = generate_2D_gaussian_splatting_step_buffer(sr_size=torch.tensor([split_size_sr, split_size_sr]), gs_parameters=gs_parameters,
                                                           scale=scale_factor, sample_coords=None,
                                                           scale_modify = scale_modify,
                                                           default_step_size = default_step_size, mode = mode,
                                                           cuda_rendering = cuda_rendering,
                                                           if_dmax = if_dmax,
                                                           dmax_mode = dmax_mode,
                                                           dmax = dmax,
                                                           buffer_size = buffer_size)
            sr_tile_list.append(b_output.unsqueeze(0))
            # sr_tile_list.append(F.interpolate(input=input_tile, size=(math.ceil(split_size * scale_factor), math.ceil(split_size * scale_factor))))

    tile_sr_h = sr_tile_list[0].shape[2]
    tile_sr_w = sr_tile_list[0].shape[3]

    assert tile_sr_w == split_size_sr and tile_sr_h == split_size_sr, \
        f'tile_sr_h-{tile_sr_w}, tile_sr_w-{tile_sr_w}, split_size_sr-{split_size_sr} is not the same'

    overlap_sr = math.ceil(overlap_size * scale_factor)

    sr_pad = torch.zeros(lq.shape[0], lq.shape[1],
                         math.ceil(lq_pad.shape[2] * scale_factor),
                         math.ceil(lq_pad.shape[3] * scale_factor),
                         device=lq.device)

    idx = 0
    
    if scale_factor != int(scale_factor):
        for h_num in range(tile_nums_h):
            for w_num in range(tile_nums_w):
                tile_sr_position_start_w = w_num * (split_size_sr - overlap_sr)
                tile_sr_position_end_w = tile_sr_position_start_w + split_size_sr
                tile_sr_position_start_h = h_num * (split_size_sr - overlap_sr)
                tile_sr_position_end_h = tile_sr_position_start_h + split_size_sr
                if h_num == 0 and w_num == 0:
                    sr_pad[:, :, tile_sr_position_start_h:tile_sr_position_end_h,
                    tile_sr_position_start_w:tile_sr_position_end_w] = sr_tile_list[idx]
                elif h_num == 0 and w_num !=0:
                    if w_num != tile_nums_w - 1:
                        sr_pad[:, :, tile_sr_position_start_h:tile_sr_position_end_h,
                        tile_sr_position_start_w+crop_size:tile_sr_position_end_w] = sr_tile_list[idx][:,:,:,crop_size:]
                    else:
                        sr_pad[:, :, tile_sr_position_start_h:tile_sr_position_end_h,
                        tile_sr_position_start_w+crop_size:sr_pad.shape[3]] = sr_tile_list[idx][:,:,:,crop_size:sr_pad.shape[3] - tile_sr_position_start_w]
                elif h_num != 0 and w_num ==0:
                    if h_num != tile_nums_h - 1:
                        sr_pad[:, :, tile_sr_position_start_h+crop_size:tile_sr_position_end_h,
                        tile_sr_position_start_w:tile_sr_position_end_w] = sr_tile_list[idx][:,:,crop_size:,:]
                    else:
                        sr_pad[:, :, tile_sr_position_start_h+crop_size:sr_pad.shape[2],
                        tile_sr_position_start_w:tile_sr_position_end_w] = sr_tile_list[idx][:,:,crop_size:sr_pad.shape[2] - tile_sr_position_start_h,:]
                else:
                    if w_num != tile_nums_w - 1 and h_num != tile_nums_h - 1:
                        sr_pad[:,:,tile_sr_position_start_h+crop_size:tile_sr_position_end_h,
                        tile_sr_position_start_w+crop_size:tile_sr_position_end_w] = sr_tile_list[idx][:,:,crop_size:,crop_size:]
                    elif w_num == tile_nums_w - 1 and h_num != tile_nums_h - 1:
                        sr_pad[:, :, tile_sr_position_start_h:tile_sr_position_end_h,
                        tile_sr_position_start_w+crop_size:sr_pad.shape[3]] = sr_tile_list[idx][:,:,:,crop_size:sr_pad.shape[3] - tile_sr_position_start_w]
                    elif w_num != tile_nums_w - 1 and h_num == tile_nums_h - 1:
                        sr_pad[:, :, tile_sr_position_start_h+crop_size:sr_pad.shape[2],
                        tile_sr_position_start_w:tile_sr_position_end_w] = sr_tile_list[idx][:,:,crop_size:sr_pad.shape[2] - tile_sr_position_start_h,:]
                    elif w_num == tile_nums_w - 1 and h_num == tile_nums_h - 1:
                        sr_pad[:,:,tile_sr_position_start_h+crop_size:sr_pad.shape[2],
                        tile_sr_position_start_w+crop_size:sr_pad.shape[3]] = sr_tile_list[idx][:,:,crop_size:sr_pad.shape[2] - tile_sr_position_start_h,crop_size:sr_pad.shape[3] - tile_sr_position_start_w]
                idx = idx + 1
    else:
        for h_num in range(tile_nums_h):
            for w_num in range(tile_nums_w):
                tile_sr_position_start_w = w_num * (split_size_sr - overlap_sr)
                tile_sr_position_end_w = tile_sr_position_start_w + split_size_sr
                tile_sr_position_start_h = h_num * (split_size_sr - overlap_sr)
                tile_sr_position_end_h = tile_sr_position_start_h + split_size_sr
                if h_num == 0 and w_num == 0:
                    sr_pad[:, :, tile_sr_position_start_h:tile_sr_position_end_h,
                    tile_sr_position_start_w:tile_sr_position_end_w] = sr_tile_list[idx]
                elif h_num == 0 and w_num !=0:
                    sr_pad[:, :, tile_sr_position_start_h:tile_sr_position_end_h,
                    tile_sr_position_start_w+crop_size:tile_sr_position_end_w] = sr_tile_list[idx][:,:,:,crop_size:]
                elif h_num != 0 and w_num ==0:
                    sr_pad[:, :, tile_sr_position_start_h+crop_size:tile_sr_position_end_h,
                    tile_sr_position_start_w:tile_sr_position_end_w] = sr_tile_list[idx][:,:,crop_size:,:]
                else:
                    sr_pad[:,:,tile_sr_position_start_h+crop_size:tile_sr_position_end_h,
                    tile_sr_position_start_w+crop_size:tile_sr_position_end_w] = sr_tile_list[idx][:,:,crop_size:,crop_size:]
                idx = idx + 1

    print(f"sr_pad shape is {sr_pad.shape}")

    # sr_final = sr_pad[:,:, 0:math.ceil(h_lq * scale_factor), 0: math.ceil(w_lq * scale_factor)]
    sr_final = sr_pad

    return sr_final







