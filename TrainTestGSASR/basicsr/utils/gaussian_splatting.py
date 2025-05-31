import torch
import numpy as np
import torch.nn.functional as F
import math
import torch.nn as nn

import torchvision.utils
from torchvision.utils import save_image
import time

def rendering_python(sigma_x, sigma_y, rho, coords, colours_with_alpha, sr_size, step_size, device):
    sr_h, sr_w = sr_size[0], sr_size[1]
    num_gs = sigma_x.shape[0]

    sigma_x = sigma_x[...,None]
    sigma_y = sigma_y[...,None]
    rho = rho[...,None]
    covariance = torch.stack(
        [torch.stack([sigma_x**2, rho*sigma_x*sigma_y], dim=-1),
        torch.stack([rho*sigma_x*sigma_y, sigma_y**2], dim=-1)],
        dim=-2
    )

    # Check for positive semi-definiteness
    determinant = (sigma_x**2) * (sigma_y**2) - (rho * sigma_x * sigma_y)**2
    if (determinant < 0).any():
        raise ValueError("Covariance matrix must be positive semi-definite")

    inv_covariance = torch.inverse(covariance)

    # Sampling progress
    num_step = int(10 * 2 / step_size)
    ax_h_batch = torch.tensor([i * step_size for i in range(num_step)]).to(device)[None]
    ax_h_batch -= ax_h_batch.mean()
    ax_w_batch = torch.tensor([i * step_size for i in range(num_step)]).to(device)[None]
    ax_w_batch -= ax_w_batch.mean()

    # Expanding dims for broadcasting
    ax_batch_expanded_x = ax_h_batch.unsqueeze(-1).expand(-1, -1, num_step)
    ax_batch_expanded_y = ax_w_batch.unsqueeze(1).expand(-1, num_step, -1)

    # Creating a batch-wise meshgrid using broadcasting
    xx, yy = ax_batch_expanded_x, ax_batch_expanded_y

    xy = torch.stack([xx, yy], dim=-1)

    max_buffer = 2000
    final_image = torch.zeros((3, sr_h, sr_w), device=device)
    for i in range(num_gs // max_buffer + 1):
        # print('processing gs buffer id:', i, num_gs // max_buffer )
        s_idx, e_idx = i * max_buffer, min((i + 1) * max_buffer, num_gs)
        buffer_size = e_idx - s_idx
        if buffer_size == 0:
            break
        # print(f"buffer_size is {buffer_size}")
        buff_inv_covariance = inv_covariance[s_idx:e_idx]
        buff_covariance = covariance[s_idx:e_idx]
        buffer_pixel_coords = coords[s_idx:e_idx]
        buffer_alpha = colours_with_alpha[s_idx:e_idx].unsqueeze(-1).unsqueeze(-1)

        z = torch.einsum('b...i,b...ij,b...j->b...', xy, -0.5 * buff_inv_covariance, xy)
        kernel = torch.exp(z) / (2 * torch.tensor(np.pi, device=device) * torch.sqrt(torch.det(buff_covariance)).view(buffer_size, 1, 1))

        kernel_max = kernel.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        kernel_normalized = kernel / (kernel_max + 1e-4)
        kernel_reshaped = kernel_normalized.repeat(1, 3, 1).view(buffer_size * 3, num_step, num_step)
        kernel_reshaped = kernel_reshaped.unsqueeze(0).reshape(buffer_size, 3, num_step, num_step)

        b, c, h, w = kernel_reshaped.shape

        # Create a batch of 2D affine matrices
        theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
        theta[:, 0, 0] = 1 * sr_w / num_step
        theta[:, 1, 1] = 1 * sr_h / num_step
        theta[:, 0, 2] = -buffer_pixel_coords[:, 0] * sr_w / num_step  # !!!!!!!! note -1
        theta[:, 1, 2] = -buffer_pixel_coords[:, 1] * sr_h / num_step  # !!!!!!!! note -1

        grid = F.affine_grid(theta, size=(b, c, sr_h, sr_w), align_corners=False)  # !!!!! align_corners=False
        kernel_reshaped_translated = F.grid_sample(kernel_reshaped, grid,
                                                   align_corners=False)  # !!!! align_corners=False
        buffer_final_image = buffer_alpha * kernel_reshaped_translated
        final_image += buffer_final_image.sum(0)

    return final_image

def rendering_cuda(sigma_x, sigma_y, rho, coords, colours_with_alpha, sr_size, step_size, device):
    from basicsr.utils.gs_cuda.gswrapper import GSCUDA
    sigmas = torch.cat([sigma_y/step_size*2/(sr_size[1] - 1), sigma_x/step_size*2/(sr_size[0] - 1),  rho], dim=-1).contiguous()  # (gs num, 3)
    coords[:, 0] = (coords[:, 0] + 1 - 1/sr_size[1]) * sr_size[1] / (sr_size[1] - 1) - 1.0
    coords[:, 1] = (coords[:, 1] + 1 - 1/sr_size[0]) * sr_size[0] / (sr_size[0] - 1) - 1.0
    colours_with_alpha = colours_with_alpha.contiguous()  # (gs num, 3)
    rendered_img = torch.zeros(sr_size[0], sr_size[1], 3).to(device).type(torch.float32).contiguous()
    # with torch.no_grad():
    #    final_image = GSCUDA.apply(sigmas, coords, colours_with_alpha, rendered_img)
    # final_image = (torch.sum(sigmas)+torch.sum(coords)+torch.sum(colours_with_alpha))*final_image
    final_image = GSCUDA.apply(sigmas, coords, colours_with_alpha, rendered_img)
    final_image = final_image.permute(2, 0, 1).contiguous()
    return final_image

def rendering_cuda_buffer(sigma_x, sigma_y, rho, coords, colours_with_alpha, sr_size, step_size, device, buffer_size = 1000000):
    from basicsr.utils.gs_cuda.gswrapper import GSCUDA
    sigmas = torch.cat([sigma_y/step_size*2/(sr_size[1] - 1), sigma_x/step_size*2/(sr_size[0] - 1),  rho], dim=-1).contiguous()  # (gs num, 3)
    coords[:, 0] = (coords[:, 0] + 1 - 1/sr_size[1]) * sr_size[1] / (sr_size[1] - 1) - 1.0
    coords[:, 1] = (coords[:, 1] + 1 - 1/sr_size[0]) * sr_size[0] / (sr_size[0] - 1) - 1.0
    colours_with_alpha = colours_with_alpha.contiguous()  # (gs num, 3)
    final_image = torch.zeros(sr_size[0], sr_size[1], 3).to(device).type(torch.float32).contiguous()

    # buffer
    buffer_num = len(sigma_x)// buffer_size+1
    for buffer_id in range(buffer_num):
        # print(f'processing{buffer_id+1}/{buffer_num}')
        idx_start, idx_end = buffer_id * buffer_size, (buffer_id+1) * buffer_size
        final_image = GSCUDA.apply(sigmas[idx_start:idx_end], coords[idx_start:idx_end],
                                    colours_with_alpha[idx_start:idx_end], final_image)
        # final_image += buffer_image
    final_image = final_image.permute(2, 0, 1).contiguous()
    return final_image

def rendering_cuda_new(sigma_x, sigma_y, rho, coords, colours_with_alpha, sr_size, step_size,  device, dmax=1):
    from basicsr.utils.gs_cuda_dmax.gswrapper import GSCUDA
    sigmas = torch.cat([sigma_y/step_size*2/(sr_size[1] - 1), sigma_x/step_size*2/(sr_size[0] - 1),  rho], dim=-1).contiguous()  # (gs num, 3)
    coords[:, 0] = (coords[:, 0] + 1 - 1/sr_size[1]) * sr_size[1] / (sr_size[1] - 1) - 1.0
    coords[:, 1] = (coords[:, 1] + 1 - 1/sr_size[0]) * sr_size[0] / (sr_size[0] - 1) - 1.0
    colours_with_alpha = colours_with_alpha.contiguous()  # (gs num, 3)
    rendered_img = torch.zeros(sr_size[0], sr_size[1], 3).to(device).type(torch.float32).contiguous()
    # with torch.no_grad():
    #     final_image = GSCUDA.apply(sigmas, coords, colours_with_alpha, rendered_img, dmax)
    # final_image = (torch.sum(sigmas)+torch.sum(coords)+torch.sum(colours_with_alpha))*final_image
    final_image = GSCUDA.apply(sigmas, coords, colours_with_alpha, rendered_img, dmax)
    final_image = final_image.permute(2, 0, 1).contiguous()
    return final_image

def rendering_cuda_new_buffer(sigma_x, sigma_y, rho, coords, colours_with_alpha, sr_size, step_size,  device, dmax=1, buffer_size = 1000000):
    from basicsr.utils.gs_cuda_dmax.gswrapper import GSCUDA
    sigmas = torch.cat([sigma_y/step_size*2/(sr_size[1] - 1), sigma_x/step_size*2/(sr_size[0] - 1),  rho], dim=-1).contiguous()  # (gs num, 3)
    coords[:, 0] = (coords[:, 0] + 1 - 1/sr_size[1]) * sr_size[1] / (sr_size[1] - 1) - 1.0
    coords[:, 1] = (coords[:, 1] + 1 - 1/sr_size[0]) * sr_size[0] / (sr_size[0] - 1) - 1.0
    colours_with_alpha = colours_with_alpha.contiguous()  # (gs num, 3)

    final_image = torch.zeros(sr_size[0], sr_size[1], 3).to(device).type(torch.float32).contiguous()
    # with torch.no_grad():
    #     final_image = GSCUDA.apply(sigmas, coords, colours_with_alpha, rendered_img, dmax)
    # final_image = (torch.sum(sigmas)+torch.sum(coords)+torch.sum(colours_with_alpha))*final_image

    # buffer
    buffer_num = len(sigma_x)// buffer_size+1
    for buffer_id in range(buffer_num):
        # print(f'processing{buffer_id+1}/{buffer_num}')
        idx_start, idx_end = buffer_id * buffer_size, (buffer_id+1) * buffer_size
        final_image = GSCUDA.apply(sigmas[idx_start:idx_end], coords[idx_start:idx_end],
                                    colours_with_alpha[idx_start:idx_end], final_image, dmax)
        # final_image += buffer_image

    final_image = final_image.permute(2, 0, 1).contiguous()
    return final_image


def generate_2D_gaussian_splatting_step(sr_size, gs_parameters, scale, scale_modify,
                                        sample_coords = None, default_step_size = 1.2, 
                                        cuda_rendering=True, mode = 'scale_modify',
                                        if_dmax = True,
                                        dmax_mode = 'fix',
                                        dmax = 25):

    # set step_size according to scale factor
    if mode == 'scale':
        final_scale = scale
    elif mode == 'scale_modify':
        assert scale_modify[0] == scale_modify[1], f"scale_modify is not the same-{scale_modify}"
        final_scale = scale_modify[0]
    step_size = default_step_size/ final_scale

    # prepare gaussian properties
    sigma_x = 0.99999 * torch.sigmoid(gs_parameters[:, 0:1]) + 1e-6
    sigma_y = 0.99999 * torch.sigmoid(gs_parameters[:, 1:2]) + 1e-6
    rho = 0.999999 * torch.tanh(gs_parameters[:, 2:3])
    alpha = torch.sigmoid(gs_parameters[:, 3:4])
    colours = torch.sigmoid(gs_parameters[:, 4:7])
    coords = (gs_parameters[:, 7:9] * 2 - 1)
    colours_with_alpha = colours * alpha


    ## todo for save GS parameters
    # GS_parameters = torch.cat([sigma_x, sigma_y, rho, alpha, colours, coords], dim = 1)
    # torch.save(GS_parameters.cpu(), "/home/notebook/code/personal/S9053766/chendu/myprojects/GSSR_20240606/results/0804_48*48.pt")
    # print(f"GS_parameter shape is {GS_parameters.shape}")
    # print(f"-------")

    # todo for visualization the position of Gaussian
    # select = (torch.randn_like(alpha[..., 0])>2.5)
    # colours_with_alpha[select, 0] = 1
    # colours_with_alpha[select, 1] = 0
    # colours_with_alpha[select, 2] = 0
    # todo for visualization the shape of Gaussian
    # sigma_x = torch.ones_like(sigma_x)*0.05
    # sigma_y = torch.ones_like(sigma_y)*0.05
    # rho = torch.ones_like(rho) * 0
    # colours_with_alpha = torch.ones_like(colours_with_alpha)*0.5

    # rendering
    if cuda_rendering:
        if if_dmax:
            if dmax_mode == 'dynamic':
                dmax = (dmax + 2) / min(sr_size[0], sr_size[1])
            final_image = rendering_cuda_new(sigma_x, sigma_y, rho, coords, colours_with_alpha, sr_size, step_size, dmax=dmax, device=sigma_x.device)
        else:
            final_image = rendering_cuda(sigma_x, sigma_y, rho, coords, colours_with_alpha, sr_size, step_size, device=sigma_x.device)
    else:
        final_image = rendering_python(sigma_x, sigma_y, rho, coords, colours_with_alpha, sr_size, step_size, device=sigma_x.device)
    if sample_coords is not None:
        sample_RGB_values = [final_image[:, coord[0], coord[1]] for coord in sample_coords]
        final_image = torch.stack(sample_RGB_values, dim = 1)
    return final_image

def generate_2D_gaussian_splatting_step_buffer(sr_size, gs_parameters, scale, scale_modify,
                                        sample_coords = None, default_step_size = 1.2, 
                                        cuda_rendering=True, mode = 'scale_modify',
                                        if_dmax = True,
                                        dmax_mode = 'fix',
                                        dmax = 25,
                                        buffer_size = 4000000):

    # set step_size according to scale factor
    if mode == 'scale':
        final_scale = scale
    elif mode == 'scale_modify':
        assert scale_modify[0] == scale_modify[1], f"scale_modify is not the same-{scale_modify}"
        final_scale = scale_modify[0]
    step_size = default_step_size/ final_scale

    # prepare gaussian properties
    sigma_x = 0.99999 * torch.sigmoid(gs_parameters[:, 0:1]) + 1e-6
    sigma_y = 0.99999 * torch.sigmoid(gs_parameters[:, 1:2]) + 1e-6
    rho = 0.999999 * torch.tanh(gs_parameters[:, 2:3])
    alpha = torch.sigmoid(gs_parameters[:, 3:4])
    colours = torch.sigmoid(gs_parameters[:, 4:7])
    coords = (gs_parameters[:, 7:9] * 2 - 1)
    colours_with_alpha = colours * alpha

    # rendering
    if cuda_rendering:
        if if_dmax:
            if dmax_mode == 'dynamic':
                dmax = (dmax + 2) / min(sr_size[0], sr_size[1])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            final_image = rendering_cuda_new_buffer(sigma_x, sigma_y, rho, coords, colours_with_alpha, 
                                                    sr_size, step_size, dmax=dmax, device=sigma_x.device,
                                                    buffer_size = buffer_size)
            torch.cuda.synchronize()
            end = time.time()
            time_cost = end - start
            used_memory = torch.cuda.max_memory_allocated()
            print(f"Rendering time cost is {(time_cost)*1000} ms, Memory is {used_memory /1024 /1024} MB")
        else:
            final_image = rendering_cuda_buffer(sigma_x, sigma_y, rho, coords, colours_with_alpha, 
                                                sr_size, step_size, device=sigma_x.device,
                                                buffer_size = buffer_size)
    else:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.time()

        final_image = rendering_python(sigma_x, sigma_y, rho, coords, colours_with_alpha, sr_size, step_size, device=sigma_x.device)

        torch.cuda.synchronize()
        end = time.time()
        time_cost = end - start
        used_memory = torch.cuda.max_memory_allocated()
        print(f"Rendering time cost is {(time_cost)*1000} ms, Memory is {used_memory /1024 /1024} MB")
    if sample_coords is not None:
        sample_RGB_values = [final_image[:, coord[0], coord[1]] for coord in sample_coords]
        final_image = torch.stack(sample_RGB_values, dim = 1)
    return final_image