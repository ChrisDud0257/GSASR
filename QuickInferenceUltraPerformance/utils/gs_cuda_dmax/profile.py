import cv2
import torch
import numpy as np
import torch.nn.functional as F

from gswrapper import gaussiansplatting_render

def generate_2D_gaussian_splatting(kernel_size, sigma_x, sigma_y, rho, coords, 
        colours, image_size=(256, 256, 3), device="cuda"):

    batch_size = colours.shape[0]

    sigma_x = sigma_x.view(batch_size, 1, 1)
    sigma_y = sigma_y.view(batch_size, 1, 1)
    rho = rho.view(batch_size, 1, 1)

    covariance = torch.stack(
        [torch.stack([sigma_x**2, rho*sigma_x*sigma_y], dim=-1),
        torch.stack([rho*sigma_x*sigma_y, sigma_y**2], dim=-1)],
        dim=-2
    )

    # Check for positive semi-definiteness
    # determinant = (sigma_x**2) * (sigma_y**2) - (rho * sigma_x * sigma_y)**2
    # if (determinant <= 0).any():
    #     raise ValueError("Covariance matrix must be positive semi-definite")

    inv_covariance = torch.inverse(covariance)

    # Choosing quite a broad range for the distribution [-5,5] to avoid any clipping
    start = torch.tensor([-5.0], device=device).view(-1, 1)
    end = torch.tensor([5.0], device=device).view(-1, 1)
    base_linspace = torch.linspace(0, 1, steps=kernel_size, device=device)
    ax_batch = start + (end - start) * base_linspace

    # Expanding dims for broadcasting
    ax_batch_expanded_x = ax_batch.unsqueeze(-1).expand(-1, -1, kernel_size)
    ax_batch_expanded_y = ax_batch.unsqueeze(1).expand(-1, kernel_size, -1)

    # Creating a batch-wise meshgrid using broadcasting
    xx, yy = ax_batch_expanded_x, ax_batch_expanded_y # (batchsize, kernelsize, kernelsize)

    xy = torch.stack([xx, yy], dim=-1) # (batchsize, kernelsize, kernelsize, 2)
    z = torch.einsum('b...i,b...ij,b...j->b...', xy, -0.5 * inv_covariance, xy) # (batchsize, kernelsize, kernelsize, 2)
    kernel = torch.exp(z) / (2 * torch.tensor(np.pi, device=device) * torch.sqrt(torch.det(covariance)).view(batch_size, 1, 1)) # (batchsize, kernelsize, kernelsize)


    kernel_max_1, _ = kernel.max(dim=-1, keepdim=True)  # Find max along the last dimension
    kernel_max_2, _ = kernel_max_1.max(dim=-2, keepdim=True)  # Find max along the second-to-last dimension
    kernel_normalized = kernel / kernel_max_2 # (batchsize, kernelsize, kernelsize)


    kernel_reshaped = kernel_normalized.repeat(1, 3, 1).view(batch_size * 3, kernel_size, kernel_size)
    kernel_rgb = kernel_reshaped.unsqueeze(0).reshape(batch_size, 3, kernel_size, kernel_size)  # (batchsize, 3, kernelsize, kernelsize)

    # Calculating the padding needed to match the image size
    pad_h = image_size[0] - kernel_size
    pad_w = image_size[1] - kernel_size

    if pad_h < 0 or pad_w < 0:
        raise ValueError("Kernel size should be smaller or equal to the image size.")

    # Adding padding to make kernel size equal to the image size
    padding = (pad_w // 2, pad_w // 2 + pad_w % 2,  # padding left and right
               pad_h // 2, pad_h // 2 + pad_h % 2)  # padding top and bottom

    kernel_rgb_padded = torch.nn.functional.pad(kernel_rgb, padding, "constant", 0) # (batchsize, 3, h, w)

    # Extracting shape information
    b, c, h, w = kernel_rgb_padded.shape

    # Create a batch of 2D affine matrices
    theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = -coords # (b, 2) - the offset of gaussian splating

    # Creating grid and performing grid sampling
    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True) # (b, 3, h, w)
    # grid_y = torch.linspace(-1, 1, steps=h, device=device).reshape(1, h, 1, 1).repeat(1, 1, w, 1)
    # grid_x = torch.linspace(-1, 1, steps=w, device=device).reshape(1, 1, w, 1).repeat(1, h, 1, 1)
    # grid = torch.cat([grid_x, grid_y], dim=-1)
    # grid = grid - coords.reshape(-1, 1, 1, 2)

    kernel_rgb_padded_translated = F.grid_sample(kernel_rgb_padded, grid, align_corners=True) # (b, 3, h, w)

    rgb_values_reshaped = colours.unsqueeze(-1).unsqueeze(-1)

    final_image_layers = rgb_values_reshaped * kernel_rgb_padded_translated
    final_image = final_image_layers.sum(dim=0)
    # final_image = torch.clamp(final_image, 0, 1)
    final_image = final_image.permute(1,2,0)

    return final_image


if __name__ == "__main__":
    from mylineprofiler import MyLineProfiler
    profiler_th = MyLineProfiler(cuda_sync=True)
    generate_2D_gaussian_splatting = profiler_th.decorate(generate_2D_gaussian_splatting)
    profiler_cuda = MyLineProfiler(cuda_sync=True)
    gaussiansplatting_render = profiler_cuda.decorate(gaussiansplatting_render)


    # --- test ---
    # s = 1000
    s = 5
    # image_size = (512, 512, 3)
    image_size = (511, 511, 3)
    # image_size = (256, 512, 3)
    # image_size = (256, 256, 3)

    sigmas = 0.999*torch.rand(s, 3).to(torch.float32).to("cuda")
    sigmas[:,:2] = 5*sigmas[:, :2]
    coords = 2*torch.rand(s, 2).to(torch.float32).to("cuda")-1.0
    colors = torch.rand(s, 3).to(torch.float32).to("cuda")

    # --- torch version ---
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    for _ in range(20):
        img = generate_2D_gaussian_splatting(101, sigmas[:,1], sigmas[:,0], sigmas[:,2], coords, colors, image_size)
    profiler_th.print("profile.log", "w")
    cv2.imwrite("th.png", 255.0 * img.detach().clamp(0, 1).cpu().numpy())
    # --- ends ---

    # --- cuda version ---
    _stepsize_of_gs_th = 10 / (101-1)
    _stepsize_of_gs_cuda_w = 2 / (image_size[1]-1)
    _stepsize_of_gs_cuda_h = 2 / (image_size[0]-1)
    sigmas[:, 0] = sigmas[:, 0] * _stepsize_of_gs_cuda_w / _stepsize_of_gs_th
    sigmas[:, 1] = sigmas[:, 1] * _stepsize_of_gs_cuda_h / _stepsize_of_gs_th
    dmax = 101/2*_stepsize_of_gs_cuda_w
    gc.collect()
    torch.cuda.empty_cache()
    for _ in range(20):
        img = gaussiansplatting_render(sigmas, coords, colors, image_size, dmax)

    profiler_cuda.print("profile.log", "a")
    cv2.imwrite("cuda.png", 255.0 * img.detach().clamp(0, 1).cpu().numpy())
    # --- ends ---
