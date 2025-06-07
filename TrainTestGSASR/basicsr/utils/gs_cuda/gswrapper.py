import os
import torch
from torch.utils.cpp_extension import load
from torch.autograd import Function
from torch.autograd.function import once_differentiable

build_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'build')
os.makedirs(build_path, exist_ok=True)

file_path = os.path.split(os.path.abspath(__file__))[0]
# GSWrapper = load(
#         name="gscuda",
#         # sources=["gs_cuda/gswrapper.cpp", "gs_cuda/gs.cu"],
#         sources=[os.path.join(file_path, "gswrapper.cpp"), 
#                  os.path.join(file_path, "gs.cu")],
#         build_directory=build_path,
#         verbose=True)

import gscuda
GSWrapper = gscuda

class GSCUDA(Function):
   
        @staticmethod
        def forward(ctx, sigmas, coords, colors, rendered_img):
            ctx.save_for_backward(sigmas, coords, colors)
            h, w, c = rendered_img.shape
            s = sigmas.shape[0]
            GSWrapper.gs_render(sigmas, coords, colors, rendered_img, s, h, w, c)
            return rendered_img

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            sigmas, coords, colors = ctx.saved_tensors
            h, w, c = grad_output.shape
            s = sigmas.shape[0]
            grads_sigmas = torch.zeros_like(sigmas)
            grads_coords = torch.zeros_like(coords)
            grads_colors = torch.zeros_like(colors)
            GSWrapper.gs_render_backward(sigmas, coords, colors, grad_output.contiguous(), grads_sigmas, grads_coords, grads_colors, s, h, w, c)
            return (grads_sigmas, grads_coords, grads_colors, None)

def gaussiansplatting_render(sigmas, coords, colors, image_size):
    sigmas = sigmas.contiguous() # (gs num, 3)
    coords = coords.contiguous() # (gs num, 2)
    colors = colors.contiguous() # (gs num, c)
    h, w = image_size[:2]
    c = colors.shape[-1]
    rendered_img = torch.zeros(h, w, c).to(colors.device).to(torch.float32)
    return GSCUDA.apply(sigmas, coords, colors, rendered_img)

