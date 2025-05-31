import torch
from gswrapper import gaussiansplatting_render

def torch_version(sigmas, coords, colors, image_size):
    h, w = image_size
    c = colors.shape[-1]

    if h >= 50 or w >= 50:
        logger.warning(f'too large values for h({h}), w({w}), torch version would be slow')

    rendered_img = torch.zeros(h, w, c).to(colors.device).to(torch.float32)

    for hi in range(h):
        for wi in range(w):
            curh = 2*hi/(h-1)-1.0
            curw = 2*wi/(w-1)-1.0

            v = (curw-coords[:,0])**2/sigmas[:,0]**2
            v -= (2*sigmas[:,2])*(curw-coords[:,0])*(curh-coords[:,1])/sigmas[:,0]/sigmas[:,1]
            v += (curh-coords[:,1])**2/sigmas[:,1]**2
            v *= -1.0/(2.0*(1-sigmas[:,2]**2))
            v = torch.exp(v)
                          
            for ci in range(c):
                rendered_img[hi, wi, ci] = torch.sum(v*colors[:, ci])

    return rendered_img


if __name__ == "__main__":
    s = 40 # the number of gs
    image_size = (49, 49)

    for _ in range(1):
        print(f"--------------------------- begins --------------------------------")

        sigmas = 0.999*torch.rand(s, 3).to(torch.float32).to("cuda")
        # sigmas[:,:2] = 5*sigmas[:, :2]
        coords = 2*torch.rand(s, 2).to(torch.float32).to("cuda")-1.0
        colors = torch.rand(s, 3).to(torch.float32).to("cuda")

        # sigmas = torch.Tensor([[0.9196, 0.3979, 0.7784]]).to(torch.float32).to("cuda")
        # coords = torch.Tensor([[-0.0469, -0.1726]]).to(torch.float32).to("cuda")
        # colors = torch.Tensor([[0.3775, 0.2346, 0.1513]]).to(torch.float32).to("cuda")
        # colors = torch.ones_like(coords[:,0:1])

        print(f"sigmas: {sigmas}, \ncoords:{coords}, \ncolors:{colors}")

        # --- check forward ---
        with torch.no_grad():
            rendered_img_th = torch_version(sigmas,coords,colors,image_size)
            rendered_img_cuda = gaussiansplatting_render(sigmas,coords,colors,image_size)

        # 
        distance = (rendered_img_th-rendered_img_cuda)**2
        print(f"check forward - torch: {rendered_img_th[:2,:2,0]}")
        print(f"check forward - cuda: {rendered_img_cuda[:2,:2,0]}")
        print(f"check forward - distance: {distance[:2, :2, 0]}")
        print(f"check forward - sum: {torch.sum(distance)}\n")
        # --- ends ---

        # --- check backward ---
        sigmas.requires_grad_(True)
        coords.requires_grad_(True)
        colors.requires_grad_(True)
        # sigmas.retain_grad()
        # coords.retain_grad()
        # colors.retain_grad()
        weight = torch.rand_like(rendered_img_th) # make each pixel has different grads

        sigmas.grad = None
        coords.grad = None
        colors.grad = None
        rendered_img_th = torch_version(sigmas,coords,colors,image_size)
        loss_th = torch.sum(weight*rendered_img_th) 
        loss_th.backward()

        sigmas_grad_th = sigmas.grad
        coords_grad_th = coords.grad
        colors_grad_th = colors.grad

        sigmas.grad = None
        coords.grad = None
        colors.grad = None
        rendered_img_cuda = gaussiansplatting_render(sigmas,coords,colors,image_size)
        loss_cuda = torch.sum(weight*rendered_img_cuda)
        # loss_cuda = torch.sum(rendered_img_cuda)
        loss_cuda.backward()

        sigmas_grad_cuda = sigmas.grad
        coords_grad_cuda = coords.grad
        colors_grad_cuda = colors.grad

        distance_sigmas_grad = (sigmas_grad_th-sigmas_grad_cuda)**2
        distance_coords_grad = (coords_grad_th-coords_grad_cuda)**2
        distance_colors_grad = (colors_grad_th-colors_grad_cuda)**2

        print(f"check backward - sigmas - torch: {sigmas_grad_th[:2]}")
        print(f"check backward - sigmas - cuda: {sigmas_grad_cuda[:2]}")
        print(f"check backward - sigmas - distance: {distance_sigmas_grad[:2]}")
        print(f"check backward - sigmas - sum: {torch.sum(distance_sigmas_grad)}\n")

        print(f"check backward - coords - torch: {coords_grad_th[:2]}")
        print(f"check backward - coords - cuda: {coords_grad_cuda[:2]}")
        print(f"check backward - coords - distance: {distance_coords_grad[:2]}")
        print(f"check backward - coords - sum: {torch.sum(distance_coords_grad)}\n")
        
        print(f"check backward - colors - torch: {colors_grad_th[:2]}")
        print(f"check backward - colors - cuda: {colors_grad_cuda[:2]}")
        print(f"check backward - colors - distance: {distance_colors_grad[:2]}")
        print(f"check backward - colors - sum: {torch.sum(distance_colors_grad)}\n")

        print(f"--------------------------- ends --------------------------------\n\n")

    
