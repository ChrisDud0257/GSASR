#include "gs.h"
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void gs_render(
        torch::Tensor &sigmas,
        torch::Tensor &coords,
        torch::Tensor &colors,
        torch::Tensor &rendered_img,
	const int s,
	const int h,
	const int w,
	const int c
        ){
      
        CHECK_INPUT(sigmas);
        CHECK_INPUT(coords);
        CHECK_INPUT(colors);
        CHECK_INPUT(rendered_img);

        // run the code at the cuda device same with the input
        const at::cuda::OptionalCUDAGuard device_guard(device_of(sigmas));

        _gs_render(
            (const float *) sigmas.data_ptr(),
            (const float *) coords.data_ptr(),
            (const float *) colors.data_ptr(),
            (float *) rendered_img.data_ptr(),
	    s, h, w, c);
}

void gs_render_backward(
        torch::Tensor &sigmas,
        torch::Tensor &coords,
        torch::Tensor &colors,
        torch::Tensor &grads,
        torch::Tensor &grads_sigmas,
        torch::Tensor &grads_coords,
        torch::Tensor &grads_colors,
	const int s,
	const int h,
	const int w,
	const int c
        ){

        CHECK_INPUT(sigmas);
        CHECK_INPUT(coords);
        CHECK_INPUT(colors);
        CHECK_INPUT(grads);
        CHECK_INPUT(grads_sigmas);
        CHECK_INPUT(grads_coords);
        CHECK_INPUT(grads_colors);


        // run the code at the cuda device same with the input
        const at::cuda::OptionalCUDAGuard device_guard(device_of(sigmas));

        _gs_render_backward(
            (const float *) sigmas.data_ptr(),
            (const float *) coords.data_ptr(),
            (const float *) colors.data_ptr(),
            (const float *) grads.data_ptr(),
            (float *) grads_sigmas.data_ptr(),
            (float *) grads_coords.data_ptr(),
            (float *) grads_colors.data_ptr(),
	    s, h, w, c);
}

PYBIND11_MODULE( TORCH_EXTENSION_NAME, m) {
        m.def( "gs_render",
                &gs_render,
                "cuda forward wrapper");
        m.def( "gs_render_backward",
                &gs_render_backward,
                "cuda backward wrapper");
}
