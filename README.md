# GSASR
Official PyTorch code for our Paper "Generalized and Efficient 2D Gaussian Splatting for
Arbitrary-scale Super-Resolution"

## [Paper and Supplementary (Arxiv Version)](https://arxiv.org/pdf/2501.06838)

> **Generalized and Efficient 2D Gaussian Splatting for
Arbitrary-scale Super-Resolution** <br>
> [Du CHEN\*](https://github.com/ChrisDud0257), [Liyi CHEN\*](https://github.com/mt-cly), [Zhengqiang ZHANG](https://github.com/xtudbxk) and [Lei ZHANG](https://www4.comp.polyu.edu.hk/~cslzhang/). <br>


## ⚡ TODO
- [ ] Hugging Face Page.
- [ ] Performance comparisons among different versions of GSASR.
- [ ] Details about Automatic Mixed Precision and Flash Attention which are utilized in our enhanced and ultra performance versions (to be updated in the supplementary).



# 1.Introduction about GSASR

## Abstract
Implicit Neural Representation (INR) has been successfully employed for Arbitrary-scale Super-Resolution (ASR). However, INR-based models need to query the multi-layer perceptron module numerous times 
and render a pixel in each query, resulting in insufficient representation capability and computational efficiency. Recently, Gaussian Splatting (GS) has shown its advantages over INR in both visual 
quality and rendering speed in 3D tasks, which motivates us to explore whether GS can be employed for the ASR task. However, directly applying GS to ASR is exceptionally challenging because the original 
GS is an optimization-based method through overfitting each single scene, while in ASR we aim to learn a single model that can generalize to different images and scaling factors. We overcome these 
challenges by developing two novel techniques. Firstly, to generalize GS for ASR, we elaborately design an architecture to predict the corresponding image-conditioned Gaussians of the input low-resolution 
image in a feed-forward manner. Each Gaussian can fit the shape and direction of an area of complex textures, showing powerful representation capability. Secondly, we implement an efficient differentiable 
2D GPU/CUDA-based scale-aware rasterization to render super-resolved images by sampling discrete RGB values from the predicted continuous Gaussians. Via end-to-end training, our optimized network, namely 
GSASR, can perform ASR for any image and unseen scaling factors. Extensive experiments validate the effectiveness of our proposed method. 

## Framework
![framework](./figures/framework.png)
In the training phase, an LR image is fed into the encoder to extract image features, conditioned on which the learnable Gaussian embeddings are passed through the Condition Injection Block and Gaussian
Interaction Block to output 2D Gaussians. These 2D Gaussians are then rendered into an SR image of a specified resolution through differential rasterization.


## Fast Rasterization
![Fast Rasterization](./figures/sampling.png)
Simply rendering an SR image by querying each pixel from all 2D Gaussians leads to a complexity of ___O(s<sup>2</sup>HWN)___, which is too high for high-resolution images.
Actually, a Gaussian generally focuses on a limited area and its contribution to pixel values decays rapidly with the increase of distance.
Therefore, we introduce a rasterization ratio ___r<1___ to control the rendering range of each Gaussian.
Specifically, we handle all Gaussians in parallel and only render the pixels that are close enough to the Gaussian centers, greatly reducing the computational complexity to ___O(r<sup>2</sup>s<sup>2</sup>HWN)___ and 
making our algorithm practical. (___r___ is set to ___0.1___ in our implementation.)
Note that our rasterization process is differential, which can be seamlessly integrated with neural networks for end-to-end optimization. This algorithm is implemented via ___CUDA C++___, which is GPU-friendly and 
achieves faster speed and low memory requirements.


The algorithm is described as follows,

![Algorithm](./figures/algorithm.png)


All of the pretrained models, the datasets can be access by the following link,

|Project|                                               Download                                                |
|:----:|:-----------------------------------------------------------------------------------------------------:|
|GSASR| [Google Drive](https://drive.google.com/drive/folders/1bja0ars9zlzo6XryZSLZ3xVJjNycl6Vg?usp=sharing)  |


## Core Codes of 2D Rendering/Rasterization

We provide both python-based and CUDA-based 2D rendering code.

### Python-based 2D Rendering/Rasterization

```bash
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
```

You could also find it [here](TrainTestGSASR/basicsr/utils/gaussian_splatting.py).

However, we don't recommend you to use python-based 2D rasterization since it is too slow, and will consume huge GPU memory. The python-based code just make you more clear with the rendering progress.

### CUDA-based 2D Rendering/Rasterization

```bash
def rendering_cuda_dmax(sigma_x, sigma_y, rho, coords, colours_with_alpha, sr_size, step_size,  device, dmax=1):
    from utils.gs_cuda_dmax.gswrapper import GSCUDA
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
```

You could also find it [here](TrainTestGSASR/basicsr/utils/gaussian_splatting.py). The CUDA operator could be fould [here](TrainTestGSASR/basicsr/utils/gs_cuda_dmax/gswrapper.cpp).

We strongly recommend you to utilize CUDA-based rasterization since it's very efficient!

# 2.Quick Inference - Paper (GSASR Models reported in the main paper)

Please note that, in paper, for each kind of encoder, we only train it on DIV2K. Note that, we don't report the performance of GSASR with SwinIR encoder, since the page is too limited. However, we still provide the pretrained models.

Please download the models from the following link,

|           Models           |lDowload| Version|
|:------------------------:|:---:|:---:|
|EDSR-Baseline-Train-with-DIV2K|[Google Drive](https://drive.google.com/drive/folders/1rSnM1HOBaI6TpfJ0XkXhHZcjjRnS95Sb?usp=sharing)|Paper|
|RDN-Train-with-DIV2K|[Google Drive](https://drive.google.com/drive/folders/1xR5JoiLG6Muav-C8XGpE4sTr2bleBxPU?usp=sharing)|Paper|
|SwinIR-Train-with-DIV2K|[Google Drive](https://drive.google.com/drive/folders/1Zv2ijlkyU0UdNz9XDvAu9HHaiUVmhkR0?usp=sharing)|Paper|

## 2.1 Installation
 - python == 3.10
 - PyTorch == 2.0
 - Anaconda
 - CUDA == 11.8

Then install the relevant environments :
```bash
git clone https://github.com/ChrisDud0257/GSASR
cd QuickInferencePaper
conda create --name gsasr python=3.10
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```


**Please note that, since we implement CUDA-based rasterization, please make sure you have installed correct CUDA version. The CUDA version from 11.1 to 11.8 all works well under our project. We utilize CUDA-11.8 in our implementation.**

**After you install CUDA, please export the CUDA path, for me, I export it as follows,**

```bash
export CUDA_HOME=/home/notebook/code/personal/S9053766/chendu/cuda-11.8
```

## 2.2 For single image

We test our results on NVIDIA A100 GPU, 

```bash
cd QuickInferencePaper/
python inference_GSASR_with_single_image.py --encoder_path [path to encoder path] --decoder_path [path to decoder path] --input_img_path [path to LR image] \
--save_sr_path [path to save path] --scale [indicate the up-scaling factor] --encoder_name [please choose between 'EDSR-Baseline', 'RDN' and 'SwinIR']
```

Please note that, when you test with EDSR-Baseline or RDN encoder, please set "denominator=12". While you test with SwinIR encoder, please set "denominator=24".

If your GPU memory is limited, please use "tile_process" to lower the GPU memory,

```bash
cd QuickInferencePaper/
python inference_GSASR_with_single_image.py --encoder_path [path to encoder path] --decoder_path [path to decoder path] --input_img_path [path to LR image] \
--save_sr_path [path to save path] --scale [indicate the up-scaling factor] --encoder_name [please choose among 'EDSR-Baseline', 'RDN' and 'SwinIR'] --tile_process \
--tile_size [tile_size must be divisible by denominator]
```
**Please note that, use tile_process will sacrifice some reconstruction fidelity, the lower tile_size, the worse performance.**


## 2.3 For standard benchmarks

If you want to test the computational cost of GSASR (which is reported in Table.2 from the main paper and Table.8 from the supplementary), we provide the cropped 720*720 size of GT images, and the corresponding LR images of DIV2K testing parts which are utilized in our paper, please download them in the following link,

|Dataset|Download|
|:--:|:--:|
|DIV2K_GT720|[Google Drive](https://drive.google.com/file/d/1FQrVcCppV_No-0BeTxUh2kBaGIb6xZ82/view?usp=sharing)|


If you want to crop images all by yourself, please follow this [instruction](TrainTestGSASR/datasets/README.MD) to prepare the data which could be utilized to test the computational cost.


After you download them, please test by the following instruction,

```bash
cd QuickInferencePaper/
python inference_GSASR_with_standard_benchmarks.py --encoder_path [path to encoder path] --decoder_path [path to decoder path] --input_img_path [path to benchmark LR image path] --save_sr_path [path to save path] --scale [indicate the up-scaling factor] --encoder_name [please choose among 'EDSR-Baseline', 'RDN' and 'SwinIR']
```
Please indicate the "input_img_path" to your downloaded DIV2K testing parts (which is provided by us).

Please note that, when you test with EDSR-Baseline or RDN encoder, please set "denominator=12". While you test with SwinIR encoder, please set "denominator=24".

If your GPU memory is limited, please use "tile_process" to lower the GPU memory,

```bash
cd QuickInferencePaper/
python inference_GSASR_with_standard_benchmarks.py --encoder_path [path to encoder path] --decoder_path [path to decoder path] --input_img_path [path to LR image] \
--save_sr_path [path to save path] --scale [indicate the up-scaling factor] --encoder_name [please choose among 'EDSR-Baseline', 'RDN' and 'SwinIR'] --tile_process \
--tile_size [tile_size must be divisible by denominator]
```

**In this code, we also provide the calculation of computational cost, including the test time (ms) and GPU memory (MB). In our paper, we calculate the computational cost on a single NVIDIA A100 GPU, and we input the full size image into the model, we don't use tile_process since it will lower the GPU memory and accelerate the speed, leading to unfair comparions when compared with other methods.**

We utilize the official code from [PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html) to record the GPU memory usage. The calculation of computational cost used in our paper is as follows, we ignore any pre-process or post-process steps, and just record the computational cost of the full pipeline of the model part, including the encoder, the decoder as well as the rendering parts.

```bash
import time
import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    output = model(input)

    torch.cuda.synchronize()
    end = time.time()
    time_cost = end - start
    used_memory = torch.cuda.max_memory_allocated()

    print(f"Time cost is {time_cost*1000:.4f} ms, GPU memory is {used_memory /1024 /1024:.4f} MB")
```


**Please note that, as for the inference time and GPU memory, different verions of GPU, CUDA, PyTorch, RAM and so on will influence the final results. Therefore, each one's results might differ from others. For fair comparison, please test your method together with other methods on the same machine.**

## 2.4 For any datasets of yours

Please use the same instructions in Section 2.3, which supports for any kinds of RGB domain datasets.





# 3.Quick Inference - AMP (Enhanced version of GSASR. Models are trained with Automatic Mixed Precision and Flash Attention, please find the details in our supplementary materials)

To further accelerate the inference speed, lower the GPU memory and improve the performance, we substitute the vanilla self attention with [Flash Attention](https://docs.pytorch.org/docs/2.0/generated/torch.nn.functional.scaled_dot_product_attention.html?highlight=scaled_dot_product_attention#torch.nn.functional.scaled_dot_product_attention). To support the Flash Attention, we substitute the relative position embedding (RPE) with [rotary position embedding (ROPE)](https://github.com/naver-ai/rope-vit). We further utilize Automatic Mixed Precision (AMP) strategy, through combining bfloat16 and fp32 precision in our training and inference stage to accelerate the speed and lower the GPU memory.

Please note that, in ROPE, the channel dimension of each attention head must be divisible by 4, thus we set the channel dimension of Gaussian embedding to 192, while the number of heads still keeps 6, then the channel dimension of each attention head is 32, which could be divisible by 4. 

Please note that, for each kind of encoder, we provide two different versions of models, one is trained with DIV2K, while the other is trained with DF2K (DIV2K+FLICKR2K).

**Please note that, Flash Attention, bfloat16 precision are only supported by some specific GPUs. If you encounter an error during the program execution, please don't use "--AMP_test" command.**

Please download the models from the following link,

|           Models           |lDowload| Version|
|:------------------------:|:---:|:---:|
|EDSR-Baseline-Train-with-DIV2K|[Google Drive](https://drive.google.com/drive/folders/1R6ZCdAd6t_2CCpjCK67F9nag9jitMhI6?usp=sharing)|Enhanced (AMP+Flash Attention)|
|EDSR-Baseline-Train-with-DF2K|[Google Drive](https://drive.google.com/drive/folders/16TV2yJt_lfNqJnATtJnEkHV1KoBuW8ww?usp=sharing)|Enhanced (AMP+Flash Attention)|
|RDN-Train-with-DIV2K|[Google Drive](https://drive.google.com/drive/folders/1guSg28c8gvrTkCvTmNbzqf9vWJfLv58Q?usp=sharing)|Enhanced (AMP+Flash Attention)|
|RDN-Train-with-DF2K|[Google Drive](https://drive.google.com/drive/folders/1vkBvsiiNqTFKmPtNjPlqMn_mh_ClUrKE?usp=sharing)|Enhanced (AMP+Flash Attention)|
|SwinIR-Train-with-DIV2K|[Google Drive](https://drive.google.com/drive/folders/1kVLkOs4KrXlXsPsh0oqvey2dvT6TxqH-?usp=sharing)|Enhanced (AMP+Flash Attention)|
|SwinIR-Train-with-DF2K|[Google Drive](https://drive.google.com/drive/folders/1kVLkOs4KrXlXsPsh0oqvey2dvT6TxqH-?usp=sharing)|Enhanced (AMP+Flash Attention)|

## 3.1 Installation
 - python == 3.10
 - PyTorch == 2.0
 - Anaconda
 - CUDA == 11.8

Then install the relevant environments :
```bash
git clone https://github.com/ChrisDud0257/GSASR
cd QuickInferenceAMP
conda create --name gsasr python=3.10
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Please note that, since we implement CUDA-based rasterization, please make sure you have installed correct CUDA version. The CUDA version from 11.1 to 11.8 all works well under our project. We utilize CUDA-11.8 in our implementation.**

**After you install CUDA, please export the CUDA path, for me, I export it as follows,**

```bash
export CUDA_HOME=/home/notebook/code/personal/S9053766/chendu/cuda-11.8
```

## 3.2 For single image

We test our results on NVIDIA A100 GPU, 

```bash
cd QuickInferenceAMP/
python inference_GSASR_with_single_image.py --encoder_path [path to encoder path] --decoder_path [path to decoder path] --input_img_path [path to LR image] \
--save_sr_path [path to save path] --scale [indicate the up-scaling factor] --encoder_name [please choose among 'EDSR-Baseline', 'RDN' and 'SwinIR'] --AMP_test
```

Please note that, when you test with EDSR-Baseline or RDN encoder, please set "denominator=12". While you test with SwinIR encoder, please set "denominator=16".

**Please note that, Flash Attention, bfloat16 precision are only supported by some specific GPUs. If you encounter an error during the program execution, please don't use "--AMP_test" command.**

If your GPU memory is limited, please use "tile_process" to lower the GPU memory,

```bash
cd QuickInferenceAMP/
python inference_GSASR_with_single_image.py --encoder_path [path to encoder path] --decoder_path [path to decoder path] --input_img_path [path to LR image] \
--save_sr_path [path to save path] --scale [indicate the up-scaling factor] --encoder_name [please choose among 'EDSR-Baseline', 'RDN' and 'SwinIR'] --tile_process \
--tile_size [tile_size must be divisible by denominator] --AMP_test
```

**Please note that, Flash Attention, bfloat16 precision are only supported by some specific GPUs. If you encounter an error during the program execution, please don't use "--AMP_test" command.**

**Please note that, use tile_process will sacrifice some reconstruction fidelity, the lower tile_size, the worse performance.**


## 3.3 For standard benchmarks

If you want to test the computational cost of GSASR (which is reported in Table.2 from the main paper and Table.8 from the supplementary), we provide the cropped 720*720 size of GT images, and the corresponding LR images of DIV2K testing parts which are utilized in our paper, please download them in the following link,

|Dataset|Download|
|:--:|:--:|
|DIV2K_GT720|[Google Drive](https://drive.google.com/file/d/1FQrVcCppV_No-0BeTxUh2kBaGIb6xZ82/view?usp=sharing)|


If you want to crop images all by yourself, please follow this [instruction](TrainTestGSASR/datasets/README.MD) to prepare the data which could be utilized to test the computational cost.

After you download them, please test by the following instruction,

```bash
cd QuickInferencePaper/
python inference_GSASR_with_standard_benchmarks.py --encoder_path [path to encoder path] --decoder_path [path to decoder path] --input_img_path [path to benchmark LR image path] --save_sr_path [path to save path] --scale [indicate the up-scaling factor] --encoder_name [please choose among 'EDSR-Baseline', 'RDN' and 'SwinIR']
```
Please indicate the "input_img_path" to your downloaded DIV2K testing parts (which is provided by us).

If your GPU memory is limited, please use "tile_process" to lower the GPU memory,

```bash
cd QuickInferencePaper/
python inference_GSASR_with_standard_benchmarks.py --encoder_path [path to encoder path] --decoder_path [path to decoder path] --input_img_path [path to LR image] \
--save_sr_path [path to save path] --scale [indicate the up-scaling factor] --encoder_name [please choose among 'EDSR-Baseline', 'RDN' and 'SwinIR'] --tile_process \
--tile_size [tile_size must be divisible by denominator]
```

**In this code, we also provide the calculation of computational cost, including the test time (ms) and GPU memory (MB). In our paper, we calculate the computational cost on a single NVIDIA A100 GPU, and we input the full size image into the model, we don't use tile_process since it will lower the GPU memory and accelerate the speed, leading to unfair comparions when compared with other methods.**

We utilize the official code from [PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html) to record the GPU memory usage. The calculation of computational cost used in our paper is as follows, we ignore any pre-process or post-process steps, and just record the computational cost of the full pipeline of the model part, including the encoder, the decoder as well as the rendering parts.

```bash
import time
import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    output = model(input)

    torch.cuda.synchronize()
    end = time.time()
    time_cost = end - start
    used_memory = torch.cuda.max_memory_allocated()

    print(f"Time cost is {time_cost*1000:.4f} ms, GPU memory is {used_memory /1024 /1024:.4f} MB")
```


**Please note that, as for the inference time and GPU memory, different verions of GPU, CUDA, PyTorch, RAM, random seed and so on will influence the final results of the computational cost. Therefore, each one's results might differ from others. For fair comparison, please test your method together with other methods on the same machine.**

## 3.4 For any datasets of yours

Please use the same instructions in Section 3.3, which supports for any kinds of RGB domain datasets.




# 4.Quick Inference - Ultra performance (The ultimost version of GSASR. Models are trained with Automatic Mixed Precision and Flash Attention, please find the details in our supplementary materials)

Based on AMP and Flash Attention, we train GSASR with [HAT-L](https://github.com/XPixelGroup/HAT) encoder to explore the ultimost performance.

The trainging setting are as follows:

|Training Details|Settings|
|:---:|:---:|
|Dataset|[SA1B]([https://ai.meta.com/datasets/segment-anything/)|
|GPUs|16 x NVIDIA A100|
|Batch Size per GPU|8|
|Iterations|500000|
|Training Time Cost|30 days|
|Encoder|[HAT-L]((https://github.com/XPixelGroup/HAT))|
|Range of Scaling Factor in Training|[1,16]|
|Input LR size|64 x 64|
|Acceleration Strategy|AMP + Flash Attention|

Please note that, the original HAT-L utilizes vanilla self attention, which is very slow and will consume large GPU memories. We optimize them with [Flash Attention](https://docs.pytorch.org/docs/2.0/generated/torch.nn.functional.scaled_dot_product_attention.html?highlight=scaled_dot_product_attention#torch.nn.functional.scaled_dot_product_attention). To support the Flash Attention, we substitute the relative position embedding (RPE) in HAT-L with [rotary position embedding (ROPE)](https://github.com/naver-ai/rope-vit). We further utilize Automatic Mixed Precision (AMP) strategy, through combining bfloat16 and fp32 precision in our training and inference stage to accelerate the speed and lower the GPU memory.


Please note that, in ROPE, the channel dimension of each attention head must be divisible by 4, thus we set the channel dimension of self attention layers in HAT-L to 192, while the number of heads still keeps 6, then the channel dimension of each attention head is 32, which could be divisible by 4. And the squeeze factor is set to 32 in HAT-L.

Please note that, for HAT-L encoder, we just train it on SA1B.

**Please note that, HAT-L are firstly pretrained on ImageNet for 800000 iterations, and then it is fine-tuned with DF2K for another 250000 iteration. The computational cost is too heavy, we just train GSASR-HATL totally from scratch for 500000 iterations, there is no pre-training or fine-tuning stage. Therefore, directly comparing GSASR-HATL with the original HAT-L models are unfair.**

**Please note that, Flash Attention, bfloat16 precision are only supported by some specific GPUs. If you encounter an error during the program execution, please don't use "--AMP_test" command.**

Please download the models from the following link,

|           Models           |                                               Download                                               | Version|
|:------------------------:|:----------------------------------------------------------------------------------------------------:|:---:|
|HATL-SA1B| [Google Drive](https://drive.google.com/drive/folders/1Pn-4JWvlMj50CulmAcBI1Hssiu-6nSYI?usp=sharing) |Ultra Performance (AMP+Flash Attention+HATL Encoder + Training on SA1B)|

## 4.1 Installation
 - python == 3.10
 - PyTorch == 2.0
 - Anaconda
 - CUDA == 11.8

Then install the relevant environments :
```bash
git clone https://github.com/ChrisDud0257/GSASR
cd QuickInferenceUltraPerformance/
conda create --name gsasr python=3.10
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Please note that, since we implement CUDA-based rasterization, please make sure you have installed correct CUDA version. The CUDA version from 11.1 to 11.8 all works well under our project. We utilize CUDA-11.8 in our implementation.**

**After you install CUDA, please export the CUDA path, for me, I export it as follows,**

```bash
export CUDA_HOME=/home/notebook/code/personal/S9053766/chendu/cuda-11.8
```

## 4.2 For single image

We test our results on NVIDIA A100 GPU, 

```bash
cd QuickInferenceUltraPerformance/
python inference_GSASR_with_single_image.py --encoder_path [path to encoder path] --decoder_path [path to decoder path] --input_img_path [path to LR image] \
--save_sr_path [path to save path] --scale [indicate the up-scaling factor] --encoder_name [it must be "HATL"] --AMP_test
```

**Please note that, Flash Attention, bfloat16 precision are only supported by some specific GPUs. If you encounter an error during the program execution, please don't use "--AMP_test" command.**

If your GPU memory is limited, please use "tile_process" to lower the GPU memory,

```bash
cd QuickInferenceUltraPerformance/
python inference_GSASR_with_single_image.py --encoder_path [path to encoder path] --decoder_path [path to decoder path] --input_img_path [path to LR image] \
--save_sr_path [path to save path] --scale [indicate the up-scaling factor] --encoder_name [it must be "HATL"] --tile_process \
--tile_size [tile_size must be divisible by denominator] --AMP_test
```

**Please note that, Flash Attention, bfloat16 precision are only supported by some specific GPUs. If you encounter an error during the program execution, please don't use "--AMP_test" command.**

**Please note that, use tile_process will sacrifice some reconstruction fidelity, the lower tile_size, the worse performance.**


## 4.3 For standard benchmarks

If you want to test the computational cost of GSASR (which is reported in Table.2 from the main paper and Table.8 from the supplementary), we provide the cropped 720*720 size of GT images, and the corresponding LR images of DIV2K testing parts which are utilized in our paper, please download them in the following link,

|Dataset|Download|
|:--:|:--:|
|DIV2K_GT720|[Google Drive](https://drive.google.com/file/d/1FQrVcCppV_No-0BeTxUh2kBaGIb6xZ82/view?usp=sharing)|


If you want to crop images all by yourself, please follow this [instruction](TrainTestGSASR/datasets/README.MD) to prepare the data which could be utilized to test the computational cost.

After you download them, please test by the following instruction,

```bash
cd QuickInferenceUltraPerformance/
python inference_GSASR_with_standard_benchmarks.py --encoder_path [path to encoder path] --decoder_path [path to decoder path] --input_img_path [path to benchmark LR image path] --save_sr_path [path to save path] --scale [indicate the up-scaling factor] --encoder_name [it must be "HATL"]
```
Please indicate the "input_img_path" to your downloaded DIV2K testing parts (which is provided by us).

If your GPU memory is limited, please use "tile_process" to lower the GPU memory,

```bash
cd QuickInferenceUltraPerformance/
python inference_GSASR_with_standard_benchmarks.py --encoder_path [path to encoder path] --decoder_path [path to decoder path] --input_img_path [path to LR image] \
--save_sr_path [path to save path] --scale [indicate the up-scaling factor] --encoder_name [it must be "HATL"] --tile_process \
--tile_size [tile_size must be divisible by denominator]
```

**In this code, we also provide the calculation of computational cost, including the test time (ms) and GPU memory (MB). In our paper, we calculate the computational cost on a single NVIDIA A100 GPU, and we input the full size image into the model, we don't use tile_process since it will lower the GPU memory and accelerate the speed, leading to unfair comparions when compared with other methods.**

We utilize the official code from [PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html) to record the GPU memory usage. The calculation of computational cost used in our paper is as follows, we ignore any pre-process or post-process steps, and just record the computational cost of the full pipeline of the model part, including the encoder, the decoder as well as the rendering parts.

```bash
import time
import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    output = model(input)

    torch.cuda.synchronize()
    end = time.time()
    time_cost = end - start
    used_memory = torch.cuda.max_memory_allocated()

    print(f"Time cost is {time_cost*1000:.4f} ms, GPU memory is {used_memory /1024 /1024:.4f} MB")
```


**Please note that, as for the inference time and GPU memory, different verions of GPU, CUDA, PyTorch, RAM, random seed and so on will influence the final results of the computational cost. Therefore, each one's results might differ from others. For fair comparison, please test your method together with other methods on the same machine.**

## 4.4 For any datasets of yours

Please use the same instructions in Section 4.3, which supports for any kinds of RGB domain datasets.


# 5. Training Code

## 5.1 Dataset preparation

Please follow this [instruction](TrainTestGSASR/datasets/README.MD) to prepare the training and testing datasets.

## 5.2 Training GSASR

Please follow this [instruction](TrainTestGSASR/README.md) to train GSASR.

# 6. Testing Code

## 6.1 Dataset preparation

Please follow this [instruction](TrainTestGSASR/datasets/README.MD) to prepare the training and testing datasets.

## 6.2 Testing GSASR

Please follow this [instruction](TrainTestGSASR/README.md) to test GSASR.

# 7.Evaluation Metrics

We provide the code of PSNR/SSIM/LPIPS/DISTS here.

```bash
cd TrainTestGSASR/scripts/metrics
python calculate_psnr_ssim.py --gt [path to GT images] --restored [path to SR images] --scale [scaling factor] --suffix [suffix of restored images] --test_y_channel
python calculate_lpips.py --gt [path to GT images] --restored [path to SR images] --scale [scaling factor] --suffix [suffix of restored images]
python calculate_dists.py --gt [path to GT images] --restored [path to SR images] --scale [scaling factor] --suffix [suffix of restored images]
```

Please note that, when calculating PSNR/SSIM, we test them on Y channel of Ycbcr space. If you want to obtain the same results as that reported in our paper, please use "--test_y_channel".

Please note that, when calculating PSNR/SSIM/LPIPS/DISTS, if the scaling factor is not larger than 8, we set "crop_border = scaling factor", else, we set "crop_border = 8".


# 8.License
This project is released under the Apache 2.0 license.


# 9.Acknowlegement

This project is built mainly based on the excellent [BasicSR](https://github.com/XPixelGroup/BasicSR), [HAT](https://github.com/XPixelGroup/HAT) and [ROPE-ViT](https://github.com/naver-ai/rope-vit) codeframe. We appreciate it a lot for their developers.

We sincerely thank [Mr.Zhengqiang Zhang](https://github.com/xtudbxk) for his support in the CUDA operator of rasterization.

# 10.Citation
If you find this research helpful for you, please follow us by
```bash
@article{chen2025generalized,
  title={Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution},
  author={Chen, Du and Chen, Liyi and Zhang, Zhengqiang and Zhang, Lei},
  journal={arXiv preprint arXiv:2501.06838},
  year={2025}
}
```
