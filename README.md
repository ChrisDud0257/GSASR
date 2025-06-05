<div align="center">
<h1>Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution</h1>

[**Du Chen**](https://github.com/ChrisDud0257)<sup>12*</sup> · [**Liyi Chen**](https://github.com/mt-cly)<sup>1*</sup> · [**Zhengqiang Zhang**](https://github.com/xtudbxk)<sup>12</sup> · [**Lei Zhang**](https://www4.comp.polyu.edu.hk/~cslzhang/)<sup>12*</sup>
<br>

<sup>1</sup>The Hong Kong Polytechnic University <sup>2</sup>OPPO Research Institute
<br>
*Equal contribution &dagger;Corresponding author &emsp; 

<a href="https://arxiv.org/abs/2501.06838"><img src='https://img.shields.io/badge/arXiv-GSASR-red' alt='Paper PDF'></a>
<a href='https://mt-cly.github.io/GSASR.github.io/'><img src='https://img.shields.io/badge/Project_Page-GSASR-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/mutou0308/GSASR'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>

</div>

This work presents GSASR. It achieve SoTA in arbitrary-scale super-resolution by representing given LR image as millions of continuous 2D Gaussians.  

![Fast Rasterization](./figures/sampling.png)


## News
- **2025-06-05:** The online demo with most powerful HATL-based GSASR is released, [click to try it](https://huggingface.co/spaces/mutou0308/GSASR).
- **2025-05-30:** The {EDSR, RDN, Swin, HATL}-based GSASR models are available.
- **2025-01-16:** GSASR [paper](https://arxiv.org/abs/2501.06838) and [project papge](https://mt-cly.github.io/GSASR.github.io/) are released.


## Pre-trained Models

We provide **models** of varying-scale encoder for GSASR:

|           Models           |        Training Dataset|                                       Download                                               | Version|
|:------------------------:|:----------------------------------------------------------------------------------------------------:|:---:|:---:|
|EDSR-Baseline-Train-with-DIV2K| DIV2K | [Google Drive](https://drive.google.com/drive/folders/1R6ZCdAd6t_2CCpjCK67F9nag9jitMhI6?usp=sharing) |Enhanced (AMP+ROPE+Flash Attention)|
|EDSR-Baseline-Train-with-DF2K| DF2K| [Google Drive](https://drive.google.com/drive/folders/16TV2yJt_lfNqJnATtJnEkHV1KoBuW8ww?usp=sharing) |Enhanced (AMP+ROPE+Flash Attention)|
|RDN-Train-with-DIV2K| DIV2K| [Google Drive](https://drive.google.com/drive/folders/1guSg28c8gvrTkCvTmNbzqf9vWJfLv58Q?usp=sharing) |Enhanced (AMP+ROPE+Flash Attention)|
|RDN-Train-with-DF2K| DF2K|  [Google Drive](https://drive.google.com/drive/folders/1vkBvsiiNqTFKmPtNjPlqMn_mh_ClUrKE?usp=sharing) |Enhanced (AMP+ROPE+Flash Attention)|
|SwinIR-Train-with-DIV2K| DIV2K| [Google Drive](https://drive.google.com/drive/folders/1kVLkOs4KrXlXsPsh0oqvey2dvT6TxqH-?usp=sharing) |Enhanced (AMP+ROPE+Flash Attention)|
|SwinIR-Train-with-DF2K| DF2K| [Google Drive](https://drive.google.com/drive/folders/1ql6dktVUlQFIoPSJkEuvvMPz9TlacMdy?usp=sharing) |Enhanced (AMP+ROPE+Flash Attention)|
|HATL-SA1B| SA1B| [Google Drive](https://drive.google.com/drive/folders/1Pn-4JWvlMj50CulmAcBI1Hssiu-6nSYI?usp=sharing) |Ultra Performance (AMP+ROPE+Flash Attention)|

Please note that these models use AMP+ROPE+Flash Attention to reduce memory and time cost. While in our paper report, we does not using these tricks for fair comparison, please refer to [paper-results-on-benchmarks](#paper-results-on-benchmarks).

## Usage

### Prepraration

```bash
git clone https://github.com/ChrisDud0257/GSASR
cd GSASR
conda create --name gsasr python=3.10
conda activate gsasr
pip install -r requirements.txt
```


We test code in CUDA 12.4 and 11.8.


### Runing
You need to properly authenticate with Hugging Face to download our model weights. Once set up, our code will handle it automatically at your first run. You can authenticate by running

```bash
# This will prompt you to enter your Hugging Face credentials.
huggingface-cli login
```

You can try GSASR easily by runing in command or lanching gradio demo. 

### :computer: CLI
```bash
python inference.py \
    --input_img_path <path_to_img> \
    --save_sr_path <path_to_saved_folder> \
    --model <{EDSR_DIV2K, EDSR_DF2K, RDN_DIV2K, RDN_DF2K, SWIN_DIV2K,SWIN_DF2K, HATL_SA1B}> \
    --scale <scale> [--tile_process] [--AMP_test]
```
using `--tile_process` and `--AMP_test` if memory is limited.

### :rocket: Gradio demo
```bash
python demo_gr.py
```


## Paper Results on Benchmarks

Please note that, in our paper, we only train GSASR on DIV2K wihtout AMP+RoPE+Flash Attention tricks for fair comparison. 
Besides {EDSR, RDN}-based GSASR present in paper, here we provide Swin-based GSASR model. 

### Pre-trained Models
Download models from the following link.

|           Models           |  Training Dataset |                                             Download                                               | Version|
|:------------------------:|:---:|:----------------------------------------------------------------------------------------------------:|:---:|
|EDSR-Baseline-Train-with-DIV2K|DIV2K| [Google Drive](https://drive.google.com/drive/folders/1rSnM1HOBaI6TpfJ0XkXhHZcjjRnS95Sb?usp=sharing) |Paper|
|RDN-Train-with-DIV2K|DIV2K| [Google Drive](https://drive.google.com/drive/folders/1xR5JoiLG6Muav-C8XGpE4sTr2bleBxPU?usp=sharing) |Paper|
|SwinIR-Train-with-DIV2K| DIV2K| [Google Drive](https://drive.google.com/drive/folders/1Zv2ijlkyU0UdNz9XDvAu9HHaiUVmhkR0?usp=sharing) |Paper|


### Inference for single image
if you have logined in the huggingface, directly execute the `inference_paper.py` as follows.
```bash
python inference_paper.py \
    --input_img_path <path_to_img> \
    --save_sr_path <path_to_saved_folder> \
    --model <{EDSR, RDN, SWIN}> \
    --scale <scale> [--tile_process]
```




### Inference on DIV2K benchmark

Download cropped 720*720 size of GT images, and the corresponding LR images of DIV2K testing parts, which are utilized in our paper.

|Dataset|Download|
|:--:|:--:|
|DIV2K_GT720|[Google Drive](https://drive.google.com/file/d/1FQrVcCppV_No-0BeTxUh2kBaGIb6xZ82/view?usp=sharing)|


If you want to crop images all by yourself, please follow this [instruction](TrainTestGSASR/datasets/README.MD) to prepare the data which could be utilized to test the computational cost.


After you download them, please test by the following command.


```bash
python inference_paper_benchmark.py \
    --input_img_path <path_to_LRx4_folder> \
    --save_sr_path <path_to_saved_folder> \
    --model <{EDSR, RDN, SWIN}> \
    --scale 4 [--tile_process]
```

Please indicate the "input_img_path" to your downloaded DIV2K testing parts (which is provided by us).

### Memory and inference time estimation
In `inference_paper_benchmarks.py`, we integrate the statistics code of test time (ms) and GPU memory (MB). In our paper, we calculate the computational cost on a single NVIDIA A100 GPU, and we input the full size image into the model, we don't use tile_process. The inference time omit the pre-processing and post-processing and record the full pipeline cost inlcuding encoder, decoder and rendering.

### Metrics
After inference,  execute the code to estimate PSNR/SSIM/LPIPS/DISTS.

```bash
pip install basicsr
# if meet the error of ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor',
# please refer to https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985#issuecomment-1813885266
python TrainTestGSASR/scripts/metrics/calculate_psnr_ssim.py --test_y_channel --gt <path_to_GT_folder> --restored <path_to_SR_folder> --scale <scale> [--suffix <suffix_of_images>]
python TrainTestGSASR/scripts/metrics/calculate_lpips.py  --gt <path_to_GT_folder> --restored <path_to_SR_folder> --scale <scale> [--suffix <suffix_of_images>]
python TrainTestGSASR/scripts/metrics/calculate_dists.py  --gt <path_to_GT_folder> --restored <path_to_SR_folder> --scale <scale> [--suffix <suffix_of_images>]
```
Please note that we test them on Y channel of Ycbcr space with `--test_y_channel`  when calculating PSNR/SSIM. When calculating PSNR/SSIM/LPIPS/DISTS,  we set `crop_border=${scale}` if the scaling factor is not larger than 8, otherwise `crop_border=8`.

## Training

### Dataset preparation

Please follow this [instruction](TrainTestGSASR/datasets/README.MD) to prepare the training and testing datasets.

### Training GSASR

Please follow this [instruction](TrainTestGSASR/README.md) to train GSASR.



# License
This project is released under the Apache 2.0 license.


# 10.Acknowlegement

This project is built mainly based on the excellent [BasicSR](https://github.com/XPixelGroup/BasicSR), [HAT](https://github.com/XPixelGroup/HAT) and [ROPE-ViT](https://github.com/naver-ai/rope-vit) codeframe. We appreciate it a lot for their developers.

We sincerely thank [Mr.Zhengqiang Zhang](https://github.com/xtudbxk) for his support in the CUDA operator of rasterization.

# 11.Citation
If you find this research helpful for you, please cite our paper.
```bash
@article{chen2025generalized,
  title={Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution},
  author={Chen, Du and Chen, Liyi and Zhang, Zhengqiang and Zhang, Lei},
  journal={arXiv preprint arXiv:2501.06838},
  year={2025}
}
```
