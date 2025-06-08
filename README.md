<div align="center">
<h1>Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution</h1>

[**Du Chen**](https://github.com/ChrisDud0257)<sup>1,2*</sup> · [**Liyi Chen**](https://github.com/mt-cly)<sup>1*</sup> · [**Zhengqiang Zhang**](https://github.com/xtudbxk)<sup>1,2</sup> · [**Lei Zhang**](https://www4.comp.polyu.edu.hk/~cslzhang/)<sup>1,2*</sup>
<br>

<sup>1</sup>The Hong Kong Polytechnic University <sup>2</sup>OPPO Research Institute
<br>
*Equal contribution &dagger;Corresponding author &emsp; 

<a href="https://arxiv.org/abs/2501.06838"><img src="https://img.shields.io/badge/%F0%9F%93%84%20arXiv-2501.06-B31B1B.svg"></a>
<a href="https://mt-cly.github.io/GSASR.github.io/"><img src="https://img.shields.io/badge/%F0%9F%8F%A0%20Project%20Page-GASAR-green.svg" alt='Project Page'></a>
<a href="https://huggingface.co/mutou0308/GSASR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a>
<a href="https://huggingface.co/spaces/mutou0308/GSASR"><img src="https://img.shields.io/badge/%F0%9F%9A%80%20Gradio%20Demo-Huggingface-blue"></a>

</div>

This work presents GSASR. It achieve SoTA in arbitrary-scale super-resolution by representing given LR image as millions of continuous 2D Gaussians.  

![Fast Rasterization](./assets/sampling.png)




**Comparisons among representative/SoTA ASR models and different versions of GSASR, PSNR/SSIM are tested on Y channel of Ycbcr space.**

<div style="overflow-x:auto; font-size:10px;">
<table>
  <tr>
    <th rowspan="2">Encoder Backbone</th>
    <th rowspan="2">Methods</th>
    <th rowspan="2">Version</th>
    <th rowspan="2">Trainging Dataset</th>
    <th colspan="3" align="center">PSNR/SSIM/LPIPS/DIST (x4 scaling factor)</th>
  </tr>
  <tr>
    <td align="center">DIV2K</td>
    <td align="center">LSDIR</td>
    <td align="center">Urban100</td>
  </tr>

  <!-- EDSR Backbone -->
  <tr>
    <td rowspan="6">EDSR</td>
    <td>LIIF</td>
    <td>Paper</td>
    <td>DIV2K</td>
    <td align="center">30.43/0.8388/0.2662/0.1403</td>
    <td align="center">26.21/0.7614/0.2978/0.1678</td>
    <td align="center">26.14/0.7885/0.2271/0.1738</td>
  </tr>
  <tr>
    <td>GaussianSR</td>
    <td>Paper</td>
    <td>DIV2K</td>
    <td align="center">30.46/0.8389/0.2684/0.1406</td>
    <td align="center">26.23/0.7615/0.3007/0.1679</td>
    <td align="center">26.19/0.7893/0.2283/0.1730</td>
  </tr>
  <tr>
    <td>CiaoSR</td>
    <td>Paper</td>
    <td>DIV2K</td>
    <td align="center">30.67/0.8431/0.2585/0.1370</td>
    <td align="center">26.42/0.7681/0.2865/0.1631</td>
    <td align="center">26.69/0.8091/0.2078/0.1659</td>
  </tr>
  <tr>
    <td>GSASR</td>
    <td>Paper Reported</td>
    <td>DIV2K</td>
    <td align="center">30.89/0.8486/0.2518/0.1301</td>
    <td align="center">26.65/0.7774/0.2777/0.1554</td>
    <td align="center">27.01/0.8142/0.1987/0.1552</td>
  </tr>
  <tr>
    <td>GSASR</td>
    <td>Enhanced</td>
    <td>DIV2K</td>
    <td align="center">31.01/0.8509/0.2508/0.1306</td>
    <td align="center">26.78/0.7813/0.2962/0.1543</td>
    <td align="center">27.34/0.8230/0.1920/0.1515</td>
  </tr>
  <tr>
    <td>GSASR</td>
    <td>Enhanced</td>
    <td>DF2K</td>
    <td align="center">31.04/0.8515/0.2512/0.1307</td>
    <td align="center">26.82/0.7827/0.2751/0.1540</td>
    <td align="center">27.45/0.8256/0.1902/0.1507</td>
  </tr>

  <!-- RDN Backbone -->
  <tr>
    <td rowspan="6">RDN</td>
    <td>LIIF</td>
    <td>Paper</td>
    <td>DIV2K</td>
    <td align="center">30.71/0.8449/0.2566/0.1354</td>
    <td align="center">26.48/0.7714/0.2838/0.1603</td>
    <td align="center">26.71/0.8055/0.2062/0.1562</td>
  </tr>
  <tr>
    <td>GaussianSR</td>
    <td>Paper</td>
    <td>DIV2K</td>
    <td align="center">30.76/0.8457/0.2570/0.1347</td>
    <td align="center">26.53/0.7727/0.2837/0.1595</td>
    <td align="center">26.77/0.8064/0.2069/0.1610</td>
  </tr>
  <tr>
    <td>CiaoSR</td>
    <td>Paper</td>
    <td>DIV2K</td>
    <td align="center">30.91/0.8481/0.2525/0.1327</td>
    <td align="center">26.66/0.7770/0.2768/0.1563</td>
    <td align="center">27.10/0.8142/0.1966/0.1559</td>
  </tr>
  <tr>
    <td>GSASR</td>
    <td>Paper Reported</td>
    <td>DIV2K</td>
    <td align="center">30.96/0.8500/0.2505/0.1288</td>
    <td align="center">26.73/0.7801/0.2752/0.1533</td>
    <td align="center">27.15/0.8177/0.1953/0.1515</td>
  </tr>
  <tr>
    <td>GSASR</td>
    <td>Enhanced</td>
    <td>DIV2K</td>
    <td align="center">31.03/0.8513/0.2499/0.1306</td>
    <td align="center">26.79/0.7819/0.2740/0.1543</td>
    <td align="center">27.37/0.8238/0.1898/0.1511</td>
  </tr>
  <tr>
    <td>GSASR</td>
    <td>Enhanced</td>
    <td>DF2K</td>
    <td align="center">31.10/0.8525/0.2482/0.1296</td>
    <td align="center">26.88/0.7848/0.2709/0.1527</td>
    <td align="center">27.58/0.8289/0.1849/0.1500</td>
  </tr>

  <!-- SWIN Backbone -->
  <tr>
    <td rowspan="4">SWIN</td>
    <td>CiaoSR</td>
    <td>Paper</td>
    <td>DIV2K</td>
    <td align="center">31.05/0.8511/0.2487/0.1316</td>
    <td align="center">26.80/0.7812/0.2724/0.1552</td>
    <td align="center">27.40/0.8231/0.1869/0.1535</td>
  </tr>
  <tr>
    <td>GSASR</td>
    <td>Paper (not Reported)</td>
    <td>DIV2K</td>
    <td align="center">31.06/0.8521/0.2487/0.1270</td>
    <td align="center">26.84/0.7837/0.2719/0.1503</td>
    <td align="center">27.39/0.8247/0.1913/0.1466</td>
  </tr>
  <tr>
    <td>GSASR</td>
    <td>Enhanced</td>
    <td>DIV2K</td>
    <td align="center">31.10/0.8530/0.2463/0.1285</td>
    <td align="center">26.88/0.7849/0.2690/0.1517</td>
    <td align="center">27.55/0.8280/0.1850/0.1475</td>
  </tr>
  <tr>
    <td>GSASR</td>
    <td>Enhanced</td>
    <td>DF2K</td>
    <td align="center">31.17/0.8541/0.2456/0.1288</td>
    <td align="center">26.96/0.7876/0.2665/0.1513</td>
    <td align="center">27.81/0.8343/0.1781/0.1465</td>
  </tr>

  <!-- HATL Backbone -->
  <tr>
    <td>HATL</td>
    <td>GSASR</td>
    <td>Ultra Performance</td>
    <td>SA1B</td>
    <td align="center">31.31/0.8570/0.2381/0.1268</td>
    <td align="center">27.17/0.7948/0.2548/0.1470</td>
    <td align="center">28.44/0.8493/0.1580/0.1394</td>
  </tr>
</table>

Maybe you are confused with different versions of GSASR models, we explain here.
 - Paper Reported: the results of which are reported in our paper.
 - Paper (not Reported): the results of which are not shown in our paper due to limited pages.
 - Enhanced: we further utilize Flash Attention to substitute the vanilla self attention in Gaussion decoder to do acceleration, and utilize [Rotary Position Embedding (ROPE)](https://github.com/naver-ai/rope-vit) to support the position embedding in Flash Attention. During training/inference stage, we utilize Automatic Mixed Precision (AMP) by combining bfloat16 and fp32 precisions to further accelerate the speed.
 - Ultra Performance: based on AMP+Flash Attention+ROPE in enhanced version, we train with [HAT-L](https://github.com/XPixelGroup/HAT) encoder on [SA1B](https://ai.meta.com/datasets/segment-anything/) dataset to explore the ultimost performance of GSASR.

## 🎉  News
- **2025-06-08:** We update the comparison results among representative/SoTA ASR models and different versions of GSASR.
- **2025-06-05:** The online demo with most powerful HATL-based GSASR is released, [click to try it](https://huggingface.co/spaces/mutou0308/GSASR).
- **2025-05-30:** The {EDSR, RDN, SWIN, HATL}-based GSASR models are available.
- **2025-01-16:** GSASR [paper](https://arxiv.org/abs/2501.06838) and [project papge](https://mt-cly.github.io/GSASR.github.io/) are released.


## ⚙️  Pre-trained Models (Enhanced and Ultra Performance Version)

Apart from the models reported in our paper, we further provide different versions of **GSASR models** here, which are trained with varied encoder backbones or training datasets.

Please note that, in the following, based on our `Paper Reported` models, `Enhanced` and `Ultra Performance` models employ AMP+ROPE+Flash Attention to reduce memory and time cost. Thoes models have superior performance than the models reported in our main paper:

|           Model Backbone           |        Training Dataset|                                       Download                                               | Version|
|:------------------------:|:----------------------------------------------------------------------------------------------------:|:---:|:---:|
|EDSR| DIV2K | [Google Drive](https://drive.google.com/drive/folders/1R6ZCdAd6t_2CCpjCK67F9nag9jitMhI6?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/EDSR_DIV2K)|Enhanced |
|EDSR| DF2K| [Google Drive](https://drive.google.com/drive/folders/16TV2yJt_lfNqJnATtJnEkHV1KoBuW8ww?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/EDSR_DF2K) |Enhanced |
|RDN| DIV2K| [Google Drive](https://drive.google.com/drive/folders/1guSg28c8gvrTkCvTmNbzqf9vWJfLv58Q?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/RDN_DIV2K) |Enhanced|
|RDN| DF2K|  [Google Drive](https://drive.google.com/drive/folders/1vkBvsiiNqTFKmPtNjPlqMn_mh_ClUrKE?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/RDN_DF2K) |Enhanced |
|SWIN| DIV2K| [Google Drive](https://drive.google.com/drive/folders/1kVLkOs4KrXlXsPsh0oqvey2dvT6TxqH-?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/SWIN_DIV2K) |Enhanced |
|SWIN| DF2K| [Google Drive](https://drive.google.com/drive/folders/1ql6dktVUlQFIoPSJkEuvvMPz9TlacMdy?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/SWIN_DF2K) |Enhanced|
|HATL| SA1B| [Google Drive](https://drive.google.com/drive/folders/1Pn-4JWvlMj50CulmAcBI1Hssiu-6nSYI?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/HATL-SA1B) |Ultra Performance|

Toward the results of our paper, we do not use these tricks (AMP+ROPE+Flash Attention/extra training datasets) for fair comparison.

As for the pretrained models reported in our paper, please refer to **Pre-trained Models (Paper Version)** in the below.

## 🔧 Usage

### Prepraration
 - Pytorch == 2.0 (PyTorch Version must >= 2.0)
 - Anaconda
 - CUDA Toolkit (necessary)


Firstly, please make sure you have installed [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)! Since we have hand-crafted CUDA operators, you need to compile them when you run GSASR.

```bash
git clone https://github.com/ChrisDud0257/GSASR
cd GSASR
conda create --name gsasr python=3.10
conda activate gsasr
export CUDA_HOME=${path_to_CUDA} ### specify the path to cuda-11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
python setup_gscuda.py install # gscuda
cd TrainTestGSASR
pip install -r requirements.txt
BASICSR_EXT=True python setup.py develop # basicsr
```

We have tested that the versions of CUDA from 11.0 to 12.4 are all OK.


### Runing
You need to properly authenticate with Hugging Face to download our model weights. Once set up, our code will handle it automatically at your first run. You can authenticate by running

```bash
# This will prompt you to enter your Hugging Face credentials.
huggingface-cli login
```

You can try GSASR easily by lanching gradio demo or runing in command. 

### 🚀 Gradio demo
```bash
python demo_gr.py
```

### 💻 CLI
```bash
python inference_enhenced.py \
    --input_img_path <path_to_img> \
    --save_sr_path <path_to_saved_folder> \
    --model <{EDSR_DIV2K, EDSR_DF2K, RDN_DIV2K, RDN_DF2K, SWIN_DIV2K,SWIN_DF2K, HATL_SA1B}> \
    --scale <scale> [--tile_process] [--AMP_test]
```
If it fails to access Huggingface, try to manually download pretrained models and specify local path with `--model_path <path_to_model_weight>`. 

Using `--tile_process` and `--AMP_test` if memory is limited.


## 📏 Pre-trained Models (Paper Version)

Please note that, in our paper, we only train GSASR on DIV2K dataset without AMP+RoPE+Flash Attention tricks for fair comparison. 
Due to the limited pages in our paper, we don't report the results of Swin-based model. Here, besides {EDSR, RDN}-based GSASR present in paper, we further provide the Swin-based GSASR model.
The {EDSR, RDN}-based GSASR models provided bellow should exactly generate the same results as that reported in our paper (Table.1 in the main paper and Table.1 ~ Table.7 in the supplementary).

### Download Pre-trained models (Paper Version)
Download models from the following link.

|           Encoder Backbone           |  Training Dataset |                                             Download                                               | Version|
|:------------------------:|:---:|:----------------------------------------------------------------------------------------------------:|:---:|
|EDSR|DIV2K| [Google Drive](https://drive.google.com/drive/folders/1rSnM1HOBaI6TpfJ0XkXhHZcjjRnS95Sb?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR_paper/tree/main/EDSR) |Paper Reported|
|RDN|DIV2K| [Google Drive](https://drive.google.com/drive/folders/1xR5JoiLG6Muav-C8XGpE4sTr2bleBxPU?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR_paper/tree/main/RDN) |Paper Reported|
|SWIN| DIV2K| [Google Drive](https://drive.google.com/drive/folders/1Zv2ijlkyU0UdNz9XDvAu9HHaiUVmhkR0?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR_paper/tree/main/SWIN) |Paper (not Reported)|


### Inference for single image
if you have logined in the huggingface, directly execute the `inference_paper.py` as follows.
```bash
python inference_paper.py \
    --input_img_path <path_to_img> \
    --save_sr_path <path_to_saved_folder> \
    --model <{EDSR, RDN, SWIN}> \
    --scale <scale> [--tile_process]
```


### Inference on standard benchmark
To get the numerical performance in Tab.2 of the main paper. Please download cropped 720*720 size of GT images, and the corresponding LR images of DIV2K testing parts, which are utilized in our paper.

The {EDSR, RDN}-based GSASR models provided bellow should exactly generate the same PSNR/SSIM/LPIPS/DISTS results as that reported in our paper (Table.2 in the main paper and Table.8 the supplementary).

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



If you want to test GSASR on standard benchmarks with full size, please use the same commands as above. We also provide the widely-used testing benchmarks, including Set5, Set14, DIV2K-val 100, LSDIR-val 250, Urban100, Manga109, BSDS100, General100, and each GT image's corresponding LR counterparts with different scaling factors which is obtained by bicubic operation.

|Dataset|Link|
|:---:|:---:|
|Testing Benchmarks|[Google Drive](https://drive.google.com/drive/folders/1ivwuFoyNwRf9FevHGlCEnjhXLD6Og7mj?usp=sharing)|

### Memory and inference time estimation
In `inference_paper_benchmarks.py`, we integrate the statistics code of test time (ms) and GPU memory (MB). In our paper, we calculate the computational cost on a single NVIDIA A100 GPU, and we input the full size image into the model, we don't use tile_process. The inference time omit the pre-processing and post-processing and record the full pipeline cost inlcuding encoder, decoder and rendering.

### Metrics
After inference,  execute the code to estimate PSNR/SSIM/LPIPS/DISTS.

```bash
cd TrainTestGSASR/scripts/metrics/
python calculate_psnr_ssim.py --test_y_channel --gt <path_to_GT_folder> --restored <path_to_SR_folder> --scale <scale> [--suffix <suffix_of_images>]
python calculate_lpips.py  --gt <path_to_GT_folder> --restored <path_to_SR_folder> --scale <scale> [--suffix <suffix_of_images>]
python calculate_dists.py  --gt <path_to_GT_folder> --restored <path_to_SR_folder> --scale <scale> [--suffix <suffix_of_images>]
```
Please note that we test them on Y channel of Ycbcr space with `--test_y_channel`  when calculating PSNR/SSIM. When calculating PSNR/SSIM/LPIPS/DISTS,  we set `crop_border=${scale}` if the scaling factor is not larger than 8, otherwise `crop_border=8`.

## 🗝️ Training and Testing

### Dataset preparation

Please follow this [instruction](TrainTestGSASR/datasets/README.MD) to prepare the training and testing datasets.

### Training GSASR

Please follow this [instruction](TrainTestGSASR/README.md) to train GSASR.

### Testing GSASR

Please follow this [instruction](TrainTestGSASR/README.md) to test GSASR if you further want to do it .


## 🙏 Acknowlegement

This project is built mainly based on the excellent [BasicSR](https://github.com/XPixelGroup/BasicSR), [HAT](https://github.com/XPixelGroup/HAT) and [ROPE-ViT](https://github.com/naver-ai/rope-vit) codeframe. We appreciate it a lot for their developers.

We sincerely thank [Mr.Zhengqiang Zhang](https://github.com/xtudbxk) for his support in the CUDA operator of rasterization.

## 📚 Citation
If you find this research helpful for you, please cite our paper.
```bash
@article{chen2025generalized,
  title={Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution},
  author={Chen, Du and Chen, Liyi and Zhang, Zhengqiang and Zhang, Lei},
  journal={arXiv preprint arXiv:2501.06838},
  year={2025}
}
```
