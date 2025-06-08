# Trainging details of GSASR

We provide three different versions of training codes here, including the paper version, the enhanced version with AMP and Flash Attention, as well as the ultra performance version.

## 1. Installation (Please skip this step if your have already installed the relevant environmets when you do quick inference)
 - python == 3.10
 - PyTorch == 2.0
 - Anaconda
 - CUDA == 11.8

**Please export the CUDA path, for me, I export it as follows,**

```bash
export CUDA_HOME=/home/notebook/code/personal/S9053766/chendu/cuda-11.8
```

Then install the relevant environments :
```bash
git clone https://github.com/ChrisDud0257/GSASR
cd GSASR/TrainTestGSASR
conda create --name gsasr python=3.10
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
BASICSR_EXT=True python setup.py develop
```

For more installation issues, please refer to the excellent [BasicSR](https://github.com/XPixelGroup/BasicSR) project.


## 2. Pretrained models

|           Encoder Backbone           |        Training Dataset|                                       Download                                               | Version|
|:------------------------:|:----------------------------------------------------------------------------------------------------:|:---:|:---:|
|EDSR|DIV2K| [Google Drive](https://drive.google.com/drive/folders/1rSnM1HOBaI6TpfJ0XkXhHZcjjRnS95Sb?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR_paper/tree/main/EDSR) |Paper Reported|
|EDSR| DIV2K | [Google Drive](https://drive.google.com/drive/folders/1R6ZCdAd6t_2CCpjCK67F9nag9jitMhI6?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/EDSR_DIV2K)|Enhanced |
|EDSR| DF2K| [Google Drive](https://drive.google.com/drive/folders/16TV2yJt_lfNqJnATtJnEkHV1KoBuW8ww?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/EDSR_DF2K) |Enhanced |
|RDN|DIV2K| [Google Drive](https://drive.google.com/drive/folders/1xR5JoiLG6Muav-C8XGpE4sTr2bleBxPU?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR_paper/tree/main/RDN) |Paper Reported|
|RDN| DIV2K| [Google Drive](https://drive.google.com/drive/folders/1guSg28c8gvrTkCvTmNbzqf9vWJfLv58Q?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/RDN_DIV2K) |Enhanced|
|RDN| DF2K|  [Google Drive](https://drive.google.com/drive/folders/1vkBvsiiNqTFKmPtNjPlqMn_mh_ClUrKE?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/RDN_DF2K) |Enhanced |
|SWIN| DIV2K| [Google Drive](https://drive.google.com/drive/folders/1Zv2ijlkyU0UdNz9XDvAu9HHaiUVmhkR0?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR_paper/tree/main/SWIN) |Paper (not reported)|
|SWIN| DIV2K| [Google Drive](https://drive.google.com/drive/folders/1kVLkOs4KrXlXsPsh0oqvey2dvT6TxqH-?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/SWIN_DIV2K) |Enhanced |
|SWIN| DF2K| [Google Drive](https://drive.google.com/drive/folders/1ql6dktVUlQFIoPSJkEuvvMPz9TlacMdy?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/SWIN_DF2K) |Enhanced|
|HATL| SA1B| [Google Drive](https://drive.google.com/drive/folders/1Pn-4JWvlMj50CulmAcBI1Hssiu-6nSYI?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/HATL-SA1B) |Ultra Performance|


## 3. Training GSASR

### 3.1 Paper Version

We provide the training codes of GSASR-EDSR-Baseline and GSASR-RDN. Besides, we also release the training codes of GSASR-SwinIR which are not reported in our paper (Due to the limited space, we do not report the results of GSASR-SwinIR). The training command is as follows, which is also written in ```demo.sh```.

Here, for example, if you want to train GSASR-EDSR-Baseline,

```bash
cd TrainTestGSASR/
### For single GPU training
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/train.py -opt ./options/train/paper/train_GSASR_EDSR-Baseline_paper_bicubic_x1_4.yml --auto_resume

### For DDP training
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 ./basicsr/train.py -opt ./options/train/paper/train_GSASR_EDSR-Baseline_paper_bicubic_x1_4.yml --launcher pytorch --auto_resume
```

in ```options/train/paper/train_GSASR_EDSR-Baseline_paper_bicubic_x1_4.yml```, you need to modify the training dataset settings and validation dataset settings :

Training dataset settings:
```bash
(line 25)       name: DIV2K
                    type: ContinuousBicubicDownsampleDataset
                    all_gt_list: ['/home/notebook/data/sharedgroup/RG_YLab/aigc_share_group_data/chendu/dataset/DIV2K/trainHR_multiscaleHR_shortest512_subimages512']
                    filename_tmpl: '{}'
                    io_backend:
                      type: disk
```

To prepare the training dataset, please follow this [instruction](datasets/README.MD).

Validation dataset settings:
```bash
(line 47)       name: DIV2K100
                    type: PairedImageDataset
                    dataroot_gt: /home/notebook/data/sharedgroup/RG_YLab/aigc_share_group_data/chendu/dataset/AnyScaleTestBicubic/DIV2K100/x4/GT
                    dataroot_lq: /home/notebook/data/sharedgroup/RG_YLab/aigc_share_group_data/chendu/dataset/AnyScaleTestBicubic/DIV2K100/x4/LR
                    io_backend:
                      type: disk

```

To prepare the validation/test dataset, please follow this [instruction](datasets/README.MD).


### 3.2 Enhanced Version (AMP + Flash Attention)

We provide the training codes of GSASR-EDSR-Baseline, GSASR-RDN and GSASR-SwinIR. The training command is as follows, which is also written in ```demo.sh```.

Here, for example, if you want to train GSASR-EDSR-Baseline,

```bash
cd TrainTestGSASR/
### For single GPU training
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/train.py -opt ./options/train/AMP/train_GSASR_EDSR-Baseline_amp_DIV2K_bicubic_x1_4.yml --auto_resume

### For DDP training
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 ./basicsr/train.py -opt ./options/train/AMP/train_GSASR_EDSR-Baseline_amp_DIV2K_bicubic_x1_4.yml --launcher pytorch --auto_resume
```

in ```options/train/AMP/train_GSASR_EDSR-Baseline_amp_DIV2K_bicubic_x1_4.yml```, you need to modify the training dataset settings and validation dataset settings :

Training dataset settings:
```bash
(line 25)       name: DIV2K
                    type: ContinuousBicubicDownsampleDataset
                    all_gt_list: ['/home/notebook/data/sharedgroup/RG_YLab/aigc_share_group_data/chendu/dataset/DIV2K/trainHR_multiscaleHR_shortest512_subimages512']
                    filename_tmpl: '{}'
                    io_backend:
                      type: disk
```

To prepare the training dataset, please follow this [instruction](datasets/README.MD).

Validation dataset settings:
```bash
(line 47)       name: DIV2K100
                    type: PairedImageDataset
                    dataroot_gt: /home/notebook/data/sharedgroup/RG_YLab/aigc_share_group_data/chendu/dataset/AnyScaleTestBicubic/DIV2K100/x4/GT
                    dataroot_lq: /home/notebook/data/sharedgroup/RG_YLab/aigc_share_group_data/chendu/dataset/AnyScaleTestBicubic/DIV2K100/x4/LR
                    io_backend:
                      type: disk

```

To prepare the validation/test dataset, please follow this [instruction](datasets/README.MD).


### 3.3 Ultra Performance Version (AMP + Flash Attention + HAT-L Encoder+ SA1B Training Dataset)

We provide the training codes of GSASR-HATL. The training command is as follows, which is also written in ```demo.sh```.

Based on AMP and Flash Attention, we train GSASR with [HAT-L](https://github.com/XPixelGroup/HAT) encoder to explore the ultimost performance.

The trainging setting are as follows:

|Training Details|Settings|
|:---:|:---:|
|Dataset|[SA1B]([https://ai.meta.com/datasets/segment-anything/)|
|GPUs|16 x NVIDIA A100|
|Batch Size per GPU|8|
|Iterations|500000|
|Training Time Cost|30 days|
|Encoder|HAT-L|
|Range of Scaling Factor in Training|[1,16]|
|Input LR size|64 x 64|
|Acceleration Strategy|AMP + Flash Attention|

Here, for example, if you want to train GSASR-HATL,

```bash
cd TrainTestGSASR/
### For single GPU training
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/train.py -opt ./options/train/UltraPerformance/train_GSASR_HATL_amp_SA1B_bicubic_x1_16.yml --auto_resume

### For DDP training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 ./basicsr/train.py -opt ./options/train/UltraPerformance/train_GSASR_HATL_amp_SA1B_bicubic_x1_16.yml --launcher pytorch --auto_resume
```

in ```options/train/UltraPerformance/train_GSASR_HATL_amp_SA1B_bicubic_x1_16.yml```, you need to modify the training dataset settings and validation dataset settings :

Training dataset settings:
```bash
(line 25)           name: SA1B
                        type: ContinuousBicubicDownsampleSA1BDataset
                        all_gt_list: ['/home/notebook/data/group/SA1B/images/']
                        filename_tmpl: '{}'
                        io_backend:
                          type: disk
```

To prepare the training dataset, please follow this [instruction](datasets/README.MD).

Validation dataset settings:
```bash
(line 49)       name: DIV2K100
                    type: PairedImageDataset
                    dataroot_gt: /home/notebook/data/sharedgroup/RG_YLab/aigc_share_group_data/chendu/dataset/AnyScaleTestBicubic/DIV2K100/x4/GT
                    dataroot_lq: /home/notebook/data/sharedgroup/RG_YLab/aigc_share_group_data/chendu/dataset/AnyScaleTestBicubic/DIV2K100/x4/LR
                    io_backend:
                      type: disk

```

To prepare the validation/test dataset, please follow this [instruction](datasets/README.MD).


# Testing details of GSASR

## 1. Pretrained models

|           Encoder Backbone           |        Training Dataset|                                       Download                                               | Version|
|:------------------------:|:----------------------------------------------------------------------------------------------------:|:---:|:---:|
|EDSR|DIV2K| [Google Drive](https://drive.google.com/drive/folders/1rSnM1HOBaI6TpfJ0XkXhHZcjjRnS95Sb?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR_paper/tree/main/EDSR) |Paper Reported|
|EDSR| DIV2K | [Google Drive](https://drive.google.com/drive/folders/1R6ZCdAd6t_2CCpjCK67F9nag9jitMhI6?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/EDSR_DIV2K)|Enhanced |
|EDSR| DF2K| [Google Drive](https://drive.google.com/drive/folders/16TV2yJt_lfNqJnATtJnEkHV1KoBuW8ww?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/EDSR_DF2K) |Enhanced |
|RDN|DIV2K| [Google Drive](https://drive.google.com/drive/folders/1xR5JoiLG6Muav-C8XGpE4sTr2bleBxPU?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR_paper/tree/main/RDN) |Paper Reported|
|RDN| DIV2K| [Google Drive](https://drive.google.com/drive/folders/1guSg28c8gvrTkCvTmNbzqf9vWJfLv58Q?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/RDN_DIV2K) |Enhanced|
|RDN| DF2K|  [Google Drive](https://drive.google.com/drive/folders/1vkBvsiiNqTFKmPtNjPlqMn_mh_ClUrKE?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/RDN_DF2K) |Enhanced |
|SWIN| DIV2K| [Google Drive](https://drive.google.com/drive/folders/1Zv2ijlkyU0UdNz9XDvAu9HHaiUVmhkR0?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR_paper/tree/main/SWIN) |Paper (not reported)|
|SWIN| DIV2K| [Google Drive](https://drive.google.com/drive/folders/1kVLkOs4KrXlXsPsh0oqvey2dvT6TxqH-?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/SWIN_DIV2K) |Enhanced |
|SWIN| DF2K| [Google Drive](https://drive.google.com/drive/folders/1ql6dktVUlQFIoPSJkEuvvMPz9TlacMdy?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/SWIN_DF2K) |Enhanced|
|HATL| SA1B| [Google Drive](https://drive.google.com/drive/folders/1Pn-4JWvlMj50CulmAcBI1Hssiu-6nSYI?usp=sharing),  [Hugging Face](https://huggingface.co/mutou0308/GSASR/tree/main/HATL-SA1B) |Ultra Performance|



## 2. Testing GSASR
### 2.1 Paper Version
We provide the testing codes of GSASR-EDSR-Baseline and GSASR-RDN. Besides, we also release the training codes of GSASR-SwinIR which are not reported in our paper (Due to the limited space, we do not report the results of GSASR-SwinIR). The training command is as follows, which is also written in ```demo.sh```.

Here, for example, if you want to test GSASR-EDSR-Baseline,

```bash
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/test.py -opt ./options/test/paper/test_GSASR_EDSR-Baseline_paper_bicubic_x1_4_x4.yml
```

in ```options/test/paper/test_GSASR_EDSR-Baseline_paper_bicubic_x1_4_x4.yml```, you need to modify the scaling factor, testing dataset and the pretrained models:

Scaling factor:
```bash
(line 3) scale: 4
```

The scaling factor must be the same as the corresponding testing dataset's scaling factor.

Testing dataset settings:
```bash
(line 22)      test_1:  # the 1st test dataset
                  name: Set5
                  type: PairedImageDataset
                  dataroot_gt: /home/notebook/data/sharedgroup/RG_YLab/aigc_share_group_data/chendu/dataset/AnyScaleTestBicubic/Set5/x4/GT
                  dataroot_lq: /home/notebook/data/sharedgroup/RG_YLab/aigc_share_group_data/chendu/dataset/AnyScaleTestBicubic/Set5/x4/LR
                  io_backend:
                    type: disk
...
```
You are supposed to indicate the image paths.
 For more details, please refer to this [instruction](datasets/README.MD) to prepare the testing datasets.


Pretrained network_g.pth and net_fea2gs.pth:
```bash
(line 107)  path:
              # pretrain_network_g: experiments/GSASR_EDSR-Baseline_paper_bicubic_x1-4/models/net_g_200.pth
              pretrain_network_g: /home/notebook/code/personal/S9053766/chendu/FinalUpload/GSASR/pretrained_models/Paper/EDSR-Baseline/net_g.pth
              strict_load_g: True
              param_key_g: params_ema

            path_fea2gs:
              # pretrain_network_fea2gs: experiments/GSASR_EDSR-Baseline_paper_bicubic_x1-4/models/net_fea2gs_200.pth
              pretrain_network_fea2gs: /home/notebook/code/personal/S9053766/chendu/FinalUpload/GSASR/pretrained_models/Paper/EDSR-Baseline/net_fea2gs.pth
              strict_load_fea2gs: True
              param_key_fea2gs: params_ema
```
Note that, if your GPU memory is limited, please set ```(line 9) tile_process: True```.

### 2.2 Enhanced Version (AMP + Flash Attention)
We provide the testing codes of GSASR-EDSR-Baseline ,GSASR-RDN and GSASR-SwinIR. The testing command is as follows, which is also written in ```demo.sh```.

Here, for example, if you want to test GSASR-EDSR-Baseline,

```bash
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/test.py -opt ./options/test/AMP/test_GSASR_EDSR-Baseline_amp_DIV2K_bicubic_x1_4_x4.yml
```

in ```options/test/AMP/test_GSASR_EDSR-Baseline_amp_DIV2K_bicubic_x1_4_x4.yml```, you need to modify the scaling factor, testing dataset, and the pretrained models:

Scaling factor:
```bash
(line 3) scale: 4
```

The scaling factor must be the same as the corresponding testing dataset's scaling factor.

Testing dataset settings:
```bash
(line 23)      test_1:  # the 1st test dataset
                  name: Set5
                  type: PairedImageDataset
                  dataroot_gt: /home/notebook/data/sharedgroup/RG_YLab/aigc_share_group_data/chendu/dataset/AnyScaleTestBicubic/Set5/x4/GT
                  dataroot_lq: /home/notebook/data/sharedgroup/RG_YLab/aigc_share_group_data/chendu/dataset/AnyScaleTestBicubic/Set5/x4/LR
                  io_backend:
                    type: disk
...
```
You are supposed to indicate the image paths.
 For more details, please refer to this [instruction](datasets/README.MD) to prepare the testing datasets.


Pretrained network_g.pth and net_fea2gs.pth:
```bash
(line 110)  path:
              # pretrain_network_g: experiments/GSASR_EDSR-Baseline_paper_bicubic_x1-4/models/net_g_200.pth
              pretrain_network_g: /home/notebook/code/personal/S9053766/chendu/FinalUpload/GSASR/pretrained_models/Paper/EDSR-Baseline/net_g.pth
              strict_load_g: True
              param_key_g: params_ema

            path_fea2gs:
              # pretrain_network_fea2gs: experiments/GSASR_EDSR-Baseline_paper_bicubic_x1-4/models/net_fea2gs_200.pth
              pretrain_network_fea2gs: /home/notebook/code/personal/S9053766/chendu/FinalUpload/GSASR/pretrained_models/Paper/EDSR-Baseline/net_fea2gs.pth
              strict_load_fea2gs: True
              param_key_fea2gs: params_ema
```

Note that, if your GPU does not support for AMP and Flash Attention, please set ```(line 12) test_AMP: False```.

Note that, if your GPU memory is limited, please set ```(line 9) tile_process: True```.


### 2.3 Ultra Performance Version (AMP + Flash Attention + HAT-L Encoder+ SA1B Training Dataset)
We provide the testing codes of GSASR-HATL. The testing command is as follows, which is also written in ```demo.sh```.

Here, for example, if you want to test GSASR-HATL,

```bash
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/test.py -opt ./options/test/UltraPerformance/test_GSASR_HATL_amp_SA1B_bicubic_x1_16_x4.yml
```

in ```options/test/UltraPerformance/test_GSASR_HATL_amp_SA1B_bicubic_x1_16_x4.yml```, you need to modify the scaling factor, testing dataset and the pretrained models:

Scaling factor:
```bash
(line 3) scale: 4
```

The scaling factor must be the same as the corresponding testing dataset's scaling factor.

Testing dataset settings:
```bash
(line 23)      test_1:  # the 1st test dataset
                  name: Set5
                  type: PairedImageDataset
                  dataroot_gt: /home/notebook/data/sharedgroup/RG_YLab/aigc_share_group_data/chendu/dataset/AnyScaleTestBicubic/Set5/x4/GT
                  dataroot_lq: /home/notebook/data/sharedgroup/RG_YLab/aigc_share_group_data/chendu/dataset/AnyScaleTestBicubic/Set5/x4/LR
                  io_backend:
                    type: disk
...
```
You are supposed to indicate the image paths.
 For more details, please refer to this [instruction](datasets/README.MD) to prepare the testing datasets.


Pretrained network_g.pth and net_fea2gs.pth:
```bash
(line 110)  path:
              # pretrain_network_g: experiments/GSASR_EDSR-Baseline_paper_bicubic_x1-4/models/net_g_200.pth
              pretrain_network_g: /home/notebook/code/personal/S9053766/chendu/FinalUpload/GSASR/pretrained_models/Paper/EDSR-Baseline/net_g.pth
              strict_load_g: True
              param_key_g: params_ema

            path_fea2gs:
              # pretrain_network_fea2gs: experiments/GSASR_EDSR-Baseline_paper_bicubic_x1-4/models/net_fea2gs_200.pth
              pretrain_network_fea2gs: /home/notebook/code/personal/S9053766/chendu/FinalUpload/GSASR/pretrained_models/Paper/EDSR-Baseline/net_fea2gs.pth
              strict_load_fea2gs: True
              param_key_fea2gs: params_ema
```

Note that, if your GPU does not support for AMP and Flash Attention, please set ```(line 12) test_AMP: False```.

Note that, if your GPU memory is limited, please set ```(line 9) tile_process: True```.
