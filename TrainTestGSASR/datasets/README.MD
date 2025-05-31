## Training dataset

We provide the download link of many widely-used datasets. Please download them from:

|Dataset|Link|
|:---:|:---:|
|DIV2K(1-800 images for training)|[Download](https://data.vision.ee.ethz.ch/cvl/DIV2K/)|
|Flickr2K|[Download](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)|
|OST|[Download](https://github.com/xinntao/SFTGAN)|
|FFHQ|[Download](https://github.com/NVlabs/ffhq-dataset)|
|LSDIR|[Download](https://ofsoundof.github.io/lsdir-data/)|
|DIV8K|[Download](https://competitions.codalab.org/competitions/22217#participate)|
|SA1B|[Download](https://ai.meta.com/datasets/segment-anything/)|

In our paper, we only use [DIV2K(1-800 images)](https://data.vision.ee.ethz.ch/cvl/DIV2K/) to train GSASR.

In our enhanced version, we use [DIV2K(1-800 images)](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and DF2K([DIV2K(1-800 images)](https://data.vision.ee.ethz.ch/cvl/DIV2K/) + [FLICKR2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)) to train GSASR.

In our ultra performance version, we use [SA1B](https://ai.meta.com/datasets/segment-anything/) to train GSASR.


## Testing dataset

We provide the download link of many widely-used datasets. Please download them from:

|Dataset|Link|
|:---:|:---:|
|DIV2K(801-900 images for testing)|[Download](https://data.vision.ee.ethz.ch/cvl/DIV2K/)|
|Set5|[Download](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)|
|Set14|[Download](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)|
|Urban100|[Download](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)|
|BSDS100|[Download](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)|
|Manga109|[Download](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)|
|General100|[Download](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)|
|LSDIR(Validation 250)|[Download](https://ofsoundof.github.io/lsdir-data/)|

Then, please use the following code to generate the low-resolution images through bicubic operator. Please note that, we
use the [bicubic operator](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/matlab_functions.py) from [BasicSR](https://github.com/XPixelGroup/BasicSR), which is the same bicubic operator as that in Matlab, while being excuted in Python. The codes are as follows,

```bash
cd GSASR/TrainTestGSASR/scripts/data_preparation
python generate_bicubic_img_use_python_matlab.py --gt [path to GT image folder] --scale_list [indicate the scaling factor] --save_path [path to your save path] 
```

We also provide our well-process version of thoes testing benchmarks, you could download them through the following link,

|Dataset|Link|
|:---:|:---:|
|Testing Benchmarks|[Google Drive](https://drive.google.com/drive/folders/1ivwuFoyNwRf9FevHGlCEnjhXLD6Og7mj?usp=sharing)|

## Benchmarks for testing the computational cost


We randomly crop 720 * 720 size images from DIV2K100 to serve as GT, and use bicubic downsampling to obtain the corresponding low-resolution images. Note that, we
use the [bicubic operator](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/matlab_functions.py) from [BasicSR](https://github.com/XPixelGroup/BasicSR), which is the same bicubic operator as that in Matlab, while being excuted in Python. The codes are as follows,

```bash
cd GSASR/TrainTestGSASR/scripts/data_preparation
python test_time_img_generate.py --gt [path to GT image folder] --scale_list [indicate the scaling factor] --save_path [path to your save path] --gt_size [set the size of GT images]
```

Please note that, we randomly crop images each time. Therefore, if you excute this code, your cropped results will be different from us.
If you want to directly use our data, please download them in the following,

|Dataset|Download|
|:---:|:---:|
|DIV2K_GT720|[Google Drive](https://drive.google.com/file/d/1FQrVcCppV_No-0BeTxUh2kBaGIb6xZ82/view?usp=sharing)|
