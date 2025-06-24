# License and Acknowledgement

This TrainTestGSASR project is released under the Apache 2.0 license.

We utilize and modify some codes from the repositories as follows,

- BasicSR
  - The codes are modified from the repository [BasicSR](https://github.com/XPixelGroup/BasicSR). The LICENSE is included as [license_BasicSR](LICENSE_BasicSR).
  - The official repository is <https://github.com/XPixelGroup/BasicSR>.
- pytorch-image-models
  - We use the implementation of `DropPath` and `trunc_normal_` from [pytorch-image-models](https://github.com/rwightman/pytorch-image-models/). The LICENSE is included as [LICENSE_pytorch-image-models](LICENSE_pytorch-image-models).
- SwinIR
  - The arch implementation of SwinIR is from [SwinIR](https://github.com/JingyunLiang/SwinIR). The LICENSE is included as [LICENSE_SwinIR](LICENSE_SwinIR).
- HAT
  - The arch implementation of HAT is from [HAT](https://github.com/XPixelGroup/HAT). The LICENSE is included as [LICENSE_HAT](LICENSE_HAT).
- ROPE-ViT
  - The arch implementation of ROPE is from [ROPE-ViT](https://github.com/XPixelGroup/HAT](https://github.com/naver-ai/rope-vit). The LICENSE is included as [LICENSE_ROPE-ViT](LICENSE_ROPE-ViT).


## References

1. NIQE metric: the codes are translated from the [official MATLAB codes](http://live.ece.utexas.edu/research/quality/niqe_release.zip)

    > A. Mittal, R. Soundararajan and A. C. Bovik, "Making a Completely Blind Image Quality Analyzer", IEEE Signal Processing Letters, 2012.

1. FID metric: the codes are modified from [pytorch-fid](https://github.com/mseitzer/pytorch-fid) and [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch).
