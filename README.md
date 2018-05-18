# Introduction

Real time segmentation inference production ready code based on DeepLab-ResNet, PSP Net.


![Demo Segmentation](https://user-images.githubusercontent.com/8921629/40209620-39e7bbb2-59f5-11e8-94ee-2c320e3062dd.png)

# Note

This repo is basically a cleaned unified re-organized code from multiple open sourced [projects](https://github.com/KleinYuan/tf-segmentation#borrowed-code). I tried to implement as good practice as possible.
 However, there are still some ugly code borrowed from the original repo requiring tons of time to refine, while I haven't got enough bandwidth yet. Ultimately, I will even rewrite all the models/network layers so that it's easier to understand, modify, deploy on all levels, with clean level similar to [this one](https://github.com/KleinYuan/cnn).

Honestly, if the pre-trained model can be frozen well, we can just use similar [architecture](https://github.com/KleinYuan/tf-object-detection) for inference. However, unfortunately, my previous efforts on freeze the graph and fetch tensor running the compute does not really give reasonable predictions. I may spend more time on that issue later so that
we can remove the ugly network constructing code.

This repo is all about inference, server, api instead of training, or research.

Therefore, you are expected to do following things with this repo:

- [X] Real Time Segmentation with multiple models against Camera

- [X] Segmentation API wrapped up with Docker container, ready to deploy

- [X] Multiple open source pre-trained segmentation model performance evaluation against single image


# Dependencies

`Actually simply using Anaconda may save you a year!`

- [X] Python 2.X

- [X] Tensorflow > 1.0.0

- [X] OpenCV

* No GPU required

# Run Demo

1. Download pre-trained model first 

- [X] [DeepLab-ResNet](https://drive.google.com/drive/folders/0B9CKOTmy0DyaQ2oxUHdtYUd2Mm8?usp=sharing)

- [X] [PSP Net](https://drive.google.com/drive/folders/1S90PWzXEX_GNzulG1f2eTHvsruITgqsm?usp=sharing)

2. Put model.* files under `/model` folder ensuring the name is consistent with model name

```
--model
  |-- deeplab
      |-- checkpoint
      |-- model.ckpt-100000.data-00000-of-00001
      |-- model.ckpt-100000.index
      |-- model.ckpt-100000.meta
  |-- pspnet50
      |-- checkpoint
      |-- model.ckpt-0.data-00000-of-00001
      |-- model.ckpt-0.index
      |-- model.ckpt-0.meta
  |-- pspnet101
      |-- checkpoint
      |-- model.ckpt-0.data-00000-of-00001
      |-- model.ckpt-0.index
      |-- model.ckpt-0.meta

```

3. run below:

```
make demo
```

Note: you can change this [line](https://github.com/KleinYuan/tf-segmentation/blob/master/demo.py#L8) to be with `DeepLab`, `PSPNet101` and `PSPNet50`.

# Freeze Model [Optional]

```
# Navigate to tf-segmentation
bash freeze.sh
```

# References

Paper: 

1. [Chen, Liang-Chieh, et al. "Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs." arXiv preprint arXiv:1606.00915 (2016).](https://arxiv.org/pdf/1606.00915.pdf)

2. [Zhao, Hengshuang, et al. "Pyramid scene parsing network." IEEE Conf. on Computer Vision and Pattern Recognition (CVPR). 2017.](https://arxiv.org/pdf/1612.01105.pdf)

# Borrowed Code

1. `model.py/network.py` are borrowed from [DrSleep's implementation](https://github.com/DrSleep/tensorflow-deeplab-resnet). The layout does not seem ideal to me and I may re-implement them later on, but for now, I will just stick with it.

2. Pre-trained weight can be referred from [Indoor-segmentation](https://github.com/hellochick/Indoor-segmentation)

3. PSP Network code borrowed but refined from [PSP-tensorflow](https://github.com/hellochick/PSPNet-tensorflow)

# Docker

```
make build run
```

# API

```
URL: http://0.0.0.0:8080/segmentation

HEADERS: {'Content-Type': application/json}

BODY: {url: ''}

```

# Future Work

- [ ] Freeze model as a google protobuf file

- [X] Wrap up this with flask as a Restful API

- [ ] Wrap up this with Docker as a micro-server