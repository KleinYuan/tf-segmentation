# Introduction

Real time segmentation inference production ready code based on DeepLab.

# Dependencies

- [X] Python 2.X

- [X] Tensorflow > 1.0.0

- [X] OpenCV

* No GPU required

# Freeze Model [Optional]

```
# Navigate to tf-segmentation
bash freeze.sh
```

# Run Demo

1. Download pre-trained model first (e.g: [Indoor-segmentation](https://github.com/hellochick/Indoor-segmentation))

```
1 (wall)      <- 9(window), 15(door), 33(fence), 43(pillar), 44(sign board), 145(bullertin board)
4 (floor)     <- 7(road), 14(ground, 30(field), 53(path), 55(runway)
5 (tree)      <- 18(plant)
8 (furniture) <- 8(bed), 11(cabinet), 14(sofa), 16(table), 19(curtain), 20(chair), 25(shelf), 34(desk) 
7 (stairs)    <- 54(stairs)
26(others)    <- class number larger than 26
```

2. Put model.* files under `/model` folder
3. run below:

```
python demo.py
```

# References

Paper: [Chen, Liang-Chieh, et al. "Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs." arXiv preprint arXiv:1606.00915 (2016).](https://arxiv.org/pdf/1606.00915.pdf)

# Borrowed Code

1. `model.py/network.py` are borrowed from [DrSleep's implementation](https://github.com/DrSleep/tensorflow-deeplab-resnet). The layout does not seem ideal to me and I may re-implement them later on, but for now, I will just stick with it.
2. Pre-trained weight can be referred from [Indoor-segmentation](https://github.com/hellochick/Indoor-segmentation)

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