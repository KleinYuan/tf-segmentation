# Introduction

Real time segmentation inference production ready code based on DeepLab.

# Dependencies

- [X] Python 2.X

- [X] Tensorflow > 1.0.0

- [X] OpenCV

* No GPU required

# Run Demo

```
python inference.py
```

# References

Paper: [Chen, Liang-Chieh, et al. "Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs." arXiv preprint arXiv:1606.00915 (2016).](https://arxiv.org/pdf/1606.00915.pdf)

# Borrowed Code

`model.py/network.py` are borrowed from [DrSleep's implementation](https://github.com/DrSleep/tensorflow-deeplab-resnet). The layout does not seem ideal to me and I may re-implement them later on, but for now, I will just stick with it.