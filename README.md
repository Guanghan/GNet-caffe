# Modified Caffe

This Caffe branch is a modified version of [Caffe from Convolutional Pose Machine](https://github.com/shihenw/caffe/tree/d154e896b48e8fb520cb4b47af8ba10bf9403382).

It is designed for [Knowledge-Guided Deep Fractal Neural Networks for Human Pose Estimation](http://github.com/Guanghan/GNet-pose)

# What's new?

- Merged the latest caffe version with the CPM caffe, so that batch normalization layers are available.
- Modified CPM Data Layer, such that the Data layer outputs transformed image data, annotation data, as well as injected features.

# Citation

The details are published as a technical report on arXiv. If you find the code and models useful, please cite the following paper:
[TMM 2017](http://ieeexplore.ieee.org/document/8064661/).

@ARTICLE{8064661, 
author={G. Ning and Z. Zhang and Z. He}, 
journal={IEEE Transactions on Multimedia}, 
title={Knowledge-Guided Deep Fractal Neural Networks for Human Pose Estimation}, 
year={2017}, 
volume={PP}, 
number={99}, 
pages={1-1}, 
keywords={Fractal Networks;Human Pose Estimation;Knowledge-Guided Learning}, 
doi={10.1109/TMM.2017.2762010}, 
ISSN={1520-9210}, 
month={},}
