# Modified Caffe

This Caffe branch is a modified version of [Caffe from Convolutional Pose Machine](https://github.com/shihenw/caffe/tree/d154e896b48e8fb520cb4b47af8ba10bf9403382).

It is designed for [Knowledge-Guided Deep Fractal Neural Networks for Human Pose Estimation](http://github.com/Guanghan/GNet-pose)

# What's new?

- Merged the latest caffe version with the CPM caffe, so that batch normalization layers are available.
- Modified CPM Data Layer, such that the Data layer outputs transformed image data, annotation data, as well as injected features.

# Citation

The details are published as a technical report on arXiv. If you find the code and models useful, please cite the following paper:
[arXiv:1607.05781](http://arxiv.org/abs/1607.05781).

                @article{ning2017knowledge,
                  title={Knowledge-Guided Deep Fractal Neural Networks for Human Pose Estimation},
                  author={Ning, Guanghan and Zhang, Zhi and He, Zhihai},
                  journal={arXiv preprint arXiv:1607.05781},
                  year={2017}
                }
