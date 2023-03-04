# pytorch-fcn

Implementation of [`Fully Convolutional Networks for Semantic Segmentation`](https://arxiv.org/abs/1411.4038) by Jonathan Long, Evan Shelhamer, Trevor Darrell, UC Berkeley

# Dataset
## PASCAL VOC 2011
I used [`PASCAL VOC 2011`](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html) as described in paper.
In order to use it without any changes just create `data` directory in the root of this repo and put the data into it. So you need to have the following directory structure:
```
| ...
| - data
|     | - TrainVal
|     |       |
|     |       | - VOCdevkit
|     |       |        |
|     |       |        | - VOC2011
|     |       |        |      |
|     |       |        |      | - Annotations
|     |       |        |      | - ImageSets
|     |       |        |      | - JPEGImages
|     |       |        |      | - SegmentationClass
|     |       |        |      | - SegmentationObject
| ...
```
If your dataset lies outside the repo directory than just modify `config.yml` file. Set
`config.dataset.path` to the path of base directory with your PASCAL VOC 2011 dataset.
For example you have your dataset inside `my_data` folder with the following structure:
```
| ...
| - my_data
|     | - TrainVal
|     |       |
|     |       | - VOCdevkit
|     |       |        |
|     |       |        | - VOC2011
|     |       |        |      |
|     |       |        |      | - Annotations
|     |       |        |      | - ImageSets
|     |       |        |      | - JPEGImages
|     |       |        |      | - SegmentationClass
|     |       |        |      | - SegmentationObject
| ...
```
than just modify `config.yml` file in the following way:
```
dataset:
    path: /path/to/my_data
```

## Custom dataset
If you want to use custom dataset than you have the following options:
1. Translate your dataset to the PASCAL VOC 2011 format;
2. Add your custom dataset class to the `source/datasets` and modify the rest of the code.

Choose the easiest option for you.

# Installation
Dev-env is the following:
1. `Python==3.10`;
2. `PyTorch==1.13.1` with `CUDA==11.6`;
3. `Ubuntu: release 20.04, codename focal`;
4. `NVidia Tesla V100 16GB`;
5. `110 GB RAM`.

I suggest to use conda for environment managing. To setup repo for your own
experiments please run the following commands:
```
conda create -n fcn python=3.10 -y
conda activate fcn
pip install -r requirements.txt # or you can install from requirements.dev.txt
```

# How to develop
Clone this repo and install development dependencies via `pip install -r requirements.dev.txt`. `Makefile` consists of the following recipies:
1. `lint` - run linter checking;
2. `format` - run formatters;
3. `verify_format` - run formatters is checking mode;
4. `run_tests` - run all tests;
5. `pre_push_test` - run formatters and tests.
Use `make lint` to run linter checking and `make format` to apply formatters.

This repo doesn't contain any integration tests (test for the whole modelling pipeline)
just because it's really annoying to adapt them to free github runners.
