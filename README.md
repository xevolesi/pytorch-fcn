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

To setup repo for your own experiments please run the following commands:
```
conda create -n fcn python=3.10 -y
conda activate fcn
pip install -r requirements.txt
```

# Results
| Architecture | Crop        | Backbone          | Trainable upsamplings | #epochs | Running time | mIOU     | ClearML URL                                                                                                                  |
|--------------|-------------|-------------------|-----------------------|---------|--------------|----------|------------------------------------------------------------------------------------------------------------------------------|
| FCN32        | center crop | torchvision VGG16 | false                 | 13      | 42.51 m      | 0.60502  | https://app.clear.ml/projects/3380a0f929714fb0a00b88fe46d44356/experiments/40ea261473874c659280e6c3de951af1/output/execution |
| FCN32        | caffe crop  | torchvision VGG16 | false                 | 13      | 42.45 m      | 0.605138 | https://app.clear.ml/projects/3380a0f929714fb0a00b88fe46d44356/experiments/54caf397c7aa4566a551f4436c7915f6/output/execution |
| FCN32        | caffe crop  | torchvision VGG16 | true                  | 13      | 43.14 m      | 0.604591 | https://app.clear.ml/projects/3380a0f929714fb0a00b88fe46d44356/experiments/1504db451e0241c083f7a6c1ddb6e3ae/output/execution |

# How to develop
Clone this repo and install development dependencies via `pip install -r requirements.dev.txt`. `Makefile` consists of the following recipies:
1. `lint` - run linter checking;
2. `format` - run formatters;
3. `verify_format` - run formatters is checking mode;
4. `run_tests` - run all tests;
5. `pre_push_test` - run formatters and tests.
Use `make lint` to run linter checking and `make format` to apply formatters.
