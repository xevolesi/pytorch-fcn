# pytorch-fcn

Implementation of [`Fully Convolutional Networks for Semantic Segmentation`](https://arxiv.org/abs/1411.4038) by Jonathan Long, Evan Shelhamer, Trevor Darrell, UC Berkeley

# Dataset
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

# How to develop
1. Clone this repo;
2. Install development dependencies via `pip install -r requirements.dev.txt`.
3. Use `make format` command to apply `black` and `isort`;
4. Use `make run_tests` command to run tests.