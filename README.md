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
# How to develop
1. Clone this repo;
2. Install development dependencies via `pip install -r requirements.dev.txt`.
3. Use `make format` command to apply `black` and `isort`;
4. Use `make run_tests` command to run tests.