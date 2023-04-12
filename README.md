# pytorch-fcn
Implementation of [`Fully Convolutional Networks for Semantic Segmentation`](https://arxiv.org/abs/1411.4038) by Jonathan Long, Evan Shelhamer, Trevor Darrell, UC Berkeley

# Results

I tried to reproduce the results from the journal version of paper, but I did not succeed completely. I think this is because I used vgg16 from torchvision as a backbone and trained
with a mini-batches without gradient accumulation (although, in theory, this should be 
almost the same). Also, because of i trained with mini-batches, the learning rate is 
slightly increased relative to the one from paper. Here are the results:

| Architecture | mIOU     |
|--------------|----------|
| FCN32s       | 0.62457  |
| FCN16s       | 0.64217  |
| FCN8s        | 0.64389  |

Actually i suppose that you can easily beat my results just by adding more augmentations
or EMA model, so do it if you want :) In `predefined_configs` you will find configs for
FCN32-FCN8 to reproduce my results.
In order to get better result you can try the following:
1. Add more standard data augmentation;
2. Add advanced data augmentation techniques like CutMix, MixUp, Mosaic (YOLO variant) and so on;
3. Replace transpose convolution with `torch.nn.Upsample(mode='nearest')` followed by 1x1 `Conv2d`;
4. Do full mixing from different upsampling paths of FCN8;
5. Use backbones from the best [`timm repo`](https://github.com/huggingface/pytorch-image-models) or at least torchvision vgg16_bn;
6. ...

## Fixed batch examples

### FCN32s
![FCN32s predictions for fixed batch](./assets/fcn32_fixed_batch.png)

### FCN16s
![FCN32s predictions for fixed batch](./assets/fcn16_fixed_batch.png)

### FCN8s
![FCN32s predictions for fixed batch](./assets/fcn8_fixed_batch.png)

# Dataset

## PASCAL VOC
Validation part is [this](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) with [this](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/data/pascal/seg11valid.txt) image set. For training [`SBD`](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz) was used. As i understood this is exact setup as in paper.

## Custom dataset
I suggest to translate your dataset to PASCAL VOC format and then just use this repo as is
to train on your own dataset, but of course you can write you own dataset implementation
and use it.

# Installation
You need `Python==3.10` and `PyTorch==1.13.1` with `CUDA==11.6`, but i think it's easy
to run with other versions of PyTorch. Note that i was training the models with `NVidia RTX 4090 24GB` and `128 GB RAM`.

## Core requirements
Core requirements are listed in `requirements.txt` file. You need them to be able to run training pipeline.
The simplest way to install them is to use conda:
```
conda create -n fcn python=3.10 -y
conda activate fcn
pip install -r requirements.txt
```
But ofcourse you can use old but gold `venv` by running the following commands:
```
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Optional requirements
I've also provided optional requirements listed in `requirements.optional.txt`. This file contains [`Weights & biases`](https://wandb.ai/site)
for logging training process. You don't need them for core functionality but i strongly recommend you to use this awesome tool. In order to use
`wandb` you need to follow [`quickstart`](https://docs.wandb.ai/quickstart) guide. `wandb` will need your `WANDB_API_KEY` environment variable and you can set by creating `.env` file in the root of this repo (see `.env-example`).

## Development requirements.
These are for development pupropse. They consist of `flake8`, `pytest`, and so on. You can read `requirements.dev.txt` file to get more information about development requirements.

# How to train
It's quite simple:
1. Modify `config.yml` file according to your desires;
2. Run `python train.py`.

It shoud works okay.

# How to develop
Clone this repo and install development dependencies via `pip install -r requirements.dev.txt`. `Makefile` consists of the following recipies:
1. `lint` - run linter checking;
2. `format` - run formatters;
3. `verify_format` - run formatters is checking mode;
4. `run_tests` - run all tests;
5. `pre_push_test` - run formatters and tests.
Use `make lint` to run linter checking and `make format` to apply formatters.

This repo contains some tests just to be sure that some tricks works correctly, 
but unfortunatelly it doesn't contain any integration tests (test for the whole 
modelling pipeline) just because it's really annoying to adapt them to free github runners.
Note that running tests takes quite a lot of time.
