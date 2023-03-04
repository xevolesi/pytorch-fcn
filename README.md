# pytorch-fcn

Implementation of [`Fully Convolutional Networks for Semantic Segmentation`](https://arxiv.org/abs/1411.4038) by Jonathan Long, Evan Shelhamer, Trevor Darrell, UC Berkeley

# Results

I tried to reproduce the results from the journal version of paper, but I did not succeed completely. I think this is because I used vgg16 from torchvision as a backbone and trained
with a mini-batches without gradient accumulation (although, in theory, this should be 
almost the same). Also, because of i trained with mini-batches, the learning rate is 
slightly increased relative to the one from paper. Here are the results:

| Architecture | mIOU     |
|--------------|----------|
| FCN32s       | 0.623813 |
| FCN16s       | 0.642105 |
| FCN8s        | 0.644392 |

Actually i suppose that you can easily beat my results just by adding more augmentations
or EMA model, so do it if you want :) In `predefined_configs` you will find configs for
FCN32-FCN8 to reproduce my results.

# Dataset


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
