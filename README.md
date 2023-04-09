# pytorch-fcn
Implementation of [`Fully Convolutional Networks for Semantic Segmentation`](https://arxiv.org/abs/1411.4038) by Jonathan Long, Evan Shelhamer, Trevor Darrell, UC Berkeley

# Results

I tried to reproduce the results from the journal version of paper, but I did not succeed completely. I think this is because I used vgg16 from torchvision as a backbone and trained
with a mini-batches without gradient accumulation (although, in theory, this should be 
almost the same). Also, because of i trained with mini-batches, the learning rate is 
slightly increased relative to the one from paper. Here are the results:

| Architecture | mIOU     |
|--------------|----------|
| FCN32s       | 0.62341  |
| FCN16s       | 0.6419   |
| FCN8s        | 0.6442   |

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
I used the same dataset as suggested here: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc_layers.py

## Custom dataset
I suggest to translate your dataset to PASCAL VOC format and then just use this repo as is
to train on your own dataset, but of course you can write you own dataset implementation
and use it.

# Installation
You need `Python==3.10` and `PyTorch==1.13.1` with `CUDA==11.6`, but i think it's easy
to run with other versions of PyTorch. Note that i was training the models with `NVidia Tesla V100 16GB` and `110 GB RAM`.
I suggest to use conda for environment managing. To setup repo for your own
experiments please run the following commands:
```
conda create -n fcn python=3.10 -y
conda activate fcn
pip install -r requirements.txt # or you can install from requirements.dev.txt
```

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
