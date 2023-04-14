# pytorch-fcn
Implementation of [`Fully Convolutional Networks for Semantic Segmentation`](https://arxiv.org/abs/1411.4038) by Jonathan Long, Evan Shelhamer, Trevor Darrell, UC Berkeley

# Results

## Paper version
I tried to reproduce the results from the journal version of paper, but I did not succeed completely. I think this is because I used vgg16 from torchvision as a backbone and trained
with a mini-batches without gradient accumulation (although, in theory, this should be 
almost the same). Also, because of i trained with mini-batches, the learning rate is 
slightly increased relative to the one from paper. Here are the results:

| Architecture | mIOU     | 			PyTorch checkpoints			      |
|--------------|----------|-----------------------------------------------|
| FCN32s       | 0.62457  |[link](https://disk.yandex.ru/d/Rz5hLoJTw9H0ag)|
| FCN16s       | 0.64217  |[link](https://disk.yandex.ru/d/4o5p-XA3XoMdAQ)|
| FCN8s        | 0.64389  |[link](https://disk.yandex.ru/d/13dK-dOx6WllNw)|

Actually i suppose that you can easily beat my results just by adding more augmentations
or EMA model, so do it if you want :) In `predefined_configs` you will find configs for
FCN32-FCN8 to reproduce my results.

## Integration with timm package
I've added integration with awesome [`timm repo`](https://github.com/huggingface/pytorch-image-models) so now you can use backbones
from it. Please see `predefined_configs/config_fcn32_densenet121.yml` for example of usage `densenet121` backbone from `timm`. Some results:

| Architecture |     Backbone    |Additional FCN head|mIOU     | 			PyTorch checkpoints			       |
|--------------|-----------------|-------------------|---------|-----------------------------------------------|
| FCN32s       |   densenet121   |       False       |0.65869  |[link](https://disk.yandex.ru/d/zYuLFl6W5n1Miw)|
| FCN32s       |   densenet121   |		 True        |0.64217  |[link](https://disk.yandex.ru/d/_bGvmVgpYVdWRg)|

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
I've also provided optional requirements listed in `requirements.optional.txt`. This file contains:
1. [`Weights & biases`](https://wandb.ai/site) for logging training process. You don't need them for core functionality,
but i strongly recommend you to use this awesome tool. In order to use `wandb` you need to follow [`quickstart`](https://docs.wandb.ai/quickstart) guide. `wandb` will need your `WANDB_API_KEY` environment variable and you can set by creating `.env` file in the root of this repo (see `.env-example`);
2. Packages for exporting to `ONNX` format.

## Development requirements.
These are for development pupropse. They consist of `flake8`, `pytest`, and so on. You can read `requirements.dev.txt` file to get more information about development requirements.

# How to train
It's quite simple:
1. Modify `config.yml` file according to your desires;
2. Run `python train.py`.

It shoud works okay.

# Predict on image
To get predictions on single image please use `predict.py` script. To run this script just do:
```
python predict.py \
		--image %path to image% \
		--config %path to config% \
		--weights %path to weights% \
		--image_size %Image height and image width separated by single comma%
```

Example:
```
python predict.py \
		--image data/VOCdevkit/VOC2012/JPEGImages/2007_000363.jpg \
		--config config.yml \
		--weights weights/fcn_sim..onnx \
		--image_size 500,500
```

# Export to ONNX
To export the model just do:
```
python export_to_onnx.py \
		--config %path to config.yml% \
		--torch_weights %path to PyTorch weights% \
		--onnx_path %path tot ONNX file% \
		--image_size %Image height and image width separated by single comma% \
		--do_check_on_validation_set
```
Example of command to export `FCN8`:
```
python export_to_onnx.py \
		--config config.yml \
		--torch_weights logs/2023-04-12\ 08\:04\:42.694413/weights/fcn_8_iou_0.6438923478126526.pt \
		--onnx_path ./fcn.onnx \
		--image_size 500,500 \
		--do_check_on_validation_set
```

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
