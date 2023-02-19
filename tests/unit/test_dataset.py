import os
import random

import cv2
import numpy as np
import pytest

from source.datasets import VOCSegmentation
from source.datasets.utils import (
    VOC_COLORMAP,
    create_voc_paths,
    voc_mask2segmentation_mask,
)
from source.utils.general import read_config

_CFG_PATH = "config.yml"

cfg = read_config(_CFG_PATH)
if not os.path.exists(cfg.dataset.path):
    pytest.skip(
        f"Unable to find `{cfg.dataset.path}`. Skipping ...", allow_module_level=True
    )

sys_random = random.SystemRandom()


@pytest.mark.parametrize("subset", ["train", "val"])
def test_create_voc_paths(subset, get_test_config):
    base_path = os.path.join(get_test_config.dataset.path, VOCSegmentation._PREFIX_PATH)
    paths = create_voc_paths(base_path, subset)
    assert os.path.exists(paths["image_set"])
    assert os.path.exists(paths["images"])
    assert os.path.exists(paths["masks"])


@pytest.mark.parametrize("subset", ["train", "val"])
def test_voc_mask2segmentation_mask(subset, get_test_config):
    base_path = os.path.join(get_test_config.dataset.path, VOCSegmentation._PREFIX_PATH)
    paths = create_voc_paths(base_path, subset)
    with open(paths["image_set"], "r") as tfs:
        sample_list = [sample.replace("\n", "") for sample in tfs.readlines()]
        random_samples = [sys_random.choice(sample_list) for _ in range(5)]
    masks_path = [
        os.path.join(paths["masks"], img_name + ".png") for img_name in random_samples
    ]
    for mask_path in masks_path:
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        segm_mask = voc_mask2segmentation_mask(mask)
        assert mask.shape[:2] == segm_mask.shape[:2]
        assert segm_mask.shape[2] == len(VOC_COLORMAP)
        assert set(np.unique(segm_mask)) == {0.0, 1.0}
        class_indices = np.nonzero(np.any((segm_mask == 1.0), axis=(0, 1)))[0]
        for class_index in class_indices:
            assert np.any(mask == VOC_COLORMAP[class_index])


@pytest.mark.parametrize("subset", ["train", "val", "something_else"])
def test_dataset(subset, get_test_config):
    try:
        dataset = VOCSegmentation(get_test_config, subset, transforms=None)
    except ValueError:
        assert subset == "something_else"
        return
    assert len(dataset.image_container) == len(dataset.mask_container)
    mask_names = [os.path.split(mask_path)[-1] for mask_path in dataset.mask_container]
    image_names = [os.path.split(img_path)[-1] for img_path in dataset.mask_container]
    assert all(
        mask_name == image_name for mask_name, image_name in zip(mask_names, image_names)
    )
    assert all(os.path.exists(path) for path in dataset.image_container)
    assert all(os.path.exists(path) for path in dataset.mask_container)
    random_indices = [sys_random.choice(range(len(dataset))) for _ in range(5)]
    for index in random_indices:
        image, mask = dataset[index]
        assert (
            image.shape[:2]
            == mask.shape[:2]
            == (get_test_config.training.image_size, get_test_config.training.image_size)
        )
        assert mask.shape[2] == len(VOC_COLORMAP)
        class_indices = np.nonzero(np.any((mask == 1.0), axis=(0, 1)))[0]
        assert set(class_indices).intersection(set(range(len(VOC_COLORMAP)))) == set(
            class_indices
        )
