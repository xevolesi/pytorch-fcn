import os.path as osp
import random
from copy import deepcopy
from itertools import product

import numpy as np
import pytest
import torch

from source.datasets import SBDDSegmentationDataset, VOCSegmentationDataset
from source.utils.augmentations import get_albumentation_augs
from source.utils.general import get_object_from_dict

_ALLOWED_SPLITS = {
    "SBDDSegmentationDataset": SBDDSegmentationDataset._ALLOWED_SPLITS,
    "VOCSegmentationDataset": VOCSegmentationDataset._ALLOWED_SPLITS,
}
_VOC_DATASETS_IMPLS = (
    "source.datasets.voc.SBDDSegmentationDataset",
    "source.datasets.voc.VOCSegmentationDataset",
)
_ALLOWED_VOC_CLASS_INDICES = set(range(21)).union({-1})
sys_random = random.SystemRandom()


@pytest.mark.parametrize(
    "dataset_impl, split, use_augs",
    list(
        product(
            _VOC_DATASETS_IMPLS,
            ["train", "val", "trainval", "seg11valid"],
            [True, False],
        )
    ),
)
def test_voc_datasets(dataset_impl, split, use_augs, get_test_config):
    """
    Patch dataset.train attr in config with different values to do some
    checks.
    """

    # Set up dataset attributes.
    config = deepcopy(get_test_config)
    data_root = config.dataset.train.root
    impl = {"__class_fullname__": dataset_impl, "split": split, "root": data_root}
    config.dataset.train = impl

    # Check that error was rised correctly.
    try:
        dataset = get_object_from_dict(config.dataset.train)
    except ValueError:
        allowed_splits = _ALLOWED_SPLITS.get(dataset_impl.split(".")[-1], None)
        if allowed_splits is not None:
            assert split not in allowed_splits
        return

    if use_augs:
        augs = get_albumentation_augs(config)
        dataset.set_transforms(augs.get(split))

    # Check that we don't have empty paths.
    assert all(osp.exists(image) for image in dataset.images)
    assert all(osp.exists(label) for label in dataset.labels)
    assert len(dataset.images) == len(dataset.labels) == len(dataset)

    # Check that each image has corresponding mask.
    for image_path, label_path in zip(dataset.images, dataset.labels):
        image_name = osp.splitext(osp.split(image_path)[-1])[0]
        label_name = osp.splitext(osp.split(label_path)[-1])[0]
        assert image_name == label_name

    random_indices = [sys_random.choice(range(len(dataset))) for _ in range(5)]
    for index in random_indices:
        image, mask = dataset[index]

        assert mask.ndim == 2
        assert image.ndim == 3

        if use_augs and dataset.transforms is not None:
            assert isinstance(image, torch.Tensor)
            assert isinstance(mask, torch.Tensor)
            assert mask.dtype == torch.int32
            assert image.dtype == torch.float
            assert image.shape[1:] == mask.shape
            assert set(torch.unique(mask).numpy()).intersection(
                _ALLOWED_VOC_CLASS_INDICES
            ) == set(torch.unique(mask).numpy())
        else:
            assert isinstance(image, np.ndarray)
            assert isinstance(mask, np.ndarray)
            assert image.shape[:2] == mask.shape[:2]
            assert set(np.unique(mask)).intersection(_ALLOWED_VOC_CLASS_INDICES) == set(
                np.unique(mask)
            )
