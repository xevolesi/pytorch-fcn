import os
import os.path as osp
from functools import partial

import addict
import albumentations as album
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset, default_collate

from source.utils.general import get_object_from_dict, reseed

from .utils import parallel_image_reader, read_image, read_mask_sbdd, read_mask_voc

DataPoint = tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]

VOC_CLASSES: list[str] = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]
VOC_COLORMAP: list[list[int]] = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


def _fix_worker_seeds(worker_id: int) -> None:
    """Fix seeds inside single worker."""
    seed = np.random.get_state()[1][0] + worker_id
    reseed(seed)


def _collate_fn(
    data: list[tuple[np.ndarray, np.ndarray]], to_channels_last: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = default_collate(data)
    if to_channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    return images, labels


def create_torch_dataloaders(
    config: addict.Dict, transforms: dict[str, album.Compose | None]
) -> dict[str, DataLoader]:
    loaders = {}
    for subset in transforms:
        dataset = get_object_from_dict(getattr(config.dataset, subset))
        dataset.set_transforms(transforms[subset])
        if config.training.overfit_single_batch:
            config.training.batch_size = 1
        to_shuffle = (subset == "train") and not config.training.overfit_single_batch
        pin_memory = "cuda" in config.training.device
        dataloader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            num_workers=config.training.dataloader_num_workers,
            shuffle=to_shuffle,
            pin_memory=pin_memory,
            worker_init_fn=_fix_worker_seeds,
            collate_fn=partial(
                _collate_fn, to_channels_last=config.training.channels_last
            ),
        )
        loaders.update({subset: dataloader})
    if config.training.overfit_single_batch:
        train_loader = [next(iter(loaders["train"]))]
        loaders = {subset: train_loader for subset in loaders}
    return loaders


class VOCSegmentationDataset(Dataset):
    _ALLOWED_SPLITS: set[str] = {"train", "val", "trainval", "seg11valid"}

    def __init__(self, root: str, split: str, cache_images: bool = False) -> None:
        if split not in self._ALLOWED_SPLITS:
            raise ValueError(
                f"Expect `split` to be one of {self._ALLOWED_SPLITS}, but got {split}"
            )
        self.cache_images = cache_images
        self.transforms = None
        data_path = osp.join(root, "VOCdevkit", "VOC2012")
        split_path = osp.join(data_path, "ImageSets", "Segmentation", f"{split}.txt")
        with open(split_path, "r") as tfs:
            names = [img.replace("\n", "") for img in tfs.readlines() if img != "\n"]

        # It's better to use NumPy arrays or PyArrow arrays to store
        # items in PyTorch dataset fields due to some memory leaks
        # causing by Python's lists or Pandas DataFrames.
        self.images = []
        self.labels = []
        for name in names:
            self.images.append(osp.join(data_path, "JPEGImages", name + ".jpg"))
            self.labels.append(osp.join(data_path, "SegmentationClass", name + ".png"))
        if self.cache_images:
            self.images = parallel_image_reader(self.images, os.cpu_count(), read_image)
            self.labels = parallel_image_reader(
                self.labels, os.cpu_count(), read_mask_voc
            )
        else:
            self.images = np.array(self.images)
            self.labels = np.array(self.labels)

    def set_transforms(self, transforms: album.Compose | None = None) -> None:
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> DataPoint:
        if self.cache_images:
            image = self.images[index]
            label = self.labels[index]
        else:
            image = np.array(Image.open(self.images[index]).convert("RGB"))

            # It's important to read images with PIL and convert it to NumPy
            # array as follows. More about it:
            # https://stackoverflow.com/questions/49629933/ground-truth-pixel-labels-in-pascal-voc-for-semantic-segmentation
            label = np.array(Image.open(self.labels[index])).astype(np.int32)
        if self.transforms is not None:
            T = self.transforms(image=image, mask=label)
            image = T["image"]
            label = T["mask"]
        return image, label


class SBDDSegmentationDataset(Dataset):
    _ALLOWED_SPLITS: set[str] = {"train", "val"}

    def __init__(self, root: str, split: str, cache_images: bool = False) -> None:
        if split not in self._ALLOWED_SPLITS:
            raise ValueError(
                f"Expect `split` to be one of {self._ALLOWED_SPLITS}, but got {split}"
            )
        self.cache_images = cache_images
        self.transforms = None
        data_path = osp.join(root, "benchmark_RELEASE", "dataset")
        split_path = osp.join(data_path, f"{split}.txt")

        with open(split_path, "r") as tfs:
            names = [img.replace("\n", "") for img in tfs.readlines() if img != "\n"]

        self.images = []
        self.labels = []
        for name in names:
            self.images.append(osp.join(data_path, "img", name + ".jpg"))
            self.labels.append(osp.join(data_path, "cls", name + ".mat"))
        if self.cache_images:
            self.images = parallel_image_reader(self.images, os.cpu_count(), read_image)
            self.labels = parallel_image_reader(
                self.labels, os.cpu_count(), read_mask_sbdd
            )
        else:
            self.images = np.array(self.images)
            self.labels = np.array(self.labels)

    def set_transforms(self, transforms: album.Compose | None = None) -> None:
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> DataPoint:
        if self.cache_images:
            image = self.images[index]
            label = self.labels[index]
        else:
            image = np.array(Image.open(self.images[index]).convert("RGB"))
            label = loadmat(self.labels[index])
            label = label["GTcls"][0]["Segmentation"][0].astype(np.int32)
        if self.transforms is not None:
            T = self.transforms(image=image, mask=label)
            image = T["image"]
            label = T["mask"]
        return image, label
