import os
import typing as ty

import addict
import albumentations as album
import cv2
import jpeg4py as jpeg
import numpy as np
from torch.utils.data import Dataset

from .utils import create_voc_paths, voc_mask2segmentation_mask


class VOCSegmentation(Dataset):
    _PREFIX_PATH: str = os.path.join("TrainVal", "VOCdevkit", "VOC2011")

    def __init__(
        self,
        config: addict.Dict,
        subset: ty.Literal["train", "val"],
        transforms: album.Compose | None = None,
    ) -> None:
        if subset not in ("train", "val"):
            raise ValueError(
                f"Expect `subset` to be one of ('train', 'val'), got {subset}"
            )
        base_path = os.path.join(config.dataset.path, self._PREFIX_PATH)
        paths = create_voc_paths(base_path, subset)
        with open(paths["image_set"], "r") as tfs:
            image_names = tfs.readlines()
        self.image_container = [
            os.path.join(paths["images"], image_name.replace("\n", "") + ".jpg")
            for image_name in image_names
        ]
        self.mask_container = [
            os.path.join(paths["masks"], image_name.replace("\n", "") + ".png")
            for image_name in image_names
        ]
        self.image_size = (config.training.image_size, config.training.image_size)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_container)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image = self._read_image(index)
        mask = self._read_mask(index)
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask

    def _read_image(self, index: int) -> np.ndarray:
        image = jpeg.JPEG(self.image_container[index]).decode()
        return cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)

    def _read_mask(self, index: int) -> np.ndarray:
        mask = cv2.imread(self.mask_container[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = voc_mask2segmentation_mask(mask)
        return cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
