import multiprocessing as mp
import typing as ty
from collections.abc import Sequence

import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat


def pad_to_square(
    image: np.ndarray,
    border_type: int = cv2.BORDER_CONSTANT,
    pad_values: tuple[int, ...] = (0, 0, 0),
) -> np.ndarray:
    """
    Pad image along shortest size to square size.

    Parameters:
        image: Source image;
        border_type: Integer number of OpenCV border types;
        pad_values: Per-channel values for padding.

    Returns:
        Padded image.
    """
    image_height, image_width = image.shape[:2]
    square_size = max(image_height, image_width)
    height_offset = (square_size - image_height) // 2
    width_offset = (square_size - image_width) // 2
    return cv2.copyMakeBorder(
        image,
        height_offset,
        height_offset,
        width_offset,
        width_offset,
        borderType=border_type,
        value=pad_values,
    )


def read_image(image_path: str, batched: bool = False) -> np.ndarray:
    image = np.array(Image.open(image_path).convert("RGB"))
    if not batched:
        return image
    return pad_to_square(image)


def read_mask_voc(mask_path: str, batched: bool = False) -> np.ndarray:
    # It's important to read images with PIL and convert it to NumPy
    # array as follows. More about it:
    # https://stackoverflow.com/questions/49629933/ground-truth-pixel-labels-in-pascal-voc-for-semantic-segmentation
    mask = np.array(Image.open(mask_path)).astype(np.int32)
    if not batched:
        return mask
    return pad_to_square(mask, pad_values=255)


def read_mask_sbdd(mask_path: str, batched: bool = False) -> np.ndarray:
    mask = loadmat(mask_path)["GTcls"][0]["Segmentation"][0].astype(np.int32)
    if not batched:
        return mask
    return pad_to_square(mask, pad_values=255)


def parallel_image_reader(
    image_path_seq: Sequence[str],
    n_processes: int,
    img_reader_fn: ty.Callable[[str], np.ndarray],
) -> list[np.ndarray]:
    with mp.Pool(n_processes) as pool:
        container = pool.map(img_reader_fn, image_path_seq)
    return container
