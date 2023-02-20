import os

import numpy as np

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


def create_voc_paths(base_path: str, subset: str) -> dict[str, str]:
    return {
        "image_set": os.path.join(
            base_path, "ImageSets", "Segmentation", subset + ".txt"
        ),
        "images": os.path.join(base_path, "JPEGImages"),
        "masks": os.path.join(base_path, "SegmentationClass"),
    }


def voc_mask2segmentation_mask(
    voc_mask: np.ndarray, exclude_background: bool = False
) -> np.ndarray:
    mask_size = voc_mask.shape[:2]
    color_match = VOC_COLORMAP
    if exclude_background:
        color_match = VOC_COLORMAP[1:]
    segmentation_mask = np.zeros((*mask_size, len(color_match)), dtype=np.float32)
    for label_index, label in enumerate(color_match):
        segmentation_mask[:, :, label_index] = np.all(voc_mask == label, axis=-1).astype(
            float
        )
    return segmentation_mask
