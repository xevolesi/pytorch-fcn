import argparse as ap
import os

import cv2
import numpy as np
import onnxruntime as ort
import torch

from source.datasets.utils import pad_to_square
from source.datasets.voc import VOC_COLORMAP
from source.models import FCN
from source.utils.general import read_config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_imagenet(
    rgb_image: np.ndarray, add_batch_dim: bool = False
) -> np.ndarray:
    tensor = rgb_image.astype(np.float32)
    tensor /= 255
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor -= np.array(IMAGENET_MEAN)[:, None, None]
    tensor /= np.array(IMAGENET_STD)[:, None, None]
    if add_batch_dim:
        return np.expand_dims(tensor, axis=0)
    return tensor


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=False,
        default="./",
        help="Path to folder for predictions",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to .YAML configuration file",
    )
    parser.add_argument(
        "--weights",
        required=True,
        type=str,
        help="Path to model weights",
    )
    parser.add_argument(
        "--image_size",
        required=True,
        type=str,
        help="Image size to work with",
    )
    args = parser.parse_args()

    config = read_config(args.config)

    image_size = tuple(map(int, args.image_size.split(",")))
    image = pad_to_square(cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB))
    image = cv2.resize(image, image_size)
    tensor = preprocess_imagenet(image, add_batch_dim=True)

    base_path, filename = os.path.split(args.weights)
    name, ext = os.path.splitext(filename)
    if ext == ".onnx":
        model = ort.InferenceSession(args.weights)
        _, masks = model.run(None, {"input_image": tensor})
    elif ext in (".pt", ".pth"):
        model = FCN(config)
        model.load_state_dict(torch.load(args.weights))
        model.eval()
        with torch.no_grad():
            masks = model(torch.from_numpy(tensor)).argmax(dim=1).detach().cpu().numpy()

    os.makedirs(args.output_folder, exist_ok=True)
    image_base_path, image_name = os.path.split(args.image)
    name, _ = os.path.splitext(image_name)
    masks = np.squeeze(masks, axis=0)
    color_mask = np.zeros((*masks.shape, 3))
    for index in range(len(VOC_COLORMAP)):
        color_mask[masks == index] = np.array(VOC_COLORMAP[index])
    color_mask = color_mask.astype(np.uint8)
    image = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    save_path = os.path.join(args.output_folder, name + "_predictions.png")
    cv2.imwrite(save_path, image)
