import argparse as ap
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch
from torch import nn
from tqdm import tqdm

from source.metrics import compute_paper_metrics, torch_hist
from source.models import FCN
from source.utils.augmentations import get_albumentation_augs
from source.utils.general import get_object_from_dict, read_config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ATOL = 1e-4


class SegmentationWrapper(nn.Module):
    """Wrapper for FCN model."""

    def __init__(self, fcn: FCN) -> None:
        super().__init__()
        self.fcn = fcn

    def forward(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.fcn(tensor)
        return {"logits": logits, "argmax_mask": logits.argmax(dim=1)}


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to .YAML configuration file",
    )
    parser.add_argument(
        "--torch_weights",
        required=True,
        type=str,
        help="Path to PyTorch .pth or .pt file",
    )
    parser.add_argument(
        "--onnx_path",
        required=True,
        type=str,
        help="Path for exported onnx file",
    )
    parser.add_argument(
        "--image_size",
        required=True,
        type=str,
        help="Image size to work with",
    )
    parser.add_argument(
        "--do_check_on_validation_set",
        required=False,
        default=False,
        type=bool,
        action=ap.BooleanOptionalAction,
    )
    args = parser.parse_args()

    config = read_config(args.config)
    fcn = FCN(config)
    fcn.load_state_dict(torch.load(args.torch_weights))
    wrapper = SegmentationWrapper(fcn)
    wrapper.eval()

    image_size = tuple(map(int, args.image_size.split(",")))
    dummy_input = torch.zeros(1, 3, *image_size, dtype=torch.float)
    torch.onnx.export(
        wrapper,
        dummy_input,
        args.onnx_path,
        verbose=True,
        input_names=["input_image"],
        output_names=["output_dict"],
        do_constant_folding=True,
    )

    # Simplify ONNX model.
    onnx_model = onnx.load(args.onnx_path)
    onnx_model, check = onnxsim.simplify(onnx_model, check_n=10)
    save_path, save_name = os.path.split(args.onnx_path)
    save_name, ext = os.path.splitext(save_name)
    save_name = ".".join((save_name + "_sim", ext))
    save_path = os.path.join(save_path, save_name)
    onnx.save(onnx_model, save_path)

    # Remove initializers from ONNX model.
    onnx_model = onnx.load(save_path)
    inputs = onnx_model.graph.input
    name_to_input = {input.name: input for input in inputs}
    for initializer in onnx_model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
    onnx.save(onnx_model, save_path)

    # Check outputs from ONNX and PyTorch models.
    if not args.do_check_on_validation_set:
        sys.exit()

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(save_path, sess_options)

    config.dataset.val.batched = False
    dataset = get_object_from_dict(config.dataset.val)
    augs = get_albumentation_augs(config)["val"]
    dataset.set_transforms(augs)

    hist_torch = torch.zeros((wrapper.fcn.n_classes, wrapper.fcn.n_classes))
    hist_onnx = torch.zeros((wrapper.fcn.n_classes, wrapper.fcn.n_classes))
    pbar = tqdm(dataset, total=len(dataset), desc="Evaluation on validation set")
    for item in pbar:
        image, label = item
        image = image.unsqueeze(dim=0)
        label = label.unsqueeze(dim=0)
        with torch.no_grad():
            torch_output = wrapper(image)
            hist_torch += torch_hist(label, torch_output["logits"])
        onnx_logits, onnx_masks = ort_session.run(None, {"input_image": image.numpy()})
        hist_onnx += torch_hist(label, torch.from_numpy(onnx_logits))
        assert np.allclose(torch_output["logits"].numpy(), onnx_logits, atol=ATOL)
    metrics_onnx = compute_paper_metrics(hist_onnx)
    metrics_torch = compute_paper_metrics(hist_torch)
    for metric in metrics_onnx:
        assert torch.allclose(metrics_onnx[metric], metrics_torch[metric])
