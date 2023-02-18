import torch
from torch import nn


class SpatialCaffeLikeCrop(nn.Module):
    def __init__(self, offset: tuple[int, int]) -> None:
        """
        This is just a try to mimic spatial cropping in Caffe:
        1) https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/voc-fcn32s/train.prototxt#L509
        2) https://groups.google.com/g/caffe-users/c/YSRYy7Nd9J8
        """
        super().__init__()
        self.offset = offset

    def forward(self, to_crop: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        ref_h, ref_w = reference.shape[2:]
        height_slice = slice(self.offset[0], self.offset[0] + ref_h)
        width_slice = slice(self.offset[1], self.offset[1] + ref_w)
        return to_crop[:, :, height_slice, width_slice]
