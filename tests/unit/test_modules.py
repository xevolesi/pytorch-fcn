import pytest
import torch
from source.modules import SpatialCaffeLikeCrop


@pytest.mark.parametrize("offset", [(19, 19), (21, 21), (13, 13)])
def test_caffe_like_crop(offset):
    cropper = SpatialCaffeLikeCrop(offset=offset)
    to_crop = torch.randn((4, 78, 128, 128))
    reference = torch.randn((4, 32, 64, 64))
    cropped_by_module = cropper(to_crop, reference)
    ref_h, ref_w = reference.shape[2:]
    cropped = to_crop[:, :, offset[0] : offset[0] + ref_h, offset[1] : offset[1] + ref_w]
    assert torch.allclose(cropped, cropped_by_module)
