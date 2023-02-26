import os

import torch
from torchvision.utils import make_grid, save_image

from source.datasets.voc import VOC_COLORMAP

IMAGENET_MEAN = torch.as_tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.as_tensor([0.229, 0.224, 0.225])


@torch.inference_mode()
def log_predictions(
    model: torch.nn.Module,
    fixed_batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    folder_path: str,
) -> None:
    model.eval()
    images, labels = fixed_batch
    batch_size, _, height, width = images.shape

    # Obtain raw masks with logits.
    outputs = model(images.to(device=device, non_blocking=True))
    outputs = outputs.argmax(dim=1).detach().cpu()

    # Transfer logits to class indices.
    out = torch.zeros((batch_size, 3, height, width))
    out_labels = torch.zeros((batch_size, 3, height, width))
    for i in range(out.shape[0]):
        for j in range(len(VOC_COLORMAP)):
            out[i, :, outputs[i] == j] = torch.as_tensor(
                VOC_COLORMAP[j], dtype=torch.float
            ).view(3, -1)
            out_labels[i, :, labels[i] == j] = torch.as_tensor(
                VOC_COLORMAP[j], dtype=torch.float
            ).view(3, -1)

    # Pack together and save image as 3-column image:
    # GT image, GT mask, predicted mask.
    images = images * IMAGENET_STD.view(1, -1, 1, 1) + IMAGENET_MEAN.view(1, -1, 1, 1)
    log_batch = torch.zeros((3 * batch_size, 3, height, width))
    for i in range(batch_size):
        log_batch[3 * i] = images[i]
        log_batch[3 * i + 1] = out_labels[i]
        log_batch[3 * i + 2] = out[i]
    grid = make_grid(log_batch, nrow=log_batch.shape[0] // 3)
    batch_path = os.path.join(folder_path, "fixed_batch.png")
    save_image(grid, batch_path)
