import torch
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

from source.metrics import compute_paper_metrics, torch_hist


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    autocast_ctx: autocast,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model.eval()
    running_loss = torch.as_tensor(0.0, device=device)
    hist = torch.zeros((model.n_classes, model.n_classes), device=device)
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        with autocast_ctx:
            outputs = model(images)
            loss = criterion(outputs, masks)
        running_loss += loss
        hist += torch_hist(masks, outputs)
    metrics = compute_paper_metrics(hist)
    metrics.update({"val_loss": running_loss / len(loader)})
    return metrics
