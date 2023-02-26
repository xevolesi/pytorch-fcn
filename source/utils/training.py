import os
import sys

import addict
import torch
from clearml import Logger
from loguru import logger
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights, vgg16

from source.datasets.voc import create_torch_dataloaders
from source.models import FCN32VGG16
from source.utils.augmentations import get_albumentation_augs
from source.utils.evaluation import validate
from source.utils.general import get_cpu_state_dict, get_object_from_dict, reseed
from source.utils.logs import log_predictions

logger.remove()
logger.add(
    sys.stdout,
    format=(
        "[<green>{time: HH:mm:ss}</green> | <blue>{level}</blue> | "
        "<magenta>training.py</magenta>:<yellow>{line}</yellow>] {message}"
    ),
    level="INFO",
    colorize=True,
)


def create_param_groups(
    model: torch.nn.Module, learning_rate: float, weight_decay: float
) -> list[dict[str, torch.Tensor]]:
    """
    Create 2 param groups:
        1) Biases only;
        2) Other parameters.

    Double the learning rate for biases and set weight decay for them
    to 0. Rest parameters are not affected.
    """
    biases = []
    others = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            biases.append(param)
        else:
            others.append(param)
    return [
        {"params": biases, "weight_decay": 0.0, "lr": 2 * learning_rate},
        {"params": others, "weight_decay": weight_decay, "lr": learning_rate},
    ]


def create_model(config: addict, device: torch.device):
    model = FCN32VGG16(config)
    model.copy_params_from_vgg16(vgg16(weights=VGG16_Weights.DEFAULT))
    model = model.to(device)
    if config.training.channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def train(config: addict.Dict, run_log_path: str, cm_logger: Logger | None) -> None:
    # Ingridients.
    device = torch.device(config.training.device)
    loaders = create_torch_dataloaders(config, get_albumentation_augs(config))
    model = create_model(config, device)
    optimizer = get_object_from_dict(
        config.optimizer,
        params=create_param_groups(
            model, config.optimizer.lr, config.optimizer.weight_decay
        ),
    )
    criterion = get_object_from_dict(config.criterion)
    scaler = GradScaler(enabled=device.type != "cpu")

    if config.training.log_fixed_batch:
        fixed_batch = next(iter(loaders["train"]))

    if config.training.overfit_single_batch:
        config.training.epochs = 100
        config.training.grad_acc_iters = 1

    best_weights = None
    best_metric = float("-inf")
    for epoch in range(config.training.epochs):
        # Reseed at the beginning to be sure that augmentation will be
        # different during each epoch.
        reseed(config.training.seed + epoch)
        training_loss = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            scaler,
            device,
            config.training.grad_acc_iters,
        )
        metrics = validate(model, loaders["val"], criterion, device)
        metrics = {name: tensor.item() for name, tensor in metrics.items()}

        logger.info(
            (
                "[EPOCH {epoch}/{total_epochs}] "
                "TL={tl:.2f}, "
                "VL={vl:.2f}, "
                "PixelAcc={pacc:.5f}, "
                "PCAcc={per_class_acc:.5f}, "
                "IoU={iou:.5f}, "
                "FWAcc={freq_acc:.5f}"
            ),
            epoch=epoch + 1,
            total_epochs=config.training.epochs,
            tl=training_loss.item(),
            vl=metrics["val_loss"],
            pacc=metrics["acc"],
            per_class_acc=metrics["per_class_acc"],
            iou=metrics["iu"],
            freq_acc=metrics["freq_acc"],
        )
        if cm_logger is not None:
            cm_logger.report_scalar("Losses", "Train loss", training_loss.item(), epoch)
            for metric_name, metric_value in metrics.items():
                if metric_name == "val_loss":
                    graph = "Losses"
                else:
                    graph = "Metrics"
                cm_logger.report_scalar(graph, metric_name, metric_value, epoch)

        batch_log_path = os.path.join(
            run_log_path, config.logs.fixed_batch_predictions, f"epoch_{epoch+1}"
        )
        os.makedirs(batch_log_path, exist_ok=True)
        log_predictions(model, fixed_batch, device, batch_log_path)

        # Determine if there was any improvement.
        if metrics["iu"] > best_metric:
            best_metric = metrics["iu"]
            best_weights = get_cpu_state_dict(model)

    # Save best model weights.
    weights_path = os.path.join(
        run_log_path, config.logs.weights_folder, f"fcn_iou_{best_metric}.pt"
    )
    torch.save(best_weights, weights_path)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scaler: GradScaler,
    device: torch.device,
    grad_acc_iters: int = 1,
) -> torch.Tensor:
    model.train()
    running_loss = torch.as_tensor(0.0, device=device)

    # I don't want to use enumerate because it will
    # implicitly trigger GPU-CPU sync.
    step = 0  # noqa: SIM113
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type != "cpu"
        ):
            outputs = model(images)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        if (step + 1) % grad_acc_iters == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running_loss += loss
        step += 1
    return running_loss / len(loader)
