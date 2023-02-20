import sys
import typing as ty

import addict
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from source.datasets import VOCSegmentation
from source.models import FCN32VGG16
from source.utils.augmentations import get_albumentation_augs
from source.utils.general import get_object_from_dict, reseed

logger.remove()
logger.add(
    sys.stdout,
    format=(
        "[<green>{time: HH:mm:ss}</green> |"
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


def _fix_worker_seeds(worker_id: int) -> None:
    """Fix seeds inside single worker."""
    seed = np.random.get_state()[1][0] + worker_id
    reseed(seed)


def train(config: addict.Dict) -> None:
    # Ingridients.
    device = torch.device(config.training.device)
    augs = get_albumentation_augs(config)
    train_set = VOCSegmentation(
        config,
        "train",
        augs["train"],
    )
    val_set = VOCSegmentation(
        config,
        "val",
        augs["val"],
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        num_workers=config.training.dataloader_num_workers,
        pin_memory="cuda" in config.training.device,
        worker_init_fn=_fix_worker_seeds,
        shuffle=True,
    )
    train_loader = DataLoader(
        val_set,
        batch_size=config.training.batch_size,
        num_workers=config.training.dataloader_num_workers,
        pin_memory="cuda" in config.training.device,
        worker_init_fn=_fix_worker_seeds,
        shuffle=False,
    )
    model = FCN32VGG16(config).to(device)
    optimizer = get_object_from_dict(
        config.optimizer,
        params=create_param_groups(
            model, config.optimizer.lr, config.optimizer.weight_decay
        ),
    )
    criterion = get_object_from_dict(config.criterion)

    # We may want to overfit single batch of 4 images just to be sure
    # that pipeline works okay and our model can learn at least something.
    if config.training.overfit_single_batch:
        train_loader = [
            next(
                iter(
                    DataLoader(
                        train_set,
                        batch_size=4,
                        num_workers=config.training.dataloader_num_workers,
                        pin_memory="cuda" in config.training.device,
                        worker_init_fn=_fix_worker_seeds,
                        shuffle=True,
                    )
                )
            )
        ]
        val_loader = train_loader
        config.training.epochs = 100

    for epoch in range(config.training.epochs):
        # Reseed at the beginning to be sure that augmentation will be
        # different during each epoch.
        reseed(config.training.seed + epoch)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        logger.info(
            "[EPOCH {epoch}/{ttl}] TL={tl:.5f}",
            epoch=epoch + 1,
            tl=train_loss,
            ttl=config.training.epochs,
        )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ty.Any,
    device: torch.device,
) -> float:
    model.train()
    running_loss = torch.as_tensor(0.0, device=device)
    for images, masks in loader:
        optimizer.zero_grad()
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss
    return (running_loss / len(loader)).item()


if __name__ == "__main__":
    train()
