import torch


@torch.inference_mode()
def torch_hist(y_true: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """
    Ported to PyTorch from here: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/score.py#L9

    Parameters:
        y_true: Batch of target masks with shape (N, H, W);
        y_pred: Batch of predicted masks with shape (N, C, H, W).

    Returns:
        Table of frequencies.
    """
    n_cls = y_hat.size(1)

    # Mask out difficult pixels which are marked as -1 in target mask.
    mask = (y_true >= 0) & (y_true < n_cls)

    # Obtain predicted labels.
    y_hat = y_hat.argmax(dim=1)

    unique_sums = n_cls * y_true[mask].view(-1) + y_hat[mask].view(-1)
    return torch.bincount(unique_sums, minlength=n_cls**2).view(n_cls, n_cls)


@torch.inference_mode()
def compute_paper_metrics(hist: torch.Tensor) -> dict[str, torch.Tensor]:
    """Ported to PyTorch from here: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/score.py#L37"""
    overall_acc = hist.diag().sum() / hist.sum()
    per_class_acc = hist.diag() / hist.sum(1)
    iu = hist.diag() / (hist.sum(1) + hist.sum(0) - hist.diag())
    freq = hist.sum(1) / hist.sum()
    freq_acc = (freq[freq > 0] * iu[freq > 0]).sum()
    return {
        "acc": overall_acc,
        "per_class_acc": torch.nanmean(per_class_acc),
        "iu": torch.nanmean(iu),
        "freq_acc": freq_acc,
    }
