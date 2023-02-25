import inspect
import os
import pydoc
import random
import types
import typing as ty

import addict
import numpy as np
import torch
import yaml


def read_config(path: str) -> addict.Dict:
    with open(path, "r") as yfs:
        return addict.Dict(yaml.safe_load(yfs))


def seed_everything(config: addict.Dict, local_rank: int = 0) -> None:
    """
    Fix all avaliable seeds to ensure reproducibility.
    Local rank is needed for distributed data parallel training.
    It is used to make seeds different for different processes.
    Each process will have `seed = config_seed + local_rank`.
    """
    random.SystemRandom().seed(config.training.seed + local_rank)
    np.random.seed(config.training.seed + local_rank)
    torch.manual_seed(config.training.seed + local_rank)
    os.environ["PYTHONHASHSEED"] = str(config.training.seed + local_rank)
    torch.cuda.manual_seed(config.training.seed + local_rank)
    torch.cuda.manual_seed_all(config.training.seed + local_rank)


def reseed(new_seed: ty.Any) -> None:
    np.random.seed(new_seed)
    random.SystemRandom().seed(int(new_seed))
    torch.manual_seed(new_seed)


def _is_iterable(instance: ty.Any) -> bool:
    try:
        _ = iter(instance)
    except TypeError:
        return False
    return True


def get_object_from_dict(  # noqa: CCR001.
    dict_repr: dict,
    parent: dict | None = None,
    **additional_kwargs,
) -> ty.Any:
    """
    Parse pydantic model and build instance of provided type.

    Parameters:
        dict_repr: Dictionary representation of object;
        parent: Parent object dictionary;
        additional_kwargs: Additional arguments for instantiation procedure.

    Returns:
        Intance of provided type.
    """
    if dict_repr is None:
        return None
    object_type = dict_repr.pop("__class_fullname__")
    for param_name, param_value in additional_kwargs.items():
        dict_repr.setdefault(param_name, param_value)
    if parent is not None:
        return getattr(parent, object_type)(**dict_repr)
    callable_ = pydoc.locate(object_type)

    # If callable is regular python function then we don't need to  instantiate it.
    if isinstance(callable_, types.FunctionType):
        return callable_

    # If parameter has kind == `VAR_POSITIONAL` then it will not possible to set it as
    # keyword argument. Thet's why we need to explicitly create an *args list.
    args = []
    signature = inspect.signature(callable_)
    for param_name, param_value in signature.parameters.items():
        if param_value.kind == param_value.VAR_POSITIONAL:
            config_value = dict_repr.get(param_name)
            if _is_iterable(config_value):
                args.extend(list(config_value))
            else:
                args.append(config_value)
            del dict_repr[param_name]
    return callable_(*args, **dict_repr)


def get_cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
