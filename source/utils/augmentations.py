import addict
import albumentations as album
from albumentations.core.serialization import Serializable


def get_albumentation_augs(config: addict.Dict) -> dict[str, Serializable | None]:
    """Build albumentations's augmentation pipelines from configuration file."""
    return {
        "train": album.from_dict(config.augmentations.train),
        "val": album.from_dict(config.augmentations.val),
    }
