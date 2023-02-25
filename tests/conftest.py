import pytest

from source.utils.general import read_config

_CONFIG_PATH = "config.yml"


@pytest.fixture(scope="session")
def get_test_config():
    config = read_config(_CONFIG_PATH)
    config.training.device = "cpu"
    config.training.dataloader_num_workers = 0
    config.training.epochs = 1
    config.training.batch_size = 1
    config.training.overfit_single_batch = False
    return config
