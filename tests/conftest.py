import pytest

from source.utils.general import read_config

_CONFIG_PATH = "config.yml"


@pytest.fixture(scope="session")
def get_test_config():
    config = read_config(_CONFIG_PATH)

    # Disable GPU for tests.
    config.training.device = "cpu"

    # Disable any parallelization.
    config.training.batch_size = 1
    config.training.dataloader_num_workers = 0

    # Disable image caching.
    for split in config.dataset:
        getattr(config.dataset, split).cache_images = False

    # Turn of logging.
    config.training.log_fixed_batch = False
    config.training.use_clearml = False

    # Simplify training procedure.
    config.training.epochs = 1
    config.training.overfit_single_batch = False

    return config
