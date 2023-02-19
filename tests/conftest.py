import pytest

from source.utils.general import read_config

_CONFIG_PATH = "config.yml"


@pytest.fixture(scope="session")
def get_test_config():
    return read_config(_CONFIG_PATH)
