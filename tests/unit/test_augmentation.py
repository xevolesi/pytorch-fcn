from source.utils.augmentations import get_albumentation_augs


def test_get_albumentation_augs(get_test_config):
    augs = get_albumentation_augs(get_test_config)
    assert augs
