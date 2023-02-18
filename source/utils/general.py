import addict
import yaml


def read_config(path: str) -> addict.Dict:
    with open(path, "r") as yfs:
        return addict.Dict(yaml.safe_load(yfs))
