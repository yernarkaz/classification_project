# utils.py
import yaml


def load_yaml_file(path) -> yaml:
    """
    Load yaml file

    :param path:
    :return:
    """

    # load yaml file
    yaml_file = yaml.load(open(path, "r"), Loader=yaml.FullLoader)

    return yaml_file
