import json
import logging
import os
import os.path as osp
from enum import Enum

from collections import defaultdict, OrderedDict

import torch
from tensorboardX import SummaryWriter

# Default configuration for logging
default_log_config = {
    "logging_file": os.path.join("./logs", "my.log"),
    "logging_level": logging.INFO,
    # "logging_level": logging.DEBUG,
    "log_format": "%(asctime)s - %(levelname)s - %(message)s",
    "date_format": "%m/%d/%Y %H:%M:%S %p"
}


class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif isinstance(o, torch.device):
            return {
                "device": "{}".format(o)
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }

        return json.JSONEncoder.default(self, o)


class Logger(object):
    def __init__(self, args, log_dir, **kwargs):
        self.logger_path = osp.join(log_dir, 'scalars.json')
        self.tb_logger = SummaryWriter(
            logdir=osp.join(log_dir, 'tflogger'),
            **kwargs,
        )
        self.log_config(vars(args))

        self.scalars = defaultdict(OrderedDict)

    def add_scalar(self, key, value, counter):
        assert self.scalars[key].get(counter, None) is None, 'counter should be distinct'
        self.scalars[key][counter] = value
        self.tb_logger.add_scalar(key, value, counter)

    def log_config(self, variant_data):
        config_filepath = osp.join(osp.dirname(self.logger_path), 'configs.json')
        with open(config_filepath, "w") as fd:
            json.dump(variant_data, fd, indent=2, sort_keys=True, cls=ConfigEncoder)

    def dump(self):
        with open(self.logger_path, 'w') as fd:
            json.dump(self.scalars, fd, indent=2)


def get_basic_logger():
    """
    Set up and return a basic logger that outputs to both the console and a file.
    :return: logger object
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(default_log_config["log_format"], datefmt=default_log_config["date_format"])

    # Create file handler for logging to file
    if not os.path.exists(os.path.split(default_log_config["logging_file"])[0]):
        os.makedirs(os.path.split(default_log_config["logging_file"])[0])

    # Ensure the log directory exists
    if not os.path.exists(default_log_config["logging_file"]):
        file = open(default_log_config["logging_file"], 'w')
        file.close()

    fh = logging.FileHandler(default_log_config["logging_file"])
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Create stream handler for logging to console
    ch = logging.StreamHandler()
    ch.setLevel(default_log_config["logging_level"])
    ch.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger