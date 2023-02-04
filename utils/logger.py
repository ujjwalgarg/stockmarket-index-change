import os
import logging
from pathlib import Path

from logging.handlers import RotatingFileHandler
from constants import LOGGER_PATH


def initialise_logger(logger_name, log_file_path):
    """
    Function to initialize the logger.

    :param str logger_name: name of the logger
    :param str log_file_path: path of the logger file

    :return: logger object
    """
    max_bytes = 1024 * 1024
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s "
                                  "%(levelname)s "
                                  "[%(module)s] "
                                  "[%(funcName)s] "
                                  "[%(lineno)d] "
                                  "%(message)s",
                                  datefmt=logging.Formatter.default_time_format)

    backup_count = 5
    handler = RotatingFileHandler(log_file_path, maxBytes=max_bytes,
                                  backupCount=backup_count)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# Creates logs folder if not existent
if not os.path.exists(LOGGER_PATH):
    os.makedirs(LOGGER_PATH)

app_logger = initialise_logger(logger_name='app_logs',
                               log_file_path=LOGGER_PATH + '/app_logs.out')

train_logger = initialise_logger(logger_name='train_logs',
                                 log_file_path=LOGGER_PATH + '/train_logs.out')

test_logger = initialise_logger(logger_name='test_logs',
                                log_file_path=LOGGER_PATH + '/test_logs.out')
