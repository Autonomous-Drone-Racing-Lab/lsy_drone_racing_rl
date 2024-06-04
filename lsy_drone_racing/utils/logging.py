import logging
import sys


def setup_log(log_name, log_config):
    logger = logging.getLogger(log_name)
    level_name = logging.getLevelName(log_config.log_level)
    logger.setLevel(level_name)

    if log_config.log_terminal:
        terminal_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(terminal_handler)
    
    if log_config.log_file:
        file_handler = logging.FileHandler(log_config.log_file)
        logger.addHandler(file_handler)
    
    return logger

def setup_test_logger(log_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    terminal_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(terminal_handler)
    return logger

def get_logger(log_name):
    return logging.getLogger(log_name)
