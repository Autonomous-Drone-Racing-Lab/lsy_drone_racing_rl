"""Logging configuration."""
import logging
import sys

from munch import Munch


def setup_log(log_name: str, log_config: Munch) -> logging.Logger:
    """Setup the logger with the given configuration. Creaing a logger of the proper name and initializing relevant folders.

    Args:
        log_name: The name of the logger
        log_config: The configuration for the logger
    """
    logger = logging.getLogger(log_name)
    level_name = logging.getLevelName(log_config.log_level)
    logger.setLevel(level_name)

    if log_config.log_terminal:
        terminal_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(terminal_handler)
    
    if log_config.log_file:
        file_handler = logging.FileHandler(log_config.log_file)
        logger.addHandler(file_handler)

    logger.debug("Testing looger debugs")
    logger.info("Testing looger infos")
    logger.warning("Testing looger warnings")
    logger.error("Testing looger errors")
    
    return logger

def setup_test_logger(log_name: str) -> logging.Logger:
    """Setup a logger for testing purposes.
    
    The test logger always writes to the console.

    Args:
        log_name: The name of the logger
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    terminal_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(terminal_handler)
    return logger