from functools import partial
from pathlib import Path
import re
from copy import deepcopy
import math

import os
from safe_control_gym.utils.registration import make
from stable_baselines3.common.env_checker import check_env
from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.utils.logging import setup_log

# At top to prevent cirular import
def save_config_to_file(config):
    """
    Utility function to save config to file in well formated order.
    The save path is automatically deduced from the config itself.
    """
    log_dir = config.log_config.log_dir
    config_path = os.path.join(log_dir, "config.yaml")

    # overwrite config ctrl_frew
    config.quadrotor_config.ctrl_freq = 30

    with open(config_path, "w") as f:
        def represent_flows_as_list(dumper, data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

        def represent_blocks_as_dict(dumper, data):
            return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=False)
        class CustomDumper(yaml.SafeDumper):
            pass
        CustomDumper.add_representer(list, represent_flows_as_list)
        CustomDumper.add_representer(dict, represent_blocks_as_dict)
        #print(config.rl_config)
        yaml.dump(config, f, sort_keys=False, default_flow_style=False, Dumper=CustomDumper)

import yaml
def create_experiment_log_folder(logs_dir, experiment_name):
    """
    Creates a log directory for an experiment with an incrementing index.
    
    Args:
    - base_dir (str): The base directory where the logs folder should be created.
    - experiment_name (str): The name of the experiment.
    
    Returns:
    - str: The path of the created log directory.
    """

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Find the highest index for the given experiment name
    highest_idx = 0
    for folder_name in os.listdir(logs_dir):
        match = re.match(rf"^{experiment_name}_(\d+)$", folder_name)
        if match:
            idx = int(match.group(1))
            if idx > highest_idx:
                highest_idx = idx
    
    # Create the new folder with the next index
    new_idx = highest_idx + 1
    new_folder_name = f"{experiment_name}_{new_idx}"
    new_folder_path = os.path.join(logs_dir, new_folder_name)
    os.makedirs(new_folder_path)
    
    return new_folder_path

def create_race_env(config, rank, is_train:bool, random_gate_init: bool=False, gui: bool = False):
    """
    Create drone racing evnrionment based on the config.
    """
    from lsy_drone_racing.wrapper import DroneRacingWrapper
    """Create the drone racing environment."""
    config = deepcopy(config) # deepcopy required because we will change argumens
    config.quadrotor_config.gui = gui
    config.quadrotor_config.seed = config.quadrotor_config.seed + rank
    CTRL_FREQ = config.quadrotor_config["ctrl_freq"]
    # Create environment
    assert config.use_firmware, "Firmware must be used for the competition."
    pyb_freq = config.quadrotor_config["pyb_freq"]
    assert pyb_freq % FIRMWARE_FREQ == 0, "pyb_freq must be a multiple of firmware freq"
    config.quadrotor_config["ctrl_freq"] = FIRMWARE_FREQ
    env_factory = partial(make, "quadrotor", **config.quadrotor_config)
    firmware_env = make("firmware", env_factory, FIRMWARE_FREQ, CTRL_FREQ)
    env =  DroneRacingWrapper(firmware_env, rank=rank, is_train=is_train, config=config, terminate_on_lap=True, random_initialization=random_gate_init)
    env.reset(seed=config.quadrotor_config.seed)
    check_env(env)
    return env
    
def make_env(config, rank: int):
    """
    Utility funciton to generate randomized verctorized environments for training
    """
    def _init():
        env = create_race_env(config, is_train=True, rank=rank, gui=False, random_gate_init=True)
        return env
    return _init

def resume_from_checkpoint(checkpoint_path: str):
    """
    Resume training from checkpoint by loading the config and setting up logging.
    """
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), f"Checkpoint {checkpoint_path} does not exist."
    config_path = checkpoint_path.parents[0] / "config.yaml"
    assert config_path.exists(), f"Config {config_path} does not exist."

    config = load_config(config_path)

    # setup logging
    setup_log('drone_rl', config.log_config)

    return config

def start_from_scratch(config_path: Path):
    """
    Start training from scratch. Load config and setup all experiment related folders.
    """
    config_path = Path(config_path)
    assert config_path.exists(), f"Config {config_path} does not exist."
    config = load_config(config_path)

    # Setup workspace folder
    log_folder = config.log_config.log_dir
    name = config.log_config.exp_name
    logs_dir = create_experiment_log_folder(log_folder, name)
    config.log_config.log_dir = logs_dir
    
    # Setup logging
    config.log_config.log_file = os.path.join(logs_dir, "log.log")
    setup_log('drone_rl', config.log_config)    

    # Dump config
    save_config_to_file(config)
    return config