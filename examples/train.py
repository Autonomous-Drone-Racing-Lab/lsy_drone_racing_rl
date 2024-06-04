"""Example training script using the stable-baselines3 library.

Note:
    This script requires you to install the stable-baselines3 library.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
import re

import fire
from safe_control_gym.utils.registration import make
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.utils.logging import setup_log
from lsy_drone_racing.wrapper import DroneRacingWrapper
from safe_control_gym.envs.env_wrappers.vectorized_env import make_vec_envs
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
import json
#from safe_control_gym.envs.env_wrappers.vectorized_env import SubprocVecEnv

logger = logging.getLogger(__name__)
import os

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

def create_race_env(config_path: Path, rank=0, random_gate_init: bool=False, gui: bool = False) -> DroneRacingWrapper:
    """Create the drone racing environment."""
    # Load configuration and check if firmare should be used.
    config = load_config(config_path)
    # Overwrite config options
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
    env =  DroneRacingWrapper(firmware_env,config=config, terminate_on_lap=True, random_initialization=random_gate_init)
    env.reset(seed=config.quadrotor_config.seed)
    check_env(env)
    #print(f" env factory id {id(env)} firmware id {id(firmware_env)} env id {id(env)}, unwrap {unwrap} unwrap id {id(unwrap)}")
    return env

    
def make_env(config_path: Path, rank: int):
    def _init():
        env = create_race_env(config_path=config_path, rank=rank, gui=False, random_gate_init=False)
        return env
    return _init


def main(checkpoint=None, config: str = "config/getting_started.yaml", gui: bool = False):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""
    logging.basicConfig(level=logging.INFO)
    config_path = Path(__file__).resolve().parents[1] / config
    config = load_config(config_path)
    #env = create_race_env(config_path=config_path, gui=True, random_gate_init=False)


    # Setup workspace folder
    log_folder = config.log_config.log_dir
    name = config.log_config.exp_name
    logs_dir = create_experiment_log_folder(log_folder, name)

    # Setup logging
    config.log_config.log_file = os.path.join(logs_dir, "log.log")
    setup_log('drone_rl', config.log_config)

    # Dump config
    dump = json.dumps(config, indent=4)
    with open(os.path.join(logs_dir, "config.json"), "w") as f:
        f.write(dump)

    #envs = make_vec_env(make_env(config_path), n_envs=4, vec_env_cls=DummyVecEnv)

    num_processes = 4
    envs = SubprocVecEnv([make_env(config_path, rank=i) for i in range(4)])
    envs = VecMonitor(envs)

    eval_frequency = 50000
    eval_frquency_scaled = eval_frequency  // num_processes
    checkpoint_frequency = 100000
    checkpoint_frequency_scaled = checkpoint_frequency // num_processes

    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_frequency_scaled, save_path=logs_dir,
                                             name_prefix='rl_model', verbose=2)
    
    eval_env = create_race_env(config_path=config_path, gui=False, random_gate_init=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path=logs_dir,
                                 log_path=logs_dir, eval_freq=eval_frquency_scaled,
                                 deterministic=True, render=False)
    if checkpoint:
        print(f"Init from checkpont {checkpoint}")
        model = PPO.load(checkpoint, envs, verbose=1, tensorboard_log=logs_dir)
    else:
        model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log=logs_dir)
    model.learn(total_timesteps=20000000, callback=[checkpoint_callback, eval_callback], progress_bar=True)


if __name__ == "__main__":
   fire.Fire(main)