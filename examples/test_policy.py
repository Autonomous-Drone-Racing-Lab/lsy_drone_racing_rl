"""Example training script using the stable-baselines3 library.

Note:
    This script requires you to install the stable-baselines3 library.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import fire
from safe_control_gym.utils.registration import make
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.wrapper import DroneRacingWrapper
from stable_baselines3.common.evaluation import evaluate_policy

logger = logging.getLogger(__name__)



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


def main(checkpoint_path:str, config: str = "config/getting_started.yaml"):
    config_path = Path(__file__).resolve().parents[1] / config
    env = create_race_env(config_path=config_path, gui=True, random_gate_init=True, rank=0)
    check_env(env)  # Sanity check to ensure the environment conforms to the sb3 API

    # load the model
    model = PPO.load(checkpoint_path, env=env)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

if __name__ == "__main__":
    fire.Fire(main)
