"""Training script using the stable-baselines3 library.

Note:
    This script requires you to install the stable-baselines3 library.
"""

from copy import deepcopy
from pathlib import Path

import fire
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from lsy_drone_racing.environment import (
    create_race_env,
    make_env,
    resume_from_checkpoint,
    start_from_scratch,
)
from lsy_drone_racing.utils.eval_callback_count_gates_passed import EvalCallbackCountGatesPassed
from lsy_drone_racing.utils.eval_callback_scale_environment_complexity import (
    EvalCallbackIncreaseEnvComplexity,
)
from lsy_drone_racing.utils.utils import load_config


def main(checkpoint: str=None, config: str = "config/getting_started.yaml"):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""
    if checkpoint:
        print(f"Resuming from checkpoint {checkpoint}")
        if config:
            print("Warning: config argument will be ignored when resuming from checkpoint")
        config = resume_from_checkpoint(checkpoint)
    else:
        print("Starting from scratch")
        config = start_from_scratch(config)

    num_processes = 4
    random_init = config.rl_config.random_gate_init
    envs = SubprocVecEnv([make_env(config, rank=i, random_init=random_init) for i in range(num_processes)])
    envs = VecMonitor(envs)

    eval_frequency = 50000
    eval_frquency_scaled = eval_frequency  // num_processes
    checkpoint_frequency = 100000
    checkpoint_frequency_scaled = checkpoint_frequency // num_processes

    logs_dir = config.log_config.log_dir
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_frequency_scaled, save_path=logs_dir,
                                             name_prefix='rl_model', verbose=2)

    if config.rl_config.increase_env_complexity:
        no_gates = len(config.quadrotor_config.gates)
        success_threshold = config.rl_config.success_threshold
        eval_env = create_race_env(config, rank=0, is_train=False, gui=False, random_gate_init=False)
        eval_callback = EvalCallbackIncreaseEnvComplexity(eval_env, no_gates=no_gates,success_threshold=success_threshold,  n_eval_episodes=50, eval_freq=eval_frquency_scaled)
    else:
        eval_config = "config/level3.yaml"
        eval_config = load_config(Path(eval_config))
        config_copy = deepcopy(config)
        eval_config.rl_config = config_copy.rl_config
        eval_env = create_race_env(config, rank=0, is_train=False, gui=False, random_gate_init=False)
        # eval_callback = EvalCallback(eval_env, best_model_save_path=logs_dir, log_path=logs_dir,
        #                           eval_freq=eval_frquency_scaled,
        #                          deterministic=True, render=False)
        eval_callback = EvalCallbackCountGatesPassed(eval_env, n_eval_episodes=10, eval_freq=eval_frquency_scaled, 
                                                     log_path=logs_dir, best_model_save_path=logs_dir, deterministic=True, render=False)
    
    if checkpoint:
        model = PPO.load(checkpoint, envs, verbose=1, tensorboard_log=logs_dir)
    else:
        model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log=logs_dir)

    model.learn(total_timesteps=20000000, callback=[checkpoint_callback, eval_callback], progress_bar=True)


if __name__ == "__main__":
   fire.Fire(main)