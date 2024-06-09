"""Example training script using the stable-baselines3 library.

Note:
    This script requires you to install the stable-baselines3 library.
"""

from __future__ import annotations

from lsy_drone_racing.environment import resume_from_checkpoint, start_from_scratch, make_env, create_race_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from safe_control_gym.envs.env_wrappers.vectorized_env import make_vec_envs
from stable_baselines3 import PPO
import fire
import matplotlib.pyplot as plt

from lsy_drone_racing.environment_tracking_wrapper import TrajectoryTrackingWrapper
from lsy_drone_racing.utils.logging import setup_test_logger
from lsy_drone_racing.utils.visualization import visualize_trajectories

def main(checkpoint_path:str, gui: bool = False, random_gate_init: bool = True, show_plot: bool = True):
    print(f"Resuming from checkpoint {checkpoint_path}")
    config = resume_from_checkpoint(checkpoint_path)
   
    env = create_race_env(config, gui=gui, random_gate_init=random_gate_init, rank=0)
    
    tracked_trajectories = []
    def save_trajectory(trajectory):
        tracked_trajectories.append(trajectory)
    
    env = TrajectoryTrackingWrapper(env, on_save_callback=save_trajectory)
    setup_test_logger('drone_rl')

    # load the model
    model = PPO.load(checkpoint_path, env=env)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # visualize the trajectories
    if show_plot:
        visualize_trajectories(tracked_trajectories)
        plt.show()

if __name__ == "__main__":
    fire.Fire(main)
