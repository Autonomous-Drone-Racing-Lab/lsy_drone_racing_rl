"""Test trained RL agent in the drone racing environment."""

from __future__ import annotations

import fire
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from lsy_drone_racing.environment import (
    create_race_env,
    resume_from_checkpoint,
)
from lsy_drone_racing.environment_tracking_wrapper import TrajectoryTrackingWrapper
from lsy_drone_racing.utils.logging import setup_test_logger
from lsy_drone_racing.utils.visualization import visualize_trajectories


def main(checkpoint:str, gui: bool = False, random_gate_init: bool = False, show_plot: bool = False):
    """Main function to test agent in the drone racing environment.
    
    Args:
        checkpoint (str): The path to the checkpoint.
        gui (bool): Whether to show the GUI.
        random_gate_init (bool): Whether to randomize the start position of the drone (i.e. the start gate).
        show_plot (bool): Whether to show the plot.
    """
    print(f"Resuming from checkpoint {checkpoint}")
    config = resume_from_checkpoint(checkpoint)
   
    env = create_race_env(config, gui=gui, is_train=False, random_gate_init=random_gate_init, rank=0)
    
    tracked_trajectories = []
    def save_trajectory(trajectory: list):
        tracked_trajectories.append(trajectory)
    
    env = TrajectoryTrackingWrapper(env, on_save_callback=save_trajectory)
    setup_test_logger('drone_rl')

    # load the model
    model = PPO.load(checkpoint, env=env)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # visualize the trajectories
    if show_plot:
        visualize_trajectories(tracked_trajectories)
        plt.show()

if __name__ == "__main__":
    fire.Fire(main)
