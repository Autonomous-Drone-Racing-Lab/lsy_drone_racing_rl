import os
from abc import ABC
import warnings
import typing
from typing import Union, List, Dict, Any, Optional

import gym
import numpy as np

from lsy_drone_racing.utils.evaluate_policy import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization, DummyVecEnv
from stable_baselines3.common.callbacks import  BaseCallback

class EvalCallbackIncreaseEnvComplexity(BaseCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
    :param callback_on_new_best: (Optional[BaseCallback]) Callback to trigger
        when there is a new best model according to the `mean_reward`
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param log_path: (str) Path to a folder where the evaluations (`evaluations.npz`)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: (str) Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: (bool) Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: (bool) Whether to render or not the environment during evaluation
    :param verbose: (int)
    """
    def __init__(self, 
                 eval_env: Union[gym.Env, VecEnv],
                 no_gates: int,
                 n_eval_episodes: int = 5,
                 success_threshold: float = 0.7,
                 eval_freq: int = 10000,
                 verbose: int = 1):
        super().__init__(verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.no_gates = no_gates
        self.success_threshold = success_threshold

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"
        self.eval_env = eval_env

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same
        if not type(self.training_env) is type(self.eval_env):
            warnings.warn("Training and eval env are not of the same type"
                          "{} != {}".format(self.training_env, self.eval_env))

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            episode_rewards, episode_lengths, no_gates_passed = evaluate_policy(
                model=self.model,
                env=self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                return_episode_rewards=True
            )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            mean_no_gates_passed = np.mean(no_gates_passed)
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward
            no_succecces = 0
            for episode_gates_passed in no_gates_passed:
                if episode_gates_passed == self.no_gates:
                       no_succecces += 1
            success_rate = no_succecces / self.n_eval_episodes

            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward, std_reward))
                print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))
                print("No gates passed: {:.2f}".format(mean_no_gates_passed))
                print("Success rate: {:.2f}".format(no_succecces / self.n_eval_episodes))
            
            if success_rate > self.success_threshold:
                print(f"Success rate of {success_rate} higher than threshold of {self.success_threshold}. Increasing environment complexity")
                self.training_env.env_method("increase_gates_obstacles_randomization") 
        return True
