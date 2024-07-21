"""Trivial Extenstion of the evaluate_policy function to also count the number of gates passed."""
from typing import List, Tuple, Union

import gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv


def evaluate_policy(
    model,  # noqa: ANN001
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    return_episode_rewards: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """Trivial Extenstion of the evaluate_policy function to also count the number of gates passed.

    :param model: (BaseRLModel) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: (Optional[float]) If True, a list of reward per episode
        will be returned instead of the mean.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths, no_gates_passed = [], [], []
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            new_obs, reward, done, _info = env.step(action)
            _info = _info[0]
            obs=new_obs
            episode_reward += reward
            episode_length += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        no_gates_passed_this_episode = _info['no_gates_passed']
        no_gates_passed.append(no_gates_passed_this_episode)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_no_gates_passed = np.mean(no_gates_passed)
    if return_episode_rewards:
        return episode_rewards, episode_lengths, no_gates_passed
    return mean_reward, std_reward, mean_no_gates_passed