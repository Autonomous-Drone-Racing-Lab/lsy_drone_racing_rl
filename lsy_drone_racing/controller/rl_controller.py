"""Controler class for simulation and deployment."""
from __future__ import annotations

from typing import Optional  # Python 3.10 type hints

import numpy as np
from stable_baselines3 import PPO

from lsy_drone_racing.action_space_wrapper import action_space_wrapper_factory
from lsy_drone_racing.command import Command
from lsy_drone_racing.experiment_trakcer import ExperimentTracker  # noqa: TCH001
from lsy_drone_racing.observation_space_wrapper import observation_space_wrapper_factory
from lsy_drone_racing.state_estimator import StateEstimator


class Controller():
    """Implementation of a controller class. It is not required for trainig, however, deployment in real-world requires it to be implemented.

    Attention: We are no longer extending the base controll, class to be able to provide more flexibility in the implementation.
    Still, you must implement the following methods:
    - compute_control
    - reset
    - step_learn
    - episode_learn
    - episode_reset
    """
    def __init__(
        self,
        checkpoint: str,
        config: str,
        init_pos: np.ndarray,
        experiment_tracker: Optional[ExperimentTracker] = None
    ):
        """Initialization of the controller.

        Args:
            checkpoint: Neural Network model checkpoint to load
            config: Path to the configuration file.
            init_pos: Initial position of the drone.
            experiment_tracker: ExperimentTracker object to log data. (Optional)
        """
        self.config = config
        self.state_estimator = StateEstimator(config.rl_config.state_estimator_buffer_size) 
        self.observation_space_wrapper = observation_space_wrapper_factory(config)
        self.action_space_wrapper = action_space_wrapper_factory(config)
        self.model = PPO.load(checkpoint)
        self.experiment_tracker = experiment_tracker

        # For state machine
        self._take_off = False
        self._take_off_time = 0.4
        self._initialized = False
        self._initialized_time = 0
        self._setpoint_land = False
        self._land = False

        self._trained_start_pos = np.array(init_pos)

        # Reset counters and buffers.
        self.reset()
        self.episode_reset()

    def compute_control(
        self,
        ep_time: float,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ) -> tuple[Command, list]:
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        Args:
            ep_time: Episode's elapsed time, in seconds.
            obs: The environment's observation [drone_xyz_yaw, gates_xyz_yaw, gates_in_range,
                obstacles_xyz, obstacles_in_range, gate_id].
            reward: The reward signal.
            done: Wether the episode has terminated.
            info: Current step information as a dictionary with keys 'constraint_violation',
                'current_target_gate_pos', etc.

        Returns:
            The command type and arguments to be sent to the quadrotor. See `Command`.
        """
        self.state_estimator.add_measurement(obs[0][:3], ep_time)
        estimated_velocity, estimated_acceleration = self.state_estimator.estimate_state()
        kwargs = {"estimated_velocity": estimated_velocity, "estimated_acceleration": estimated_acceleration}
        transformed_obs = self.observation_space_wrapper.transform_observation(obs, **kwargs)
        action, _ = self.model.predict(transformed_obs, deterministic=True)

        if self.experiment_tracker is not None:
            drone_pos = obs[0][:3]
            self.experiment_tracker.add_drone_obs(drone_pos, ep_time)

        max_time = 20 # To abort in case it tkes too long
        # Calculate compute command based on current flight state
        if not self._take_off:

            command_type = Command.TAKEOFF
            args = [0.3, self._take_off_time]  # Height, duration
            self._take_off = True  # Only send takeoff command once
        elif not self._initialized:
            if ep_time - self._take_off_time > 0:
                print("Initializing")
                command_type = Command.FULLSTATE
                args = [self._trained_start_pos, np.zeros(3), np.zeros(3), 0.0, np.zeros(3), ep_time]
                self._initialized = True
            else:
                command_type = Command.NONE
                args = []
        else:
            if ep_time - (self._take_off_time + self._initialized_time) > 0:
                estimated_velocity, estimated_acceleration = self.state_estimator.estimate_state()
                kwargs = {"estimated_velocity": estimated_velocity, "estimated_acceleration": estimated_acceleration}
                transformed_obs = self.observation_space_wrapper.transform_observation(obs, **kwargs)
                action, _ = self.model.predict(transformed_obs, deterministic=True)
                drone_pose = obs[0]
                action = self.action_space_wrapper.scale_action(action, drone_pose)
                target_pos = np.array([action[0], action[1], action[2]])

                target_vel = np.zeros(3)
                target_acc = np.zeros(3)
                target_yaw = 0.0
                target_rpy_rates = np.zeros(3)
                command_type = Command.FULLSTATE
                args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates, ep_time]
            # Notify set point stop has to be called every time we transition from low-level
            # commands to high-level ones. Prepares for landing
            elif ep_time >max_time and not self._setpoint_land:
                command_type = Command.NOTIFYSETPOINTSTOP
                args = []
                self._setpoint_land = True
            elif ep_time >max_time and not self._land:
                command_type = Command.LAND
                args = [0.0, 2.0]  # Height, duration
                self._land = True  # Send landing command only once
            elif self._land:
                command_type = Command.FINISHED
                args = []
            else:
                command_type = Command.NONE
                args = []

        return command_type, args

    def reset(self):
        """Initialize/reset data buffers and counters."""
        self.state_estimator.reset()

    def step_learn(
        self,
        action: list,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ):
        """Learning and controller updates called between control steps.

        Args:
            action: Most recent applied action.
            obs: Most recent observation of the quadrotor state.
            reward: Most recent reward.
            done: Most recent done flag.
            info: Most recent information dictionary.
        """
        pass

    def episode_learn(self):
        """Learning and controller updates called between episodes."""
        pass

    def episode_reset(self):
        """Reset the controller's internal state and models if necessary."""
        pass