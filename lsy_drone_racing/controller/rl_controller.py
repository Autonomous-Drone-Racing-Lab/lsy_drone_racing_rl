from __future__ import annotations  # Python 3.10 type hints

import numpy as np
from scipy import interpolate

from lsy_drone_racing.command import Command
from lsy_drone_racing.action_space_wrapper import action_space_wrapper_factory
from lsy_drone_racing.experiment_trakcer import ExperimentTracker
from lsy_drone_racing.observation_space_wrapper import observation_space_wrapper_factory
from lsy_drone_racing.state_estimator import StateEstimator
import numpy as np
from stable_baselines3 import PPO 


class Controller():
    """Template controller class."""

    def __init__(
        self,
        checkpoint: str,
        config,
        init_pos: np.ndarray,
        experiment_tracker: ExperimentTracker = None
    ):
        #########################
        # REPLACE THIS (START) ##
        #########################
        self.config = config
        self.state_estimator = StateEstimator(config.rl_config.state_estimator_buffer_size) 
        self.observation_space_wrapper = observation_space_wrapper_factory(config)
        self.action_space_wrapper = action_space_wrapper_factory(config)
        self.model = PPO.load(checkpoint)
        self.experiment_tracker = experiment_tracker

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
        #########################
        # REPLACE THIS (END) ####
        #########################

    def compute_control(
        self,
        ep_time: float,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ) -> tuple[Command, list]:
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration,
            attitude, and attitude rates to be sent from Crazyswarm to the Crazyflie using, e.g., a
            `cmdFullState` call.

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

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Handcrafted solution for getting_stated scenario.
        max_time = 20
        if not self._take_off:
            print("Taking off")
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

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def reset(self):
        self.state_estimator.reset()

    def step_learn(
        self,
        action: list,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ):
        pass

    def episode_learn(self):
        pass

    def episode_reset(self):
        """Reset the controller's internal state and models if necessary."""
        pass