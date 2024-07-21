"""Wrapper to make the environment compatible with the gymnasium API.

The drone simulator does not conform to the gymnasium API, which is used by most RL frameworks. This
wrapper can be used as a translation layer between these modules and the simulation.

RL environments are expected to have a uniform action interface. However, the Crazyflie commands are
highly heterogeneous. Users have to make a discrete action choice, each of which comes with varying
additional arguments. Such an interface is impractical for most standard RL algorithms. Therefore,
we restrict the action space to only include FullStateCommands.

We also include the gate pose and range in the observation space. This information is usually
available in the info dict, but since it is vital information for the agent, we include it directly
in the observation space.

Warning:
    The RL wrapper uses a reduced action space and a transformed observation space!
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from gymnasium import Wrapper
from gymnasium.error import InvalidAction
from munch import Munch, munchify
from safe_control_gym.controllers.firmware.firmware_wrapper import FirmwareWrapper

from lsy_drone_racing.action_space_wrapper import action_space_wrapper_factory
from lsy_drone_racing.environment import save_config_to_file
from lsy_drone_racing.observation_space_wrapper import observation_space_wrapper_factory
from lsy_drone_racing.state_estimator import StateEstimator
from lsy_drone_racing.utils.delayed_reward import DelayedReward
from lsy_drone_racing.utils.rewards import (
    progress_reward,
    safety_reward,
    state_limits_exceeding_penalty,
)


class DroneRacingWrapper(Wrapper):
    """Drone racing firmware wrapper to make the environment compatible with the gymnasium API.

    In contrast to the underlying environment, this wrapper only accepts FullState commands as
    actions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: FirmwareWrapper, config: Munch, rank: int, is_train: bool, random_initialization:bool = True, terminate_on_lap: bool = True):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
            config: The configuration object.
            rank: The rank of the environment.
            is_train: Whether the environment is used for training or evaluation.
            random_initialization: Whether to randomize the start position of the drone (i.e. the start gate)
            terminate_on_lap: Whether to terminate the episode when the drone completes a lap.
        """
        if not isinstance(env, FirmwareWrapper):
            raise TypeError(f"`env` must be an instance of `FirmwareWrapper`, is {type(env)}")
        super().__init__(env)
        # Patch the FirmwareWrapper to add any missing attributes required by the gymnasium API.
        self.config = config
        self.env = env
        self.env.unwrapped = []  # Changed this to allow for vectorized environments
        self.env.render_mode = None

        # Action space
        self.action_space_wrapper = action_space_wrapper_factory(config)
        self.action_space = self.action_space_wrapper.get_action_space()

        # Observation space provided by environment transformation
        self.observation_space_wrapper = observation_space_wrapper_factory(config)
        self.observation_space = self.observation_space_wrapper.get_observation_space()

        self.pyb_client_id: int = env.env.PYB_CLIENT
        # Config and helper flags
        self.terminate_on_lap = terminate_on_lap
        self._reset_required = False
        # The original firmware wrapper requires a sim time as input to the step function. This
        # breaks the gymnasium interface. Instead, we keep track of the sim time here. On each step,
        # it is incremented by the control time step. On env reset, it is reset to 0.
        self._sim_time = 0.0
        # The firmware quadrotor env requires the rotor forces as input to the step function. These
        # are zero initially and updated by the step function. We automatically insert them to
        # ensure compatibility with the gymnasium interface.
        # TODO: It is not clear if the rotor forces are even used in the firmware env. Initial tests
        #       suggest otherwise.
        self._f_rotors = np.zeros(4)
        self.rank = rank
        self.is_train = is_train
        self.logger = logging.getLogger("drone_rl")


        # Custom by me extra values we must keep track of because they are only available in the initial info dict
        self.random_initialization = random_initialization
        self.no_gates = len(config.quadrotor_config["gates"])
        self.rng = np.random.default_rng()
        self.state_estimator = StateEstimator(self.config.rl_config.state_estimator_buffer_size) 
        self.delayed_gate_reward = DelayedReward() 

        
    @property
    def time(self) -> float:
        """Return the current simulation time in seconds."""
        return self._sim_time

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        Args:
            seed: The random seed to use for the environment. Not used in this wrapper.
            options: Additional options to pass to the environment. Not used in this wrapper.

        Returns:
            The initial observation and info dict of the next episode.
        """
        if seed is not None:
            self.logger.info("Setting seed of wrapper to %d", seed)
            self.rng = np.random.default_rng(seed)


        self._reset_required = False
        self._sim_time = 0.0
        self._f_rotors[:] = 0.0
        self.state_estimator.reset()

        reset_kwargs = {}
        if self.random_initialization:
            random_gate_id = self.rng.integers(0, self.no_gates)
            reset_kwargs["initial_target_gate_id"] = random_gate_id

        obs, info = self.env.reset(**reset_kwargs)
        # Store obstacle height for observation expansion during env steps.
        obs = self.observation_transform(obs, info)
        self.state_estimator.add_measurement(obs[0][:3], self._sim_time)
        estimated_velocity, estimated_acceleration = self.state_estimator.estimate_state()
        assert (estimated_velocity == np.zeros(3)).all() and (estimated_acceleration == np.zeros(3)).all(), "Initial state estimation must be zero"
        reset_kwargs = {"estimated_velocity": estimated_velocity, "estimated_acceleration": estimated_acceleration}
        transformed_obs = self.observation_space_wrapper.transform_observation(obs, **reset_kwargs)


        self.prev_drone_pose = obs[0]
        self.initial_drone_pose = obs[0]
        self.prev_action = None

        self.no_gates_passed = 0
        self.init_gate_id_offset = obs[5]
        self.last_gate_to_pass_id = obs[5]
        self.delayed_gate_reward.reset()
        self.gate_passed_set = set()
       
        return transformed_obs, self.info_transform(info)
    


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: The action to take in the environment. See action space for details.

        Returns:
            The next observation, the reward, the terminated and truncated flags, and the info dict.
        """
        assert not self._reset_required, "Environment must be reset before taking a step"
        # Progress counter objects
        self.delayed_gate_reward.step()
        
        if action not in self.action_space:
            # Wrapper has a reduced action space compared to the firmware env to make it compatible
            # with the gymnasium interface and popular RL libraries.
            raise InvalidAction(f"Invalid action: {action}")
        action = self.action_space_wrapper.scale_action(action, self.prev_drone_pose)
        # The firmware does not use the action input in the step function
        zeros = np.zeros(3)
        self.env.sendFullStateCmd(action[:3], zeros, zeros, action[3], zeros, self._sim_time)
        # The firmware quadrotor env requires the sim time as input to the step function. It also
        # returns the desired rotor forces. Both modifications are not part of the gymnasium
        # interface. We automatically insert the sim time and reuse the last rotor forces.
        obs, _, done, info, f_rotors = self.env.step(self._sim_time, action=self._f_rotors)
        self._f_rotors[:] = f_rotors
        obs = self.observation_transform(obs, info)
        self.state_estimator.add_measurement(obs[0][:3], self._sim_time)

        estimated_velocity, estimated_acceleration = self.state_estimator.estimate_state()
        kwargs = {"estimated_velocity": estimated_velocity, "estimated_acceleration": estimated_acceleration}
        transformed_obs = self.observation_space_wrapper.transform_observation(obs, **kwargs)
        # We set truncated to True if the task is completed but the drone has not yet passed the
        # final gate. We set terminated to True if the task is completed and the drone has passed
        # the final gate.
        terminated, truncated = False, False 
        if info["task_completed"] and info["current_gate_id"] != -1:
            self.logger.debug("Task completed, but last gate not passed")
            truncated = True
        elif self.terminate_on_lap and info["current_gate_id"] == -1:
            self.logger.debug("Task completed, i.e. terminated")
            info["task_completed"] = True
            terminated = True
        elif self.terminate_on_lap and done:  # Done, but last gate not passed -> terminate
            # self.logger.debug("Task not completed, but done, i.e. terminated (probably time out)")
            terminated = True
        
        # Get the reward
        current_drone_pose = obs[0]
        next_gate_id = obs[5]
        next_gate_pose_world = obs[1][next_gate_id]
        progress_reward_value = progress_reward(current_drone_pose, self.prev_drone_pose, next_gate_pose_world)
        #print(f"Target gate position {next_gate_pose_world}, prev drone pose {self.prev_drone_pose}, current drone pose {current_drone_pose}, progress reward {progress_reward_value}")

        # Terminations
        col_id, did_collide = info["collision"]
        has_error = info["has_error"]
        
        termination_penalty = 0
        if has_error:
            self.logger.debug("Drone tumbling, aborting")
            assert terminated
            termination_penalty = -1
        elif did_collide:
            self.logger.debug(f"Drone collided with gate {col_id}, aborting")
            assert terminated
            termination_penalty = -1
        elif transformed_obs not in self.observation_space:
            self.logger.debug(f"Drone out of bounds at pos {transformed_obs}, aborting")
            termination_penalty = -1
            terminated = True
        
        if next_gate_id != self.last_gate_to_pass_id:
            if next_gate_id in self.gate_passed_set:
                print(f"Double passing gate {next_gate_id}!")
            else:
                delay = self.config.rl_config.get("delay_gate_passed_reward", 0)
                self.no_gates_passed += 1
                self.logger.debug(f"Drone passed gate {self.last_gate_to_pass_id}. Delaying reward for {delay} steps")
                self.delayed_gate_reward.add_reward(1, delay)
                self.last_gate_to_pass_id = next_gate_id
                self.gate_passed_set.add(next_gate_id)

        
    
        velocity_limit_penalty = state_limits_exceeding_penalty(estimated_velocity, self.config.rl_config.desirable_vel_bound)

        safety_reward_value = safety_reward(current_drone_pose, next_gate_pose_world)

        # Reward calculation
        lambda_progress = self.config.rl_config.lambda_progress
        lambda_termination = self.config.rl_config.lambda_termination
        lambda_gate_passed = self.config.rl_config.lambda_gate_passed
        lambda_velocity_limit = self.config.rl_config.lambda_velocity_limit
        lambda_safety = self.config.rl_config.get("lambda_safety", 0)

        flush_reward = info["task_completed"] and info["current_gate_id"] == -1
        reward = lambda_progress * progress_reward_value + lambda_termination * termination_penalty + lambda_gate_passed * self.delayed_gate_reward.get_value(flush=flush_reward) + lambda_velocity_limit * velocity_limit_penalty + lambda_safety * safety_reward_value
        # print(f"Progress reward {progress_reward_value}, termination penalty {termination_penalty}, gate passed reward {gate_passed_reward}, total reward {reward}")

        self._reset_required = terminated or truncated
        
        # Keep track of extra values-------------------------------------------------
        self.prev_drone_pose = current_drone_pose
        self.prev_action = action

        # Increment the sim time after the step if we are not yet done.
        if not terminated and not truncated:
            self._sim_time += self.env.ctrl_dt

        # Update info
        info["no_gates_passed"] = self.no_gates_passed
        return transformed_obs, reward, terminated, truncated, self.info_transform(info)


    def render(self):
        """Render the environment.

        Used for compatibility with the gymnasium API. Checks if PyBullet was launched with an
        active GUI.

        Raises:
            AssertionError: If PyBullet was not launched with an active GUI.
        """
        assert self.pyb_client_id != -1, "PyBullet not initialized with active GUI"

    @staticmethod
    def observation_transform(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """Transform the observation to include additional information and restructure the format.

        Args:
            obs: The observation to transform.
            info: Additional information to include in the observation.

        Returns:
            The transformed observation.
        """
        drone_pos = obs[0:6:2]
        drone_yaw = obs[8]
        drone_xyz_yaw = np.concatenate([drone_pos, [drone_yaw]])
        drone_pos = obs[0:6:2]
        drone_yaw = obs[8]
        drone_rpy = obs[6:9]
        drone_ang_vel = obs[8:11]


        obs = [
            drone_xyz_yaw,
            info["gates_pose"][:, [0, 1, 2, 5]],
            info["gates_in_range"],
            info["obstacles_pose"][:, :3],
            #obstacle_pose_grounded,
            info["obstacles_in_range"],
            info["current_gate_id"],
            drone_rpy,
            drone_ang_vel,
        ]
        return obs
    
    @staticmethod
    def info_transform(info: dict[str, Any]) -> dict[str, Any]:
        """Transform the info dict for multithreading.
        
        To allow multithreading we must strip all non-pickable information from it.This mainly means that we remove all casadi objects from the info dict.
        For now until we know how to do that, only return keys where value is either primitive type, numpy array or dict.

        Args:
            info: The info dict to transform.
        
        Returns:
            The transformed info dict.
        """
        allowed_types = [int, float, np.ndarray, dict, tuple, bool, str]
        transformed_info = {}
        for key, value in info.items():
            if type(value) in allowed_types:
                transformed_info[key] = value
        return transformed_info
    
    def increase_env_complexity(self):
        """Increases the environment complexity of the simulator of either the gates and obstacles or the initial state."""
        assert self.config.rl_config.increase_env_complexity, "Increase environment complexity must be set to True"
        assert self.config.rl_config.env_complexity_stages, "Env complexity stages must be set"
        env_complexity_cur_stage = self.config.rl_config.get("env_complexity_cur_stage", 0)
        stage = self.config.rl_config.env_complexity_stages[env_complexity_cur_stage]
        for func_str in stage:
            if func_str == "gate_obstacle":
                self.increase_gates_obstacles_randomization()
            elif func_str == "init_state":
                self.increase_init_state_randomization()
            else:
                raise ValueError(f"Unknown function {func_str} in env complexity stage {env_complexity_cur_stage}")
        self.config.rl_config.env_complexity_cur_stage = (env_complexity_cur_stage + 1) % len(self.config.rl_config.env_complexity_stages)
        if self.rank == 0 and self.is_train:
            save_config_to_file(self.config)

    def increase_gates_obstacles_randomization(self):
        """Increases the environment complexity by increasing the randomization of the gates and obstacles."""
        assert self.config.rl_config.increase_env_complexity, "Increase environment complexity must be set to True"
        randomization_step_size = self.config.rl_config.increase_gates_obstacles_randomization_step_size
        assert randomization_step_size > 0, "Randomization step size must be greater than 0"

        control_gym_env = self.env.env
        # get previous label from config
        prev_gate_obstacle_randomization = self.config.quadrotor_config.get("gates_and_obstacles_randomization_info", None)
        randomized_gates_and_obstacles = self.config.quadrotor_config.get("randomized_gates_and_obstacles", None)

        if not randomized_gates_and_obstacles or prev_gate_obstacle_randomization is None:
            gate_bound_high = randomization_step_size
            obstacle_bound_high = randomization_step_size
        else:

            gate_bound_high = prev_gate_obstacle_randomization.gates.high
            gate_bound_low = prev_gate_obstacle_randomization.gates.low
            obstacle_bound_high = prev_gate_obstacle_randomization.obstacles.high
            obstacle_bound_low = prev_gate_obstacle_randomization.obstacles.low
            assert gate_bound_high == -gate_bound_low, "Gates must have symmetric bounds"
            assert obstacle_bound_high == -obstacle_bound_low, "Obstacles must have symmetric bounds"

            gate_bound_high += randomization_step_size
            obstacle_bound_high += randomization_step_size

        control_gym_env.set_gate_obstacle_randomization(gate_bound_high, obstacle_bound_high)

        # update config
        updated_randomization_info = {
            "gates": {"high": gate_bound_high, "low": -gate_bound_high, "distrib": "uniform"},
            "obstacles": {"high": obstacle_bound_high, "low": -obstacle_bound_high, "distrib": "uniform"},
        }
        self.config.quadrotor_config.gates_and_obstacles_randomization_info = munchify(updated_randomization_info)
        self.config.quadrotor_config.randomized_gates_and_obstacles = True
        print(f"This environment is rank {self.rank}.Increased randomization to: gates {gate_bound_high}, obstacles {obstacle_bound_high}")

        # Save the updated config
        if self.rank == 0 and self.is_train:
            save_config_to_file(self.config)

    def increase_init_state_randomization(self):
        """Increases the environment complexity by increasing the randomization of the initial state."""
        assert self.config.rl_config.increase_env_complexity, "Increase environment complexity must be set to True"
        randomization_step_size = self.config.rl_config.increase_init_state_randomization_step_size
        assert randomization_step_size > 0, "Randomization step size must be greater than 0"

        control_gym_env = self.env.env
        # get previous label from config
        prev_init_state_randomization = self.config.quadrotor_config.get("init_state_randomization_info", None)
        randomized_init_state = self.config.quadrotor_config.get("randomized_init", None)
        
        if randomized_init_state:
            try:
                init_x_rand = prev_init_state_randomization.init_x.high
            except:  # noqa: E722
                init_x_rand = 0
            try:
                init_y_rand = prev_init_state_randomization.init_y.high
            except:  # noqa: E722
                init_y_rand = 0
            try:
                init_z_rand = prev_init_state_randomization.init_z.high
            except:  # noqa: E722
                init_z_rand = 0
        else:
            init_x_rand = 0
            init_y_rand = 0
            init_z_rand = 0

        init_x_rand = init_x_rand + randomization_step_size
        init_y_rand = init_y_rand + randomization_step_size
        init_z_rand = init_z_rand + randomization_step_size
        control_gym_env.set_init_state_randomization(init_x_rand, init_y_rand, init_z_rand)

        rand_info = {
            "init_x": {"high": init_x_rand + randomization_step_size, "low": -init_x_rand - randomization_step_size, "distrib": "uniform"},
            "init_y": {"high": init_y_rand + randomization_step_size, "low": -init_y_rand - randomization_step_size, "distrib": "uniform"},
            "init_z": {"high": init_z_rand + randomization_step_size, "low": -init_z_rand - randomization_step_size, "distrib": "uniform"},
        }

        self.config.quadrotor_config.init_state_randomization_info = munchify(rand_info)
        self.config.quadrotor_config.randomized_init = True

        # Save the updated config
        if self.rank == 0 and self.is_train:
            save_config_to_file(self.config)
    

class DroneRacingObservationWrapper:
    """Wrapper to transform the observation space the firmware wrapper.

    This wrapper matches the observation space of the DroneRacingWrapper. See its definition for
    more details. While we want to transform the observation space, we do not want to change the API
    of the firmware wrapper. Therefore, we create a separate wrapper for the observation space.

    Note:
        This wrapper is not a subclass of the gymnasium ObservationWrapper because the firmware is
        not compatible with the gymnasium API.
    """

    def __init__(self, env: FirmwareWrapper):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
        """
        if not isinstance(env, FirmwareWrapper):
            raise TypeError(f"`env` must be an instance of `FirmwareWrapper`, is {type(env)}")
        self.env = env
        self.pyb_client_id: int = env.env.PYB_CLIENT

    def __getattribute__(self, name: str) -> Any:
        """Get an attribute from the object.

        If the attribute is not found in the wrapper, it is fetched from the firmware wrapper.

        Args:
            name: The name of the attribute.

        Returns:
            The attribute value.
        """
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.env, name)

    def reset(self, *args: Any, **kwargs: dict[str, Any]) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            The transformed observation and the info dict.
        """
        obs, info = self.env.reset(*args, **kwargs)
        obs = DroneRacingWrapper.observation_transform(obs, info)
        return obs, info

    def step(
        self, *args: Any, **kwargs: dict[str, Any]
    ) -> tuple[np.ndarray, float, bool, dict, np.ndarray]:
        """Take a step in the current environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            The transformed observation and the info dict.
        """
        obs, reward, done, info, action = self.env.step(*args, **kwargs)
        obs = DroneRacingWrapper.observation_transform(obs, info)
        info = DroneRacingWrapper.info_transform(info)
        return obs, reward, done, info, action
