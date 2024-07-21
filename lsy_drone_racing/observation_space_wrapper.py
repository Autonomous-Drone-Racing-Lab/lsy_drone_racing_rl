"""Observation space wrapper to convert the observation from the environment into an observation that can be used by the RL agent (NN)."""
from abc import ABC

import numpy as np
from gymnasium.spaces import Box
from munch import Munch

from lsy_drone_racing.coordinate_transformation import (
    convert_gate_to_corners,
    translate_points_in_local_frame,
)

EDGE_LENGTH = 0.45

class ObervationSpaceWrapper(ABC):
    """Abstract base class for observation space wrappers."""
    
    def __init__(self, config: Munch):
        """Create an observation space wrapper.

        Args:
            config: The configuration object.
        """
        self.config = config
    
    def transform_observation(self, obs: np.ndarray, **kwargs: dict):
        """Transform the given environment observation into an observation that can be used by the RL agent (NN).

        Args:
            obs (np.ndarray): The environment observation.
            kwargs: Additional keyword arguments.
        
        Returns:
            The transformed observation
        """
        raise NotImplementedError

    def _get_next_gate_one_hot(self, current_gate_id: int) -> np.ndarray:
        """Helper function to transform gate id into one hot encoding.

        Args:
            current_gate_id: The id of the current gate.
        
        Returns:
            The one hot encoding of the current gate id.
        """
        no_gates = len(self.config.quadrotor_config.gates)
        one_hot = np.zeros(no_gates)
        if current_gate_id != -1:
            one_hot[current_gate_id] = 1
        return one_hot
    
    def get_observation_space(self) -> Box:
        """Get the observation space.

        Returns:
            The observation space.
        """
        raise NotImplementedError
    

def observation_space_wrapper_factory(config: Munch) -> ObervationSpaceWrapper:
    """Factory function to create an observation space wrapper based on the configuration."""
    observation_space = config.rl_config.observation_space
    if observation_space == "relative_only_next_goal":
        return ObservationSpaceWrapperRelativeOnlyNextGoalAllObstacles(config)
    elif observation_space == "relative_only_next_goal_rpy":
        return ObservationSpaceWrapperRelativeOnlyNextGoalAllObstaclesRPY(config)
    elif observation_space == "relative_only_next_goal_rp_relative_vel":
        return ObservationSpaceWrapperRelativeNextGoalAllObstaclesRPRelativeVel(config)
    elif observation_space == "relative_next_goal_next_obstacle_rp_relative_vel":
        return ObservationSpaceWrapperRelativeNextGoalNextObstacleRPRelativeVel(config)
    elif observation_space == "relative_next_goal_next_obstacle_rp_relative_vel_nearest_gate":
        return ObservationSpaceWrapperRelativeNextGoalNextObstacleRPRelativeVelNearestGate(config)
    elif observation_space == "relative_all_goals_rp":
        return ObservationSpaceWrapperRelativeAllGoalAllObstaclesRP(config)
    elif observation_space == "relative_all_goals_rp_relative_vel":
        return ObservationSpaceWrapperRelativeAllGoalAllObstaclesRPRelativeVel(config)
    elif observation_space == "absolute_all_goals":
        return ObservationSpaceWrapperAbsoluteAllGoals(config)
    elif observation_space == "relative_all_goals_all_obstacles":
        return ObservationSpaceWrapperRelativeAllGoalsAllObstacles(config)
    else:
        raise ValueError(f"Observation space {observation_space} not supported.")
    
       
class ObservationSpaceWrapperRelativeOnlyNextGoalAllObstacles(ObervationSpaceWrapper):
    """Implementation on Observation space wrapper.
    
    - Next goal and obstacle are provided in relative coordinates.
    """
    def __init__(self, config: Munch):
        """Create an observation space wrapper.

        Args:
            config: The configuration object.
        """
        super().__init__(config)

    def transform_observation(self, obs: np.ndarray, **kwargs: dict) -> np.ndarray:
        """Transform the given environment observation into an observation that can be used by the RL agent (NN).

        Args:
            obs (np.ndarray): The environment observation.
            kwargs: Additional keyword arguments.
        
        Returns:
            The transformed observation
        """
        drone_xyz_yaw = obs[0]
        gates_xyz_yaw = obs[1]
        obstacles_xyz = obs[3]
        current_gate_id = obs[5]

        estimated_vel = kwargs["estimated_velocity"]

        
        current_gate = gates_xyz_yaw[current_gate_id]
        current_gate_xyz_yaw = current_gate[:4]
        gate_corners = convert_gate_to_corners(current_gate_xyz_yaw, edge_length=EDGE_LENGTH)
        gate_corners = translate_points_in_local_frame(drone_xyz_yaw, gate_corners).flatten()  #Todo, update this

        all_obstacles_local = translate_points_in_local_frame(drone_xyz_yaw, obstacles_xyz).flatten()

        obs = np.concatenate([drone_xyz_yaw[:-1], estimated_vel, gate_corners, all_obstacles_local]).astype(np.float32)
        return obs

    def get_observation_space(self) -> Box:
        """Get the observation space.

        Returns:
            The observation space.
        """
        no_obstacles = len(self.config.quadrotor_config.obstacles)

        world_lower_bound = np.array(self.config.rl_config.world_lower_bound)
        world_upper_bound = np.array(self.config.rl_config.world_upper_bound)
        vel_bound = self.config.rl_config.vel_bound
        drone_vel_limits_upper = np.array([vel_bound, vel_bound, vel_bound])
        drone_vel_limits_lower = -drone_vel_limits_upper

        world_diff = world_upper_bound - world_lower_bound
        single_gate_limit_upper =np.tile(world_diff, 4)# 4 corners
        single_gate_limit_lower = -single_gate_limit_upper

        singe_obstacle_limit_lower = -world_diff
        singe_obstacle_limit_upper = world_diff
        obstacle_limit_lower = np.tile(singe_obstacle_limit_lower, no_obstacles)
        obstacle_limit_upper = np.tile(singe_obstacle_limit_upper, no_obstacles)

        obs_limit_low = np.concatenate([world_lower_bound, drone_vel_limits_lower, single_gate_limit_lower, obstacle_limit_lower])
        obs_limits_high = np.concatenate([world_upper_bound, drone_vel_limits_upper, single_gate_limit_upper, obstacle_limit_upper])

        return Box(obs_limit_low, obs_limits_high, dtype=np.float32)
    

class ObservationSpaceWrapperRelativeOnlyNextGoalAllObstaclesRPY(ObervationSpaceWrapper):
    """Implementation of Observation space wrapper.
    
    - Next goal and all obstacles are provided in the observation space in local frame of the drone.
    - Drone angles and velocities are provided.
    """
    def __init__(self, config: Munch):
        """Create an observation space wrapper.

        Args:
            config: The configuration object.
        """
        super().__init__(config)

    def transform_observation(self, obs: np.ndarray, **kwargs: dict) -> np.ndarray:
        """Transform the given environment observation into an observation that can be used by the RL agent (NN).

        Args:
            obs (np.ndarray): The environment observation.
            kwargs: Additional keyword arguments.
        
        Returns:
            The transformed observation
        """
        drone_xyz_yaw = obs[0]
        gates_xyz_yaw = obs[1]
        obstacles_xyz = obs[3]
        current_gate_id = obs[5]

        estimated_vel = kwargs["estimated_velocity"]
        rpy = obs[6]
        ang_vel = obs[7]

        
        current_gate = gates_xyz_yaw[current_gate_id]
        current_gate_xyz_yaw = current_gate[:4]
        gate_corners = convert_gate_to_corners(current_gate_xyz_yaw, edge_length=EDGE_LENGTH)
        gate_corners = translate_points_in_local_frame(drone_xyz_yaw, gate_corners).flatten()  #Todo, update this

        all_obstacles_local = translate_points_in_local_frame(drone_xyz_yaw, obstacles_xyz).flatten()

        obs = np.concatenate([drone_xyz_yaw[:-1], rpy, estimated_vel, ang_vel, gate_corners, all_obstacles_local]).astype(np.float32)
       # obs = np.concatenate([drone_xyz_yaw[:-1], estimated_vel, estimated_acc, [drone_xyz_yaw[-1]], all_gate_corners, [current_gate_id]]).astype(np.float32)
        return obs

    
    def get_observation_space(self) -> Box:
        """Get the observation space.

        Returns:
            The observation space.
        """
        no_obstacles = len(self.config.quadrotor_config.obstacles)

        world_lower_bound = np.array(self.config.rl_config.world_lower_bound)
        world_upper_bound = np.array(self.config.rl_config.world_upper_bound)
        rpy_lower_bound = np.array([-np.pi, -np.pi, -np.pi])
        rpy_upper_bound = np.array([np.pi, np.pi, np.pi])
        vel_bound = self.config.rl_config.vel_bound
        drone_vel_limits_upper = np.array([vel_bound, vel_bound, vel_bound])
        drone_vel_limits_lower = -drone_vel_limits_upper
        ang_vel_bound_lower = np.array([-np.inf, -np.inf, -np.inf])
        ang_vel_bound_upper = np.array([np.inf, np.inf, np.inf])

        world_diff = world_upper_bound - world_lower_bound
        single_gate_limit_upper =np.tile(world_diff, 4)# 4 corners
        single_gate_limit_lower = -single_gate_limit_upper

        singe_obstacle_limit_lower = -world_diff
        singe_obstacle_limit_upper = world_diff
        obstacle_limit_lower = np.tile(singe_obstacle_limit_lower, no_obstacles)
        obstacle_limit_upper = np.tile(singe_obstacle_limit_upper, no_obstacles)

        obs_limit_low = np.concatenate([world_lower_bound, rpy_lower_bound, drone_vel_limits_lower, ang_vel_bound_lower, single_gate_limit_lower, obstacle_limit_lower])
        obs_limits_high = np.concatenate([world_upper_bound, rpy_upper_bound, drone_vel_limits_upper, ang_vel_bound_upper, single_gate_limit_upper, obstacle_limit_upper])

        return Box(obs_limit_low, obs_limits_high, dtype=np.float32)
    

class ObservationSpaceWrapperRelativeAllGoalAllObstaclesRP(ObervationSpaceWrapper):
    """Implementation of Observation space wrapper.

    - All goals and all obstacles are provided in the observation space in local frame of the drone.
    - Drone angles and velocities are provided. (yaw excluded)
    """
    def __init__(self, config: Munch):
        """Create an observation space wrapper.

        Args:
            config: The configuration object.
        """
        super().__init__(config)
    
    def transform_observation(self, obs: np.ndarray, **kwargs: dict) -> np.ndarray:
        """Transform the given environment observation into an observation that can be used by the RL agent (NN).

        Args:
            obs (np.ndarray): The environment observation.
            kwargs: Additional keyword arguments.
        
        Returns:
            The transformed observation
        """
        drone_xyz_yaw = obs[0]
        gates_xyz_yaw = obs[1]
        obstacles_xyz = obs[3]
        current_gate_id = obs[5]

        estimated_vel = kwargs["estimated_velocity"]
        rpy = obs[6][:-1] # dont care about yaw
        ang_vel = obs[7][:-1] # dont care about yaw

        
        # Extract next gate to pass
        all_gate_corners = []
        for gate in gates_xyz_yaw:
            assert len(gate) == 4, "Gate must have 4 elements."
            gate_corners = convert_gate_to_corners(gate, edge_length=EDGE_LENGTH)
            gate_corners = translate_points_in_local_frame(drone_xyz_yaw, gate_corners).flatten()
            all_gate_corners.extend(gate_corners)
        current_gate_id_one_hot = self._get_next_gate_one_hot(current_gate_id)
        

        all_obstacles_local = translate_points_in_local_frame(drone_xyz_yaw, obstacles_xyz).flatten()
        
        obs = np.concatenate([drone_xyz_yaw[:-1], rpy, estimated_vel, ang_vel, all_gate_corners, all_obstacles_local, current_gate_id_one_hot]).astype(np.float32)
       # obs = np.concatenate([drone_xyz_yaw[:-1], estimated_vel, estimated_acc, [drone_xyz_yaw[-1]], all_gate_corners, [current_gate_id]]).astype(np.float32)
        return obs

    def get_observation_space(self) -> Box:
        """Get the observation space.

        Returns:
            The observation space.
        """
        no_gates = len(self.config.quadrotor_config.gates)
        no_obstacles = len(self.config.quadrotor_config.obstacles)

        world_lower_bound = np.array(self.config.rl_config.world_lower_bound)
        world_upper_bound = np.array(self.config.rl_config.world_upper_bound)
        rpy_lower_bound = np.array([-np.pi, -np.pi])
        rpy_upper_bound = np.array([np.pi, np.pi])
        
        vel_bound = self.config.rl_config.vel_bound
        drone_vel_limits_upper = np.array([vel_bound, vel_bound, vel_bound])
        drone_vel_limits_lower = -drone_vel_limits_upper
        ang_vel_bound_lower = np.array([-np.inf, -np.inf])
        ang_vel_bound_upper = np.array([np.inf, np.inf])

        world_diff = world_upper_bound - world_lower_bound
        single_gate_limit_upper =np.tile(world_diff, 4)# 4 corners
        single_gate_limit_lower = -single_gate_limit_upper
        gate_limit_lower = np.tile(single_gate_limit_lower, no_gates)
        gate_limit_upper = np.tile(single_gate_limit_upper, no_gates)

        singe_obstacle_limit_lower = -world_diff
        singe_obstacle_limit_upper = world_diff
        obstacle_limit_lower = np.tile(singe_obstacle_limit_lower, no_obstacles)
        obstacle_limit_upper = np.tile(singe_obstacle_limit_upper, no_obstacles)

        obs_limit_low = np.concatenate([world_lower_bound, rpy_lower_bound, drone_vel_limits_lower, ang_vel_bound_lower, gate_limit_lower, obstacle_limit_lower, np.zeros(no_gates)])
        obs_limits_high = np.concatenate([world_upper_bound, rpy_upper_bound, drone_vel_limits_upper, ang_vel_bound_upper, gate_limit_upper, obstacle_limit_upper, np.ones(no_gates)])

        return Box(obs_limit_low, obs_limits_high, dtype=np.float32)
    
class ObservationSpaceWrapperRelativeAllGoalAllObstaclesRPRelativeVel(ObervationSpaceWrapper):
    """Implementation of Observation space wrapper.

    - All goals and all obstacles are provided in the observation space in local frame of the drone.
    - Drone angles and velocities are provided
    - Velocity is provided in local frame
    """
    def __init__(self, config: Munch):
        """Create an observation space wrapper.

        Args:
            config: The configuration object.
        """
        super().__init__(config)

    def transform_observation(self, obs: np.ndarray, **kwargs: dict) -> np.ndarray:
        """Transform the given environment observation into an observation that can be used by the RL agent (NN).

        Args:
            obs (np.ndarray): The environment observation.
            kwargs: Additional keyword arguments.
        
        Returns:
            The transformed observation
        """
        drone_xyz_yaw = obs[0]
        gates_xyz_yaw = obs[1]
        obstacles_xyz = obs[3]
        current_gate_id = obs[5]

        estimated_vel = np.array(kwargs["estimated_velocity"])
        #estimated_acc = kwargs["estimated_acceleration"]
        rpy = obs[6][:-1] # dont care about yaw
        ang_vel = obs[7][:-1] # dont care about yaw

        yaw = drone_xyz_yaw[-1]
        rot_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        vel_local_frame = (rot_matrix.T @ estimated_vel.reshape(3, 1)).flatten()

        # Extract next gate to pass
        all_gate_corners = []
        for gate in gates_xyz_yaw:
            assert len(gate) == 4, "Gate must have 4 elements."
            gate_corners = convert_gate_to_corners(gate, edge_length=EDGE_LENGTH)
            gate_corners = translate_points_in_local_frame(drone_xyz_yaw, gate_corners).flatten()
            all_gate_corners.extend(gate_corners)
        current_gate_id_one_hot = self._get_next_gate_one_hot(current_gate_id)
        

        all_obstacles_local = translate_points_in_local_frame(drone_xyz_yaw, obstacles_xyz).flatten()
        
        obs = np.concatenate([drone_xyz_yaw[:-1], rpy, vel_local_frame, ang_vel, all_gate_corners, all_obstacles_local, current_gate_id_one_hot]).astype(np.float32)
       # obs = np.concatenate([drone_xyz_yaw[:-1], estimated_vel, estimated_acc, [drone_xyz_yaw[-1]], all_gate_corners, [current_gate_id]]).astype(np.float32)
        return obs

    def get_observation_space(self) -> Box:
        """Get the observation space.

        Returns:
            The observation space.
        """
        no_gates = len(self.config.quadrotor_config.gates)
        no_obstacles = len(self.config.quadrotor_config.obstacles)

        world_lower_bound = np.array(self.config.rl_config.world_lower_bound)
        world_upper_bound = np.array(self.config.rl_config.world_upper_bound)
        rpy_lower_bound = np.array([-np.pi, -np.pi])
        rpy_upper_bound = np.array([np.pi, np.pi])
        
        vel_bound = self.config.rl_config.vel_bound
        drone_vel_limits_upper = np.array([vel_bound, vel_bound, vel_bound])
        drone_vel_limits_lower = -drone_vel_limits_upper
        ang_vel_bound_lower = np.array([-np.inf, -np.inf])
        ang_vel_bound_upper = np.array([np.inf, np.inf])

        world_diff = world_upper_bound - world_lower_bound
        single_gate_limit_upper =np.tile(world_diff, 4)# 4 corners
        single_gate_limit_lower = -single_gate_limit_upper
        gate_limit_lower = np.tile(single_gate_limit_lower, no_gates)
        gate_limit_upper = np.tile(single_gate_limit_upper, no_gates)

        singe_obstacle_limit_lower = -world_diff
        singe_obstacle_limit_upper = world_diff
        obstacle_limit_lower = np.tile(singe_obstacle_limit_lower, no_obstacles)
        obstacle_limit_upper = np.tile(singe_obstacle_limit_upper, no_obstacles)

        obs_limit_low = np.concatenate([world_lower_bound, rpy_lower_bound, drone_vel_limits_lower, ang_vel_bound_lower, gate_limit_lower, obstacle_limit_lower, np.zeros(no_gates)])
        obs_limits_high = np.concatenate([world_upper_bound, rpy_upper_bound, drone_vel_limits_upper, ang_vel_bound_upper, gate_limit_upper, obstacle_limit_upper, np.ones(no_gates)])

        return Box(obs_limit_low, obs_limits_high, dtype=np.float32)
    
class ObservationSpaceWrapperRelativeNextGoalAllObstaclesRPRelativeVel(ObervationSpaceWrapper):
    """Implementation of Observation space wrapper.

    - Next goal and all obstacles are provided in the observation space in local frame of the drone.
    - Drone angles and velocities are provided
    - Velocity is provided in local frame
    """
    def __init__(self, config: Munch):
        """Create an observation space wrapper.

        Args:
            config: The configuration object.
        """
        super().__init__(config)

    def transform_observation(self, obs: np.ndarray, **kwargs: dict) -> np.ndarray:
        """Transform the given environment observation into an observation that can be used by the RL agent (NN).

        Args:
            obs (np.ndarray): The environment observation.
            kwargs: Additional keyword arguments.
        
        Returns:
            The transformed observation
        """
        drone_xyz_yaw = obs[0]
        gates_xyz_yaw = obs[1]
        obstacles_xyz = obs[3]
        current_gate_id = obs[5]

        estimated_vel = np.array(kwargs["estimated_velocity"])
        #estimated_acc = kwargs["estimated_acceleration"]
        rpy = obs[6][:-1] # dont care about yaw
        ang_vel = obs[7][:-1] # dont care about yaw

        yaw = drone_xyz_yaw[-1]
        rot_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        vel_local_frame = (rot_matrix.T @ estimated_vel.reshape(3, 1)).flatten()
        
        gate_corners = convert_gate_to_corners(gates_xyz_yaw[current_gate_id], edge_length=EDGE_LENGTH)
        gate_corners = translate_points_in_local_frame(drone_xyz_yaw, gate_corners).flatten()
        

        all_obstacles_local = translate_points_in_local_frame(drone_xyz_yaw, obstacles_xyz).flatten()
        
        obs = np.concatenate([drone_xyz_yaw[:-1], rpy, vel_local_frame, ang_vel, gate_corners, all_obstacles_local]).astype(np.float32)
        return obs

    def get_observation_space(self) -> Box:
        """Get the observation space.

        Returns:
            The observation space.
        """
        no_obstacles = len(self.config.quadrotor_config.obstacles)

        world_lower_bound = np.array(self.config.rl_config.world_lower_bound)
        world_upper_bound = np.array(self.config.rl_config.world_upper_bound)
        rpy_lower_bound = np.array([-np.pi, -np.pi])
        rpy_upper_bound = np.array([np.pi, np.pi])
        
        vel_bound = self.config.rl_config.vel_bound
        drone_vel_limits_upper = np.array([vel_bound, vel_bound, vel_bound])
        drone_vel_limits_lower = -drone_vel_limits_upper
        ang_vel_bound_lower = np.array([-np.inf, -np.inf])
        ang_vel_bound_upper = np.array([np.inf, np.inf])

        world_diff = world_upper_bound - world_lower_bound
        single_gate_limit_upper =np.tile(world_diff, 4)# 4 corners
        single_gate_limit_lower = -single_gate_limit_upper
        gate_limit_lower = np.tile(single_gate_limit_lower, 1)
        gate_limit_upper = np.tile(single_gate_limit_upper, 1)

        singe_obstacle_limit_lower = -world_diff
        singe_obstacle_limit_upper = world_diff
        obstacle_limit_lower = np.tile(singe_obstacle_limit_lower, no_obstacles)
        obstacle_limit_upper = np.tile(singe_obstacle_limit_upper, no_obstacles)

        obs_limit_low = np.concatenate([world_lower_bound, rpy_lower_bound, drone_vel_limits_lower, ang_vel_bound_lower, gate_limit_lower, obstacle_limit_lower])
        obs_limits_high = np.concatenate([world_upper_bound, rpy_upper_bound, drone_vel_limits_upper, ang_vel_bound_upper, gate_limit_upper, obstacle_limit_upper])

        return Box(obs_limit_low, obs_limits_high, dtype=np.float32)
    
class ObservationSpaceWrapperRelativeNextGoalNextObstacleRPRelativeVel(ObervationSpaceWrapper):
    """Implementation of Observation space wrapper.

    - Next goal and nearest obstacle are provided in the observation space in local frame of the drone.
    - Drone angles and velocities are provided (yaw excluded)
    - Velocity is provided in local frame
    """
    def __init__(self, config: Munch):
        """Create an observation space wrapper.

        Args:
            config: The configuration object.
        """
        super().__init__(config)

    def transform_observation(self, obs: np.ndarray, **kwargs: dict) -> np.ndarray:
        """Transform the given environment observation into an observation that can be used by the RL agent (NN).

        Args:
            obs (np.ndarray): The environment observation.
            kwargs: Additional keyword arguments.
        
        Returns:
            The transformed observation
        """
        drone_xyz_yaw = obs[0]
        gates_xyz_yaw = obs[1]
        obstacles_xyz = obs[3]
        current_gate_id = obs[5]

        estimated_vel = np.array(kwargs["estimated_velocity"])
        rpy = obs[6][:-1] # dont care about yaw
        ang_vel = obs[7][:-1] # dont care about yaw

        yaw = drone_xyz_yaw[-1]
        rot_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        vel_local_frame = (rot_matrix.T @ estimated_vel.reshape(3, 1)).flatten()
        
        gate_corners = convert_gate_to_corners(gates_xyz_yaw[current_gate_id], edge_length=EDGE_LENGTH)
        gate_corners = translate_points_in_local_frame(drone_xyz_yaw, gate_corners).flatten()
        

        all_obstacles_local = translate_points_in_local_frame(drone_xyz_yaw, obstacles_xyz)
        all_obstacles_norm = np.linalg.norm(all_obstacles_local, axis=1)
        closest_obstacle_idx = np.argmin(all_obstacles_norm)
        closest_obstacle = all_obstacles_local[closest_obstacle_idx]
        
        obs = np.concatenate([drone_xyz_yaw[:-1], rpy, vel_local_frame, ang_vel, gate_corners, closest_obstacle]).astype(np.float32)
        return obs

    def get_observation_space(self) -> Box:
        """Get the observation space.

        Returns:
            The observation space.
        """
        world_lower_bound = np.array(self.config.rl_config.world_lower_bound)
        world_upper_bound = np.array(self.config.rl_config.world_upper_bound)
        rpy_lower_bound = np.array([-np.pi, -np.pi])
        rpy_upper_bound = np.array([np.pi, np.pi])
        
        vel_bound = self.config.rl_config.vel_bound
        drone_vel_limits_upper = np.array([vel_bound, vel_bound, vel_bound])
        drone_vel_limits_lower = -drone_vel_limits_upper
        ang_vel_bound_lower = np.array([-np.inf, -np.inf])
        ang_vel_bound_upper = np.array([np.inf, np.inf])

        world_diff = world_upper_bound - world_lower_bound
        single_gate_limit_upper =np.tile(world_diff, 4)# 4 corners
        single_gate_limit_lower = -single_gate_limit_upper
        gate_limit_lower = np.tile(single_gate_limit_lower, 1)
        gate_limit_upper = np.tile(single_gate_limit_upper, 1)

        singe_obstacle_limit_lower = -world_diff
        singe_obstacle_limit_upper = world_diff
        obstacle_limit_lower = np.tile(singe_obstacle_limit_lower, 1)
        obstacle_limit_upper = np.tile(singe_obstacle_limit_upper, 1)

        obs_limit_low = np.concatenate([world_lower_bound, rpy_lower_bound, drone_vel_limits_lower, ang_vel_bound_lower, gate_limit_lower, obstacle_limit_lower])
        obs_limits_high = np.concatenate([world_upper_bound, rpy_upper_bound, drone_vel_limits_upper, ang_vel_bound_upper, gate_limit_upper, obstacle_limit_upper])

        return Box(obs_limit_low, obs_limits_high, dtype=np.float32)
    
class ObservationSpaceWrapperRelativeNextGoalNextObstacleRPRelativeVelNearestGate(ObervationSpaceWrapper):
    """Implementation of Observation space wrapper.

    - Next goal, nearest obstacle and nearest gate are provided in the observation space in local frame of the drone.
    - Drone angles and velocities are provided, excluding yaw
    - Velocity is provided in local frame
    """
    def __init__(self, config: Munch):
        """Create an observation space wrapper.

        Args:
            config: The configuration object.
        """
        super().__init__(config)

    def transform_observation(self, obs: np.ndarray, **kwargs: dict) -> np.ndarray:
        """Transform the given environment observation into an observation that can be used by the RL agent (NN).

        Args:
            obs (np.ndarray): The environment observation.
            kwargs: Additional keyword arguments.
        
        Returns:
            The transformed observation
        """
        drone_xyz_yaw = obs[0]
        gates_xyz_yaw = obs[1]
        obstacles_xyz = obs[3]
        current_gate_id = obs[5]

        estimated_vel = np.array(kwargs["estimated_velocity"])
        rpy = obs[6][:-1] # dont care about yaw
        ang_vel = obs[7][:-1] # dont care about yaw

        yaw = drone_xyz_yaw[-1]
        rot_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        vel_local_frame = (rot_matrix.T @ estimated_vel.reshape(3, 1)).flatten()
        
        gate_corners = convert_gate_to_corners(gates_xyz_yaw[current_gate_id], edge_length=EDGE_LENGTH)
        gate_corners = translate_points_in_local_frame(drone_xyz_yaw, gate_corners).flatten()
        

        all_obstacles_local = translate_points_in_local_frame(drone_xyz_yaw, obstacles_xyz)
        all_obstacles_norm = np.linalg.norm(all_obstacles_local, axis=1)
        closest_obstacle_idx = np.argmin(all_obstacles_norm)
        closest_obstacle = all_obstacles_local[closest_obstacle_idx]

        # find nearest gate to not collide
        gates_distances = np.linalg.norm(gates_xyz_yaw[:, :3] - drone_xyz_yaw[:3], axis=1)
        nearest_gate_idx = np.argmin(gates_distances)
        nearest_gate = gates_xyz_yaw[nearest_gate_idx]
        nearest_gate_corners = convert_gate_to_corners(nearest_gate, edge_length=EDGE_LENGTH)
        nearest_gate_corners = translate_points_in_local_frame(drone_xyz_yaw, nearest_gate_corners).flatten()
        
        obs = np.concatenate([drone_xyz_yaw[:-1], rpy, vel_local_frame, ang_vel, gate_corners, nearest_gate_corners, closest_obstacle]).astype(np.float32)
        return obs

    def get_observation_space(self) -> Box:
        """Get the observation space.

        Returns:
            The observation space.
        """
        world_lower_bound = np.array(self.config.rl_config.world_lower_bound)
        world_upper_bound = np.array(self.config.rl_config.world_upper_bound)
        rpy_lower_bound = np.array([-np.pi, -np.pi])
        rpy_upper_bound = np.array([np.pi, np.pi])
        
        vel_bound = self.config.rl_config.vel_bound
        drone_vel_limits_upper = np.array([vel_bound, vel_bound, vel_bound])
        drone_vel_limits_lower = -drone_vel_limits_upper
        ang_vel_bound_lower = np.array([-np.inf, -np.inf])
        ang_vel_bound_upper = np.array([np.inf, np.inf])

        world_diff = world_upper_bound - world_lower_bound
        single_gate_limit_upper =np.tile(world_diff, 4)# 4 corners
        single_gate_limit_lower = -single_gate_limit_upper
        gate_limit_lower = np.tile(single_gate_limit_lower, 1)
        gate_limit_upper = np.tile(single_gate_limit_upper, 1)

        singe_obstacle_limit_lower = -world_diff
        singe_obstacle_limit_upper = world_diff
        obstacle_limit_lower = np.tile(singe_obstacle_limit_lower, 1)
        obstacle_limit_upper = np.tile(singe_obstacle_limit_upper, 1)

        obs_limit_low = np.concatenate([world_lower_bound, rpy_lower_bound, drone_vel_limits_lower, ang_vel_bound_lower, gate_limit_lower, gate_limit_lower, obstacle_limit_lower])
        obs_limits_high = np.concatenate([world_upper_bound, rpy_upper_bound, drone_vel_limits_upper, ang_vel_bound_upper, gate_limit_upper, gate_limit_upper, obstacle_limit_upper])

        return Box(obs_limit_low, obs_limits_high, dtype=np.float32)


class ObservationSpaceWrapperRelativeAllGoalsAllObstacles(ObervationSpaceWrapper):
    """Implementation of Observation space wrapper.

    - All goals and all obstacles are provided in the observation space in local frame of the drone.
    """
    def __init__(self, config: Munch):
        """Create an observation space wrapper.

        Args:
            config: The configuration object.
        """
        super().__init__(config)

    def transform_observation(self, obs: np.ndarray, **kwargs: dict) -> np.ndarray:
        """Transform the given environment observation into an observation that can be used by the RL agent (NN).

        Args:
            obs (np.ndarray): The environment observation.
            kwargs: Additional keyword arguments.
        
        Returns:
            The transformed observation
        """
        drone_xyz_yaw = obs[0]
        gates_xyz_yaw = obs[1]
        obstacles_xyz = obs[3]
        current_gate_id = obs[5]

        estimated_vel = kwargs["estimated_velocity"]

        # Extract next gate to pass
        all_gate_corners = []
        for gate in gates_xyz_yaw:
            assert len(gate) == 4, "Gate must have 4 elements."
            gate_corners = convert_gate_to_corners(gate, edge_length=EDGE_LENGTH)
            gate_corners = translate_points_in_local_frame(drone_xyz_yaw, gate_corners).flatten()
            all_gate_corners.extend(gate_corners)
        

        all_obstacles_local = translate_points_in_local_frame(drone_xyz_yaw, obstacles_xyz).flatten()
        
        current_gate_id_one_hot = self._get_next_gate_one_hot(current_gate_id)
        obs = np.concatenate([drone_xyz_yaw[:-1], estimated_vel, all_gate_corners, all_obstacles_local, current_gate_id_one_hot]).astype(np.float32)

        return obs
    
    def get_observation_space(self) -> Box:
        """Get the observation space.

        Returns:
            The observation space.
        """
        no_gates = len(self.config.quadrotor_config.gates)
        no_obstacles = len(self.config.quadrotor_config.obstacles)

        world_lower_bound = np.array(self.config.rl_config.world_lower_bound)
        world_upper_bound = np.array(self.config.rl_config.world_upper_bound)
        vel_bound = self.config.rl_config.vel_bound
        drone_vel_limits_upper = np.array([vel_bound, vel_bound, vel_bound])
        drone_vel_limits_lower = -drone_vel_limits_upper

        world_diff = world_upper_bound - world_lower_bound
        single_gate_limit_upper =np.tile(world_diff, 4)# 4 corners
        single_gate_limit_lower = -single_gate_limit_upper
        gate_limit_lower = np.tile(single_gate_limit_lower, no_gates)
        gate_limit_upper = np.tile(single_gate_limit_upper, no_gates)

        singe_obstacle_limit_lower = -world_diff
        singe_obstacle_limit_upper = world_diff
        obstacle_limit_lower = np.tile(singe_obstacle_limit_lower, no_obstacles)
        obstacle_limit_upper = np.tile(singe_obstacle_limit_upper, no_obstacles)

        obs_limit_low = np.concatenate([world_lower_bound, drone_vel_limits_lower, gate_limit_lower, obstacle_limit_lower, np.zeros(no_gates)])
        obs_limits_high = np.concatenate([world_upper_bound, drone_vel_limits_upper, gate_limit_upper, obstacle_limit_upper, np.ones(no_gates)])

        return Box(obs_limit_low, obs_limits_high, dtype=np.float32)

    

class ObservationSpaceWrapperAbsoluteAllGoals(ObervationSpaceWrapper):
    """'Implementation of Observation space wrapper.

    - All goals are provided in the observation space in absolute coordinates.
    - No obstacles are provided
    - Velocity and Acceleration are provided
    """

    def __init__(self, config: Munch):
        """Create an observation space wrapper.

        Args:
            config: The configuration object.
        """
        super().__init__(config)

    def transform_observation(self, obs: np.ndarray, **kwargs: dict) -> np.ndarray:
        """Transform the given environment observation into an observation that can be used by the RL agent (NN).

        Args:
            obs (np.ndarray): The environment observation.
            kwargs: Additional keyword arguments.
        
        Returns:
            The transformed observation
        """
        drone_xyz_yaw = obs[0]
        gates_xyz_yaw = obs[1]
        current_gate_id = obs[5]

        estimated_vel = kwargs["estimated_velocity"]
        estimated_acc = kwargs["estimated_acceleration"]
        
        obs = np.concatenate([drone_xyz_yaw[:-1], estimated_vel, estimated_acc, gates_xyz_yaw.flatten(), [current_gate_id]]).astype(np.float32)
        return obs

    def get_observation_space(self) -> Box:
        """Get the observation space.

        Returns:
            The observation space.
        """
        no_gates = len(self.config.quadrotor_config.gates)

        world_lower_bound = np.array(self.config.rl_config.world_lower_bound)
        world_upper_bound = np.array(self.config.rl_config.world_upper_bound)
        vel_bound = self.config.rl_config.vel_bound
        drone_vel_limits_upper = np.array([vel_bound, vel_bound, vel_bound])
        drone_vel_limits_lower = -drone_vel_limits_upper
        acc_bound = self.config.rl_config.acc_bound
        drone_acc_limits_upper = np.array([acc_bound, acc_bound, acc_bound])
        drone_acc_limits_lower = -drone_acc_limits_upper
        gate_limit_lower = np.tile(np.concatenate([world_lower_bound, [-np.pi]]), no_gates)
        gate_limit_upper = np.tile(np.concatenate([world_upper_bound, [np.pi]]), no_gates)

        obs_limit_low = np.concatenate([world_lower_bound, drone_vel_limits_lower, drone_acc_limits_lower, gate_limit_lower, [-1]])
        obs_limits_high = np.concatenate([world_upper_bound, drone_vel_limits_upper, drone_acc_limits_upper, gate_limit_upper, [no_gates -1]]) # Todo, check if requires -1

        return Box(obs_limit_low, obs_limits_high, dtype=np.float32)
