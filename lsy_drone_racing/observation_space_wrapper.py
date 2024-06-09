import numpy as np

from lsy_drone_racing.coordinate_transformation import convert_gate_to_corners, translate_points_in_local_frame
from gymnasium.spaces import Box
from abc import ABC

EDGE_LENGTH = 0.45

class ObervationSpaceWrapper(ABC):
    def __init__(self, config):
        self.config = config
    
    def transform_observation(self, obs: np.ndarray, **kwargs):
        raise NotImplementedError

    def _get_next_gate_one_hot(self, current_gate_id):
        no_gates = len(self.config.quadrotor_config.gates)
        one_hot = np.zeros(no_gates)
        if current_gate_id != -1:
            one_hot[current_gate_id] = 1
        return one_hot
    
    def get_observation_space(self):
        raise NotImplementedError
    

def observation_space_wrapper_factory(config):
    observation_space = config.rl_config.observation_space
    if observation_space == "relative_only_next_goal":
        return ObservationSpaceWrapperRelativeOnlyNextGoal(config)
    elif observation_space == "relative_all_goals":
        return ObservationSpaceWrapperRelativeAllGoals(config)
    elif observation_space == "relative_all_goals_no_acc":
        return ObservationSpaceWrapperRelativeAllGoalsNoAcc(config)
    elif observation_space == "absolute_all_goals":
        return ObservationSpaceWrapperAbsoluteAllGoals(config)
    elif observation_space == "relative_all_goals_all_obstacles_no_acc":
        return ObservationSpaceWrapperRelativeAllGoalsAllObstaclesNoAcc(config)
    else:
        raise ValueError(f"Observation space {observation_space} not supported.")
    
       
class ObservationSpaceWrapperRelativeOnlyNextGoal(ObervationSpaceWrapper):
    def __init__(self, config):
        super().__init__(config)

    def transform_observation(self, obs: np.ndarray, **kwargs):
        """
        : param obs: Observation space, i.e. [drone_xyz_yaw, gates_xyz_yaw, gates_in_range, obstacles_xyz, obstacles_in_range, gate_id], see wrapper for definition
        : returns observation [drone_xyz, drone_vel_xyz, drone_acc_xyz, yaw, cornes of next gate in local frame of drone]
        """
        drone_xyz_yaw = obs[0]
        gates_xyz_yaw = obs[1]
        gates_in_range = obs[2]
        obstacles_xyz = obs[3]
        obstacles_in_range = obs[4]
        current_gate_id = obs[5]

        estimated_vel = kwargs["estimated_velocity"]
        estimated_acc = kwargs["estimated_acceleration"]

        
        current_gate = gates_xyz_yaw[current_gate_id]
        current_gate_xyz_yaw = current_gate[:4]
        gate_corners = convert_gate_to_corners(current_gate_xyz_yaw, edge_length=EDGE_LENGTH)
        gate_corners = translate_points_in_local_frame(drone_xyz_yaw, gate_corners).flatten()  #Todo, update this

        obs = np.concatenate([drone_xyz_yaw[:-1], estimated_vel, estimated_acc, [drone_xyz_yaw[-1]], gate_corners]).astype(np.float32)
       # obs = np.concatenate([drone_xyz_yaw[:-1], estimated_vel, estimated_acc, [drone_xyz_yaw[-1]], all_gate_corners, [current_gate_id]]).astype(np.float32)
        return obs

    def get_observation_space(self):
        world_lower_bound = np.array(self.config.rl_config.world_lower_bound)
        world_upper_bound = np.array(self.config.rl_config.world_upper_bound)
        vel_bound = self.config.rl_config.vel_bound
        drone_vel_limits_upper = np.array([vel_bound, vel_bound, vel_bound])
        drone_vel_limits_lower = -drone_vel_limits_upper
        acc_bound = self.config.rl_config.acc_bound
        drone_acc_limits_upper = np.array([acc_bound, acc_bound, acc_bound])
        drone_acc_limits_lower = -drone_acc_limits_upper
        drone_yaw_limits_upper = np.array([np.pi])
        drone_yaw_limits_lower = -drone_yaw_limits_upper

        max_difference = world_upper_bound - world_lower_bound
        gate_upper_limits = np.concatenate([max_difference, max_difference, max_difference, max_difference]) # 4 corners
        gate_lower_limits = -gate_upper_limits

        obs_limit_low = np.concatenate([world_lower_bound, drone_vel_limits_lower, drone_acc_limits_lower, drone_yaw_limits_lower,  gate_lower_limits])
        obs_limits_high = np.concatenate([world_upper_bound, drone_vel_limits_upper, drone_acc_limits_upper, drone_yaw_limits_upper, gate_upper_limits])

        return Box(obs_limit_low, obs_limits_high, dtype=np.float32)   

class ObservationSpaceWrapperRelativeAllGoals(ObervationSpaceWrapper):
    def __init__(self, config):
        super().__init__(config)

    def transform_observation(self, obs: np.ndarray, **kwargs):
        drone_xyz_yaw = obs[0]
        gates_xyz_yaw = obs[1]
        gates_in_range = obs[2]
        obstacles_xyz = obs[3]
        obstacles_in_range = obs[4]
        current_gate_id = obs[5]

        estimated_vel = kwargs["estimated_velocity"]
        estimated_acc = kwargs["estimated_acceleration"]

        # Extract next gate to pass
        all_gate_corners = []
        for gate in gates_xyz_yaw:
            assert len(gate) == 4, "Gate must have 4 elements."
            gate_corners = convert_gate_to_corners(gate, edge_length=EDGE_LENGTH)
            gate_corners = translate_points_in_local_frame(drone_xyz_yaw, gate_corners).flatten()
            all_gate_corners.extend(gate_corners)
        
        current_gate_id_one_hot = self._get_next_gate_one_hot(current_gate_id)
        obs = np.concatenate([drone_xyz_yaw[:-1], estimated_vel, estimated_acc, all_gate_corners, current_gate_id_one_hot]).astype(np.float32)

        return obs
    
    def get_observation_space(self):
        no_gates = len(self.config.quadrotor_config.gates)

        world_lower_bound = np.array(self.config.rl_config.world_lower_bound)
        world_upper_bound = np.array(self.config.rl_config.world_upper_bound)
        vel_bound = self.config.rl_config.vel_bound
        drone_vel_limits_upper = np.array([vel_bound, vel_bound, vel_bound])
        drone_vel_limits_lower = -drone_vel_limits_upper
        acc_bound = self.config.rl_config.acc_bound
        drone_acc_limits_upper = np.array([acc_bound, acc_bound, acc_bound])
        drone_acc_limits_lower = -drone_acc_limits_upper

        world_diff = world_upper_bound - world_lower_bound
        single_gate_limit_upper =np.tile(world_diff, 4)# 4 corners
        single_gate_limit_lower = -single_gate_limit_upper
        gate_limit_lower = np.tile(single_gate_limit_lower, no_gates)
        gate_limit_upper = np.tile(single_gate_limit_upper, no_gates)

        obs_limit_low = np.concatenate([world_lower_bound, drone_vel_limits_lower, drone_acc_limits_lower, gate_limit_lower, np.zeros(no_gates)])
        obs_limits_high = np.concatenate([world_upper_bound, drone_vel_limits_upper, drone_acc_limits_upper, gate_limit_upper, np.ones(no_gates)])

        return Box(obs_limit_low, obs_limits_high, dtype=np.float32)

class ObservationSpaceWrapperRelativeAllGoalsNoAcc(ObervationSpaceWrapper):
    def __init__(self, config):
        super().__init__(config)

    def transform_observation(self, obs: np.ndarray, **kwargs):
        drone_xyz_yaw = obs[0]
        gates_xyz_yaw = obs[1]
        gates_in_range = obs[2]
        obstacles_xyz = obs[3]
        obstacles_in_range = obs[4]
        current_gate_id = obs[5]

        estimated_vel = kwargs["estimated_velocity"]

        # Extract next gate to pass
        all_gate_corners = []
        for gate in gates_xyz_yaw:
            assert len(gate) == 4, "Gate must have 4 elements."
            gate_corners = convert_gate_to_corners(gate, edge_length=EDGE_LENGTH)
            gate_corners = translate_points_in_local_frame(drone_xyz_yaw, gate_corners).flatten()
            all_gate_corners.extend(gate_corners)
        
        current_gate_id_one_hot = self._get_next_gate_one_hot(current_gate_id)
        obs = np.concatenate([drone_xyz_yaw[:-1], estimated_vel, all_gate_corners, current_gate_id_one_hot]).astype(np.float32)

        return obs
    
    def get_observation_space(self):
        no_gates = len(self.config.quadrotor_config.gates)

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

        obs_limit_low = np.concatenate([world_lower_bound, drone_vel_limits_lower, gate_limit_lower, np.zeros(no_gates)])
        obs_limits_high = np.concatenate([world_upper_bound, drone_vel_limits_upper, gate_limit_upper, np.ones(no_gates)])

        return Box(obs_limit_low, obs_limits_high, dtype=np.float32)

class ObservationSpaceWrapperRelativeAllGoalsAllObstaclesNoAcc(ObervationSpaceWrapper):
    def __init__(self, config):
        super().__init__(config)

    def transform_observation(self, obs: np.ndarray, **kwargs):
        drone_xyz_yaw = obs[0]
        gates_xyz_yaw = obs[1]
        gates_in_range = obs[2]
        obstacles_xyz = obs[3]
        obstacles_in_range = obs[4]
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
    
    def get_observation_space(self):
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

    def __init__(self, config):
        super().__init__(config)

    def transform_observation(self, obs:np.ndarray, **kwargs):
        drone_xyz_yaw = obs[0]
        gates_xyz_yaw = obs[1]
        gates_in_range = obs[2]
        obstacles_xyz = obs[3]
        obstacles_in_range = obs[4]
        current_gate_id = obs[5]

        estimated_vel = kwargs["estimated_velocity"]
        estimated_acc = kwargs["estimated_acceleration"]
        
        obs = np.concatenate([drone_xyz_yaw[:-1], estimated_vel, estimated_acc, gates_xyz_yaw.flatten(), [current_gate_id]]).astype(np.float32)
        return obs

    def get_observation_space(self):
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


        

    
