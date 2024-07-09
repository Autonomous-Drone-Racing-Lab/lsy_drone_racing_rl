import numpy as np
from gymnasium.spaces import Box

from abc import ABC

class ActionSpaceWrapper(ABC):
    def __init__(self, config):
        self.config = config
    
    def scale_action(self, action: np.ndarray, drone_pose):
        raise NotImplementedError

    def get_action_space(self):
        raise NotImplementedError

    def _fill_full_state_array(self, xyz: np.ndarray, yaw: float):
        assert xyz.shape == (3,), "xyz must have 3 elements."

        full_state_action = np.zeros(14)
        full_state_action[:3] = xyz
        full_state_action[9] = yaw
        return full_state_action

def action_space_wrapper_factory(config):
    action_space = config.rl_config.action_space
    if action_space == "xyz":
        return ActionSpaceWrapperXYZ(config)
    elif action_space == "xyz_relative":
        return ActionSpaceWrapperXYZRelative(config)
    elif action_space == "xyz_relative_yaw":
        return ActionSpaceWrapperXYZRelativeYaw(config)
    elif action_space == "xyz_yaw":
        return ActionSpaceWrapperXYZYaw(config)
    else:
        raise ValueError(f"Action space {action_space} not supported.")
    
class ActionSpaceWrapperXYZ(ActionSpaceWrapper):
    def __init__(self, config):
        super().__init__(config)
    
    def scale_action(self, action: np.ndarray, drone_pose):
        assert action.shape == (3,), "Action must have 3 elements."
        
        scale_factor = self.config.rl_config.action_bound
        scaled_xyz = action * scale_factor

        scaled_space = np.concatenate([scaled_xyz, [0]])

        return scaled_space
        
    
    def get_action_space(self):
        action_limits = np.ones(3)
        return Box(low=-action_limits, high=action_limits, dtype=np.float32)

class ActionSpaceWrapperXYZRelative(ActionSpaceWrapper):
    def __init__(self, config):
        super().__init__(config)
    
    def scale_action(self, action: np.ndarray, drone_pose):
        assert action.shape == (3,), "Action must have 3 elements."
        assert drone_pose.shape == (4,), "Drone pose must have 4 elements."

        scale_factor = self.config.rl_config.action_bound
        scaled_xyz = action * scale_factor

        scaled_space = np.concatenate([scaled_xyz, [0]])
        return drone_pose + scaled_space
    
    # def scale_action(self, action: np.ndarray, drone_pose):
    #     assert action.shape == (3,), "Action must have 3 elements."
    #     assert drone_pose.shape == (4,), "Drone pose must have 4 elements."

    #     scale_factor = self.config.rl_config.action_bound
    #     drone_pos = drone_pose[:3]
    #     yaw = drone_pose[3]
        
    #     rot_matrix = np.array([
    #         [np.cos(yaw), -np.sin(yaw), 0],
    #         [np.sin(yaw), np.cos(yaw), 0],
    #         [0, 0, 1]
    #     ])

    #     action_global_frame = (rot_matrix @ action.reshape(-1, 1)).flatten()
    #     scaled_xyz = action_global_frame * scale_factor
    #     scaled_space = np.concatenate([scaled_xyz, [0]])
    #     #print(scaled_space.shape)
        

    #     return drone_pose + scaled_space


    def get_action_space(self):
        action_limits = np.ones(3)
        return Box(low=-action_limits, high=action_limits, dtype=np.float32)
    
class ActionSpaceWrapperXYZRelativeYaw(ActionSpaceWrapper):
    def __init__(self, config):
        super().__init__(config)
    
    def scale_action(self, action: np.ndarray, drone_pose):
        assert action.shape == (4,), "Action must have 4 elements [x,y,z,yaw]."
        assert drone_pose.shape == (4,), "Drone pose must have 4 elements."

        scale_factor = self.config.rl_config.action_bound
        scaled_xyz = action[0:3] * scale_factor
        drone_xyz_shifted = drone_pose[0:3] + scaled_xyz

        yaw = action[3] * np.pi
        # apply yaw rotation relative to current rotation, i.e. add the yaw action to the current yaw
        # however we need to make sure that the yaw is within the range of [-pi, pi]
        yaw_shifted = drone_pose[3] + yaw
        yaw_shifted = (yaw_shifted + np.pi) % (2 * np.pi) - np.pi # wrap yaw to [-pi, pi]

        return np.concatenate([drone_xyz_shifted, [yaw_shifted]])
    
    def get_action_space(self):
        action_limits = np.ones(4)
        return Box(low=-action_limits, high=action_limits, dtype=np.float32)

class ActionSpaceWrapperXYZYaw(ActionSpaceWrapper):
    def __init__(self, config):
        super().__init__(config)

    def scale_action(self, action: np.ndarray, drone_pose):
        assert action.shape == (4,), "Action must have 4 elements."
        scale_factor = self.config.rl_config.action_bound
        scaled_xyz = action[:3] * scale_factor
        scaled_phi = action[3] * np.pi

        scaled_space = np.concatenate([scaled_xyz, [scaled_phi]])
        print(f"Action space: {scaled_space}")

        return scaled_space
    
    def get_action_space(self):
        action_limits = np.ones(4)
        return Box(low=-action_limits, high=action_limits, dtype=np.float32)