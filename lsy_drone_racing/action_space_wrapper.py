import numpy as np
from gymnasium.spaces import Box

from abc import ABC

class ActionSpaceWrapper(ABC):
    """
        Abstract base class for action space wrappers.

        An action space wrapper is an object which takes the action from the RL agent (NN) as input can conevrts it into an action that can be executed by the drone, i.e.
        [x, x_dot, x_ddot, y, y_dot, y_ddot, z, z_dot, z_ddot, yaw]
        """
    def __init__(self, config):
        self.config = config

    
    def scale_action(self, action: np.ndarray, drone_pose):
        """
        Scale the given action based on the drone's pose.

        Args:
            action (np.ndarray): The action to be scaled.
            drone_pose: The pose of the drone.

        Returns:
            The scaled action.
        """
        raise NotImplementedError

    def get_action_space(self):
        """
        Get the action space.

        Returns:
            The action space.
        """
        raise NotImplementedError

    def _fill_full_state_array(self, xyz: np.ndarray, yaw: float):
        """
        Fill the full state array with the given xyz coordinates and yaw angle.

        Args:
            xyz (np.ndarray): The xyz coordinates.
            yaw (float): The yaw angle.

        Returns:
            The filled full state array.
        """
        assert xyz.shape == (3,), "xyz must have 3 elements."

        full_state_action = np.zeros(14)
        full_state_action[:3] = xyz
        full_state_action[9] = yaw
        return full_state_action

def action_space_wrapper_factory(config):
    """
    Factory function to create an action space wrapper based on the configuration.
    """
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
    """
    Action Space Wrapper converting global coordinates in normed [-1,1] space to unnormliazed coordinates.
    """
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
    

class ActionSpaceWrapperXYZYaw(ActionSpaceWrapper):
    """
    Action Space Wrapper like (ActionSpaceWrapperXYZ). However, the action space includes a yaw action.
    Yaw is provided as normalized global [-1, 1] ) (no offset) and is scaled to the range [-pi, pi].
    """
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

class ActionSpaceWrapperXYZRelative(ActionSpaceWrapper):
    """
    Action space wrapper converting relative coordinates to global unnormalized coordinates.

    Relative coordinates are provided as normalized offsets [dx, dy, dz] in the range [-1, 1].
    Global actions are computed by adding offset to the current drone position unnormalizing based on an action range.
    """
    def __init__(self, config):
        super().__init__(config)
    
    # def scale_action(self, action: np.ndarray, drone_pose):
    #     assert action.shape == (3,), "Action must have 3 elements."
    #     assert drone_pose.shape == (4,), "Drone pose must have 4 elements."

    #     scale_factor = self.config.rl_config.action_bound
    #     scaled_xyz = action * scale_factor

    #     scaled_space = np.concatenate([scaled_xyz, [0]])
    #     return drone_pose + scaled_space
    
    def scale_action(self, action: np.ndarray, drone_pose):
        assert action.shape == (3,), "Action must have 3 elements."
        assert drone_pose.shape == (4,), "Drone pose must have 4 elements."

        scale_factor = self.config.rl_config.action_bound
        drone_pos = drone_pose[:3]
        yaw = drone_pose[3]
        
        rot_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        action_global_frame = (rot_matrix @ action.reshape(-1, 1)).flatten()
        scaled_xyz = action_global_frame * scale_factor
        scaled_space = np.concatenate([scaled_xyz, [0]])

        return drone_pose + scaled_space


    def get_action_space(self):
        action_limits = np.ones(3)
        return Box(low=-action_limits, high=action_limits, dtype=np.float32)
    
class ActionSpaceWrapperXYZRelativeYaw(ActionSpaceWrapper):
    """
    Relative ActionSpaceWrapper like (ActionSpaceWrapperXYZRelative). However, the action space includes a yaw action.
    Yaw is provided as normalized global [-1, 1] ) (no offset) and is scaled to the range [-pi, pi].
    """
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
