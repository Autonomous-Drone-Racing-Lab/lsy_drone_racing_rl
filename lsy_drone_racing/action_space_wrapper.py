import numpy as np
from gymnasium.spaces import Box

from abc import ABC

class ActionSpaceWrapper(ABC):
    def __init__(self, config):
        self.config = config
    
    def scale_action(self, action: np.ndarray):
        raise NotImplementedError
    
    def _scale_xyz(self, xyz: np.ndarray):
        """
        scale xyz action limit from range [-1, 1]x[-1, 1]x[-1, 1] to world bounds

        For example scale from [-1, 1]x[-1, 1]x[-1, 1] to [-2, 2]x[-2, 2]x[0, 2]
        """
        world_lower_bound = np.array(self.config.rl_config.world_lower_bound)
        world_upper_bound = np.array(self.config.rl_config.world_upper_bound)
        assert len(world_lower_bound) == len(world_upper_bound) == 3, "World bounds must have 3 elements each."

        scale = (world_upper_bound - world_lower_bound) / 2
        offset = (world_upper_bound + world_lower_bound) / 2
        coordinate_trnasformed = xyz * scale + offset
        print(f"Scale: {scale}, Offset: {offset}")
        print(f"Transformed coordinate from {xyz} to {coordinate_trnasformed}")
        return coordinate_trnasformed

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
    elif action_space == "xyz_yaw":
        return ActionSpaceWrapperXYZYaw(config)
    else:
        raise ValueError(f"Action space {action_space} not supported.")
    
class ActionSpaceWrapperXYZ(ActionSpaceWrapper):
    def __init__(self, config):
        super().__init__(config)
    
    def scale_action(self, action: np.ndarray):
        assert action.shape == (3,), "Action must have 3 elements."
        scaled_xyz = self._scale_xyz(action)

        return self._fill_full_state_array(scaled_xyz, 0.0)
        
    
    def get_action_space(self):
        action_limits = np.ones(3)
        return Box(low=-action_limits, high=action_limits, dtype=np.float32)
    

class ActionSpaceWrapperXYZYaw(ActionSpaceWrapper):
    def __init__(self, config):
        super().__init__(config)

    def scale_action(self, action: np.ndarray):
        assert action.shape == (4,), "Action must have 4 elements."
        scaled_xyz = self._scale_xyz(action[:3])

        return self._fill_full_state_array(scaled_xyz, action[3])
    
    def get_action_space(self):
        action_limits = np.ones(4)
        return Box(low=-action_limits, high=action_limits, dtype=np.float32)