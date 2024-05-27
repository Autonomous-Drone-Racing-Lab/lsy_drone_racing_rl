import numpy as np

class Normalizer:
    def __init__(self):
        self.world_lower_bound = np.array([-2, -2, 0])
        self.world_upper_bound = np.array([2, 2, 2])
    
    def normalize_world_xyz(self, world_xyz: np.ndarray):
        """
        Normalize 3d coordinates to be within -1 and 1
        """
        normalized_xyz = (world_xyz - self.world_lower_bound) / (self.world_upper_bound - self.world_lower_bound) * 2 - 1

        return normalized_xyz
    
    def denormalize_world_xyz(self, normalized_xyz: np.ndarray):
        """
        Denormalize 3d coordinates from -1 to 1 to world
        """
        world_xyz = (normalized_xyz + 1) / 2 * (self.world_upper_bound - self.world_lower_bound) + self.world_lower_bound

        return world_xyz
    

    

