import gymnasium
import os
import numpy as np
from typing import Any, Dict

class TrajectoryTrackingWrapper(gymnasium.Wrapper):
    """
    Wrapper object. Tracks the trajectory of the drone and saves it to a file.

    Args:
        env: The environment to wrap.
        save_dir: The directory to save the trajectory to. If None, no file will be created.
        filename: The filename to save the trajectory to. If None, the filename will be "trajectory_tracking.txt".
        on_save_callback: Callback function that is called when the trajectory is saved. The callback function must accept a list of floats.
    """
    def __init__(self, env, save_dir=None, filename=None, on_save_callback=None):
        super(TrajectoryTrackingWrapper, self).__init__(env)
        if save_dir:
            self.log_to_file = True
            if filename:
                self.save_path = os.path.join(save_dir, filename)
            else:
                self.save_path = os.path.join(save_dir, "trajectory_tracking.txt")
        else:
            self.log_to_file = False
            assert filename is None, "Providing a filename without a save_dir does nto make sense."
        
        self.on_save_callback = on_save_callback

        self.trajectory_buffer = None
    
    def reset(self, seed= None):
        obs = self.env.reset(
            seed=seed
        )
        # append to file, if buffer is not empty
        if self.trajectory_buffer:
            trajectory_flattened = np.array(self.trajectory_buffer).flatten().tolist()
            if self.log_to_file:
                self.save_trajectory_to_file(trajectory_flattened)
            if self.on_save_callback:
                self.on_save_callback(trajectory_flattened)
        
        self.trajectory_buffer = []
        return obs

    def save_trajectory_to_file(self, traj_flattened):
        assert self.log_to_file
        assert self.save_path
        # write one trajectory as space separated list
        with open(self.save_path, "a") as f:
            f.write(" ".join(map(str, traj_flattened)))
            f.write("\n")
        

    def step(self, action: np.ndarray):
        transformed_obs, reward, terminated, truncated, info = self.env.step(action)
        pos = transformed_obs[:3]
        self.trajectory_buffer.append(pos)
        return transformed_obs, reward, terminated, truncated, info
    

