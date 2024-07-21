"""Visualize racing trajectories in 3d plot."""
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def visualize_trajectories(trajectories: List[List[float]]):
    """Visualize the given trajectories in a 3D plot.

    Args:
        trajectories (List[List[float]]): The trajectories to visualize.
    """
    # create 3d plot, with axis description, ...
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # plot all trajectories
    for trajectory in trajectories:
        traj_np = np.array(trajectory)
        x = traj_np[::3]
        y = traj_np[1::3]
        z = traj_np[2::3]
        ax.plot(x, y, z)
    
        
