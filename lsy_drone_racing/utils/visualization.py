import matplotlib.pyplot as plt
from typing import List
import numpy as np

def visualize_trajectories(trajectories: List[List[float]]):
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
    
        
