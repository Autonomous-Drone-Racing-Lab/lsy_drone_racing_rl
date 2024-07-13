import numpy as np
import matplotlib.pyplot as plt
import pickle

class Experiment:
    def __init__(self):
        self.drone_obs = []
        self.time_step = []

    def add_drone_obs(self, drone_obs: np.ndarray):
        self.drone_obs.append(drone_obs)
    
    def add_time_step(self, time):
        self.time_step.append(time)
    
    def get_drone_pos(self):
        return np.array(self.drone_obs)
    
    # def plot_experiment(self, verbose=False):
    #     # we want 4x2 plots
    #     # col 1, row1: drone_x, traj_x over t
    #     # col 1, row2: drone_y, traj_y over t
    #     # col 1, row3: drone_z, traj_z over t
    #     # col 1, row4: difference(dronePos, traj_pos) over t
    #     # col 2, row1: traj_vel_x over t
    #     # col 2, row2: traj_vel_y over t
    #     # col 2, row3: traj_vel_z over t
        
        
    #     fig, axs = plt.subplots(4, 2)
    #     fig.suptitle('Experiment')
    #     t = self.get_traj_time()
    #     drone_pos = self.get_drone_pos()
    #     traj_pos = self.get_traj_pos()
    #     traj_vel = self.get_traj_vel()
    #     traj_acc = self.get_traj_acc()

    #     print(f"traj pos shape {traj_pos.shape}")
    #     print(f"drone pos shape {drone_pos.shape}")
    #     print(f"traj vel shape {traj_vel.shape}")
    #     print(f"traj acc shape {traj_acc.shape}")
        
    #     # drone_x, traj_x over t
    #     axs[0, 0].plot(t, drone_pos[:, 0], label='drone_x')
    #     axs[0, 0].plot(t, traj_pos[:, 0], label='traj_x')
    #     axs[0,0].title.set_text('X position')
    #     axs[0, 0].set_xlabel('t')  
    #     axs[0, 0].set_ylabel('x') 


    #     # drone_y, traj_y over t
    #     axs[1, 0].plot(t, drone_pos[:, 1], label='drone_y')
    #     axs[1, 0].plot(t, traj_pos[:, 1], label='traj_y')
    #     axs[1,0].title.set_text('Y position')
    #     axs[1, 0].set_xlabel('t')  
    #     axs[1, 0].set_ylabel('y')

    #     # drone_z, traj_z over t
    #     axs[2, 0].plot(t, drone_pos[:, 2], label='drone_z')
    #     axs[2, 0].plot(t, traj_pos[:, 2], label='traj_z')
    #     axs[2,0].title.set_text('Z position')
    #     axs[2, 0].set_xlabel('t')  
    #     axs[2, 0].set_ylabel('z')

    #     # difference(dronePos, traj_pos) over t
    #     difference = np.linalg.norm(drone_pos - traj_pos, axis=1)
    #     axs[3, 0].plot(t, difference)
    #     axs[3,0].title.set_text('Pos difference')
    #     axs[3, 0].set_xlabel('t')  
    #     axs[3, 0].set_ylabel('pos diff')

    #     # traj_vel_x over t
    #     axs[0, 1].plot(t, traj_vel[:, 0])
    #     axs[0,1].title.set_text('X velocity')
    #     axs[0, 1].set_xlabel('t')  
    #     axs[0, 1].set_ylabel('x_dot')

    #     # traj_vel_y over t
    #     axs[1, 1].plot(t,traj_vel[:, 1])
    #     axs[1,1].title.set_text('Y velocity')
    #     axs[1, 1].set_xlabel('t')  
    #     axs[1, 1].set_ylabel('x_dot')

    #     # traj_vel_z over t
    #     axs[2, 1].plot(t, traj_vel[:, 2])
    #     axs[2,1].title.set_text('Z velocity')
    #     axs[2, 1].set_xlabel('t')  
    #     axs[2, 1].set_ylabel('z_dot')

    #     # norm acc
    #     acc_norm = np.linalg.norm(traj_acc, axis=1)
    #     axs[3, 1].plot(t, acc_norm)
    #     axs[3,1].title.set_text('Acc Norm')
    #     axs[3, 1].set_xlabel('t')  
    #     axs[3, 1].set_ylabel('acc norm')

    #     plt.show()

class ExperimentTracker:
    def __init__(self):
        self.experiments = []
    
    def add_experiment(self):
        self.experiments.append(Experiment())

    def add_drone_obs(self, drone_obs: np.ndarray, time_step):
        self.experiments[-1].add_drone_obs(drone_obs)
        self.experiments[-1].add_time_step(time_step)

    def save_experiment(self, path):
        experiments = []
        for experiment in self.experiments:
            experiments.append({
                "drone_obs": experiment.drone_obs,
                "time_step": experiment.time_step
            })
        with open(path, 'wb') as f:
            pickle.dump(experiments, f)
        
        
