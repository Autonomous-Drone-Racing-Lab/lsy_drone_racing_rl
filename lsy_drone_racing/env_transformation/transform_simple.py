import numpy as np

from lsy_drone_racing.coordinate_transformation import convert_gate_to_corners, translate_points_in_local_frame

EDGE_LENGTH = 0.45

class Experiment_1_Environment_Transformation:

    @staticmethod
    def transform_observation(obs: np.ndarray, **kwargs):
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

        # Extract next gate to pass
        current_gate = gates_xyz_yaw[current_gate_id]
        current_gate_xyz_yaw = current_gate[:4]

        gate_corners = convert_gate_to_corners(current_gate_xyz_yaw, edge_length=EDGE_LENGTH)
        gate_corners = translate_points_in_local_frame(drone_xyz_yaw, gate_corners).flatten()  #Todo, update this

        obs = np.concatenate([drone_xyz_yaw[:-1], estimated_vel, estimated_acc, [drone_xyz_yaw[-1]], gate_corners]).astype(np.float32)
        return obs
    




