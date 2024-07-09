import numpy as np
import math

def progress_reward(current_drone_pose, prev_drone_pose, next_gate_pose):
    current_drone_pos = current_drone_pose[:3]
    prev_drone_pos = prev_drone_pose[:3]
    next_gate_pos = next_gate_pose[:3]

    prev_dist = np.linalg.norm(prev_drone_pos - next_gate_pos)
    current_dist = np.linalg.norm(current_drone_pos - next_gate_pos)

    reward = prev_dist - current_dist
    return reward


def smooth_action_reward(current_action, prev_action):
    if prev_action is None:
        return 0
    action_difference = np.linalg.norm(current_action - prev_action)

    return -(action_difference ** 2) 

def state_limits_exceeding_penalty(state, desirable_max_state):
    if all(state < desirable_max_state):
        return 0
    
    per_element_difference = np.maximum(state - desirable_max_state, 0)
    difference = np.linalg.norm(per_element_difference)
    return - math.exp(difference)

def distance_reward(current_drone_pose, to_be_pos):
    current_drone_pos = current_drone_pose[:3]
    distance = np.linalg.norm(current_drone_pos - to_be_pos)
    reward = 0
    if distance < 0.1:
        reward = 10
    elif distance < 0.3:
        reward = 5
    elif distance < 0.5:
        reward = 2
    elif distance < 1:
        reward = 1

    return reward / 10

def safety_reward(current_drone_pose, next_gate_pose):
    EDGE_LENGTH = 0.45
    gate_center = next_gate_pose[:3]
    dist_to_gate = np.linalg.norm(current_drone_pose[:3] - gate_center)

    yaw_rot = next_gate_pose[3]
    normal_x = -np.sin(yaw_rot)
    normal_y = np.cos(yaw_rot)
    normal_z = 0
    normal = np.array([normal_x, normal_y, normal_z])
    
    # find drone distance to normal
    p1 = gate_center
    p2 = gate_center + normal
    p3 = current_drone_pose[:3]

    distance_to_normal = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

    max_dist = 0.10 # activate reward when drone is within 0.5m of the gate
    f = max(0, 1 - distance_to_normal / max_dist)
    v = max((1-f) * (EDGE_LENGTH / 6), 0.05)

    reward = - (f**2) * (1- math.exp(- (0.5 * dist_to_gate**2) / v))
    return reward

        

if __name__ == "__main__":
    print("Testin state exceeding penalty")
    state_limitations = np.array([5,5,5])
    state = np.array([4,4,4])
    print(state_limits_exceeding_penalty(state, state_limitations))
    state = np.array([6,6,6])
    print(state_limits_exceeding_penalty(state, state_limitations))