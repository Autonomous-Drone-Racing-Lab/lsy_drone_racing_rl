import numpy as np

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
        