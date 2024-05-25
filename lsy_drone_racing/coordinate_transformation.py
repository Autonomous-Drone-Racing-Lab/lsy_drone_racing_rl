import numpy as np

def convert_gate_to_corners(gate: np.ndarray, edge_length: float):
    """
    Convert a gate to its four corners.

    :param gate: [x, y, z, yaw] of the gate
    :param gate_geometry: dict with keys 'edge', i.e. width and 'height' of the gate
    :return: np.ndarray of shape (4, 3) representing the four corners of the gate
    """
   
    x, y, height, yaw = gate
   
    half_size = edge_length / 2
    corners = np.array([
        [-half_size, 0, height - half_size],
        [-half_size, 0, height + half_size],
        [half_size, 0, height + half_size],
        [half_size, 0, height - half_size]
    ])

    # rotation matrix for yaw (radias)
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # rotate the corners
    corners = (R @ corners.T).T
    corners += np.array([x, y, 0])

    return corners

def translate_points_in_local_frame(local_frame: np.ndarray, points: np.ndarray):
    """
    Convert list of points into the local frame

    :param local_frame: [x, y, z, yaw] of the local frame
    :param points: np.ndarray of shape (n, 3) representing the points
    :return: np.ndarray of shape (n, 3) representing the points in the local frame
    """

    frame_center = local_frame[:3]
    yaw = local_frame[3]
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    points = points - frame_center # Todo check, wheter broadcasting is correct
    points = (R.T @ points.T).T

    return points

def visualize_points(points: np.ndarray, axis):
    """
    Visualize points in the PyBullet environment.

    :param points: np.ndarray of shape (n, 3) representing the points
    :param axis: matplotlib axis
    """

    for point in points:
        axis.scatter(*point, c='r')


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rotation = 0
    gate = np.array([0, 0, 0, rotation])
    gate_geometry = {"edge": 0.5, "height": 1}
    corners = convert_gate_to_corners(gate, gate_geometry)
    print(corners)
    
    rotation = np.pi / 4
    gate = np.array([0, 0, 0, rotation])
    gate_geometry = {"edge": 0.5, "height": 1}
    corners = convert_gate_to_corners(gate, gate_geometry)
    print(corners)

   

    frame = np.array([0, 0, 0, rotation])
    corners_local = translate_points_in_local_frame(frame, corners)
    print(corners_local)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    visualize_points(corners, ax)

    # add axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

