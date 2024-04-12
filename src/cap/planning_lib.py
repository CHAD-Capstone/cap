"""
Utilities for planning paths between the tags
Mostly just a wrapper around networkx TSP solver
"""

import networkx as nx
import numpy as np

from typing import Dict, Union

from cap.apriltag_pose_estimation_lib import AprilTagMap

def plan_path(tag_map: AprilTagMap, height_m: Union[Dict[int, float], float] = 1.0, start_tag_id: int = None, close_cycle=False, tsp=False):
    """
    Plans a path through the tag poses in the tag map
    Parameters:
        tag_map: AprilTagMap
            The map of the tag poses
        height_m: Union[Dict[int, float], float]
            The height of the drone above each tag in meters
            If a float, then the height is constant
            If a dictionary, then the height can be specified for each tag
        start_tag_id: int
            The tag id to start the path from
    """
    # Generate a list of tuples (tag_id, position) for the tags
    tag_positions = []
    for tag_id in tag_map.tag_ids():
        tag_pose = tag_map.get_pose(tag_id)  # [x, y, z, qx, qy, qz, qw]
        position = tag_pose[:3].copy()
        if isinstance(height_m, dict):
            position[2] += height_m[tag_id]
        else:
            position[2] += height_m
        tag_positions.append((tag_id, position))

    if tsp:
        # Create a graph with the tag positions as nodes
        start_node_index = 0
        G = nx.complete_graph(len(tag_positions))
        for i, (tag_id, position) in enumerate(tag_positions):
            G.nodes[i]["tag_id"] = tag_id
            G.nodes[i]["position"] = position
            if tag_id == start_tag_id:
                start_node_index = i

        # Compute the edge weights
        for i in range(len(tag_positions)):
            for j in range(i + 1, len(tag_positions)):
                position_i = G.nodes[i]["position"]
                position_j = G.nodes[j]["position"]
                distance = np.linalg.norm(np.array(position_i) - np.array(position_j))
                G.edges[i, j]["weight"] = distance

        # Compute the path
        path = nx.approximation.traveling_salesman_problem(G, cycle=True, weight="weight")

        if not close_cycle:
            # Remove the last node in the path, which is the same as the first node
            path = path[:-1]

        # Find the position of the start node in the path
        start_node_index_in_path = path.index(start_node_index)
        # Reorder the path so that the start node is first
        path = path[start_node_index_in_path:] + path[:start_node_index_in_path]

        # Extract the tag ids from the path
        tag_ids = [G.nodes[node_index]["tag_id"] for node_index in path]

        # Extract the positions from the path
        positions = [G.nodes[node_index]["position"] for node_index in path]
    else:
        # Then we just use a greedy algorithm
        tag_ids = []
        positions = []
        current_tag_id = start_tag_id
        current_position = None
        while len(tag_ids) < len(tag_positions):
            # Add the current tag to the path
            print(f"Length of tag_ids: {len(tag_ids)}. Total tags: {len(tag_positions)}")
            tag_ids.append(current_tag_id)
            positions.append(tag_map.get_pose(current_tag_id)[:3].copy())
            # Find the closest tag to the current tag
            closest_tag_id = None
            closest_distance = float("inf")
            for tag_id, position in tag_positions:
                if tag_id in tag_ids:
                    print(f"Skipping tag {tag_id}")
                    continue
                distance = np.linalg.norm(position - current_position) if current_position is not None else 0
                print(f"tag_id: {tag_id}, distance: {distance}")
                if distance < closest_distance:
                    closest_distance = distance
                    closest_tag_id = tag_id
            if closest_tag_id is None:
                break
            current_tag_id = closest_tag_id
            current_position = tag_map.get_pose(current_tag_id)[:3].copy()
            print(sorted(tag_ids))

    # Get the total distance of the path
    total_distance = 0
    for i in range(len(positions) - 1):
        position_i = positions[i]
        position_j = positions[i + 1]
        total_distance += np.linalg.norm(np.array(position_i) - np.array(position_j))

    return tag_ids, positions, total_distance

def get_closest_tag(current_pose: np.ndarray, tag_map: AprilTagMap, height_m: Union[Dict[int, float], float] = 1.0):
    """
    Finds the id in the tag map of the closest tag to the current pose

    Parameters:
        current_pose: np.ndarray
            The current pose of the drone. 4x4 homogenous transformation matrix
        tag_map: AprilTagMap
            The map of the tag poses
    """
    closest_tag_id = None
    closest_distance = float("inf")
    for tag_id in tag_map.tag_ids():
        tag_pose = tag_map.get_pose(tag_id)
        tag_position = tag_pose[:3].copy()
        if isinstance(height_m, dict):
            tag_position[2] += height_m[tag_id]
        else:
            tag_position[2] += height_m
        distance = np.linalg.norm(current_pose[:3, 3] - tag_position)
        print(f"{tag_id} - tag_position: {tag_position}, current_position: {current_pose[:3, 3]}, distance: {distance}")
        if distance < closest_distance:
            closest_distance = distance
            closest_tag_id = tag_id
    return closest_tag_id

if __name__ == "__main__":
    test_map = AprilTagMap()
    randomize_tag_positions = True
    if randomize_tag_positions:
        np.random.seed(42)
        for tag_id in range(50):
            position = np.random.rand(3)
            position[:2] *= 2
            # position[2] = (position[2] * 0.5) + 1
            position[2] = 0
            params = np.concatenate((position, np.zeros(4)))
            test_map.add_tag_pose(tag_id, params)
    else:
        tag_positions = [
            (0, [0, 0, 0]),
            (1, [1, 0, 0]),
            (2, [1.5, 1, 0]),
            (3, [0, 1, 0]),
            (4, [0.5, 0.5, 1]),
        ]
        for tag_id, position in tag_positions:
            params = np.concatenate((position, np.zeros(4)))
            test_map.add_tag_pose(tag_id, params)

    current_pose = np.eye(4)
    current_pose[:3, 3] = np.array([0.2, 0.4, 1.3])

    closest_tag_id = get_closest_tag(current_pose, test_map, height_m=1.0)
    print(f"closest_tag_id: {closest_tag_id}")

    tag_ids, positions, total_distance = plan_path(test_map, height_m=1.0, start_tag_id=closest_tag_id)
    print(f"tag_ids: {tag_ids}")
    print(f"positions: {positions}")
    print(f"total_distance: {total_distance}")

    import matplotlib.pyplot as plt
    # Plot the path in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the start position in green
    ax.scatter(current_pose[0, 3], current_pose[1, 3], current_pose[2, 3], c='g', marker='o')
    # Plot the line from the start position to the first tag
    ax.plot([current_pose[0, 3], positions[0][0]], [current_pose[1, 3], positions[0][1]], [current_pose[2, 3], positions[0][2]], 'g')
    for i in range(len(positions) - 1):
        position_i = positions[i]
        position_j = positions[i + 1]
        ax.plot([position_i[0], position_j[0]], [position_i[1], position_j[1]], [position_i[2], position_j[2]], 'b')
    ax.scatter([position[0] for position in positions], [position[1] for position in positions], [position[2] for position in positions], c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the aspect ratio of the plot to be equal
    max_range_x = np.array([position[0] for position in positions]).max() - np.array([position[0] for position in positions]).min()
    max_range_y = np.array([position[1] for position in positions]).max() - np.array([position[1] for position in positions]).min()
    max_range_z = np.array([position[2] for position in positions]).max() - np.array([position[2] for position in positions]).min()
    max_range = max(max_range_x, max_range_y, max_range_z, 1)
    mid_x = (np.array([position[0] for position in positions]).max() + np.array([position[0] for position in positions]).min()) * 0.5
    mid_y = (np.array([position[1] for position in positions]).max() + np.array([position[1] for position in positions]).min()) * 0.5
    mid_z = (np.array([position[2] for position in positions]).max() + np.array([position[2] for position in positions]).min()) * 0.5
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)

    plt.show()
