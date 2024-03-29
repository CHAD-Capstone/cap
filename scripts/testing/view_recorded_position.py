"""
Can be run anywhere. Does not require ROS.
"""

from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

from cap.apriltag_pose_estimation_lib import AprilTagMap
from cap.data_lib import FLIGHT_DATA_DIR, load_current_flight_tag_map
from cap.transformation_lib import params_to_matrix

class Viewer:
    """
    A class for visualizing extrinsics calibration data
    It can visualize two different types of objects:
    1. Frames: A frame is passed in as a 4x4 homogeneous transformation matrix and is visualized as an x, y, z axis
      The x axis is red, the y axis is green, and the z axis is blue
    2. Tag: A tag is a defined by both a pose (4x4 transformation matrix) and a size (in meters)
      The frame is also visualized as an x, y, z axis, but we also put a square on the x, y plane to represent the tag
      The square is centered at the origin of the tag frame and has a side length equal to the size of the tag
    """
    def __init__(self, size):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Set the y, z, and x limits
        self.ax.set_xlim(-size, size)
        self.ax.set_ylim(-size, size)
        self.ax.set_zlim(0, size)

        # Set us to be looking from above along the +x axis
        self.ax.view_init(90, 180)

    def add_frame(self, T, label=""):
        """
        Add a frame to the plot
        """
        if T.shape[0] == 7:
            T = params_to_matrix(T, type="quaternion")
        # Extract the origin of the frame
        origin = T[:3, 3]
        # Extract the axes of the frame
        x_axis = T[:3, 0] / 2
        y_axis = T[:3, 1] / 2
        z_axis = T[:3, 2] / 2
        # Plot the axes
        self.ax.quiver(*origin, *x_axis, color='r')
        self.ax.quiver(*origin, *y_axis, color='g')
        self.ax.quiver(*origin, *z_axis, color='b')
        # Add a label
        self.ax.text(*origin, label)

    def add_tag(self, T, size, label=""):
        """
        Add a tag to the plot
        """
        # Extract the origin of the tag
        origin = T[:3, 3]
        # Extract the axes of the tag
        x_axis = T[:3, 0] / 2
        y_axis = T[:3, 1] / 2
        z_axis = T[:3, 2] / 2
        # Plot the axes
        self.ax.quiver(*origin, *x_axis, color='r')
        self.ax.quiver(*origin, *y_axis, color='g')
        self.ax.quiver(*origin, *z_axis, color='b')
        # Plot the tag
        self.ax.plot([origin[0] - size/2, origin[0] + size/2], [origin[1] - size/2, origin[1] - size/2], [origin[2]]*2, color='k')
        self.ax.plot([origin[0] - size/2, origin[0] + size/2], [origin[1] + size/2, origin[1] + size/2], [origin[2]]*2, color='k')
        self.ax.plot([origin[0] - size/2]*2, [origin[1] - size/2, origin[1] + size/2], [origin[2]]*2, color='k')
        self.ax.plot([origin[0] + size/2]*2, [origin[1] - size/2, origin[1] + size/2], [origin[2]]*2, color='k')
        # Add a label
        self.ax.text(*origin, label)

    def add_waypoint(self, T, label=""):
        """
        Plots a waypoint as a red circle larger than the frame axes
        """
        origin = T[:3, 3]
        self.ax.scatter(*origin, color='r', s=10)
        self.ax.text(*origin, label)

    def add_path(self, timestamps, Ts, label="", color='b'):
        """
        Plots a path of frames. By default only plots positions and disregards frame axes and timestamps
        """
        positions = [T[:3, 3] for T in Ts]
        x = [p[0] for p in positions]
        y = [p[1] for p in positions]
        z = [p[2] for p in positions]
        self.ax.plot(x, y, z, label=label, color=color)

    def show(self):
        self.ax.legend()
        plt.show()

def load_flight_data(flight_data_file: Path):
    """
    Returns an object
    flight_data = {
        'setpoint_position_local': [],
        'vicon': [],
        'local_position': [],
        'corrected_position': []
    }
    The elements of each array are tuples of (timestamp_s, pose_params [x, y, z, qx, qy, qz, qw])
    """
    flight_data = np.load(flight_data_file, allow_pickle=True).item()
    return flight_data

def graph_flight_data(flight_data, tag_map: AprilTagMap):
    # Process the flight data
    setpoint_data = flight_data['setpoint_position_local']
    uncorrected_setpoint_data = flight_data['setpoint_uncorrected'] if 'setpoint_uncorrected' in flight_data else None
    vicon_data = flight_data['vicon']
    local_data = flight_data['local_position']
    corrected_data = flight_data['corrected_position']

    frame_transform_data = flight_data['frame_transform']

    vicon_position_timestamps = np.array([pose[0] for pose in vicon_data])
    print(f"VICON Timestamp Min: {vicon_position_timestamps.min()}. Max: {vicon_position_timestamps.max()}")
    local_position_timestamps = np.array([pose[0] for pose in local_data])
    print(f"Local Timestamp Min: {local_position_timestamps.min()}. Max: {local_position_timestamps.max()}")

    start_time_s = 0
    data_start_time = start_time_s + vicon_position_timestamps.min()

    setpoint_positions = np.array([params_to_matrix(pose[1], type="quaternion") for pose in setpoint_data if pose[0] > data_start_time])
    uncorrected_setpoint_positions = np.array([params_to_matrix(pose[1], type="quaternion") for pose in uncorrected_setpoint_data if pose[0] > data_start_time]) if uncorrected_setpoint_data is not None else None
    vicon_positions = np.array([params_to_matrix(pose[1], type="quaternion") for pose in vicon_data if pose[0] > data_start_time])
    local_positions = np.array([params_to_matrix(pose[1], type="quaternion") for pose in local_data if pose[0] > data_start_time])
    corrected_positions = np.array([params_to_matrix(pose[1], type="quaternion") for pose in corrected_data if pose[0] > data_start_time])

    frame_transforms = np.array([params_to_matrix(pose[1], type="quaternion") for pose in frame_transform_data if pose[0] > data_start_time])

    viz = Viewer(size=2)

    # Plot the apriltags
    if tag_map is not None:
        for tag_id in tag_map.tag_ids:
            T = tag_map.get_pose_homogeneous(tag_id)
            viz.add_tag(T, size=130/1000, label=f"Tag {tag_id}")

    # Plot the setpoint positions as waypoints
    c = 0
    prev_waypoint = None
    for i, T in enumerate(setpoint_positions):
        if prev_waypoint is None or not np.allclose(T, prev_waypoint):
            prev_waypoint = T
            viz.add_waypoint(T, label=f"Setpoint {c}")
            c += 1

    if uncorrected_setpoint_positions is not None:
        for i, T in enumerate(uncorrected_setpoint_positions[::10]):
            viz.add_waypoint(T, label=f"UC {i}")

    # Plot the three position estimates
    viz.add_path(None, vicon_positions, label="Vicon", color='g')
    viz.add_path(None, local_positions, label="Local", color='b')
    viz.add_path(None, corrected_positions, label="Corrected", color='r')

    viz.add_path(None, frame_transforms, label="Frame Transforms", color="k")

    viz.show()

def graph_transform_solver(transform_data):
    vicon_pose_data = transform_data["vicon_pose"]  #[(timestamp, pose_params),...]
    local_pose_data = transform_data["local_pose"]  #[(timestamp, pose_params),...]
    frame_transform_data = transform_data["frame_transform"]  #[(timestamp, pose_params),...]

    start_time_offset = vicon_pose_data[0][0]
    start_time = 0

    data_start_time = start_time + start_time_offset

    vicon_pose_positions = np.array([params_to_matrix(pose[1], type="quaternion") for pose in vicon_pose_data if pose[0] > data_start_time])
    local_pose_positions = np.array([params_to_matrix(pose[1], type="quaternion") for pose in local_pose_data if pose[0] > data_start_time])
    frame_transform_positions = np.array([params_to_matrix(pose[1], type="quaternion") for pose in frame_transform_data if pose[0] > data_start_time])

    viz = Viewer(size=2)

    print(vicon_pose_positions[:, :3, 3])

    viz.add_path(None, vicon_pose_positions, label="VICON", color='g')
    viz.add_path(None, local_pose_positions, label="Local", color='b')
    viz.add_path(None, frame_transform_positions, label="Transform", color='k')

    viz.show()



if __name__ == "__main__":
    FLIGHT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    flight_data_file = FLIGHT_DATA_DIR / f"flight_data.npy"
    transform_solver_file = FLIGHT_DATA_DIR / f"transform_solver_test.npy"

    try:
        current_tag_map = load_current_flight_tag_map()
    except FileNotFoundError:
        print("No tag map found.")
        current_tag_map = None

    # flight_data = load_flight_data(flight_data_file)

    # graph_flight_data(flight_data, current_tag_map)


    transform_solver_data = load_flight_data(transform_solver_file)
    graph_transform_solver(transform_solver_data)