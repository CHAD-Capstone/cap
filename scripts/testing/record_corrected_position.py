#!/usr/bin/python3
"""
This node records data from the flight that will be useful for visualization.
All poses are saved as [x, y, z, qx, qy, qz, qw]

Subscribers:
/capdrone/setpoint_position/local - This is the position in the VICON frame that the drone is trying to reach

/vicon/ROB498_Drone/ROB498_Drone - This is the ground truth position of the drone
/mavros/local_position/pose - This is the uncorrected position of the drone
/capdrone/local_position/pose - This is the corrected position of the drone

Input Files:
apriltag_map.npy - This is the map of apriltag positions in the VICON frame

Output Files:
flight_data.npy - This is the data that will be used for visualization.
"""

from pathlib import Path
from geometry_msgs.msg import TransformStamped, PoseStamped

from cap.apriltag_pose_estimation_lib import load_apriltag_map
from cap.transformation_lib import transform_stamped_to_matrix, pose_stamped_to_matrix, matrix_to_params

file_dir = Path(__file__).resolve().parent
cap_pkg_dir = file_dir.parent.parent
assert (cap_pkg_dir / "CMakeLists.txt").exists(), f"cap_pkg_dir: {cap_pkg_dir} does not appear to be the root of the cap package."

class RecordCorrectedPositionNode:
    def __init__(group_number, apriltag_map_file: Path, flight_data_file: Path):
        node_name = 'record_corrected_position_{:02d}'.format(group_number)
        rospy.init_node(node_name)
        print("Starting Record Corrected Position Node with name", node_name)

        # Make sure the input files exist
        if not apriltag_map_file.exists():
            raise FileNotFoundError(f"Could not load apriltag map file: {apriltag_map_file}")
        self.apriltag_map = load_apriltag_map(apriltag_map_file)

        # Make sure the output folder exists
        flight_data_file.parent.mkdir(parents=True, exist_ok=True)
        # Check if the output file already exists
        if flight_data_file.exists():
            overwrite = input(f"Output file {flight_data_file} already exists. Overwrite? (y/n): ")
            if overwrite.lower() != 'y':
                print("Exiting")
                return
        self.flight_data_file = flight_data_file

        # Subscribers
        self.setpoint_position_local_sub = rospy.Subscriber('/capdrone/setpoint_position/local', PoseStamped, callback=self.setpoint_position_local_cb)
        self.vicon_sub = rospy.Subscriber('/vicon/ROB498_Drone/ROB498_Drone', TransformStamped, callback=self.vicon_cb)
        self.local_position_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, callback=self.local_position_cb)
        self.corrected_position_sub = rospy.Subscriber('/capdrone/local_position/pose', PoseStamped, callback=self.corrected_position_cb)

        # Data
        self.flight_data = {
            'setpoint_position_local': [],
            'vicon': [],
            'local_position': [],
            'corrected_position': []
        }
        # Data is stored as tuples of (timestamp, pose_params)

    def setpoint_position_local_cb(self, msg: PoseStamped):
        current_time_sec = rospy.get_time()
        pose_matrix = pose_stamped_to_matrix(msg)
        pose_params = matrix_to_params(pose_matrix, type='quaternion')
        self.flight_data['setpoint_position_local'].append((current_time_sec, pose_params))

    def vicon_cb(self, msg: TransformStamped):
        current_time_sec = rospy.get_time()
        transform_matrix = transform_stamped_to_matrix(msg)
        transform_params = matrix_to_params(transform_matrix, type='quaternion')
        self.flight_data['vicon'].append((current_time_sec, transform_params))

    def local_position_cb(self, msg: PoseStamped):
        current_time_sec = rospy.get_time()
        pose_matrix = pose_stamped_to_matrix(msg)
        pose_params = matrix_to_params(pose_matrix, type='quaternion')
        self.flight_data['local_position'].append((current_time_sec, pose_params))

    def corrected_position_cb(self, msg: PoseStamped):
        current_time_sec = rospy.get_time()
        pose_matrix = pose_stamped_to_matrix(msg)
        pose_params = matrix_to_params(pose_matrix, type='quaternion')
        self.flight_data['corrected_position'].append((current_time_sec, pose_params))

    def save_data(self):
        np.save(self.flight_data_file, self.flight_data)

    def run(self):
        # Save once every 10 seconds
        rate = rospy.Rate(1/10)
        while not rospy.is_shutdown():
            self.save_data()
            rate.sleep()
        
        self.save_data()

if __name__ == "__main__":
    group_number = 6
    apriltag_map_file = cap_pkg_dir / "data" / "flight_data" / "apriltag_map.npy"
    flight_data_file = cap_pkg_dir / "data" / "flight_data" / f"flight_data.npy"
    node = RecordCorrectedPositionNode(group_number, apriltag_map_file, flight_data_file)
    node.run()