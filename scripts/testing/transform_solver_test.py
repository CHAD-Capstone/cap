#!/usr/bin/python3

"""
A test script for solving for the transform between the VICON frame and the drone local frame
"""

import rospy
import numpy as np

from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Transform
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool

from threading import Thread

from cap.srv import SetLocalizationMode, SetLocalizationModeResponse
from cap.srv import IsReady, IsReadyResponse

from cap.msg import TagTransformArray

from cap.transformation_lib import transform_stamped_to_matrix, pose_stamped_to_matrix, matrix_to_params, matrix_to_pose_stamped, params_to_matrix
from cap.timestamp_queue import TimestampQueue

from cap.data_lib import FLIGHT_DATA_DIR

import tf2_ros

class TransformSolverNode:
    def __init__(self, group_number: int = 6):
        node_name = f'transform_solver_{group_number}'
        rospy.init_node(node_name)

        self.current_local_position = (-1, None)  # Timestamp & transform
        self.current_vicon_position = (-1, None)
        self.T_realsense_VICON = (-1, None)

        self.local_position_queue = TimestampQueue(max_length=100)

        self.flight_data = {
            "vicon_pose": [],
            "local_pose": [],
            "frame_transform": []
        }

        self.flight_data_file = FLIGHT_DATA_DIR / f"transform_solver_test.npy"

        self.vicon_sub = rospy.Subscriber('/vicon/ROB498_Drone/ROB498_Drone', TransformStamped, callback=self.vicon_cb)
        self.local_position_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, callback=self.local_position_cb)

        print("Waiting for VICON...")
        while self.current_vicon_position[1] == -1 and not rospy.is_shutdown():
            rospy.sleep(0.1)
        print("Got VICON")

        print("Waiting for local position...")
        while self.current_local_position[1] == -1 and not rospy.is_shutdown():
            rospy.sleep(0.1)
        print("Got local position")

        rate = rospy.Rate(1/2)
        while not rospy.is_shutdown():
            self.save_data()
            rate.sleep()
        self.save_data()

    def vicon_cb(self, msg):
        """
        Computes T_VICON_marker
        """
        # vicon_timestamp = rospy.get_time()
        def add_with_delay(msg):
            rospy.sleep(1)
            vicon_timestamp = msg.header.stamp.to_sec()
            T_matrix = transform_stamped_to_matrix(msg)
            T_params = matrix_to_params(T_matrix, type="quaternion")
            self.current_vicon_position = (vicon_timestamp, T_params)
            self.flight_data["vicon_pose"].append(self.current_vicon_position)
            self.update_transform()
            

        # vicon_timestamp = msg.header.stamp.to_sec()
        # T_matrix = transform_stamped_to_matrix(msg)
        # T_params = matrix_to_params(T_matrix, type="quaternion")
        # self.current_vicon_position = (vicon_timestamp, T_params)
        # self.flight_data["vicon_pose"].append(self.current_vicon_position)
        # self.update_transform()
        Thread(target=add_with_delay, args=(msg,), daemon=True).start()

    def local_position_cb(self, msg):
        """
        Computes T_realsense_marker
        """
        # pose_timestamp = rospy.get_time()
        pose_timestamp = msg.header.stamp.to_sec()
        T_matrix = pose_stamped_to_matrix(msg)
        T_params = matrix_to_params(T_matrix, type="quaternion")
        self.current_local_position = (pose_timestamp, T_params)
        self.local_position_queue.enqueue(pose_timestamp, T_params)
        self.flight_data["local_pose"].append(self.current_local_position)
        self.update_transform()

    def update_transform(self):
        vicon_ts, T_VICON_marker_params = self.current_vicon_position

        if len(self.local_position_queue.queue) == 0:
            rospy.logwarn("No Local Positions in Queue")
            return False

        local_position_elems = self.local_position_queue.search(vicon_ts)
        T_local_marker_params = local_position_elems[0].data
        local_ts = local_position_elems[0].timestamp

        if abs(vicon_ts - local_ts) > 0.1:
            # rospy.logerr("TIMESTAMPS MISALIGNED")
            return False

        #print(f"\n\nVicon: {T_VICON_marker_params[:3]}")
        #print(f"local: {T_local_marker_params[:3]}")

        T_local_marker = params_to_matrix(T_local_marker_params)
        T_VICON_marker = params_to_matrix(T_VICON_marker_params)
        T_local_VICON = T_local_marker @ np.linalg.inv(T_VICON_marker)

        T_ts = (vicon_ts + local_ts) / 2
        self.T_realsense_VICON = (T_ts, matrix_to_params(T_local_VICON, type="quaternion"))
        self.flight_data["frame_transform"].append(self.T_realsense_VICON)

    def save_data(self):
        rospy.loginfo(f"Saving data")
        for key, arr in self.flight_data.items():
            print(f"\t{key}: {len(arr)}")
        np.save(self.flight_data_file, self.flight_data)

if __name__ == "__main__":
    n = TransformSolverNode()