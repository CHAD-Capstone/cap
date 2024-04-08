#!/usr/bin/python3

"""
A test script for solving for the transform between the VICON frame and the drone local frame
"""

import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped

from threading import Thread

from cap.transformation_lib import transform_stamped_to_matrix, pose_stamped_to_matrix, matrix_to_params, params_to_matrix
from cap.timestamp_queue import TimestampQueue

from cap.data_lib import FLIGHT_DATA_DIR

import tf2_ros as tf2

class TransformSolverNode:
    def __init__(self, group_number: int = 6):
        node_name = f'transform_solver_{group_number}'
        rospy.init_node(node_name)

        self.current_local_position = (-1, None)  # Timestamp & transform
        self.current_vicon_position = (-1, None)
        self.T_realsense_VICON = (-1, None)
        self.start_time = rospy.get_time()

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
        while self.current_vicon_position[1] is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        print("Got VICON")

        print("Waiting for local position...")
        while self.current_local_position[1] is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        print("Got local position")

        rate = rospy.Rate(1/2)
        # while not rospy.is_shutdown():
        #     self.save_data()
        #     rate.sleep()
        self.got_update = False
        self.update_loop_thread = None
        self.should_exit_loop = False
        self.start_update_transform_thread()

        rospy.spin()

        self.should_exit_loop = True
        if self.update_loop_thread is not None:
            self.update_loop_thread.join()
        self.save_data()

    def publish_local_pose_transform(self, pose_msg):
        frame_id = "realsense_world"
        child_frame_id = "drone"
        transform_stamped = TransformStamped()

        # Copy the header from the pose message and update the frames
        transform_stamped.header.stamp = pose_msg.header.stamp
        transform_stamped.header.frame_id = frame_id
        transform_stamped.child_frame_id = child_frame_id

        # Copy the pose from the pose message
        transform_stamped.transform.translation = pose_msg.pose.position
        transform_stamped.transform.rotation = pose_msg.pose.orientation

        self.local_pose_br.sendTransform(transform_stamped)

    def get_local_pose_at_time(self, timestamp):
        """
        Get the local pose at a given timestamp using the tfBuffer

        timestamp: time object. Extracted from header.stamp of a message

        Returns the pose as [x, y, z, qx, qy, qz, qw]
        """
        trans = self.tfBuffer.lookup_transform("realsense_world", "drone", timestamp)
        translation = trans.transform.translation
        rotation = trans.transform.rotation
        return np.array([
            translation.x, translation.y, translation.z,
            rotation.x, rotation.y, rotation.z, rotation.w
        ])

        def add_flight_data(self, key, data):
        timestamp, transform = data
        timestamp_s = timestamp.to_sec()
        self.flight_data[key].append((timestamp_s, transform))

    def vicon_cb(self, msg):
        """
        Computes T_VICON_marker
        """
        # # vicon_timestamp = rospy.get_time()
        # def add_with_delay(msg):
        #     rospy.sleep(0)
        #     vicon_timestamp = msg.header.stamp.to_sec()
        #     T_matrix = transform_stamped_to_matrix(msg)
        #     T_params = matrix_to_params(T_matrix, type="quaternion")
        #     self.current_vicon_position = (vicon_timestamp, T_params)
        #     self.flight_data["vicon_pose"].append(self.current_vicon_position)
        #     self.update_transform()
        # Thread(target=add_with_delay, args=(msg,), daemon=True).start()

        vicon_timestamp = msg.header.stamp.to_sec()
        drone_time = rospy.get_time()
        T_matrix = transform_stamped_to_matrix(msg)
        T_params = matrix_to_params(T_matrix, type="quaternion")
        self.current_vicon_position = (vicon_timestamp, T_params)
        print(f"VICON Latency: {drone_time - vicon_timestamp}. V: {vicon_timestamp - self.start_time}, D: {drone_time - self.start_time}")
        self.flight_data["vicon_pose"].append(self.current_vicon_position)
        self.got_update = True
        # self.update_transform()

    def local_position_cb(self, msg):
        """
        Computes T_realsense_marker
        """
        # pose_timestamp = rospy.get_time()
        pose_timestamp = msg.header.stamp
        T_matrix = pose_stamped_to_matrix(msg)
        T_params = matrix_to_params(T_matrix, type="quaternion")
        self.current_local_position = (pose_timestamp, T_params)
        self.local_position_queue.enqueue(pose_timestamp, T_params)
        self.flight_data["local_pose"].append(self.current_local_position)
        self.got_update = True
        # self.update_transform()

    def start_update_transform_thread(self):
        def start_update_loop():
            rate = rospy.Rate(50)
            while not rospy.is_shutdown() and not self.should_exit_loop:
                if self.got_update:
                    self.update_transform()
                    self.got_update = False
                rate.sleep()
        
        self.update_loop_thread = Thread(target=start_update_loop, daemon=True)
        self.update_loop_thread.start()

    def update_transform(self):
        vicon_ts, T_VICON_marker_params = self.current_vicon_position
        vicon_ts_sec = vicon_ts.to_sec()

        if len(self.local_position_queue.queue) == 0:
            rospy.logwarn("No Local Positions in Queue")
            return False

        local_position_elems = self.local_position_queue.search(vicon_ts)
        T_local_marker_params = local_position_elems[0].data
        local_ts = local_position_elems[0].timestamp

        if abs(vicon_ts - local_ts) > 0.1:
            rospy.logerr(f"TIMESTAMPS MISALIGNED. VICON: {vicon_ts}, Local: {local_ts}")
            return False

        #print(f"\n\nVicon: {T_VICON_marker_params[:3]}")
        #print(f"local: {T_local_marker_params[:3]}")

        T_local_marker = params_to_matrix(T_local_marker_params, type="quaternion")
        T_VICON_marker = params_to_matrix(T_VICON_marker_params, type="quaternion")

        # print(f"\n\nVicon:\n{T_VICON_marker}\n")
        # print(f"Local:\n{T_local_marker}\n")
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