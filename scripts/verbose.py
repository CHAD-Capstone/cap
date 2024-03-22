#!/usr/bin/python3

"""
A node that configurably prints out a bunch of information about the drone's state
"""

from typing import Tuple

import rospy
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

class VerboseNode:
    def __init__(
        self,
        group_number: int,
        print_pose: bool = True,
        print_odometry: bool = False,
        print_rate: float = 1.0,
    ):
        """
        Constructor for the VerboseNode class
        """
        print("Starting")
        node_name = f'rob498_verbose_{group_number:02d}'
        rospy.init_node(node_name)

        self.current_state = (-1, State())
        state_sub = rospy.Subscriber("mavros/state", State, callback=self.state_cb)

        self.current_pose = (-1, PoseStamped())
        pose_sub = rospy.Subscriber("mavros/local_position/pose", PoseStamped, callback=self.pose_cb)
        self.do_print_pose = print_pose
        
        self.current_odometry_out = (-1, Odometry())
        odometry_out_sub = rospy.Subscriber("mavros/odometry/out", Odometry, callback=self.odometry_out_cb)
        self.do_print_odometry = print_odometry

        self.print_rate = print_rate

        self.start_printing()
        print("Finished")

    def state_cb(self, msg):
        """
        Callback for the state subscriber
        """
        current_time_sec = rospy.get_time()
        self.current_state = (current_time_sec, msg)

    def pose_cb(self, msg):
        """
        Callback for the pose subscriber
        """
        current_time_sec = rospy.get_time()
        self.current_pose = (current_time_sec, msg)

    def odometry_out_cb(self, msg):
        """
        Callback for the odometry subscriber
        """
        current_time_sec = rospy.get_time()
        self.current_odometry_out = (current_time_sec, msg)

    def print_pose(self) -> None:
        """
        Uses the current odometry to get pose in the local frame in ENU coordinates
        """
        current_time_sec = rospy.get_time()
        update_time_sec, pose_msg = self.current_pose
        time_since_update = current_time_sec - update_time_sec
        pose = pose_msg.pose
        position = pose.position
        orientation = pose.orientation

        print(f'Time since pose update: {time_since_update}')
        print(f'Local Position: (x={position.x}, y={position.y}, z={position.z})')

    def print_odometry(self):
        current_time_sec = rospy.get_time()
        update_time_sec, odometry_out_msg = self.current_odometry_out
        time_since_update = current_time_sec - update_time_sec
        # Odometry has both pose and twist
        # Pose has position and orientation while twist has linear and angular
        # We will be printing out the position only here
        pose = odometry_out_msg.pose.pose
        twist = odometry_out_msg.twist.twist
        position = pose.position
        print(f'Time since odom update: {time_since_update}')
        print(f'Odometry Out: (x={position.x}, y={position.y}, z={position.z})')

    def start_printing(self):
        """
        Starts the printing thread
        """
        rate = rospy.Rate(self.print_rate)
        while not rospy.is_shutdown():
            if self.do_print_pose:
                self.print_pose()
                # update_time_sec, pose = self.get_pose()
                # print(f'Time since pose update: {update_time_sec}')
                # print(f'Odometry: {pose}')
            if self.do_print_odometry:
                self.print_odometry()
            rate.sleep()
            print("\n\n")
        print("Shutting down")

if __name__ == "__main__":
    v = VerboseNode(
        group_number=6,
        print_pose=True,
        print_odometry=True,
        print_rate=1
    )