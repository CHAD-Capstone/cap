#!/usr/bin/python3

import rospy
import nanocamera as nano
import cv2
from pathlib import Path
import threading
import time
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import State
import numpy as np

def start_camera(flip=0, width=1280, height=720):
    # Connect to another CSI camera on the board with ID 1
    camera = nano.Camera(device_id=0, flip=flip, width=width, height=height, debug=False, enforce_fps=True)
    status = camera.hasError()
    codes, has_error = status
    if has_error:
        return False, codes, None
    else:
        return True, None, camera

class ImagingNode:
    def __init__(self, group_number=6, verbose=False, save_folder=Path("/home/rob498/catkin_ws/src/cap/data/flight_images"), imaging_interval=2):
        node_name = 'imaging_node_{:02d}'.format(group_number)
        rospy.init_node(node_name)

        self.last_imaging_time = -1
        self.imaging_interval = imaging_interval

        # Velocity Thresholding
        self.velocity_sub = rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, self.velocity_cal)
        self.linear_velocity_threshold = 0.1  # Only image when all velocity components are below this threshold
        self.rot_velocity_threshold = np.pi / 12 # only image when all rotational velocity components are below this threshold
        self.current_linear_velocity = None  # (x, y, z) numpy array
        self.current_rot_velocity = None  # (roll, pitch, yaw) numpy array

        # Drone State
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_cal)
        self.current_mode = None
        self.imaging_mode = "OFFBOARD"  # Only image when in this mode

        # Drone Local Position
        self.position_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.position_cal)
        self.current_position = None # (x, y, z) numpy array
        self.height_threshold = 0.5  # Only image when the drone is above this height

        # Drone Vicon Position
        # TODO: Add Vicon Position Subscriber
        self.vicon_sub = rospy.Subscriber('/vicon/ROB498_Drone/ROB498_Drone', TransformStamped, callback=self.vicon_cal)
        self.current_vicon_position = None # (x, y, z) numpy array


        # Publisher
        # When a new image is processed
        # The local position of the drone at the time of the image
        # The time of the image
        # For each tag
        # The tag id and
        # The position of the drone with respect to the april tag

        self.calibration_mode_flag = False  # If true, saves a file that contains the VICON position, the images, and the local position


        self.save_folder = save_folder
        if not save_folder.exists():
            rospy.error("Save folder does not exist. Please create it before running this node.")
            raise RuntimeError("Failed to find image folder")

        cam_success, cam_codes, camera = start_camera(flip=0, width=1280, height=720)

        if not cam_success:
            rospy.logerror("Failed to initialize camera. Information on camera codes here: https://github.com/thehapyone/NanoCamera?tab=readme-ov-file#errors-and-exceptions-handling")
            rospy.logerror(cam_codes)
            raise RuntimeError("Failed to initialize camera")

        self.camera = camera

        print("Starting imaging")
        self.start_image_loop()

    def state_cal(self, msg):
        """
        MavRos State 
        """
        self.current_mode = msg.mode

    def velocity_cal(self, msg):
        """
        Local Velocity
        """
        self.current_linear_velocity = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.current_rot_velocity = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])

    def position_cal(self, msg):
        """
        Drone Local Position
        """
        self.current_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def should_image(self):
        # If we have not waited long enough since the last imaging, we should not image
        time_exceeded = (rospy.get_time() - self.last_imaging_time) > self.imaging_interval
        if not time_exceeded:
            rospy.loginfo(f"Time since last imaging ({rospy.get_time() - self.last_imaging_time}) has not exceeded interval ({self.imaging_interval}). Not imaging.")
            return False

        # If our current velocity exceeds the threshold, we should not image
        if self.linear_velocity_threshold is not None:
            if self.current_linear_velocity is None:
                rospy.loginfo("Current velocity is None. Cannot threshold.")
                return False
            if np.any(np.abs(self.current_linear_velocity) > self.linear_velocity_threshold):
                rospy.loginfo(f"Current velocity ({self.current_linear_velocity}) exceeds threshold ({self.linear_velocity_threshold}). Not imaging.")
                return False
        
        # If our current rotational velocity exceeds the threshold, we should not image
        if self.rot_velocity_threshold is not None:
            if self.current_rot_velocity is None:
                rospy.loginfo("Current rotational velocity is None. Cannot threshold.")
                return False
            if np.any(np.abs(self.current_rot_velocity) > self.rot_velocity_threshold):
                rospy.loginfo(f"Current rotational velocity ({self.current_rot_velocity}) exceeds threshold ({self.rot_velocity_threshold}). Not imaging.")
                return False

        # If we are not in the correct mode, we should not image
        if self.imaging_mode is not None:
            if self.current_mode != self.imaging_mode:
                rospy.loginfo(f"Current mode ({self.current_mode}) does not match imaging mode ({self.imaging_mode}). Not imaging.")
                return False
        
        return True

    def start_image_loop(self):
        rate = rospy.Rate(10)
        img_count = 0
        while not rospy.is_shutdown():
            if not self.should_image():
                rate.sleep()
                continue
            image_file = self.save_folder / f"image_{img_count}.jpg"
            img = self.camera.read()
            cv2.imwrite(str(image_file), img)
            img_count += 1
            self.last_imaging_time = rospy.get_time()
            rate.sleep()
        self.camera.release()
            


if __name__ == "__main__":
    n = ImagingNode(
        group_number=6,
        save_folder=Path("/home/rob498/catkin_ws/src/cap/data/flight_images"),
        imaging_interval=2
    )