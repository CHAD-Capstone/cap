#!/usr/bin/python3

import rospy
import nanocamera as nano
import cv2
from pathlib import Path
import numpy as np

from geometry_msgs.msg import TransformStamped, TwistStamped
from mavros_msgs.msg import State
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty, EmptyResponse
from cap.msg import TagTransform, TagTransformArray
from cap.srv import IsReady, IsReadyResponse

import cap.data_lib as data_lib
from cap.apriltag_pose_estimation_lib import detect_tags, get_marker_pose
from cap.transformation_lib import matrix_to_params

def start_camera(flip=0, width=1280, height=720):
    # Connect to another CSI camera on the board with ID 1
    camera = nano.Camera(device_id=0, flip=flip, width=width, height=height, debug=False, enforce_fps=True)
    status = camera.hasError()
    codes, has_error = status
    if has_error:
        return False, codes, None
    else:
        return True, None, camera

class AprilTagLocalizationNode:
    def __init__(self, group_number=6, verbose=False, imaging_interval=2, use_approximate_map=False):
        node_name = 'apriltag_localization_{:02d}'.format(group_number)
        rospy.init_node(node_name)

        self.last_imaging_time = -1
        self.imaging_interval = imaging_interval

        # Get camera parameters
        self.camera_matrix, self.dist_coeffs = data_lib.load_final_intrinsics_calibration()
        self.extrinsics = data_lib.load_final_extrinsics()  # T_camera_marker. Transformation from the marker frame to the camera frame.

        self.use_approximate_map = use_approximate_map
        self.load_apriltag_map()

        self.tag_size_m = 130/1000

        # Velocity Thresholding
        self.velocity_sub = rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, self.velocity_cal)
        self.linear_velocity_threshold = 0.05  # Only image when all velocity components are below this threshold
        self.rot_velocity_threshold = np.pi / 12 # only image when all rotational velocity components are below this threshold
        self.current_linear_velocity = None  # (x, y, z) numpy array
        self.current_rot_velocity = None  # (roll, pitch, yaw) numpy array

        # Drone State
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_cal)
        self.current_mode = None
        self.imaging_mode = "OFFBOARD"  # Only image when in this mode


        # Estimated Position Publisher
        self.tag_transform_publisher = rospy.Publisher('/capdrone/apriltag_localization/pose', TagTransformArray, queue_size=10)
    

        # Service Servers
        self.is_ready_srv = rospy.Service('/capdrone/apriltag_localization/is_ready', IsReady, self.is_ready_cb)
        self.refresh_tag_locations_srv = rospy.Service('/capdrone/apriltag_localization/refresh_tag_locations', Empty, self.load_apriltag_map)

        cam_success, cam_codes, camera = start_camera(flip=0, width=1280, height=720)

        if not cam_success:
            rospy.logerror("Failed to initialize camera. Information on camera codes here: https://github.com/thehapyone/NanoCamera?tab=readme-ov-file#errors-and-exceptions-handling")
            rospy.logerror(cam_codes)
            raise RuntimeError("Failed to initialize camera")

        self.camera = camera

    def load_apriltag_map(self, msg=None):
        try:
            if self.use_approximate_map:
                rospy.logwarn("USING APPROXIMATE TAG MAP!!")
                self.apriltag_map = data_lib.load_approximate_tag_poses()
            else:
                self.apriltag_map = data_lib.load_current_flight_tag_map()
            rospy.loginfo(f"Loaded new map with tags: {self.apriltag_map.tag_ids()}")
        except FileNotFoundError:
            rospy.logwarn("No tag map found. Will need to be loaded on the fly.")
            self.apriltag_map = None
        return EmptyResponse()

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

    def should_image(self):
        # If we have not waited long enough since the last imaging, we should not image
        time_exceeded = (rospy.get_time() - self.last_imaging_time) > self.imaging_interval
        if not time_exceeded:
            # rospy.loginfo(f"Time since last imaging ({rospy.get_time() - self.last_imaging_time}) has not exceeded interval ({self.imaging_interval}). Not imaging.")
            return False

        # If our current velocity exceeds the threshold, we should not image
        if self.linear_velocity_threshold is not None:
            if self.current_linear_velocity is None:
                # rospy.loginfo("Current velocity is None. Cannot threshold.")
                return False
            if np.any(np.abs(self.current_linear_velocity) > self.linear_velocity_threshold):
                # rospy.loginfo(f"Current velocity ({self.current_linear_velocity}) exceeds threshold ({self.linear_velocity_threshold}). Not imaging.")
                return False
        
        # If our current rotational velocity exceeds the threshold, we should not image
        if self.rot_velocity_threshold is not None:
            if self.current_rot_velocity is None:
                # rospy.loginfo("Current rotational velocity is None. Cannot threshold.")
                return False
            if np.any(np.abs(self.current_rot_velocity) > self.rot_velocity_threshold):
                # rospy.loginfo(f"Current rotational velocity ({self.current_rot_velocity}) exceeds threshold ({self.rot_velocity_threshold}). Not imaging.")
                return False

        # # If we are not in the correct mode, we should not image
        # if self.imaging_mode is not None:
        #     if self.current_mode != self.imaging_mode:
        #         rospy.loginfo(f"Current mode ({self.current_mode}) does not match imaging mode ({self.imaging_mode}). Not imaging.")
        #         return False
        
        return True

    def take_and_process_img(self):
        """
        Takes an image and, for each tag visible in the image that is also part of the tag map,
        we compute the estimated pose of the marker with respect to the 
        """
        imaging_time = rospy.Time.now()
        img = self.camera.read()
        if img is None:
            rospy.logerr("Camera failed to capture image. Try restarting the Jetson.")
            raise Exception("Camera failed to capture image.")

        tags = detect_tags(img)  # Returns a map from tag id to a list of corner px coordinates

        estimated_poses = []  # List of (tag_id, estimated_pose 4x4 matrix)

        map_tags = self.apriltag_map.tag_ids()
        for detected_tag_id, detected_tag_corners in tags.items():
            if detected_tag_id not in map_tags:
                rospy.logwarn(f"Detected tag {detected_tag_id} in image, but it was not in the tag map {map_tags}")
                continue

            tag_pose_VICON = self.apriltag_map.get_pose_homogeneous(detected_tag_id)  # T_VICON_tag

            estimated_marker_pose = get_marker_pose(
                tag_pose_VICON,
                detected_tag_corners,
                self.tag_size_m,
                self.camera_matrix,
                self.dist_coeffs,
                self.extrinsics
            )
            estimated_marker_pose_params = matrix_to_params(estimated_marker_pose, type="quaternion")
            # estimated_marker_pose_params_euler = matrix_to_params(estimated_marker_pose, type="euler")
            # print(f"Pose Euler: {estimated_marker_pose_params_euler}")

            estimated_poses.append((detected_tag_id, estimated_marker_pose_params))

        print(f"Estimated Poses: {estimated_poses}")
        return imaging_time, estimated_poses

    def start_image_loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.should_image():
                rate.sleep()
                continue

            rospy.loginfo("Taking Image")
            imaging_time, tag_poses = self.take_and_process_img()
            
            if len(tag_poses) > 0:
                # Construct the message
                tag_transforms = []
                for tag_id, tag_pose in tag_poses:
                    transform_stamped = TransformStamped()
                    transform_stamped.header.stamp = imaging_time
                    transform_stamped.header.frame_id = "map"
                    transform_stamped.child_frame_id = f"tag_{tag_id}"
                    transform_stamped.transform.translation.x = tag_pose[0]
                    transform_stamped.transform.translation.y = tag_pose[1]
                    transform_stamped.transform.translation.z = tag_pose[2]
                    transform_stamped.transform.rotation.x = tag_pose[3]
                    transform_stamped.transform.rotation.y = tag_pose[4]
                    transform_stamped.transform.rotation.z = tag_pose[5]
                    transform_stamped.transform.rotation.w = tag_pose[6]

                    tag_transform = TagTransform()
                    tag_transform.tag_id = tag_id
                    tag_transform.transform = transform_stamped

                    tag_transforms.append(tag_transform)

                transform_array = TagTransformArray()
                transform_array.tags = tag_transforms

                self.tag_transform_publisher.publish(transform_array)
            else:
                rospy.loginfo("No tags visible")

            self.last_imaging_time = rospy.get_time()
            rate.sleep()
        self.camera.release()

    def is_ready_cb(self, req):
        """
        Callback function for the is_ready service
        """
        ready = self.apriltag_map is not None
        msg = "Tag Map is ready." if ready else "Tag Map is not ready."
        rospy.loginfo(msg)
        return IsReadyResponse(ready, msg)
            

if __name__ == "__main__":
    n = None
    try:
        n = AprilTagLocalizationNode(
            group_number=6,
            imaging_interval=2,
            use_approximate_map=False
        )
        print("Starting imaging")
        n.start_image_loop()
    except Exception as e:
        n.camera.release()
        raise e
    if n is not None:
        n.camera.release()