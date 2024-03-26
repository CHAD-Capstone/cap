#!/usr/bin/python3
"""

"""

import rospy
from geometry_msgs.msg import TransformStamped

import nanocamera as nano
import numpy as np

from std_srvs.srv import Empty, EmptyResponse
from cap_srvs.srv import FindTag, FindTagResponse, NewTag, NewTagResponse, IsReady, IsReadyResponse

from cap.apriltag_pose_estimation_lib import AprilTagMap, detect_tags, estimate_T_C_A, optimize_tag_pose
import cap.data_lib as data_lib
from cap.transformation_lib import matrix_to_params, transform_stamped_to_matrix,  matrix_to_transform_stamped

def start_camera(flip=0, width=1280, height=720):
    # Connect to another CSI camera on the board with ID 1
    camera = nano.Camera(device_id=0, flip=flip, width=width, height=height, debug=False, enforce_fps=True)
    status = camera.hasError()
    codes, has_error = status
    if has_error:
        return False, codes, None
    else:
        return True, None, camera

class AprilTagMappingNode:
    def __init__(self, group_number):
        node_name = 'apriltag_mapping_node_{:02d}'.format(group_id)
        rospy.init_node(node_name)

        # Get camera parameters
        self.camera_matrix, self.dist_coeffs = data_lib.load_final_intrinsics_calibration()
        self.extrinsics = data_lib.load_final_extrinsics()  # T_camera_marker. Transformation from the marker frame to the camera frame.

        # Set up default values
        self.tag_size = 130 / 1000  # Tag size in meters
        self.current_tag_id = None
        self.current_tag_data = []
        self.tag_map = AprilTagMap()

        # Setup up subscribers
        # VICON pose subscriber gives us ground truth pose of the marker
        self.current_vicon_pose = None
        self.vicon_sub = rospy.Subscriber('/vicon/ROB498_Drone/ROB498_Drone', TransformStamped, callback=self.vicon_cb)

        # Set up Service Servers
        # find_tag immediately returns the tag pose of a given tag id if it is visible in the camera frame
        self.find_tag_srv = rospy.Service('/capdrone/apriltag_mapping/find_tag', FindTag, self.find_tag_cb)
        # new_tag begins the process of adding a new tag to the tag map
        self.new_tag_srv = rospy.Service('/capdrone/apriltag_mapping/new_tag', NewTag, self.new_tag_cb)
        # capture_img captures one image and processes it to find the tag
        self.capture_img_srv = rospy.Service('/capdrone/apriltag_mapping/capture_img', Empty, self.capture_img_cb)
        # process_tag uses the captured images to estimate the tag pose
        self.process_tag_srv = rospy.Service('/capdrone/apriltag_mapping/process_tag', Empty, self.process_tag_cb)
        # ready returns true if VICON is publishing poses
        self.is_ready_srv = rospy.Service('/capdrone/apriltag_mapping/is_ready', IsReady, self.is_ready_cb)

        # Set up the camera
        cam_up, codes, self.camera = start_camera()
        if not cam_up:
            rospy.logerr("Camera not available. Error: {}".format(codes))
            return

        if self.camera.read() is None:
            rospy.logerr("Camera failed to capture image. Try restarting the Jetson.")
            return

        if not confirm_non_black_img(self.camera):
            rospy.logerr("Camera is not capturing images. Confirm that the lens cap is removed.")
            return

        # Spin until we have valid VICON data
        while not rospy.is_shutdown():
            if self.current_vicon_pose is not None:
                break
            rospy.sleep(0.1)
        
        rospy.loginfo("AprilTag Mapping Node is ready.")
        rospy.spin()

    def vicon_cb(self, msg):
        """
        Callback function for the VICON pose subscriber
        """
        self.current_vicon_pose = msg

    def take_and_process_img(self, tag_id):
        """

        """
        img_position = self.current_vicon_pose
        img = self.camera.read()
        if img is None:
            rospy.logerr("Camera failed to capture image. Try restarting the Jetson.")
            raise Exception("Camera failed to capture image.")

        tags = detect_tags(img)  # Returns a map from tag id to a list of corner px coordinates

        if tag_id not in tags:
            rospy.loginfo("Tag not found in image.")
            return None, None, None

        tag_corners = tags[tag_id]

        T_camera_tag = estimate_T_C_A(tag_corners, self.tag_size, self.camera_matrix, self.dist_coeffs)
        # T_marker_tag = T_marker_camera * T_camera_tag = (T_camera_marker)^-1 * T_camera_tag
        T_marker_tag = np.linalg.inv(self.extrinsics) @ T_camera_tag

        T_VICON_marker = transform_stamped_to_matrix(img_position)
        T_VICON_tag = T_VICON_marker @ T_marker_tag

        return T_VICON_tag, tag_corners, img

    def find_tag_cb(self, req):
        """
        Callback function for the find_tag service
        """
        tag_id = req.tag_id
        tag_pose, tag_corners, img = self.take_and_process_img(tag_id)
        if tag_pose is None:
            return FindTagResponse(found=False)
        else:
            transform = matrix_to_transform_stamped(tag_pose, "vicon", "tag_{}".format(tag_id))
            return FindTagResponse(found=True, transform=transform)

    def new_tag_cb(self, req):
        """
        Callback function for the new_tag service
        """
        tag_id = req.tag_id
        self.current_tag_id = tag_id
        self.current_tag_data = []
        return NewTagResponse()

    def capture_img_cb(self, req):
        """
        Callback function for the capture_img service
        """
        if self.current_tag_id is None:
            rospy.logwarn("No tag id set. Use the new_tag service to set the tag id.")
            return EmptyResponse()

        img_position = self.current_vicon_pose
        tag_pose, tag_corners, img = self.take_and_process_img(self.current_tag_id)
        if tag_pose is None:
            rospy.logwarn("Tag not found in image.")
            return EmptyResponse()

        T_VICON_marker = transform_stamped_to_matrix(img_position)
        self.current_tag_data.append((tag_pose, tag_corners, img, T_VICON_marker))

        return EmptyResponse()

    def process_tag_cb(self, req):
        """
        Callback function for the process_tag service
        """
        if self.current_tag_id is None:
            rospy.logwarn("No tag id set. Use the new_tag service to set the tag id.")
            return EmptyResponse()

        if len(self.current_tag_data) == 0:
            rospy.logwarn("No images captured. Use the capture_img service to capture images.")
            return EmptyResponse()

        initial_tag_pose = self.current_tag_data[0][0]
        tag_px_positions = {}
        drone_poses = {}
        for img_idx in range(len(self.current_tag_data)):
            tag_pose, tag_corners, img, T_VICON_marker = self.current_tag_data[img_idx]
            tag_px_positions[img_idx] = tag_corners
            drone_poses[img_idx] = T_VICON_marker

        optimized_tag_pose = optimize_tag_pose(
            initial_tag_pose,
            tag_px_positions,
            drone_poses,
            tag_size=self.tag_size,
            camera_matrix=self.camera_matrix,
            distortion_coefficients=self.dist_coeffs,
            camera_extrinsics=self.extrinsics
        )  # 4x4 homogeneous transformation matrix

        self.tag_map.add_tag_pose(self.current_tag_id, optimized_tag_pose)
        rospy.loginfo(f"Tag {self.current_tag_id} pose optimized and added to tag map.")

        self.current_tag_id = None
        self.current_tag_data = []

        return EmptyResponse()

    def is_ready_cb(self, req):
        """
        Callback function for the is_ready service
        """
        ready = self.current_vicon_pose is not None
        msg = "VICON is ready." if ready else "VICON is not ready."
        rospy.loginfo(msg)
        return IsReadyResponse(ready=ready, message=msg)
        

