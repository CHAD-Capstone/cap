#!/usr/bin/python3
"""

"""

import rospy
from geometry_msgs.msg import TransformStamped, PoseStamped

import nanocamera as nano
import numpy as np

from std_msgs.msg import Bool, String
from std_srvs.srv import Empty, EmptyResponse
from cap.srv import FindTag, NewTag, NewTagResponse, IsReady, IsReadyResponse

from cap.apriltag_pose_estimation_lib import AprilTagMap, detect_tags, estimate_T_C_A, optimize_tag_pose
import cap.data_lib as data_lib
from cap.data_lib import FLIGHT_DATA_DIR
from cap.transformation_lib import matrix_to_params, params_to_matrix, transform_stamped_to_matrix,  matrix_to_transform_stamped, pose_stamped_to_matrix, inv_matrix

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
    def __init__(self, group_id, use_local_pose=False, optimize=False):
        node_name = 'apriltag_mapping_node_{:02d}'.format(group_id)
        rospy.init_node(node_name)

        self.flushing = True

        self.optimize = optimize

        self.mapping_data_file = FLIGHT_DATA_DIR / f"mapping_data.npy"
        self.mapping_data = {
            "drone_pose": [],  # (timestamp, [x y z qx qy qz qw]). Records all drone positions
            "find_tag_cmd": [],  # (timestamp, (int, [x y z qx qy qz qw])). Records which tag was captured and the tag pose
            "new_tag_cmd": [],  # (timestamp, int). Records the tag id when we start capturing a new tag
            "capture_img_cmd": [],  # (timestamp, (int, [x y z qx qy qz qw], [x y z qx qy qz qw])). Record the tag id, drone pose and tag pose when we capture a new image
            "process_tag_cmd": [],  # (timestamp, (int, [x y z qx qy qz qw])). Record the optimized tag id and pose
        }

        # Get camera parameters
        self.camera_matrix, self.dist_coeffs = data_lib.load_final_intrinsics_calibration()
        self.extrinsics = data_lib.load_final_extrinsics()  # T_camera_marker. Transformation from the marker frame to the camera frame.

        # Set up default values
        self.tag_size = 130 / 1000  # Tag size in meters
        self.current_tag_id = None
        self.current_tag_data = []
        self.tag_map = AprilTagMap()

        self.use_local_pose = use_local_pose

        # Setup up subscribers
        # VICON pose subscriber gives us ground truth pose of the marker
        self.current_vicon_pose = None
        if self.use_local_pose:
            rospy.logwarn("Using local pose")
            self.local_pose_sub = rospy.Subscriber("mavros/local_position/pose", PoseStamped, callback=self.local_pose_cb)
        else:
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
        self.is_ready_srv = rospy.Service('/capdrone/apriltag_mapping/ready', IsReady, self.is_ready_cb)

        self.tick = 0

        # Set up the camera
        cam_up, codes, self.camera = start_camera()
        if not cam_up:
            rospy.logerr("Camera not available. Error: {}".format(codes))
            return

        if self.camera.read() is None:
            rospy.logerr("Camera failed to capture image. Try restarting the Jetson.")
            return

        # Spin until we have valid VICON data
        while not rospy.is_shutdown():
            if self.current_vicon_pose is not None:
                break
            rospy.sleep(0.1)

        input("Press enter to start mapping")
        self.flushing = False

    def add_mapping_data(self, key, data):
        timestamp, meta = data
        timestamp_s = timestamp.to_sec()
        self.mapping_data[key].append((timestamp_s, meta))

    def save_data(self):
        rospy.loginfo(f"Saving data")
        for key, arr in self.mapping_data.items():
            print(f"\t{key}: {len(arr)}")
        np.save(self.mapping_data_file, self.mapping_data)

    def save_tag_map(self):
        data_lib.save_current_flight_tag_map(self.tag_map)

    def vicon_cb(self, msg):
        """
        Callback function for the VICON pose subscriber
        """
        if not self.use_local_pose:
            self.current_vicon_pose = msg
            if self.tick < 20:
                self.tick += 1
                return
            self.tick = 0
            # rospy.loginfo("Saving data")
            T = transform_stamped_to_matrix(msg)
            T_params = matrix_to_params(T, type="quaternion")
            self.add_mapping_data("drone_pose", (msg.header.stamp, T_params))

    def local_pose_cb(self, msg):
        if self.use_local_pose:
            self.current_vicon_pose = msg
            T = pose_stamped_to_matrix(msg)
            T_params = matrix_to_params(T, type="quaternion")
            self.add_mapping_data("drone_pose", (msg.header.stamp, T_params))

    def take_and_process_img(self, tag_id):
        """

        """
        if self.current_vicon_pose is None:
            rospy.logerr("Cannot get marker pose since no VICON pose availible")
            return None, None, None
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
        T_marker_tag = inv_matrix(self.extrinsics) @ T_camera_tag

        if self.use_local_pose:
            T_VICON_marker = pose_stamped_to_matrix(img_position)
        else:
            T_VICON_marker = transform_stamped_to_matrix(img_position)
        T_VICON_tag = T_VICON_marker @ T_marker_tag

        return T_VICON_tag, tag_corners, img

    def find_tag_cb(self, req):
        """
        Callback function for the find_tag service
        """
        tag_id = req.tag_id.data
        tag_pose, tag_corners, img = self.take_and_process_img(tag_id)
        if tag_pose is None:
            transform = TransformStamped()
            return Bool(False), transform
        else:
            T_params = matrix_to_params(tag_pose, type="quaternion")
            self.add_mapping_data("find_tag_cmd", (rospy.Time.now(), (tag_id, T_params)))
            transform = matrix_to_transform_stamped(tag_pose, "vicon", "tag_{}".format(tag_id))
            return Bool(True), transform

    def new_tag_cb(self, req):
        """
        Callback function for the new_tag service
        """
        rospy.loginfo("New Tag")
        tag_id = req.tag_id
        self.current_tag_id = tag_id.data
        self.current_tag_data = []
        rospy.loginfo("Reset tag info")
        self.add_mapping_data("new_tag_cmd", (rospy.Time.now(), self.current_tag_id))
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

        if self.use_local_pose:
            T_VICON_marker = pose_stamped_to_matrix(img_position)
        else:
            T_VICON_marker = transform_stamped_to_matrix(img_position)
        self.current_tag_data.append((tag_pose, tag_corners, img, T_VICON_marker))
        rospy.loginfo(f"Drone pose at time of img:\n{T_VICON_marker}")
        rospy.loginfo(f"Tag pose:\n{tag_pose}")

        T_VICON_marker_params = matrix_to_params(T_VICON_marker, type="quaternion")
        tag_pose_params = matrix_to_params(tag_pose, type="quaternion")
        self.add_mapping_data("capture_img_cmd", (rospy.Time.now(), (self.current_tag_id, T_VICON_marker_params, tag_pose_params)))

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

        if self.optimize:
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
        else:
            # Then instead we take the median of the tag poses
            tag_poses = np.zeros((len(self.current_tag_data), 7))
            for img_idx in range(len(self.current_tag_data)):
                tag_pose = self.current_tag_data[img_idx][0]
                tag_poses[img_idx] = matrix_to_params(tag_pose, type="quaternion")
            optimized_tag_pose_params = np.median(tag_poses, axis=0)
            optimized_tag_pose = params_to_matrix(optimized_tag_pose_params, type="quaternion")

        self.tag_map.add_tag_pose(self.current_tag_id, optimized_tag_pose)
        rospy.loginfo(f"Tag {self.current_tag_id} pose optimized and added to tag map.")

        optimized_tag_pose_params = matrix_to_params(optimized_tag_pose, type="quaternion")
        self.add_mapping_data("process_tag_cmd", (rospy.Time.now(), (self.current_tag_id, optimized_tag_pose_params)))

        self.current_tag_id = None
        self.current_tag_data = []

        self.save_tag_map()

        return EmptyResponse()

    def is_ready_cb(self, req):
        """
        Callback function for the is_ready service
        """
        ready = self.current_vicon_pose is not None
        msg = "VICON is ready." if ready else "VICON is not ready."
        rospy.loginfo(msg)
        return IsReadyResponse(ready, msg)
        

if __name__ == "__main__":
    n = None
    try:
        n = AprilTagMappingNode(
            group_id=6,
            use_local_pose=False,
            optimize=False
        )
        rospy.loginfo("AprilTag Mapping Node is ready.")
        rospy.spin()
    except Exception as e:
        n.camera.release()
        raise e
    if n is not None:
        n.camera.release()
        n.save_data()
    