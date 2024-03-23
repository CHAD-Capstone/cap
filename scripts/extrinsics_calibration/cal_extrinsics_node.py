#!/usr/bin/python3

"""
A node for capturing the data necessary to calibrate the camera extrinsics

Captures images of the AprilTag and records the VICON pose at the same instant. The pose of the camera with
respect to the tag is also recorded.

Usage (Connected to remote ROS master for VICON):
1. Run the realsense camera node
2. Ensure VICON is running
3. Run this node (`rosrun cap extrinsics_calibration/cal_extrinsics_node.py`)
4. Place the tag on the ground such that the center is at the origin of the VICON frame
5. Hold the drone above the tag and move it around to capture images from different angles

The drone will take pictures only when it is perfectly still and it has moved since the last time it took a picture.

Every time a picture is taken it is saved to the out_folder and a new extrinsics calibration matrix is computed.

In order to use already captured data to recompute the extrinsics, pass the path to the data file as the recompute argument
down in the main block.
"""

try:
    import rospy
    from geometry_msgs.msg import TwistStamped, PoseStamped, TransformStamped
    from mavros_msgs.msg import State
except ImportError:
    print("ROS NOT INSTALLED. RUNNING IN NON-ROS MODE. IF A ROS NODE IS REQUIRED, THIS WILL FAIL.")
    rospy = None
import nanocamera as nano
import cv2
from pathlib import Path
import threading
import time
import numpy as np
from pupil_apriltags import Detector
from typing import Union, Tuple
from scipy.optimize import least_squares

from cap.transformation_lib import ros_message_to_matrix, matrix_to_params, params_to_matrix
from cap.apriltag_pose_estimation_lib import get_pose_relative_to_apriltag, get_corner_A_m, get_expected_pixels, detect_tags

file_dir = Path(__file__).resolve().parent
cap_pkg_dir = file_dir.parent.parent
assert (cap_pkg_dir / "CMakeLists.txt").exists(), f"cap_pkg_dir: {cap_pkg_dir} does not appear to be the root of the cap package."

def start_camera(flip=0, width=1280, height=720):
    # Connect to another CSI camera on the board with ID 1
    camera = nano.Camera(device_id=0, flip=flip, width=width, height=height, debug=False, enforce_fps=True)
    status = camera.hasError()
    codes, has_error = status
    if has_error:
        return False, codes, None
    else:
        return True, None, camera

def confirm_non_black_img(camera, black_threshold=10):
    """
    Takes a photo with the camera
    If the maximum pixel value is below the threshold, the image is considered black
    """
    img = camera.read()
    return img.max() > black_threshold

def load_calibration_data(cal_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the camera calibration data from the given file.

    Parameters:
    cal_file (Path): Path to the calibration data file.

    Returns:
    tuple[np.ndarray, np.ndarray]: Tuple containing the camera matrix and distortion coefficients.
    """
    cal_data = np.load(cal_file.absolute().as_posix())
    camera_matrix = cal_data['mtx']
    dist_coeffs = cal_data['dist']
    return camera_matrix, dist_coeffs

def compute_extrinsics_avg(tag_pose, tag_size, corners, vicon_poses, camera_matrix, distortion_coefficients, imgs=None, out_dir=None):
    """
    Computes the camera extrinsics without optimizing the tag pose

    Computes the transformation from the camera frame to the drone frame by averaging the relative poses
    """
    # Compute the relative poses of the camera with respect to the tag
    relative_poses = []
    for i in range(len(corners)):
        img_corners = corners[i]
        relative_pose = get_pose_relative_to_apriltag(img_corners, tag_size, camera_matrix, distortion_coefficients)
        relative_poses.append(relative_pose)

    # Relative poses now contains the transformation from the camera frame to the tag frame
    # We now want to compute T_marker_camera, the transformation from the camera frame to the marker frame
    # We currently have T_world_marker, the transformation from the marker frame to the world frame and
    # T_tag_camera, the transformation from the camera frame to the tag frame so
    # T_marker_camera = (T_world_marker)^-1 @ T_world_tag @ T_tag_camera
    T_marker_cameras = []
    for i in range(len(vicon_poses)):
        T_world_marker = vicon_poses[i]
        T_tag_camera = relative_poses[i]
        T_marker_world = np.linalg.inv(T_world_marker)
        T_marker_camera = T_marker_world @ tag_pose @ T_tag_camera
        T_marker_cameras.append(T_marker_camera)

    # Now we average the T_marker_camera matrices
    # To do this we convert them to xyz, xyzw and average both (and renormalize the xyzw)
    T_params = [matrix_to_params(T, type="quaternion") for T in T_marker_cameras]
    # These are now [x y z qx qy qz qw] arrays
    T_xyz = np.mean([T[:3] for T in T_params], axis=0)
    T_xyzw = np.mean([T[3:] for T in T_params], axis=0)
    T_xyzw /= np.linalg.norm(T_xyzw)
    T_avg = params_to_matrix(np.concatenate([T_xyz, T_xyzw]), type="quaternion")

    return T_avg, tag_pose

def compute_extrinsics_optimize(initial_tag_pose, tag_size, corners_px, vicon_poses, camera_matrix, distortion_coefficients, imgs=None, out_dir=None):
    """
    Jointly optimizes the tag pose and the camera extrinsics
    """
    # Params are stored as [x, y, z, roll, pitch, yaw] first for the tag pose and then for the camera extrinsics
    tag_params = matrix_to_params(initial_tag_pose)

    # Get initial extrinsics by using the average method
    initial_extrinsics, _ = compute_extrinsics_avg(initial_tag_pose, tag_size, corners_px, vicon_poses, camera_matrix, distortion_coefficients)
    extrinsics_params = matrix_to_params(initial_extrinsics)  # Encodes T_marker_camera as [x, y, z, roll, pitch, yaw]

    params = np.concatenate([tag_params, extrinsics_params])

    drone_poses = {i: vicon_poses[i] for i in range(len(vicon_poses))}
    tag_px_positions = {i: corners_px[i] for i in range(len(corners_px))}
    tag_corners_m_Ai = {i: get_corner_A_m(tag_size) for i in range(len(vicon_poses))}

    def error_func(params):
        """
        Computes the reprojection error for the given parameters
        """
        tag_params = params[:6]
        extrinsics_params = params[6:]

        tag_pose = params_to_matrix(tag_params, type="euler")
        camera_extrinsics = params_to_matrix(extrinsics_params, type="euler")

        expected_pixels = get_expected_pixels(tag_pose, tag_px_positions, drone_poses, tag_corners_m_Ai, camera_matrix, distortion_coefficients, camera_extrinsics)
        # Expected pixels is a map from image index to the expected pixel positions of the tag corners
        all_expected_pixels = []
        all_actual_pixels = []
        for img_idx in range(len(vicon_poses)):
            expected = expected_pixels[img_idx]
            actual = tag_px_positions[img_idx]

            all_expected_pixels.append(expected)
            all_actual_pixels.append(actual)
        all_expected_pixels = np.array(all_expected_pixels)
        all_actual_pixels = np.array(all_actual_pixels)

        error = all_expected_pixels - all_actual_pixels

        return error.flatten()

    result = least_squares(error_func, params, verbose=2)
    optimized_params = result.x

    optimized_tag_params = optimized_params[:6]
    optimized_extrinsics_params = optimized_params[6:]

    optimized_tag_pose = params_to_matrix(optimized_tag_params, type="euler")
    optimized_extrinsics = params_to_matrix(optimized_extrinsics_params, type="euler")

    return optimized_extrinsics, optimized_tag_pose

def compute_extrinsics(tag_pose, tag_size, imgs, corners_px, vicon_poses, camera_matrix, distortion_coefficients, optimize=False):
    """
    Computes the extrinsics calibration matrix from the recorded data

    Parameters:
    tag_pose (np.ndarray): The pose of the AprilTag in the VICON frame. As a homogenous transformation matrix
    imgs (list[np.ndarray]): List of images
    corners_px (list[np.ndarray]): List of corner positions in the images
    vicon_poses (list[np.ndarray]): List of VICON poses at the time of each image capture
    camera_matrix (np.ndarray): The camera intrinsics matrix
    distortion_coefficients (np.ndarray): The camera distortion coefficients
    optimize (bool): Whether to jointly optimize the tag pose and the camera extrinsics

    Returns:
    np.ndarray: The extrinsics calibration matrix as a 4x4 homogenous transformation matrix. The pose of the camera frame with respect to the drone frame
    """
    # We get in VICON poses as [x, y, z, qx, qy, qz, qw]
    # The compute extrinsics functions expect them as 4x4 matrices
    vicon_poses = [params_to_matrix(p, type="quaternion") for p in vicon_poses]
    if optimize:
        return compute_extrinsics_optimize(tag_pose, tag_size, corners_px, vicon_poses, camera_matrix, distortion_coefficients)
    else:
        return compute_extrinsics_avg(tag_pose, tag_size, corners_px, vicon_poses, camera_matrix, distortion_coefficients)

def compute_and_test_extrinsics(tag_id, tag_pose, tag_size_m, imgs, corners, vicon_poses, camera_matrix, distortion_coefficients, out_folder: Path, optimize=False):
    out_folder.mkdir(parents=True, exist_ok=True)

    # Recompute the corners
    new_corners = []
    for img in imgs:
        tags = detect_tags(img, use_ippe=True)
        if tag_id not in tags:
            rospy.logwarn("Tag not found in image.")
            return False
        new_corners.append(tags[tag_id])
    corners = new_corners

    # Compute the extrinsics
    extrinsics, tag_pose = compute_extrinsics(
        tag_pose,
        tag_size_m,
        imgs,
        corners,
        vicon_poses,
        camera_matrix,
        distortion_coefficients,
        optimize=optimize
    )

    print(f"Extracted extrinsics: {extrinsics}")

    # Compute the expected positions in each image, plot the corners, and save to a /results folder
    img_folder = out_folder / "img_results"
    img_folder.mkdir(parents=True, exist_ok=True)

    # Convert the drone poses to homogenous transformation matrices
    vicon_poses_homog = [params_to_matrix(p, type="quaternion") for p in vicon_poses]
    
    drone_poses = {i: vicon_poses_homog[i] for i in range(len(vicon_poses))}
    tag_px_positions = {i: corners[i] for i in range(len(corners))}
    tag_corners_m_Ai = {img_idx: get_corner_A_m(tag_size_m) for img_idx in range(len(vicon_poses))}
    expected_pixels = get_expected_pixels(tag_pose, tag_px_positions, drone_poses, tag_corners_m_Ai, camera_matrix, distortion_coefficients, extrinsics)

    # color_order = ['r', 'g', 'b', 'k']  # Used to color the true corners
    color_order = [(0, 0, 255), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
    # Show the images with the corners plotted
    for img_idx in range(len(imgs)):
        img = imgs[img_idx].copy()
        img_corners = corners[img_idx]
        expected = expected_pixels[img_idx]
        # Convert to pixels
        expected = np.array(expected).astype(int)
        for i in range(len(img_corners)):
            color = color_order[i]
            corner = img_corners[i]
            corner = (int(corner[0]), int(corner[1]))
            expected_corner = expected[i]
            # Draw the true corner with the correct color
            cv2.circle(img, corner, 5, color, -1)
            # Draw the expected corner
            cv2.circle(img, (expected_corner[0], expected_corner[1]), 5, color, -1)
        cv2.imwrite(str(img_folder / f"image_{img_idx}.jpg"), img)

    np.save(out_folder / "extrinsics_calibration.npy", extrinsics)
class ExtrinsicsCalibrationNode:
    def __init__(
        self,
        group_id: int = 6,
        tag_id: int = 1,
        tag_size_m: float = 130 / 1000,
        tag_pose: Union[np.ndarray, None] = None,
        out_folder: Path = cap_pkg_dir / "data" / "extrinsics_calibration",
        camera_intrinsics_path: Path = cap_pkg_dir / "data" / "final_calibration" / "cal.npz",
        use_optimize: bool = False,
        recompute: Path = None
    ):
        """
        Params:
        tag_id: The id of the AprilTag
        tag_pose: The pose of the AprilTag in the VICON frame. As a homogenous transformation matrix. If none, the tag is assumed to be at the origin of the VICON frame
        """

        if tag_pose is None:
            tag_pose = np.eye(4)
        self.tag_pose = tag_pose
        self.tag_id = tag_id
        self.tag_size_m = tag_size_m
        self.use_optimize = use_optimize

        # Prepare structures for holding the calibration information
        self.out_folder = out_folder
        self.out_folder.mkdir(parents=True, exist_ok=True)

        if recompute is not None:
            data = np.load(recompute, allow_pickle=True).item()
            self.imgs = data['imgs']
            self.corners = data['corners']
            self.vicon_poses = data['vicon_poses']

            self.camera_matrix, self.dist_coeffs = load_calibration_data(camera_intrinsics_path)
            self.tag_pose = tag_pose
            self.tag_id = tag_id
            self.tag_size_m = tag_size_m

            compute_and_test_extrinsics(self.tag_id, self.tag_pose, self.tag_size_m, self.imgs, self.corners, self.vicon_poses, self.camera_matrix, self.dist_coeffs, self.out_folder, optimize=self.use_optimize)
            return

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

        node_name = 'extrinsics_calibration_node_{:02d}'.format(group_id)
        rospy.init_node(node_name)

        self.last_imaging_time = -1
        self.imaging_interval = 2

        # Velocity Thresholding
        self.velocity_sub = rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, self.velocity_callback)
        self.current_velocity = None  # (x, y, z) numpy array
        self.current_angular_velocity = None  # (x, y, z) numpy array
        # We need to ensure very low velocity so that the VICON pose matches the pose at the time of the image as closely as possible
        self.velocity_threshold = 0.05  # Only image when all velocity components are below this threshold

        # Drone Local Position
        self.position_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.position_cal)
        self.current_height = None  # z position
        self.current_horizontal_position = None  # (x, y) position
        self.previous_img_position = None  # (x, y) position
        self.height_threshold = 0.5  # Only image when the drone is above this height
        self.distance_threshold = 0.1  # Only image when the drone has moved more than this distance from the previous image

        # Initialize VICON Position Subscriber
        self.vicon_sub = rospy.Subscriber('/vicon/ROB498_Drone/ROB498_Drone', TransformStamped, callback=self.vicon_cb)
        self.vicon_pose = None  # 4x4 homogenous transformation matrix

        # AprilTag Detection
        self.detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

        # Load the camera intrinsics
        self.camera_matrix, self.dist_coeffs = load_calibration_data(camera_intrinsics_path)

        self.imgs = []
        self.corners = []
        self.vicon_poses = []
        # self.tag_relative_poses = []

        print("Waiting for VICON pose...")
        while self.vicon_pose is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        print("Got VICON pose.")

        print("Waiting for local position...")
        while self.current_height is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        print("Got local position.")


    def velocity_callback(self, msg):
        self.current_velocity = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.current_angular_velocity = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])

    def position_cal(self, msg):
        self.current_height = msg.pose.position.z
        self.current_horizontal_position = np.array([msg.pose.position.x, msg.pose.position.y])

    def vicon_cb(self, msg):
        # self.vicon_pose = ros_message_to_matrix(msg)
        # Instead save as [x, y, z, qx, qy, qz, qw]
        self.vicon_pose = np.array([
            msg.transform.translation.x,
            msg.transform.translation.y,
            msg.transform.translation.z,
            msg.transform.rotation.x,
            msg.transform.rotation.y,
            msg.transform.rotation.z,
            msg.transform.rotation.w
        ])

    def capture_image(self):
        img = self.camera.read()
        current_vicon_pose = self.vicon_pose

        if current_vicon_pose is None:
            rospy.logerr("VICON pose not available. Skipping image capture.")
            return False, None, None

        if img.max() < 10:
            rospy.logerr("Image is black. Skipping image capture.")
            return False, None, None

        self.previous_img_position = self.current_horizontal_position
        return True, img, current_vicon_pose

    def process_tag_img(self, img):
        """
        Finds the tag with the specified id in the image and computes our pose with respect to the tag
        """
        # convert to greyscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"Shape of image: {img.shape}")

        tags = detect_tags(img, use_ippe=True)
        if self.tag_id not in tags:
            rospy.logwarn("Tag not found in image.")
            return False, None
        
        return True, tags[self.tag_id]

    def process_img(self, img, vicon_pose):
        """
        Processes the image and records the necessary data
        """
        success, corners = self.process_tag_img(img)
        if not success:
            # Save the image as a failed img
            cv2.imwrite(str(self.out_folder / f"no_tag.jpg"), img)
            return False

        self.imgs.append(img)
        self.corners.append(corners)
        self.vicon_poses.append(vicon_pose)
        # self.tag_relative_poses.append(relative_pose)

        return True

    def should_image(self):
        # If we have not waited long enough since the last imaging, we should not image
        time_exceeded = (rospy.get_time() - self.last_imaging_time) > self.imaging_interval
        if not time_exceeded:
            # rospy.loginfo(f"Time since last imaging ({rospy.get_time() - self.last_imaging_time}) has not exceeded interval ({self.imaging_interval}). Not imaging.")
            return False

        # If our current velocity exceeds the threshold, we should not image
        if self.velocity_threshold is not None:
            if self.current_velocity is None:
                rospy.loginfo("Current velocity is None. Cannot threshold.")
                return False
            if np.any(np.abs(self.current_velocity) > self.velocity_threshold):
                rospy.loginfo(f"Current velocity ({self.current_velocity}) exceeds threshold ({self.velocity_threshold}). Not imaging.")
                return False

        # If we have not moved far enough since the last imaging, we should not image
        if self.distance_threshold is not None:
            # If previous position is none then we have never taken a picture so we should take one
            # Otherwise
            if self.previous_img_position is not None:
                if np.linalg.norm(self.current_horizontal_position - self.previous_img_position) < self.distance_threshold:
                    rospy.loginfo(f"Current position ({self.current_horizontal_position}) has not moved far enough from previous position ({self.previous_img_position}). Not imaging.")
                    return False

        # Check for height threshold
        if self.height_threshold is not None:
            if self.current_height < self.height_threshold:
                rospy.loginfo(f"Current height ({self.current_height}) is below threshold ({self.height_threshold}). Not imaging.")
                return False
        
        return True

    def process_outputs(self):
        """
        Saves the recorded data to the output folder as a numpy file and computes the
        extrinsics calibration matrix 
        """
        np.save(
            self.out_folder / "extrinsics_calibration_data.npy",
            {
                "imgs": self.imgs,
                "corners": self.corners,
                "vicon_poses": self.vicon_poses,
                # "tag_relative_poses": self.tag_relative_poses,
            }
        )

        compute_and_test_extrinsics(
            self.tag_id,
            self.tag_pose,
            self.tag_size_m,
            self.imgs,
            self.corners,
            self.vicon_poses,
            self.camera_matrix,
            self.dist_coeffs,
            self.out_folder,
            optimize=self.use_optimize
        )

        # Save the images to the output folder
        img_folder = self.out_folder / "images"
        img_folder.mkdir(parents=True, exist_ok=True)
        # Only save the last one, the rest should already be saved
        img_index = len(self.imgs) - 1
        cv2.imwrite(str(img_folder / f"image_{img_index}.jpg"), self.imgs[-1])

    def start_image_loop(self):
        rate = rospy.Rate(10)
        img_count = 0
        while not rospy.is_shutdown():
            if not self.should_image():
                rate.sleep()
                continue
            success, img, vicon_pose = self.capture_image()
            if success:
                success = self.process_img(img, vicon_pose)  # Finds the corners and saves the img, corners, and vicon pose
                if success:
                    img_count += 1
                    self.process_outputs()
            else:
                rospy.logerr("Image capture failed.")

            # Even if the capture was unsuccessful, we should update the last imaging time so that we do not
            # attempt to image again too soon
            self.last_imaging_time = rospy.get_time()
            rate.sleep()


if __name__ == "__main__":
    node = None
    recompute = cap_pkg_dir / "data" / "extrinsics_calibration" / "extrinsics_calibration_data.npy"
    # recompute = None
    try:
        node = ExtrinsicsCalibrationNode(
            tag_id=2,  # The large tag on each tag group has id 2*the number on the paper. So paper 1 has tag 2, paper 2 has tag 4, etc.
            tag_size_m=130 / 1000,
            recompute=recompute,
            use_optimize=True
        )
        if recompute is None:
            print("Starting image loop.")
            node.start_image_loop()
    except Exception as e:
        node.camera.release()
        raise(e)
    node.camera.release()
    print("Node shutdown.")