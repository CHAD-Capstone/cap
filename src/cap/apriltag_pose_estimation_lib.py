from typing import List, Tuple, Dict
import numpy as np
from scipy.optimize import least_squares, approx_fprime
import cv2
from pupil_apriltags import Detector
from pathlib import Path
from typing import Union

from cap.transformation_lib import matrix_to_params, params_to_matrix

PxPositionMap = Dict[int, List[Tuple[int, int]]]  # image_index -> [(x, y)]
MPositionMap = Dict[int, List[Tuple[float, float, float]]]  # image_index -> [(x, y, z=0)]
Pose = np.ndarray  # 4x4 homogeneous transformation matrix

detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

class AprilTagMap:
    """
    Stores the poses of AprilTags in the VICON frame and allows for easy saving and loading of the data
    Poses are stored as [x, y, z, qx, qy, qz, qw]
    """
    def __init__(self):
        self.tag_poses = {}  # Maps from tag_id to pose

    def tag_ids(self):
        return sorted(list(self.tag_poses.keys()))

    def add_tag_pose(self, tag_id, pose):
        if tag_id in self.tag_poses:
            print(f"Tag with id {tag_id} already exists. Overwriting the pose.")
        if pose.shape == (7,):
            self.tag_poses[tag_id] = pose
        elif pose.shape == (4, 4):
            self.tag_poses[tag_id] = matrix_to_params(pose, type='quaternion')
        else:
            raise ValueError("Pose must be in either quaternion or matrix form")

    def get_pose(self, tag_id):
        return self.tag_poses[tag_id]

    def get_pose_homogeneous(self, tag_id):
        return params_to_matrix(self.tag_poses[tag_id], type='quaternion')

    def save_to_file(self, dir_name: Union[Path, str]):
        if isinstance(dir_name, str):
            dir_name = Path(dir_name)
        if dir_name.is_dir():
            filepath = dir_name / "tag_poses.npy"
        elif dir_name.suffix == ".npy":
            filepath = dir_name
        else:
            raise ValueError("Invalid file path. Must be an existing directory or a .npy file.")
        print(f"Saving tag poses to {filepath}")
        np.save(filepath, self.tag_poses)

    @classmethod
    def load_from_file(cls, dir_name: Union[Path, str]):
        if isinstance(dir_name, str):
            dir_name = Path(dir_name)
        if dir_name.is_dir():
            filename = dir_name / "tag_poses.npy"
        else:
            filename = dir_name
        if not filename.exists():
            raise FileNotFoundError(f"File {filename} does not exist.")
        tag_poses = np.load(filename, allow_pickle=True).item()
        tag_map = cls()
        tag_map.tag_poses = tag_poses
        return tag_map

def detect_tags(img, use_ippe=True):
    """
    Finds the AprilTags in the image and returns their pixel location.
    Pixel locations can be permuted to match the order needed to use SOLVEPNP_IPPE_SQUARE in the solvePnP function.

    Returns:
    tag_positions: Map of tag_id -> corners_px
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(img)
    tag_positions = {}
    for tag in tags:
        corners_px = tag.corners
        if use_ippe:
            corners_px = corners_px[[0, 3, 2, 1]]  # Gets into the order defined by https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
        tag_positions[tag.tag_id] = corners_px
    return tag_positions

def get_corner_A_m(tag_size, use_ippe=True):
    """
    Gets the 3D positions of the tag corners in the tag frame. The z component is always 0.
    """
    base = np.array([
        [tag_size/2, tag_size/2, 0],
        [-tag_size/2, tag_size/2, 0],
        [-tag_size/2, -tag_size/2, 0],
        [tag_size/2, -tag_size/2, 0],
    ])
    if use_ippe:
        # For use with the SOLVEPNP_IPPE_SQUARE flag
        return base[[1, 0, 3, 2]]  # As defined by https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
    else:
        return base[[3, 2, 1, 0]]

def get_tag_corners_m(T_1_Ai, tag_corners_m_Ai):
    """
    Gets the 3D coordinates of the corners of the tag in the base frame.
    """
    # Transform the tag corners from the tag frame to the base frame
    tag_corners_m_1 = T_1_Ai @ np.hstack((tag_corners_m_Ai, np.ones((len(tag_corners_m_Ai),1)))).T
    tag_corners_m_1 = tag_corners_m_1[:3,:].T
    return tag_corners_m_1

def estimate_T_C_A(tag_corners_px, tag_size, camera_matrix, dist_coeffs, use_ippe=True):
    """
    Estimates the transformation from the tag frame to the camera frame.
    tag_corners_px: nx2 numpy array of the corners of the tag in the image in pixel coordinates.
    tag_size: The size of the tag in m.
    camera_matrix: 3x3 numpy array of the camera intrinsic parameters.
    dist_coeffs: 5x1 numpy array of the camera distortion coefficients.
    Returns:
    T_Ci_Ai: 4x4 numpy array of the transformation from the tag frame to the camera frame.
    """
    tag_corners_m_Ai = get_corner_A_m(tag_size, use_ippe)

    # Define the 3D points of the tag corners in the tag frame. We assume that the tag is centered at the origin.
    tag_corners_m = get_tag_corners_m(np.eye(4), tag_corners_m_Ai)

    if use_ippe:
        # For use with the SOLVEPNP_IPPE_SQUARE flag
        _, R, t = cv2.solvePnP(tag_corners_m, tag_corners_px, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    else:
        _, R, t = cv2.solvePnP(tag_corners_m, tag_corners_px, camera_matrix, dist_coeffs)
    
    # Convert the rotation vector to a rotation matrix
    R, _ = cv2.Rodrigues(R)

    # Convert the rotation matrix and translation vector to a transformation matrix
    T_Ci_Ai = np.eye(4)
    T_Ci_Ai[:3,:3] = R
    T_Ci_Ai[:3,3] = t.flatten()

    return T_Ci_Ai

def get_pose_relative_to_apriltag(tag_corners_px, tag_size, camera_matrix, dist_coeffs, use_ippe=True):
    """
    Estimates the transformation from the tag frame to the camera frame.
    tag_corners_px: nx2 numpy array of the corners of the tag in the image in pixel coordinates.
    tag_size: The size of the tag in m.
    camera_matrix: 3x3 numpy array of the camera intrinsic parameters.
    dist_coeffs: 5x1 numpy array of the camera distortion coefficients.
    Returns:
    T_A_C: 4x4 numpy array of the transformation from the camera frame to the tag frame
    """
    T_C_A = estimate_T_C_A(tag_corners_px, tag_size, camera_matrix, dist_coeffs, use_ippe)
    T_A_C = np.linalg.inv(T_C_A)
    return T_A_C

def get_marker_pose(tag_pose, tag_corners_px, tag_size, camera_matrix, dist_coeffs, extrinsics, use_ippe=True):
    """
    Estimates the pose of the VICON marker in the VICON frame
    Parameters:
    tag_pose: The pose of the tag in the VICON frame (T_VICON_tag)
    tag_corners_px: The pixel positions of the tag corners in the image
    tag_size: The size of the tag in m. Used to compute the 3D positions of the tag corners in the tag frame
    camera_matrix: The camera intrinsics matrix
    dist_coeffs: The camera distortion coefficients
    extrinsics: The extrinsics of the camera in the Drone frame. Given as the transformation matrix T_marker_camera, that is the pose of the camera in the drone frame / transformation from camera to drone frame
    """
    # Get the pose of the camera with respect to the AprilTag frame (T_tag_cam)
    T_tag_cam = get_pose_relative_to_apriltag(tag_corners_px, tag_size, camera_matrix, dist_coeffs, use_ippe)
    # Get the pose of the marker with respect to the tag frame
    # T_tag_marker = T_tag_cam @ T_marker_cam
    T_tag_marker = T_tag_cam @ extrinsics
    # Then we get the position of the marker with respect to the VICON frame
    # T_VICON_marker = T_VICON_tag @ T_tag_marker
    T_VICON_marker = tag_pose @ T_tag_marker
    return T_VICON_marker


def optimize_tag_pose(
    initial_tag_pose: Pose,
    tag_px_positions: PxPositionMap,
    drone_poses: Dict[int, Pose],
    tag_size: float,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    camera_extrinsics: Pose,
):
    """
    Optimize the estimated location of the tag in the VICON frame using the AprilTag corners and the drone poses
    Parameters:
    initial_tag_pose: The initial estimate of the tag pose in the VICON frame
    tag_px_positions: The pixel positions of the tag corners in the image
    drone_poses: The poses of the drone when the images were taken. Given as the transformation matrix T_VICON_drone, that is the pose of the drone in the VICON frame / transformation from drone to VICON frame
    tag_size: The size of the tag in meters. Used to compute the 3D positions of the tag corners in the tag frame
    camera_matrix: The camera intrinsics matrix
    distortion_coefficients: The camera distortion coefficients
    camera_extrinsics: The extrinsics of the camera in the Drone frame. Given as the transformation matrix T_drone_camera, that is the pose of the camera in the drone frame / transformation from camera to drone frame

    Returns:
    The optimized tag pose in the VICON frame
    """
    all_img_idxs = sorted(tag_px_positions.keys())
    # params = params_from_T(initial_tag_pose)  # [x, y, z, roll, pitch, yaw]
    params = matrix_to_params(initial_tag_pose, type='euler')

    tag_corners_m_Ai = {img_idx: get_corner_A_m(tag_size) for img_idx in all_img_idxs}

    def err_func(params):
        # tag_pose = T_from_params(params)
        tag_pose = params_to_matrix(params, type='euler')

        expected_pixels: PxPositionMap = get_expected_pixels(tag_pose, tag_px_positions, drone_poses, tag_corners_m_Ai, camera_matrix, distortion_coefficients, camera_extrinsics)

        all_expected_pixels = []
        all_actual_pixels = []
        for img_idx in all_img_idxs:
            expected = expected_pixels[img_idx]
            actual = tag_px_positions[img_idx]
            all_expected_pixels.extend(expected)
            all_actual_pixels.extend(actual)
        all_expected_pixels = np.array(all_expected_pixels)
        all_actual_pixels = np.array(all_actual_pixels)

        error = all_expected_pixels - all_actual_pixels

        return error

    def jac_func(params):
        # tag_pose = T_from_params(params)
        tag_pose = params_to_matrix(params, type='euler')

        jacobian = find_jacobian(tag_pose, tag_px_positions, drone_poses, tag_corners_m_Ai, camera_matrix, distortion_coefficients, camera_extrinsics)

        return jacobian

    # result = least_squares(error_func, params, jac=jacobian_func, verbose=2)
    result = least_squares(error_func, params, jac='3-point', verbose=2)

    optimal_params = result.x
    # optimal_tag_pose = T_from_params(optimal_params)
    optimal_tag_pose = params_to_matrix(optimal_params, type='euler')

    return optimal_tag_pose

    
def get_rotation_and_translation(pose: Pose) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the rotation and translation from the pose matrix
    Parameters:
    pose: The pose matrix

    Returns:
    The rotation and translation
    """
    return pose[:3, :3], pose[:3, 3]

def get_expected_pixels(
    tag_pose: Pose,
    tag_px_positions: PxPositionMap,
    drone_poses: Dict[int, Pose],
    tag_corners_m_Ai: MPositionMap,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    camera_extrinsics: Pose,
) -> PxPositionMap:
    """
    Get the expected pixel positions of the tag corners in the image
    Parameters:
    tag_pose: The pose of the tag in the VICON frame
    tag_px_positions: The pixel positions of the tag corners in the image
    drone_poses: The poses of the drone when the images were taken. Given as the transformation matrix T_VICON_drone, that is the pose of the drone in the VICON frame / transformation from drone to VICON frame
    tag_corners_mm_Ai: The 3D positions of the tag corners in the tag frame. The z component is always 0.
    camera_matrix: The camera intrinsics matrix
    distortion_coefficients: The camera distortion coefficients
    camera_extrinsics: The extrinsics of the camera in the Drone frame. Given as the transformation matrix T_drone_camera, that is the pose of the camera in the drone frame / transformation from camera to drone frame

    Returns:
    The expected pixel positions of the tag corners in the image
    """
    expected_pixels = {}
    R_V_A, t_V_A = get_rotation_and_translation(tag_pose)  # Transformation to VICON from AprilTag

    for img_idx, drone_pose in drone_poses.items():
        tag_corner_position_m = tag_corners_m_Ai[img_idx]
        curr_corner_px_positions = tag_px_positions[img_idx]
        expected_pixels[img_idx] = []
        for corner_idx in range(len(tag_corner_position_m)):
            # Get the position of the corner in the VICON frame
            tag_corner_pose_m = tag_corner_position_m[corner_idx]
            p_V = R_V_A @ tag_corner_pose_m + t_V_A

            # Get the position of the corner in the camera frame
            curr_drone_pose = drone_poses[img_idx]
            T_V_Ci = curr_drone_pose @ camera_extrinsics  # Transformation to VICON from camera frame
            R_V_Ci, t_V_Ci = get_rotation_and_translation(T_V_Ci)

            p_Ci = R_V_Ci.T @ (p_V - t_V_Ci)  # Taking T_Ci_V @ p_V = (T_V_Ci)^-1 @ p_V = R_V_Ci.T @ (p_V - t_V_Ci)

            # Project the position of the corner in the camera frame to the image plane and undistort them
            p_px = cv2.projectPoints(p_Ci, np.zeros(3), np.zeros(3), camera_matrix, distortion_coefficients)[0][0][0]

            expected_pixels[img_idx].append(p_px)

    return expected_pixels
    
def find_jacobian(
    tag_pose: Pose,
    tag_px_positions: PxPositionMap,
    drone_poses: Dict[int, Pose],
    tag_corners_m_Ai: MPositionMap,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    camera_extrinsics: Pose,
) -> np.ndarray:
    """
    Find the Jacobian matrix for the error function
    Parameters:
    tag_pose: The pose of the tag in the VICON frame
    tag_px_positions: The pixel positions of the tag corners in the image
    drone_poses: The poses of the drone when the images were taken. Given as the transformation matrix T_VICON_drone, that is the pose of the drone in the VICON frame / transformation from drone to VICON frame
    tag_corners_m_Ai: The 3D positions of the tag corners in the tag frame. The z component is always 0.
    camera_matrix: The camera intrinsics matrix
    distortion_coefficients: The camera distortion coefficients
    camera_extrinsics: The extrinsics of the camera in the Drone frame. Given as the transformation matrix T_drone_camera, that is the pose of the camera in the drone frame / transformation from camera to drone frame

    Returns:
    The Jacobian matrix (2 * num_images * corners_per_tag, 6) for the error function
    Variable order: [x, y, z, roll, pitch, yaw]
    """
    all_img_idxs = sorted(tag_px_positions.keys())
    num_corners_per_tag = len(tag_corners_m_Ai[all_img_idxs[0]])

    jacobian = np.zeros((2 * len(all_img_idxs) * num_corners_per_tag, 6))

    # Pre-baked matrices used to take derivatives of 1 axis rotation matrices
    X_bar_z = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 0]])
    X_bar_y = np.array([[0, 0, 1],[0, 0, 0],[-1, 0, 0]])
    X_bar_x = np.array([[0, 0, 0],[0, 0, -1],[0, 1, 0]])

    R_V_A, t_V_A = get_rotation_and_translation(tag_pose)  # Transformation to VICON from AprilTag
    # Split the rotation matrix into the 3 axis rotations
    r, p, y = rpy_from_dcm(R_V_A).flatten()
    R_A_z = dcm_from_rpy(np.array([0, 0, y]))
    R_A_y = dcm_from_rpy(np.array([0, p, 0]))
    R_A_x = dcm_from_rpy(np.array([r, 0, 0]))

    for img_idx in all_img_idxs:
        drone_pose = drone_poses[img_idx]
        curr_corner_px_positions = tag_px_positions[img_idx]
        for corner_idx in range(len(tag_corners_m_Ai[img_idx])):
            corner_pos_m = tag_corners_m_Ai[img_idx][corner_idx]

            # Compute intermediate values
            T_V_Ci = drone_pose @ camera_extrinsics
            R_V_Ci, t_V_Ci = get_rotation_and_translation(T_V_Ci)

            p_v = R_V_A @ corner_pos_m + t_V_A  # (3,)
            p_ci = R_V_Ci.T @ (p_v - t_V_Ci)  # (3,)
            yi = camera_matrix @ p_ci  # (3,)

            # Intermediate derivatives
            d_xi_d_yi = np.array([
                [1 / yi[2], 0, -yi[0] / yi[2]**2],
                [0, 1 / yi[2], -yi[1] / yi[2]**2]
            ])  # (2, 3)

            d_yi_d_p_ci = camera_matrix  # (3, 3)

            d_p_ci_d_p_v = R_V_Ci.T  # (3, 3)

            d_xi_d_pv = d_xi_d_yi @ d_yi_d_p_ci @ d_p_ci_d_p_v  # (2, 3)

            d_pv_d_yaw   = X_bar_z @ R_A_z   @ R_A_y   @ R_A_x @ corner_pos_m  # (3,)
            d_pv_d_pitch = R_A_z   @ X_bar_y @ R_A_y   @ R_A_x @ corner_pos_m  # (3,)
            d_pv_d_roll  = R_A_z   @ R_A_y   @ X_bar_x @ R_A_x @ corner_pos_m  # (3,)

            # Compute rotational derivatives
            d_xi_d_yaw = d_xi_d_pv @ d_pv_d_yaw  # (2,)
            d_xi_d_pitch = d_xi_d_pv @ d_pv_d_pitch  # (2,)
            d_xi_d_roll = d_xi_d_pv @ d_pv_d_roll  # (2,)

            # Compute translational derivatives
            d_xi_d_tA = d_xi_d_pv @ np.eye(3)  # (2, 3)

            # Arrange the final jacobian
            start_index = (img_idx * num_corners_per_tag + corner_idx) * 2
            end_index = start_index + 2
            jacobian[start_index:start_index+2, 0] = d_xi_d_tA[0]
            jacobian[start_index:start_index+2, 1] = d_xi_d_tA[1]
            jacobian[start_index:start_index+2, 2] = d_xi_d_tA[2]
            jacobian[start_index:start_index+2, 3] = d_xi_d_roll
            jacobian[start_index:start_index+2, 4] = d_xi_d_pitch
            jacobian[start_index:start_index+2, 5] = d_xi_d_yaw

    return jacobian
    