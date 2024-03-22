from typing import List, Tuple, Dict
import numpy as np
from scipy.optimize import least_squares, approx_fprime
import cv2

from cap.transformation_lib import matrix_to_params, params_to_matrix

PxPositionMap = Dict[int, List[Tuple[int, int]]]  # image_index -> [(x, y)]
MmPositionMap = Dict[int, List[Tuple[float, float, float]]]  # image_index -> [(x, y, z=0)]
Pose = np.ndarray  # 4x4 homogeneous transformation matrix

def get_corner_A_mm(tag_size):
    """
    Gets the 3D positions of the tag corners in the tag frame. The z component is always 0.
    """
    # return np.array([
    #     [-tag_size/2, -tag_size/2, 0],
    #     [tag_size/2, -tag_size/2, 0],
    #     [tag_size/2, tag_size/2, 0],
    #     [-tag_size/2, tag_size/2, 0],
    # ])

    return np.array([
        [tag_size/2, -tag_size/2, 0],
        [-tag_size/2, -tag_size/2, 0],
        [-tag_size/2, tag_size/2, 0],
        [tag_size/2, tag_size/2, 0]
    ])

def get_tag_corners_mm(T_1_Ai, tag_corners_mm_Ai):
    """
    Gets the 3D coordinates of the corners of the tag in the base frame.
    """
    # Transform the tag corners from the tag frame to the base frame
    tag_corners_mm_1 = T_1_Ai @ np.hstack((tag_corners_mm_Ai, np.ones((len(tag_corners_mm_Ai),1)))).T
    tag_corners_mm_1 = tag_corners_mm_1[:3,:].T
    return tag_corners_mm_1

def estimate_T_Ci_Ai(tag_corners_px, tag_corners_mm_Ai, camera_matrix, dist_coeffs):
    """
    Estimates the transformation from the tag frame to the camera frame.
    tag_corners_px: nx2 numpy array of the corners of the tag in the image in pixel coordinates.
    tag_size: The size of the tag in mm.
    camera_matrix: 3x3 numpy array of the camera intrinsic parameters.
    dist_coeffs: 5x1 numpy array of the camera distortion coefficients.
    Returns:
    T_Ci_Ai: 4x4 numpy array of the transformation from the tag frame to the camera frame.
    """
    # Define the 3D points of the tag corners in the tag frame. We assume that the tag is centered at the origin.
    tag_corners_mm = get_tag_corners_mm(np.eye(4), tag_corners_mm_Ai)

    _, R, t = cv2.solvePnP(tag_corners_mm, tag_corners_px, camera_matrix, dist_coeffs)#, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    
    # Convert the rotation vector to a rotation matrix
    R, _ = cv2.Rodrigues(R)

    # Convert the rotation matrix and translation vector to a transformation matrix
    T_Ci_Ai = np.eye(4)
    T_Ci_Ai[:3,:3] = R
    T_Ci_Ai[:3,3] = t.flatten()

    return T_Ci_Ai

def get_pose_relative_to_apriltag(tag_corners_px, tag_size, camera_matrix, dist_coeffs):
    """
    Estimates the transformation from the tag frame to the camera frame.
    tag_corners_px: nx2 numpy array of the corners of the tag in the image in pixel coordinates.
    tag_size: The size of the tag in mm.
    camera_matrix: 3x3 numpy array of the camera intrinsic parameters.
    dist_coeffs: 5x1 numpy array of the camera distortion coefficients.
    Returns:
    T_A_C: 4x4 numpy array of the transformation from the camera frame to the tag frame
    """
    tag_corners_mm_Ai = get_corner_A_mm(tag_size)
    T_Ci_Ai = estimate_T_Ci_Ai(tag_corners_px, tag_corners_mm_Ai, camera_matrix, dist_coeffs)
    T_A_C = np.linalg.inv(T_Ci_Ai)
    return T_A_C

def optimize_tag_pose(
    initial_tag_pose: Pose,
    tag_px_positions: PxPositionMap,
    drone_poses: Dict[int, Pose],
    tag_size: float,
    # tag_corners_mm_Ai: MmPositionMap,
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
    Deprecated - tag_corners_mm_Ai: The 3D positions of the tag corners in the tag frame. The z component is always 0.
    camera_matrix: The camera intrinsics matrix
    distortion_coefficients: The camera distortion coefficients
    camera_extrinsics: The extrinsics of the camera in the Drone frame. Given as the transformation matrix T_drone_camera, that is the pose of the camera in the drone frame / transformation from camera to drone frame

    Returns:
    The optimized tag pose in the VICON frame
    """
    all_img_idxs = sorted(tag_px_positions.keys())
    # params = params_from_T(initial_tag_pose)  # [x, y, z, roll, pitch, yaw]
    params = matrix_to_params(initial_tag_pose, type='euler')

    tag_corners_mm_Ai = {img_idx: get_corner_A_mm(tag_size) for img_idx in all_img_idxs}

    def err_func(params):
        # tag_pose = T_from_params(params)
        tag_pose = params_to_matrix(params, type='euler')

        expected_pixels: PxPositionMap = get_expected_pixels(tag_pose, tag_px_positions, drone_poses, tag_corners_mm_Ai, camera_matrix, distortion_coefficients, camera_extrinsics)

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

        jacobian = find_jacobian(tag_pose, tag_px_positions, drone_poses, tag_corners_mm_Ai, camera_matrix, distortion_coefficients, camera_extrinsics)

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
    tag_corners_mm_Ai: MmPositionMap,
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
        tag_corner_position_mm = tag_corners_mm_Ai[img_idx]
        curr_corner_px_positions = tag_px_positions[img_idx]
        expected_pixels[img_idx] = []
        for corner_idx in range(len(tag_corner_position_mm)):
            # Get the position of the corner in the VICON frame
            tag_corner_pose_m = tag_corner_position_mm[corner_idx] / 1000  # Convert from mm to m
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
    tag_corners_mm_Ai: MmPositionMap,
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
    tag_corners_mm_Ai: The 3D positions of the tag corners in the tag frame. The z component is always 0.
    camera_matrix: The camera intrinsics matrix
    distortion_coefficients: The camera distortion coefficients
    camera_extrinsics: The extrinsics of the camera in the Drone frame. Given as the transformation matrix T_drone_camera, that is the pose of the camera in the drone frame / transformation from camera to drone frame

    Returns:
    The Jacobian matrix (2 * num_images * corners_per_tag, 6) for the error function
    Variable order: [x, y, z, roll, pitch, yaw]
    """
    all_img_idxs = sorted(tag_px_positions.keys())
    num_corners_per_tag = len(tag_corners_mm_Ai[all_img_idxs[0]])

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
        for corner_idx in range(len(tag_corners_mm_Ai[img_idx])):
            corner_pos_mm = tag_corners_mm_Ai[img_idx][corner_idx]

            # Compute intermediate values
            T_V_Ci = drone_pose @ camera_extrinsics
            R_V_Ci, t_V_Ci = get_rotation_and_translation(T_V_Ci)

            p_v = R_V_A @ corner_pos_mm + t_V_A  # (3,)
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

            d_pv_d_yaw   = X_bar_z @ R_A_z   @ R_A_y   @ R_A_x @ corner_pos_mm  # (3,)
            d_pv_d_pitch = R_A_z   @ X_bar_y @ R_A_y   @ R_A_x @ corner_pos_mm  # (3,)
            d_pv_d_roll  = R_A_z   @ R_A_y   @ X_bar_x @ R_A_x @ corner_pos_mm  # (3,)

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
    