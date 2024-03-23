try:
    from geometry_msgs.msg import TransformStamped, PoseStamped
except ImportError:
    print("RUNNING IN NON-ROS ENVIRONMENT")

import numpy as np
from scipy.spatial.transform import Rotation


def transform_stamped_to_matrix(transform_stamped):
    """
    Convert a TransformStamped object to a homogeneous transformation matrix.
    """
    t = transform_stamped.transform.translation
    r = transform_stamped.transform.rotation
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix

def pose_stamped_to_matrix(pose_stamped):
    """
    Convert a PoseStamped object to a homogeneous transformation matrix.
    """
    p = pose_stamped.pose.position
    q = pose_stamped.pose.orientation
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    matrix[:3, 3] = [p.x, p.y, p.z]
    return matrix

def matrix_to_transform_stamped(matrix, frame_id, child_frame_id):
    """
    Convert a homogeneous transformation matrix to a TransformStamped object.
    """
    transform_stamped = TransformStamped()
    transform_stamped.header.frame_id = frame_id
    transform_stamped.child_frame_id = child_frame_id
    transform_stamped.transform.translation.x, transform_stamped.transform.translation.y, transform_stamped.transform.translation.z = matrix[:3, 3]
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    transform_stamped.transform.rotation.x, transform_stamped.transform.rotation.y, transform_stamped.transform.rotation.z, transform_stamped.transform.rotation.w = quat
    return transform_stamped

def matrix_to_pose_stamped(matrix, header=None):
    """
    Convert a homogeneous transformation matrix to a PoseStamped object.
    """
    pose_stamped = PoseStamped()
    if header is not None:
        pose_stamped.header = header
    pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z = matrix[:3, 3]
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    pose_stamped.pose.orientation.x, pose_stamped.pose.orientation.y, pose_stamped.pose.orientation.z, pose_stamped.pose.orientation.w = quat
    return pose_stamped

def ros_message_to_matrix(message):
    """
    Convert a ROS message (TransformStamped or PoseStamped) to a homogeneous transformation matrix.
    
    Args:
        message (geometry_msgs.msg.TransformStamped or geometry_msgs.msg.PoseStamped): ROS message.
        
    Returns:
        numpy.ndarray: 4x4 homogeneous transformation matrix.
    """
    if isinstance(message, TransformStamped):
        t = message.transform.translation
        r = message.transform.rotation
    elif isinstance(message, PoseStamped):
        t = message.pose.position
        r = message.pose.orientation
    else:
        raise ValueError("Unsupported message type. Expected TransformStamped or PoseStamped.")

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix

def euler_to_matrix(roll, pitch, yaw, degrees=False):
    """
    Convert roll-pitch-yaw Euler angles to a rotation matrix.
    
    Args:
        roll (float): Roll angle.
        pitch (float): Pitch angle.
        yaw (float): Yaw angle.
        degrees (bool): Whether the angles are in degrees (True) or radians (False).
        
    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """
    rotation = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    matrix = rotation.as_matrix()
    return matrix

def matrix_to_euler(matrix, degrees=False):
    """
    Convert a rotation matrix to roll-pitch-yaw Euler angles.
    
    Args:
        matrix (numpy.ndarray): 3x3 rotation matrix.
        degrees (bool): Whether to return the angles in degrees (True) or radians (False).
        
    Returns:
        tuple: Roll, pitch, and yaw angles.
    """
    rotation = Rotation.from_matrix(matrix)
    roll, pitch, yaw = rotation.as_euler('xyz', degrees=degrees)
    return roll, pitch, yaw

def matrix_to_params(matrix, type="euler"):
    """
    Convert a 4x4 homogeneous transformation matrix to a list of parameters.
    if type is "euler", the parameters are [x, y, z, roll, pitch, yaw].
    if type is "quaternion", the parameters are [x, y, z, qx, qy, qz, qw].
    """
    t = matrix[:3, 3]
    if type == "euler":
        rpy = matrix_to_euler(matrix[:3, :3])
        return np.concatenate([t, rpy])
    elif type == "quaternion":
        quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        return np.concatenate([t, quat])
    else:
        raise ValueError("Unsupported type. Expected 'euler' or 'quaternion'.")

def params_to_matrix(params, type="euler"):
    """
    Convert a list of parameters to a 4x4 homogeneous transformation matrix.
    if type is "euler", the parameters are [x, y, z, roll, pitch, yaw].
    if type is "quaternion", the parameters are [x, y, z, qx, qy, qz, qw].
    """
    t = params[:3]
    if type == "euler":
        rpy = params[3:]
        matrix = np.eye(4)
        matrix[:3, :3] = euler_to_matrix(*rpy)
        matrix[:3, 3] = t
        return matrix
    elif type == "quaternion":
        quat = params[3:]
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(quat).as_matrix()
        matrix[:3, 3] = t
        return matrix
    else:
        raise ValueError("Unsupported type. Expected 'euler' or 'quaternion'.")