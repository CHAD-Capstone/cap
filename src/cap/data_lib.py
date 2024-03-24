"""
This file contains default values for locations of data files and
functions to easily load and save data.
"""

import numpy as np
from pathlib import Path
import shutil
import datetime

from cap.apriltag_pose_estimation_lib import AprilTagMap

file_dir = Path(__file__).resolve().parent
cap_pkg_dir = file_dir.parent.parent
assert (cap_pkg_dir / "CMakeLists.txt").exists(), f"cap_pkg_dir: {cap_pkg_dir} does not appear to be the root of the cap package."

DATA_DIR = cap_pkg_dir / "data"

FINAL_CALIBRATION_DIR = DATA_DIR / "final_calibration"
FLIGHT_DATA_DIR = DATA_DIR / "flight_data"
FLIGHT_DATA_ARCHIVE_DIR = DATA_DIR / "flight_data_archive"

def load_final_intrinsics_calibration():
    """
    Loads the camera matrix and distortion coefficients from the final calibration directory.
    Returns a tuple of (camera_matrix, dist_coeffs).

    camera_matrix: 3x3 numpy array
    dist_coeffs: 1x5 numpy array
    """
    cal_file = FINAL_CALIBRATION_DIR / "cal.npz"
    cal_data = np.load(cal_file.absolute().as_posix())
    camera_matrix = cal_data['mtx']
    dist_coeffs = cal_data['dist']
    return camera_matrix, dist_coeffs

def load_final_extrinsics():
    """
    Returns a 4x4 homogenous transformation matrix representing
    T_marker_camera, the transformation from the camera frame to the marker frame.
        Or in other words the pose of the camera in the marker frame.
    """
    extrinsics_file = FINAL_CALIBRATION_DIR / "extrinsics_calibration.npy"
    T_marker_camera = np.load(extrinsics_file.absolute().as_posix())
    return T_marker_camera

def archive_existing_current_flight_data():
    """
    Moves the current flight data directory to the flight data archive directory and creates a new
    empty current flight data directory.

    Uses the current date and time as the name of the archived directory.

    NOTE: If the time has not been updated since the last startup the date will be wrong. We fix that by
    running `ssh capdrone 'sudo date -s "$(date +"%Y-%m-%d %T")"'` after the drone starts up
    """
    if not FLIGHT_DATA_DIR.exists():
        FLIGHT_DATA_DIR.mkdir()
        return

    if not FLIGHT_DATA_ARCHIVE_DIR.exists():
        print(f"Creating flight data archive directory: {FLIGHT_DATA_ARCHIVE_DIR}")
        FLIGHT_DATA_ARCHIVE_DIR.mkdir()

    now = datetime.datetime.now()
    archive_dir = FLIGHT_DATA_ARCHIVE_DIR / now.strftime("%Y-%m-%d_%H-%M-%S")
    shutil.move(FLIGHT_DATA_DIR, archive_dir)
    FLIGHT_DATA_DIR.mkdir()

def save_current_flight_tag_map(tag_map: AprilTagMap):
    """
    Saves the current flight tag map to the flight data directory.
    """
    tag_map.save_to_file(FLIGHT_DATA_DIR)

def load_current_flight_tag_map():
    """
    Returns a dictionary mapping the tag id to the tag name for the current flight.
    """
    tag_map = AprilTagMap.load_from_file(FLIGHT_DATA_DIR)
    return tag_map

if __name__ == "__main__":
    # Archive existing flight data
    archive_existing_current_flight_data()

    camera_matrix, dist_coeffs = load_final_intrinsics_calibration()
    assert camera_matrix.shape == (3, 3), f"camera_matrix.shape: {camera_matrix.shape}"
    assert dist_coeffs.shape == (1, 5), f"dist_coeffs.shape: {dist_coeffs.shape}"

    T_marker_camera = load_final_extrinsics()
    assert T_marker_camera.shape == (4, 4)

    tag_map = AprilTagMap()
    tag_map.add_tag_pose(0, np.eye(4))
    save_current_flight_tag_map(tag_map)
    loaded_tag_map = load_current_flight_tag_map()
    assert np.allclose(loaded_tag_map.get_pose_homogeneous(0), np.eye(4))
    assert np.allclose(loaded_tag_map.get_pose(0), np.array([0, 0, 0, 0, 0, 0, 1]))  # [x, y, z, qx, qy, qz, qw]

    print("All tests passed!")