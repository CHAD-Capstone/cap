# CHAD Drone Main Node

Nodes are found in the `/scripts` folder and library files are in `/src/cap`. Files in `/src/cap` can be imported by nodes in `/scripts` (or other library files) by specifying the `cap` package. So to import transformation helpers you could do `from cap import transformation_lib`.

## Library Files:
### transformation_lib
This library defines helper functions for transforming between different representations of frames. Specifically, it can transform between the ROS msg types `TransformStamped`, `PoseStamped`, and a 4x4 homogenous transformation matrix (`transform_stamped_to_matrix`, `pose_stamped_to_matrix`, `matrix_to_transform_stamped`, `matrix_to_pose_stamped`). For optimization problems where the parameters are encoded as `[x, y, z, rx, ry, rz]` or `[x, y, z, qx, qy, qz, qw]` you can use `matrix_to_params` or `params_to_matrix` to transformation back and forth between homogenous transformation matrices and parameters.

### apriltag_pose_estimation_lib
This library has helper functions for:
1. Estimating the pose of an AprilTag in the camera frame (`estimate_T_C_A`)
2. Getting the pose of the camera in the AprilTag frame (`get_pose_relative_to_apriltag`)
3. Optimizing the pose of an AprilTag in the VICON frame given multiple images of the tag (`optimize_tag_pose`)
4. Extract the corner locations of AprilTags from an image using the IPPE order (`detect_tags`)
5. Get the position of the drone (marker) in the VICON frame (`get_marker_pose`)