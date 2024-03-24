#!/usr/bin/python3

"""
A node that allows for setting positions in the VICON frame even when the local position estimate is coming from realsense
We do this by defining a transformation matrix from the VICON frame to the local frame
Then when we get a message from the /capdrone/setpoint_position/local topic we transform the position from the VICON frame to the local frame

Finally, we also publish a PoseStamped to /capdrone/local_position/pose which is the local position estimate in the VICON frame
"""

import rospy
import numpy as np

from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Transform
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool

from cap_srvs.srv import SetLocalizationMode, SetLocalizationModeResponse
from cap_srvs.srv import IsReady, IsReadyResponse

from cap_msgs.msg import TagTransformArray

from cap.transformation_lib import transform_stamped_to_matrix, pose_stamped_to_matrix, matrix_to_params, matrix_to_pose_stamped
from cap.timestamp_queue import TimestampQueue

class ViconPositionSetNode:
    def __init__(self, group_number, pass_though, use_vicon, use_apriltag_loc, queue_length=100):
        node_name = 'vicon_set_position_{:02d}'.format(group_number)
        rospy.init_node(node_name)
        print("Starting VICON Node with name", node_name)
        self.ready = False
        self.ready_state = "Starting"

        # If pass_through is true then it is assumed that the realsense and VICON frames always coincide
        self.pass_through = pass_though
        # If use_vicon is true then the transform will be based off of ground truth VICON data
        self.use_vicon = use_vicon
        # If use_apriltag_loc is true then the transform will be based off of the estimated VICON data from the localization node
        self.use_apriltag_loc = use_apriltag_loc

        # VICON Inputs
        # Ground truth VICON from the real system
        self.vicon_sub = rospy.Subscriber('/vicon/ROB498_Drone/ROB498_Drone', TransformStamped, callback=self.vicon_cb)
        # Estimated VICON positions from the localization node
        self.apriltag_localization_sub = rospy.Subscriber('/capdrone/apriltag_localization/pose', TagTransformArray, callback=self.apriltag_localization_cb)

        # Outputs
        # MAVROS Setpoint Position Publisher for publishing in the local/realsense frame
        self.setpoint_publisher = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)

        self.current_corrected_position = None  # P^VICON - The current corrected position in the VICON frame
        self.current_corrected_velocity = None  # V^VICON - The current corrected velocity in the VICON frame
        # Publisher for the position that has been transformed from the local frame to the VICON frame
        self.corrected_position_pub = rospy.Publisher('/capdrone/local_position/pose', PoseStamped, queue_size=10)
        self.corrected_velocity_pub = rospy.Publisher('/capdrone/local_position/velocity_local', TwistStamped, queue_size=10)

        # Drone Local Position Input
        self.current_uncorrected_position = None  # P^realsense - The current uncorrected position in the local frame
        self.current_uncorrected_velocity = None  # V^realsense - The current uncorrected velocity in the local frame
        # Subscriber for the local position estimate from the realsense. This is the same as the uncurrected position.
        self.local_position_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, callback=self.local_position_cb)
        self.local_velocity_sub = rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, callback=self.local_velocity_cb)

        # Setpoint Input
        # Subscriber for the desired position in the VICON frame. This is the position we want the drone to move to.
        self.capdrone_set_position_local_sub = rospy.Subscriber('/capdrone/setpoint_position/local', PoseStamped, callback=self.set_desired_position_local_cb)
        self.current_desired_corrected_pose = None

        self.local_position_queue = TimestampQueue(max_length=queue_length)

        # Service call to set the localization mode (VICON, APRILTAG, REALSENSE)
        self.set_localization_mode_service = rospy.Service('/capdrone/set_localization_mode', SetLocalizationMode, self.set_localization_mode_cb)
        self.is_ready_service = rospy.Service('/capdrone/vicon_set_position/ready', IsReady, self.is_ready_cb)

        # At the start, we assume that the VICON frame coincides with the local frame
        self.T_realsense_VICON = np.eye(4)

        self.got_VICON = False

        # If we are using VICON, we need to wait for the first VICON message before we can start
        if self.use_vicon:
            self.ready_state = "Waiting for VICON"
            print("Waiting for VICON...")
            while not self.got_VICON and not rospy.is_shutdown():
                rospy.sleep(0.1)
        
        # In every case we need to have at least one local position message before we can start
        print("Waiting for local position...")
        self.ready_state = "Waiting for local position"
        while self.current_uncorrected_position is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        print("Ready to set positions")
        self.ready = True
        self.ready_state = "Ready"
        rospy.spin()

    def is_ready_cb(self, req):
        return IsReadyResponse(self.ready, self.ready_state)

    def set_localization_mode_cb(self, req):
        """
        Service callback for setting the localization mode
        """
        if req.data == "VICON":
            # Verify that we are getting VICON data
            if not self.got_VICON:
                rospy.logwarn("Cannot set localization mode to VICON because we have not received any VICON data")
                return SetLocalizationModeResponse(False)
            rospy.loginfo("Setting localization mode to VICON")
            if self.use_vicon:
                return SetLocalizationModeResponse(True)
            self.use_vicon = True
            self.use_apriltag_loc = False
            self.pass_through = False
            return SetLocalizationModeResponse(True)
        elif req.data == "APRILTAG":
            rospy.loginfo("Setting localization mode to APRILTAG")
            if self.use_apriltag_loc:
                return SetLocalizationModeResponse(True)
            self.use_vicon = False
            self.use_apriltag_loc = True
            self.pass_through = False
            return SetLocalizationModeResponse(True)
        elif req.data == "REALSENSE":
            rospy.loginfo("Setting localization mode to REALSENSE")
            self.use_vicon = False
            self.use_apriltag_loc = False
            self.pass_through = True
            return SetLocalizationModeResponse(True)
        else:
            rospy.logwarn(f"Invalid localization mode: {req.data}")
            return SetLocalizationModeResponse(False)

    def set_desired_position_local_cb(self, msg):
        """
        Callback for the setpoint position subscriber
        """
        self.current_desired_corrected_pose = msg
        # Now we project the desired position from the VICON frame into the local frame
        # Transform the desired pose in a matrix. This matrix can be though of as T_VICON_desired
        desired_pose_matrix = pose_stamped_to_matrix(msg)
        # Now we want T_realseanse_desired = T_realsense_VICON @ T_VICON_desired
        T_realsense_desired = self.T_realsense_VICON @ desired_pose_matrix
        corrected_pose_msg = matrix_to_pose_stamped(T_realsense_desired, header=msg.header)
        # And then we publish this to the mavros setpoint position to get the drone to move there
        self.setpoint_publisher.publish(corrected_pose_msg)

    def update_transform(self, vicon_timestamp, vicon_transform: TransformStamped):
        """
        Takes in a VICON transform and updates the transformation matrix
        The VICON transform is T_VICON_marker, that is the transform from the marker frame to the VICON frame, or the position of the marker in the VICON frame
        """
        T_VICON_marker = transform_stamped_to_matrix(vicon_transform)
        # We get the corresponding local position by searching in the local position queue for the closest timestamp
        local_position_elems = self.local_position_queue.search(vicon_timestamp)
        if len(local_position_elems) == 1:
            local_position = pose_stamped_to_matrix(local_position_elems[0].data)
        else:
            # Then we should interpolate between the two closest local positions, but that is hard so for now we just use the closest one
            ts1, ts2 = local_position_elems[0].timestamp, local_position_elems[1].timestamp
            if abs(ts1 - vicon_timestamp) < abs(ts2 - vicon_timestamp):
                local_position = pose_stamped_to_matrix(local_position_elems[0].data)
            else:
                local_position = pose_stamped_to_matrix(local_position_elems[1].data)
        
        # The local position is T_realsense_marker
        # Then to get the transform from the VICON frame to the realsense frame we need
        # T_realsense_VICON = T_realsense_marker @ T_marker_VICON
        T_realsense_VICON = local_position @ np.linalg.inv(T_VICON_marker)
        self.T_realsense_VICON = T_realsense_VICON
        self.update_corrected_position()

    def update_corrected_position(self):
        """
        Uses the current uncorrected position and the current transformation matrix to update the corrected position
        and publishes it
        """
        if self.current_uncorrected_position is None:
            return
        if self.pass_through:
            # Then our corrected position is the same as the uncorrected position
            self.corrected_position_pub.publish(self.current_uncorrected_position)
            self.current_corrected_position = self.current_uncorrected_position
        else:
            # Then we need to project this position into the VICON frame
            current_header = self.current_uncorrected_position.header
            # Get the transform from the local frame to the VICON frame
            T_VICON_realsense = np.linalg.inv(self.T_realsense_VICON)

            corrected_position = T_VICON_realsense @ pose_stamped_to_matrix(self.current_uncorrected_position)
            corrected_pose_msg = matrix_to_pose_stamped(corrected_position, header=current_header)

            self.corrected_position_pub.publish(corrected_pose_msg)
            self.current_corrected_position = corrected_pose_msg

    def local_position_cb(self, msg):
        """
        Updates the current corrected and uncorrected local positions
        """
        self.local_position_queue.enqueue(msg.header.stamp.to_sec(), msg)
        self.current_uncorrected_position = msg
        self.update_corrected_position()

    def update_corrected_velocity(self):
        """
        Uses the current uncorrected velocity and the current transformation matrix to update the corrected velocity
        and publishes it
        """
        if self.current_uncorrected_velocity is None:
            return
        if self.pass_through:
            # Then our corrected velocity is the same as the uncorrected velocity
            self.corrected_velocity_pub.publish(self.current_uncorrected_velocity)
            self.current_corrected_velocity = self.current_uncorrected_velocity
        else:
            # Then we need to project this velocity into the VICON frame
            current_header = self.current_uncorrected_velocity.header
            # Get the transform from the local frame to the VICON frame
            T_VICON_realsense = np.linalg.inv(self.T_realsense_VICON)

            twist = self.current_uncorrected_velocity.twist
            linear_velocity = np.array([twist.linear.x, twist.linear.y, twist.linear.z])
            angular_velocity = np.array([twist.angular.x, twist.angular.y, twist.angular.z])

            R = T_VICON_realsense[:3, :3]
            corrected_linear_velocity = R @ linear_velocity
            corrected_angular_velocity = R @ angular_velocity

            corrected_twist = TwistStamped()
            corrected_twist.header = current_header
            corrected_twist.twist.linear = Vector3(*corrected_linear_velocity)
            corrected_twist.twist.angular = Vector3(*corrected_angular_velocity)

            self.corrected_velocity_pub.publish(corrected_twist)
            self.current_corrected_velocity = corrected_twist

    def local_velocity_cb(self, msg):
        self.current_uncorrected_velocity = msg
        self.update_corrected_velocity()

    def vicon_cb(self, msg):
        if self.use_vicon:
            # vicon_timestamp = msg.header.stamp.to_sec()
            vicon_timestamp = rospy.get_time()  # The VICON timestamp is misaligned with our local ros time
            vicon_transform = msg.transform
            self.update_transform(vicon_timestamp, vicon_transform)
            self.got_VICON = True

    def apriltag_localization_cb(self, msg):
        if self.use_apriltag_loc:
            # We can get multiple transforms from the localization node and we need to select one to do the transformation
            # We use the heuristic that poses that are closer to our current estimated position are more likely to be correct
            if current_corrected_position is None:
                # Then we just use the first one
                apriltag_transform = msg.transforms[0]
            else:
                # For now, we only use the position part of the transform
                current_corrected_position_matrix = pose_stamped_to_matrix(current_corrected_position)
                current_corrected_position_position = current_corrected_position_matrix[:3, 3]

                # We select the transform that is closest to our current position
                closest_transform = None
                closest_distance = np.inf
                for transform in msg.transforms:
                    position = transform.transform.translation
                    distance = np.linalg.norm(np.array([position.x, position.y, position.z]) - current_corrected_position_position)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_transform = transform
                apriltag_transform = closest_transform

            # We need to update the transformation matrix
            # This time our header should be correct since this is coming from on the drone so we can use the timestamp
            apriltag_timestamp = apriltag_transform.header.stamp.to_sec()
            self.update_transform(apriltag_timestamp, apriltag_transform)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start the VICON position set node")
    parser.add_argument("group_number", type=int, help="The group number of the drone")
    parser.add_argument("--pass_through", action="store_true", help="Whether to pass through the setpoint position without modification")
    parser.add_argument("--use_vicon", action="store_true", help="Whether to use VICON or not")
    parser.add_argument("--use_apriltag_loc", action="store_true", help="Whether to use AprilTag localization or not")
    args = parser.parse_args()

    # Only one of pass_through, use_vicon, and use_apriltag_loc can be true
    num_true = sum([args.pass_through, args.use_vicon, args.use_apriltag_loc])
    if num_true != 1:
        raise ValueError("Exactly one of pass_through, use_vicon, and use_apriltag_loc must be true")

    vicon_position_set_node = ViconPositionSetNode(
        args.group_number,      
        args.pass_through,
        args.use_vicon,
        args.use_apriltag_loc,
    )
