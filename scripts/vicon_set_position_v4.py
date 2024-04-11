#!/usr/bin/python3

"""
A test script for solving for the transform between the VICON frame and the drone local frame
"""

import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool
from cap.msg import TagTransformArray
from cap.srv import SetLocalizationMode, SetLocalizationModeResponse
from cap.srv import IsReady, IsReadyResponse

from threading import Thread

from cap.transformation_lib import transform_stamped_to_matrix, pose_stamped_to_matrix, matrix_to_params, matrix_to_pose_stamped, params_to_matrix

from cap.data_lib import FLIGHT_DATA_DIR

import tf2_ros as tf2

class ViconSetPositionNode:
    def __init__(self, pass_through, use_vicon, use_apriltag_loc, group_number=6):
        node_name = f"vicon_set_position_{group_number}"
        rospy.init_node(node_name)

        #### Set up initial state ####
        self.last_transform_update_ts = None
        self.transform_queue = np.zeros((100, 7))
        self.transform_queue_len = 0
        self.transform_update_period = 0.5

        self.ready = False
        self.is_ready_service = rospy.Service('/capdrone/vicon_set_position/ready', IsReady, self.is_ready_cb)
        self.set_localization_mode_service = rospy.Service('/capdrone/set_localization_mode', SetLocalizationMode, self.set_localization_mode_cb)
        # Transform localization type
        self.pass_through = pass_through  # Makes the transform identity
        self.use_vicon = use_vicon  # Uses the VICON position
        self.use_apriltag_loc = use_apriltag_loc  # Uses the apriltag localization
        # Timestamp & transform
        self.current_local_position = (None, None)  # Pose in realsense frame
        self.current_vicon_position = (None, None)  # Pose in VICON frame
        self.current_apriltag_derived_position = (None, None)  # Pose in VICON frame
        self.current_corrected_local_position = (None, None)  # Pose in VICON frame. This is the corrected position using the transform.
        if self.pass_through or self.use_apriltag_loc:
            self.T_realsense_VICON = (rospy.Time.now(), np.array([0, 0, 0, 0, 0, 0, 1]))
            self.T_VICON_realsense = (rospy.Time.now(), np.array([0, 0, 0, 0, 0, 0, 1]))
        else:
            self.T_realsense_VICON = (None, None)  # Pose of VICON origin in realsense frame
            self.T_VICON_realsense = (None, None)
        # Get the startup time so that our timestamp print-outs can be in relation to this
        self.start_time = rospy.get_time()
        # Prep for saving flight data
        self.flight_data_file = FLIGHT_DATA_DIR / f"transform_solver_test.npy"
        self.flight_data = {
            "vicon_pose": [],  # Ground truth VICON pose
            "local_pose": [],  # Incoming from the realsense
            "frame_transform": [],  # Transform from VICON to realsense
            "corrected_pose": [],  # Updated whenever we get local pose. Pose in VICON frame.
            "apriltag_derived_position": [],  # Incoming from apriltag localization node. Pose in VICON frame
            "vicon_position_setpoint": [],  # Incoming from the setpoint publisher. Pose in VICON frame
            "local_position_setpoint": []  # Computed from the VICON setpoint. Pose in realsense frame
        }

        self.vicon_tick = 0
        self.local_tick = 0
        ##############

        #### Set up local transform ####
        # Set up the frame broadcaster
        self.local_pose_br = tf2.TransformBroadcaster()
        # And set up the listener
        self.tfBuffer = tf2.Buffer()
        self.listener = tf2.TransformListener(self.tfBuffer)
        # Finally set up the local pose subscriber that gets us our position in the realsense frame
        # TODO: Think about if this should actually be odometry out
        self.local_position_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, callback=self.local_position_cb)
        self.await_local_position()
        ##############

        #### Set up VICON transform ####
        # Ground truth VICON from the real system
        self.vicon_sub = rospy.Subscriber('/vicon/ROB498_Drone/ROB498_Drone', TransformStamped, callback=self.vicon_cb)
        if self.use_vicon:
            self.await_vicon_position()
        ##############

        #### Set up VICON transform from apriltags ####
        # Estimated VICON positions from the localization node
        self.apriltag_localization_sub = rospy.Subscriber('/capdrone/apriltag_localization/pose', TagTransformArray, callback=self.apriltag_localization_cb)
        # We don't await the apriltag localization because it is assumed to not be available at startup
        ##############

        #### Set up corrected position publisher ####
        # Publisher for the position that has been transformed from the local frame to the VICON frame
        self.corrected_position_pub = rospy.Publisher('/capdrone/local_position/pose', PoseStamped, queue_size=10)
        ####

        #### Set up ros connections for position setpoints ####
        # Set up the subscriber for the position setpoint
        self.capdrone_set_position_local_sub = rospy.Subscriber('/capdrone/setpoint_position/local', PoseStamped, callback=self.set_desired_position_local_cb)
        # Set up the publisher for the corrected setpoint
        self.setpoint_publisher = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        ##############

        #### Update Transform Thread Handler ####
        self.ready = True
        self.got_update = False  # Set to true when we should compute a new transform.
        self.update_loop_thread = None
        self.should_exit_update_loop = False
        self.start_update_transform_thread()

        rospy.loginfo("Starting spinning")
        rospy.spin()
        rospy.loginfo("Shutting down node")

        self.should_exit_update_loop = True
        if self.update_loop_thread is not None:
            rospy.loginfo("Joining update loop thread")
            self.update_loop_thread.join()
        self.save_data()
        ##############

    #### Utility Functions #####
    def await_local_position(self):
        # Waits until we have a local position
        rospy.loginfo("Waiting for local position")
        while self.current_local_position[0] is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        if rospy.is_shutdown():
            rospy.loginfo("Shutting down")
        else:
            rospy.loginfo("Got local position")

    def await_vicon_position(self):
        # Waits until we have a VICON position
        rospy.loginfo("Waiting for VICON position")
        while self.current_vicon_position[0] is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        if rospy.is_shutdown():
            rospy.loginfo("Shutting down")
        else:
            rospy.loginfo("Got VICON position")

    def add_flight_data(self, key, data):
        timestamp, transform = data
        timestamp_s = timestamp.to_sec()
        self.flight_data[key].append((timestamp_s, transform))

    def save_data(self):
        rospy.loginfo(f"Saving data")
        for key, arr in self.flight_data.items():
            print(f"\t{key}: {len(arr)}")
        np.save(self.flight_data_file, self.flight_data)
    ##############

    #### Localization mode update functions ####
    def set_localization_mode(self, localization_mode):
        rospy.loginfo(f"Setting localization mode to: {localization_mode}")
        if localization_mode == "REALSENSE":
            # Then we need to set pass through to true and also update the transform to be the identity
            rospy.loginfo("Setting localization mode to REALSENSE")
            self.pass_through = True
            self.use_vicon = False
            self.use_apriltag_loc = False
            self.T_realsense_VICON = (rospy.Time.now(), np.array([0, 0, 0, 0, 0, 0, 1]))
            self.T_VICON_realsense = (rospy.Time.now(), np.array([0, 0, 0, 0, 0, 0, 1]))
            return True
        elif localization_mode == "VICON":
            # Then we need to check if we have VICON position
            if self.current_vicon_position[0] is not None:
                rospy.loginfo("Setting localization mode to VICON")
                self.pass_through = False
                self.use_vicon = True
                self.use_apriltag_loc = False
                return True
            else:
                rospy.logwarn("Cannot set localization mode to VICON without VICON position")
                return False
        elif localization_mode == "APRILTAG":
            # No checks necessary. We will update the transform when and if we see an apriltag
            rospy.loginfo("Setting localization mode to APRILTAG")
            self.pass_through = False
            self.use_vicon = False
            self.use_apriltag_loc = True
            return True
        else:
            rospy.logerr(f"Invalid localization mode: {localization_mode}")
            return False

    def set_localization_mode_cb(self, req):
        """
        Service callback for setting the localization mode
        """
        rospy.loginfo(f"Got request to change localization mode: {req}")
        new_mode = req.data.data
        rospy.loginfo(f"Changing to mode: {new_mode}")
        success = self.set_localization_mode(new_mode)
        rospy.loginfo(f"Mode change success: {success}")
        return SetLocalizationModeResponse(Bool(success))
    ##############

    #### Transform Solver Functions ####
    def get_local_pose_at_time(self, timestamp):
        """
        Get the local pose at a given timestamp using the tfBuffer

        timestamp: time object. Extracted from header.stamp of a message

        Returns the pose as [x, y, z, qx, qy, qz, qw]
        """
        try:
            trans = self.tfBuffer.lookup_transform("realsense_world", "drone", timestamp)
            translation = trans.transform.translation
            rotation = trans.transform.rotation
            return np.array([
                translation.x, translation.y, translation.z,
                rotation.x, rotation.y, rotation.z, rotation.w
            ])
        except tf2.ExtrapolationException:
            # rospy.logwarn("Doing extrapolation")
            return self.current_local_position[1]
        except tf2.LookupException as err:
            rospy.logwarn(f"Couldn't look up transform. ERR:\n{err}")
            return None
    
    def publish_local_pose_transform(self, pose_msg):
        frame_id = "realsense_world"
        child_frame_id = "drone"
        transform_stamped = TransformStamped()

        # Copy the header from the pose message and update the frames
        transform_stamped.header.stamp = pose_msg.header.stamp
        transform_stamped.header.frame_id = frame_id
        transform_stamped.child_frame_id = child_frame_id

        # Copy the pose from the pose message
        transform_stamped.transform.translation = pose_msg.pose.position
        transform_stamped.transform.rotation = pose_msg.pose.orientation

        self.local_pose_br.sendTransform(transform_stamped)

    def vicon_cb(self, msg):
        """
        Computes T_VICON_marker
        """
        # if self.vicon_tick < 5:
        #     self.vicon_tick += 1
        #     return
        # self.vicon_tick = 0
        vicon_timestamp = msg.header.stamp
        drone_time = rospy.get_time()
        if drone_time - vicon_timestamp.to_sec() < -1:
            # This means the date on the drone has dropped behind the true date
            raise ValueError("Drone time is behind VICON time")
        elif drone_time - vicon_timestamp.to_sec() > 0.5:
            # This means we are running behind and we should refrain from costly computation for a bit
            rospy.logwarn(f"RUNNING BEHIND: {drone_time - vicon_timestamp.to_sec()}s")
            return
        else:
            rospy.loginfo(f"Latency: {drone_time - vicon_timestamp.to_sec()}")
            pass
        T_matrix = transform_stamped_to_matrix(msg)
        T_params = matrix_to_params(T_matrix, type="quaternion")
        self.current_vicon_position = (vicon_timestamp, T_params)
        # print(f"VICON Latency: {drone_time - vicon_timestamp.to_sec()}. V: {vicon_timestamp.to_sec() - self.start_time}, D: {drone_time - self.start_time}")
        self.add_flight_data("vicon_pose", self.current_vicon_position)
        if self.use_vicon:
            # We only use this information to update the transform if we are using VICON
            if not self.ready:
                rospy.logwarn(f"Ignoring VICON callback due to not ready")
            else:
                self.got_update = True

    def apriltag_localization_cb(self, msg):
        print(f"Got apriltag callback\n{msg}")
        # We can get multiple transforms from the localization node and we need to select one to do the transformation
        # We use the heuristic that poses that are closer to our current estimated position are more likely to be correct
        current_corrected_pos_ts, current_corrected_position = self.current_corrected_local_position
        if current_corrected_position is None:
            # Then we just use the first one
            apriltag_transform = msg.tags[0].transform
        else:
            # For now, we only use the position part of the transform
            current_corrected_position_matrix = params_to_matrix(current_corrected_position, type='quaternion')
            current_corrected_position_position = current_corrected_position_matrix[:3, 3]

            # We select the transform that is closest to our current position
            closest_transform = None
            closest_tag_id = None
            closest_distance = np.inf
            for tag_info in msg.tags:
                tag_id = tag_info.tag_id
                transform_stamped = tag_info.transform
                position = transform_stamped.transform.translation
                distance = np.linalg.norm(np.array([position.x, position.y, position.z]) - current_corrected_position_position)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_transform = transform_stamped
                    closest_tag_id = tag_id
            apriltag_transform = closest_transform

        # We need to update the transformation matrix
        # This time our header should be correct since this is coming from on the drone so we can use the timestamp
        apriltag_ts = apriltag_transform.header.stamp
        T_matrix = transform_stamped_to_matrix(apriltag_transform)
        T_params = matrix_to_params(T_matrix, type="quaternion")
        self.current_apriltag_derived_position = (apriltag_ts, T_params)
        rospy.loginfo(f"Apriltag Derived Position: {T_params}")
        self.add_flight_data("apriltag_derived_position", self.current_apriltag_derived_position)

        if self.use_apriltag_loc:
            if not self.ready:
                rospy.logwarn(f"Ignoring apriltag localization callback due to not ready")
            else:
                self.got_update = True

    def local_position_cb(self, msg):
        """
        Computes T_realsense_marker
        """
        # Update the corrected pose
        if self.T_realsense_VICON[0] is not None and self.ready:
            # If we have a current transform, then we can correct the pose
            self.publish_corrected_pose(msg)  # Saves the pose to the flight data and publishes the corrected pose
        if not self.ready:
            rospy.logwarn(f"Ignoring local position callback due to not ready")

        # if self.local_tick < 5:
        #     self.local_tick += 1
        #     return
        # self.local_tick = 0
        pose_timestamp = msg.header.stamp
        T_matrix = pose_stamped_to_matrix(msg)
        T_params = matrix_to_params(T_matrix, type="quaternion")
        self.current_local_position = (pose_timestamp, T_params)
        self.add_flight_data("local_pose", self.current_local_position)
        self.publish_local_pose_transform(msg)  # Saves the pose to TF2 so that we can recall it later to compute the transform
        # else:
        #     self.got_update = True  # Upon reflection, I think we should only be updating when we get a VICON position

    def start_update_transform_thread(self):
        def start_update_loop():
            rate = rospy.Rate(50)
            while not rospy.is_shutdown() and not self.should_exit_update_loop:
                if self.got_update:
                    try:
                        self.update_transform()
                    except Exception as err:
                        print(f"ERROR: {err}")
                    self.got_update = False
                rate.sleep()
        
        self.update_loop_thread = Thread(target=start_update_loop, daemon=True)
        self.update_loop_thread.start()

    def update_transform(self):
        """
        Updates the T_realsense_VICON transformation using the latest VICON pose

        If use_vicon is true, then we use the ground truth VICON pose to update the transform
        If use_apriltag_loc is true, then we use the apriltag localization to update the transform
        """
        if not self.ready:
            rospy.logwarn("Ignoring update transform due to not ready")
            return False

        if self.use_vicon:
            vicon_ts, T_VICON_marker_params = self.current_vicon_position
        elif self.use_apriltag_loc:
            vicon_ts, T_VICON_marker_params = self.current_apriltag_derived_position
        elif self.pass_though:
            # Then we can just set the transform to identity and be done with it
            self.T_realsense_VICON = (rospy.Time.now(), np.array([0, 0, 0, 0, 0, 0, 1]))
            return True

        if vicon_ts is None:
            rospy.logwarn("Tried to update transform with no VICON position")
            return False

        T_local_marker_params = self.get_local_pose_at_time(vicon_ts)
        if T_local_marker_params is None:
            rospy.logwarn("Tried to update transform with no local position")
            return False

        # Compute the new transform
        T_local_marker = params_to_matrix(T_local_marker_params, type="quaternion")
        T_VICON_marker = params_to_matrix(T_VICON_marker_params, type="quaternion")
        T_local_VICON = T_local_marker @ np.linalg.inv(T_VICON_marker)

        # Santity check. The z axis of the transform should be up
        # We check the cosine similarity with the expected z axis and reject if it is off by too much
        cosine_sim_threshold = np.cos(np.deg2rad(10))
        cosine_sim = np.dot(T_local_VICON[:3, 2], np.array([0, 0, 1]))
        if cosine_sim < cosine_sim_threshold:
            rospy.logwarn(f"Rejecting transform due to cosine similarity {cosine_sim}")
            rospy.logwarn(f"VICON Pose: {T_VICON_marker_params}")
            rospy.logwarn(f"Local Pose: {T_local_marker_params}")
            return False

        # And save it
        self.T_realsense_VICON = (vicon_ts, matrix_to_params(T_local_VICON, type="quaternion"))
        T_VICON_local = np.linalg.inv(T_local_VICON)
        self.T_VICON_realsense = (vicon_ts, matrix_to_params(T_VICON_local, type="quaternion"))
        self.add_flight_data("frame_transform", self.T_realsense_VICON)
        return True

        # # Add the transform to the queue
        # if self.transform_queue_len == 100:
        #     rospy.logwarn("Transform queue is full. Dropping oldest transform")
        #     # Roll the queue
        #     # self.transform_queue[:99] = self.transform_queue[1:]
        #     self.transform_queue = np.roll(self.transform_queue, -1, axis=0)
        #     self.transform_queue_len -= 1
        # self.transform_queue[self.transform_queue_len] = matrix_to_params(T_local_VICON, type="quaternion")
        # self.transform_queue_len += 1

        # current_ts_s = rospy.get_time()
        # if self.last_transform_update_ts is None or current_ts_s - self.last_transform_update_ts > self.transform_update_period:
        #     # Then we will update the transform taking the median of the last 100 transforms
        #     rospy.loginfo("Updating transform")
        #     self.last_transform_update_ts = current_ts_s
        #     T_local_VICON = np.median(self.transform_queue[:self.transform_queue_len], axis=0)
        #     self.T_realsense_VICON = (vicon_ts, T_local_VICON)
        #     T_realsense_VICON_mat = params_to_matrix(T_local_VICON, type="quaternion")
        #     T_VICON_realsense_mat = np.linalg.inv(T_realsense_VICON_mat)
        #     self.T_VICON_realsense = (vicon_ts, matrix_to_params(T_VICON_realsense_mat, type="quaternion"))
        #     self.add_flight_data("frame_transform", self.T_realsense_VICON)
        #     self.transform_queue_len = 0
        # return True

    ##############

    #### External Interaction Handlers ####
    def is_ready_cb(self, req):
        return IsReadyResponse(self.ready, "")
        
    def publish_corrected_pose(self, msg):
        """
        Uses the most up to date transform to correct the pose and publishes it
        """
        ts = msg.header.stamp
        # Convert the msg to a matrix
        T_local_marker = pose_stamped_to_matrix(msg)
        # Get the transform
        T_VICON_local = params_to_matrix(self.T_VICON_realsense[1], type="quaternion")
        # Compute the corrected pose
        T_VICON_marker = T_VICON_local @ T_local_marker
        # Save the corrected pose
        T_VICON_marker_params = matrix_to_params(T_VICON_marker, type="quaternion")
        self.add_flight_data("corrected_pose", (ts, T_VICON_marker_params))
        self.current_corrected_local_position = (ts, T_VICON_marker_params)
        # Convert the corrected pose back into a pose object and publish it
        corrected_pose = matrix_to_pose_stamped(T_VICON_marker, header=msg.header)
        self.corrected_position_pub.publish(corrected_pose)

    def set_desired_position_local_cb(self, msg):
        """
        Uses the most up to date transform to project the desired pose from the VICON frame into the local frame
        """
        print(f"Setting desired local position {msg}")
        if not self.ready:
            rospy.logwarn("Ignoring set position due to not ready")
        # Now we project the desired position from the VICON frame into the local frame
        # Transform the desired pose in a matrix. This matrix can be though of as T_VICON_desired
        desired_pose_matrix = pose_stamped_to_matrix(msg)
        # Now we want T_realseanse_desired = T_realsense_VICON @ T_VICON_desired
        T_realsense_VICON_matrix = params_to_matrix(self.T_realsense_VICON[1], type="quaternion")
        T_realsense_desired = T_realsense_VICON_matrix @ desired_pose_matrix
        corrected_pose_msg = matrix_to_pose_stamped(T_realsense_desired, header=msg.header)
        # And then we publish this to the mavros setpoint position to get the drone to move there
        self.setpoint_publisher.publish(corrected_pose_msg)

        desired_pose_local_params = matrix_to_params(T_realsense_desired, type="quaternion")
        desired_pose_vicon_params = matrix_to_params(desired_pose_matrix, type="quaternion")
        self.add_flight_data("local_position_setpoint", (msg.header.stamp, desired_pose_local_params))
        self.add_flight_data("vicon_position_setpoint", (msg.header.stamp, desired_pose_vicon_params))
    ##############




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

    vicon_position_set_node = ViconSetPositionNode(      
        args.pass_through,
        args.use_vicon,
        args.use_apriltag_loc,
        args.group_number
    )