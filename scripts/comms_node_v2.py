#!/usr/bin/python3

"""
Main node for the capdrone. This node keeps a state machine that controls the drone's behavior.
It also coordiantes the other nodes in the system.
MUST BE STARTED BEFORE ANY OTHER NODES.
"""

import rospy
import threading
from enum import Enum
import numpy as np
from typing import Tuple

# Cap libraries
from cap.data_lib import archive_existing_current_flight_data, load_current_flight_tag_map
from cap.planning_lib import plan_path, get_closest_tag
from cap.apriltag_pose_estimation_lib import AprilTagMap
from cap.transformation_lib import pose_stamped_to_matrix, transform_stamped_to_matrix

# Service messages
from cap.srv import IsReady, IsReadyResponse
from cap.srv import SetLocalizationMode, SetLocalizationModeResponse
from cap.srv import NewTag, NewTagResponse
from cap.srv import TagPoses, TagPosesResponse
from cap.srv import FindTag, FindTagResponse
from cap.srv import SetPosition, SetPositionResponse
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Bool, String
from mavros_msgs.srv import CommandBool, SetMode, SetModeRequest, CommandBoolRequest

# Message types
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped

class CommsMode(Enum):
    NON_OFFBOARD = 0  # Not under the control of the offboard controller
    IDLE = 1  # Ready for a new command. Will respond to requests to switch comms mode
    MAPPING = 2  # Flies to rough estimates of tag positions and images them. Drops into IDLE mode when done
    INSPECTING = 3  # Flies to the mapped tag positions and images them. Continues until switched to IDLE mode
    ABORT = 4  # Aborts the current operation and returns to IDLE mode
    BUSY = 5  # Currently executing a command. Will not respond to requests to switch comms mode.


class CommsNode:
    def __init__(
        self,
        group_id: int = 6,
        takeoff_altitude_m: float = 1.5,
        archive_previous_flight_data: bool = False,
    ):
        node_name = f'rob498_drone_{group_id:02d}'
        rospy.init_node(node_name)

        self.home_position = (0, 0, takeoff_altitude_m)
        self.pose_update_rate = 20  # Hz

        if archive_previous_flight_data:
            archive_existing_current_flight_data()

        # Set up arming services
        rospy.wait_for_service("/mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)

        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        self.local_pos_pub = rospy.Publisher("/capdrone/setpoint_position/local", PoseStamped, queue_size=10)

        # Wait for other nodes to be ready
        self.ready_services = [
            "/capdrone/vicon_set_position/ready",  # The vicon_set_position node
            # "/capdrone/apriltag_mapping/ready",  # The apriltag_mapping node
            # "/capdrone/apriltag_localization/ready",  # The apriltag_localization node
        ]
        for service_name in self.ready_services:
            self.wait_for_service_ready(service_name)

        self.comms_mode = CommsMode.NON_OFFBOARD
        self.offboard_thread = None

        ##### Service Servers #####
        print("Setting up service servers")
        self.srv_ping = rospy.Service(node_name + '/comm/ping', Empty, self.callback_ping)
        self.srv_launch = rospy.Service(node_name + '/comm/launch', Empty, self.callback_launch)
        self.srv_land = rospy.Service(node_name + '/comm/land', Empty, self.callback_land)
        self.srv_abort = rospy.Service(node_name + '/comm/abort', Empty, self.callback_abort)
        self.srv_set_position = rospy.Service(node_name + '/comm/set_position', SetPosition, self.callback_set_position)

        # Service to begin mapping
        self.src_begin_mapping = rospy.Service(node_name + '/comm/begin_mapping', TagPoses, self.callback_begin_mapping)
        self.mapping_height = 2.0  # Height above the tag to fly to for mapping
        self.mapping_imaging_offsets = np.array([
            [0, 0, 0],
            [0, 1, 0],
            # [1, 1, 0],
            [1, 0, 0],
            # [1, -1, 0],
            [0, -1, 0],
            # [-1, -1, 0],
            [-1, 0, 0],
            # [-1, 1, 0],
        ]) * 0.3
        # Service to begin inspecting
        self.src_begin_inspecting = rospy.Service(node_name + '/comm/begin_inspecting', Empty, self.callback_begin_inspecting)
        self.src_stop_inspecting = rospy.Service(node_name + '/comm/stop_inspecting', Empty, self.callback_stop_inspecting)

        #### Service Clients ####
        print("Setting up service clients")
        ## VICON Set Position Services
        # Sets the drone to use a different estimate for its global position
        self.srv_set_localization_mode = rospy.ServiceProxy("/capdrone/set_localization_mode", SetLocalizationMode)

        ## AprilTag Mapping Services
        # Instruct the mapping node to find the location of a tag in the world based on a single image
        self.srv_apriltag_mapping_find_tag = rospy.ServiceProxy("/capdrone/apriltag_mapping/find_tag", TagPoses)
        # Instruct the mapping node to expect a new tag id
        self.srv_apriltag_mapping_new_tag = rospy.ServiceProxy("/capdrone/apriltag_mapping/new_tag", NewTag)
        # Instruct the mapping node to capture an image
        self.srv_apriltag_mapping_capture_img = rospy.ServiceProxy("/capdrone/apriltag_mapping/capture_img", Empty)
        # Instruct the mapping node to process the tag
        # Once this returns, we will have a new tag map in the flight data directory
        self.src_apriltag_mapping_process_tag = rospy.ServiceProxy("/capdrone/apriltag_mapping/process_tag", Empty)

        ## AprilTag Localization Services
        # Instruct the localization node to refresh its tag map
        self.srv_apriltag_localization_refresh_tag_map = rospy.ServiceProxy("/capdrone/apriltag_localization/refresh_tag_locations", Empty)

        ##### State Management #####
        print("Setting up state subscribers")
        # Keep track of the current state so we can tell what mode we are in
        self.current_state: State = None
        state_sub = rospy.Subscriber("/mavros/state", State, callback = self.state_cb)

        # Keep track of the corrected pose that is coming out of the VICON set position node
        self.current_pose_VICON: PoseStamped = None
        pose_sub = rospy.Subscriber("/capdrone/local_position/pose", PoseStamped, callback = self.pose_cb)
        self.current_velocity_VICON: TwistStamped = None
        velocity_sub = rospy.Subscriber("/capdrone/local_position/local_velocity", TwistStamped, callback = self.velocity_cb)

    ##### Setup Functions #####
    def wait_for_service_ready(self, service_name: str):
        """
        There are multiple nodes that will respond to a service request checking if they are ready.
        In each case, we send a IsReady request and expect a IsReadyResponse ({ "ready": bool, "message": str }) response.
        This function waits until we receive a response with ready = True.
        """
        print(f"Waiting for service {service_name} to return ready")
        rospy.wait_for_service(service_name)
        service = rospy.ServiceProxy(service_name, IsReady)
        msg = IsReady()
        response = service()
        while not response.ready:
            print(f"Service {service_name} not ready: {response.message}")
            rospy.sleep(1)
            response = service()
        print(f"{service_name} returned ready")
        return response

    ##### Mode Transition Functions #####
    def set_comms_mode(self, mode: CommsMode):
        if mode == CommsMode.NON_OFFBOARD:
            # Then we should stop the pose thread
            self._should_run_pose_thread = False
        self.comms_mode = mode

    ##### Mode Exit Managers #####
    # These functions are called by loops that are managing the state machine to check whether we should exit the loop
    # If the mode no longer cooresponds to the loop, the loop should exit
    # If ros is shutting down, the loop should exit

    def in_offboard(self):
        # Check if we are in offboard mode
        return self.current_state.mode == "OFFBOARD"

    def should_exit_mapping(self):
        in_incorrect_mode = self.comms_mode != CommsMode.MAPPING
        has_completed_mapping = False  # TODO: Get the logic for whether we have completed mapping
        is_shutdown = rospy.is_shutdown()
        has_left_offboard = not self.in_offboard()
        return in_incorrect_mode or has_completed_mapping or is_shutdown or has_left_offboard

    def should_exit_inspecting(self):
        in_incorrect_mode = self.comms_mode != CommsMode.INSPECTING
        is_shutdown = rospy.is_shutdown()
        has_left_offboard = not self.in_offboard()
        return in_incorrect_mode or is_shutdown or has_left_offboard

    ##### Service Callbacks #####
    def callback_ping(self, request):
        print("Got ping")
        return EmptyResponse()

    def callback_launch(self, request):
        print("Got Launch")
        if self.comms_mode == CommsMode.NON_OFFBOARD:
            self.launch()
        else:
            # We are already in a mode where we can't launch
            rospy.logwarn(f"Launch requested while already flying. Current mode: {self.comms_mode}")
        return EmptyResponse()

    def callback_land(self, request):
        print("Got Land")
        # We are allowed to land in any mode
        self.land()
        return EmptyResponse()

    def callback_abort(self, request):
        print("Got Abort")
        # We are allowed to abort in any mode
        self.abort()
        return EmptyResponse()

    def callback_set_position(self, request):
        position = request.position
        print(f"Got set position {(position.x, position.y, position.z)}")
        # We can only set the position in IDLE mode
        if self.comms_mode == CommsMode.IDLE:
            position = (position.x, position.y, position.z)
            success = self.move_and_wait(lambda : False, position, velocity_threshold=0.1)
            if not success:
                rospy.logerr("Failed to move to requested position")
                return Bool(False), String("Failed to move to requested position")
            else:
                return Bool(True), String("Moved to requested position")
        else:
            rospy.logwarn(f"Set position requested while not in IDLE mode. Current mode: {self.comms_mode}")
            return Bool(False), String("Not in IDLE mode")
        return res

    def callback_begin_mapping(self, request):
        # We can only begin mapping in IDLE mode
        if self.comms_mode == CommsMode.IDLE:
            self.begin_mapping(request)
        else:
            rospy.logwarn(f"Begin mapping requested while not in IDLE mode. Current mode: {self.comms_mode}")
        return TagPosesResponse()

    def callback_begin_inspecting(self, request):
        # We can only begin inspecting in IDLE mode
        if self.comms_mode == CommsMode.IDLE:
            self.begin_inspecting()
        else:
            rospy.logwarn(f"Begin inspecting requested while not in IDLE mode. Current mode: {self.comms_mode}")
        return EmptyResponse()

    def callback_stop_inspecting(self, request):
        # We can only stop inspecting in INSPECTING mode
        if self.comms_mode == CommsMode.INSPECTING:
            self.set_comms_mode(CommsMode.IDLE)  # This should cause us to drop out of the inspecting loop
            self.move_and_wait(lambda : False, self.home_position, velocity_threshold=0.1)
        else:
            rospy.logwarn(f"Stop inspecting requested while not in INSPECTING mode. Current mode: {self.comms_mode}")
        return EmptyResponse()
    
    ##### State Callbacks #####
    def state_cb(self, data):
        self.current_state = data
        if self.current_state.mode != "OFFBOARD":
            # Immediately drop out of offboard mode
            self.set_comms_mode(CommsMode.NON_OFFBOARD)

    def pose_cb(self, data):
        self.current_pose_VICON = data

    def velocity_cb(self, data):
        self.current_velocity_VICON = data

    ##### Mode Loops #####
    def begin_mapping(self, request: TagPoses):
        """
        The steps for mapping are as follows:
        0. Instruct the vicon_set_position node to start using VICON for localization
        1. Extract the approximate tag poses from the request and put them into a tag map
        2. Plan a path of close to minimum distance to visit all the tags
        3. For each tag:
            a. Fly to the approximate position high above the tag
            b. Use self.srv_apriltag_mapping_find_tag to refine the tag position
            c. Fly lower and on top of the tag
            d. Use self.srv_apriltag_mapping_new_tag to start a new capture session
            e. For each offset in [(0, 0), (0, o), (o, o), (o, 0), (0, -o),...]
                i. Fly to the offset position and wait until our velocity is low
                ii. Use self.srv_apriltag_mapping_capture_img to capture an image
            f. Use self.srv_apriltag_mapping_process_tag to process the tag
        4. Fly home and drop into IDLE mode
        """
        # Instruct the vicon_set_position node to start using VICON for localization
        localization_mode_request = SetLocalizationModeRequest()
        localization_mode_request.mode = "VICON"
        res = self.srv_set_localization_mode(localization_mode_request)
        if not res.success:
            rospy.logerr("Failed to set localization mode")
            return

        # Extract the approximate tag poses from the request and put them into a tag map
        tag_map = AprilTagMap()
        for tag_transform in request.tags:
            tag_id = tag_transform.tag_id
            tag_transform_stamped = tag_transform.transform
            t = tag_transform_stamped.transform.translation
            q = tag_transform_stamped.transform.rotation
            tag_pose_params = np.array([
                t.x, t.y, t.z,
                q.x, q.y, q.z, q.w
            ])
            tag_map.add_tag_pose(tag_id, tag_pose_params)
        
        # Convert our current location to a matrix
        current_pose = pose_stamped_to_matrix(self.current_pose_VICON)
        closest_tag_id = get_closest_tag(current_pose, tag_map, height_m=self.mapping_height)

        # Solve the TSP to get the order of the tags
        tag_ids, positions, total_distance = plan_path(
            tag_map,
            height_m=self.mapping_height,
            start_tag_id=closest_tag_id
        )
        rospy.loginfo(f"Planned path: {tag_ids}. Total distance: {total_distance}")

        tag_idx = 0
        while tag_idx < len(tag_ids) and not self.should_exit_mapping():
            tag_id = tag_ids[tag_idx]
            flight_position = positions[tag_idx]
            success = self.move_and_wait(self.should_exit_mapping, flight_position, velocity_threshold=0.1)
            if not success:
                rospy.logerr("Failed to move to tag position")
                break

            # Use the apriltag_mapping node to find the tag
            find_tag_request = FindTagRequest()
            find_tag_request.tag_id = tag_id
            find_tag_response = self.srv_apriltag_mapping_find_tag(find_tag_request)
            if not find_tag_response.success:
                rospy.logerr(f"Failed to find tag {tag_id}")
                continue
            tag_pose_VICON = transform_stamped_to_matrix(find_tag_response.transform)

            # Move to the tag position (+ mapping height)
            tag_position = tag_pose_VICON[:3, 3].copy()
            tag_position[2] += self.mapping_height

            success = self.move_and_wait(self.should_exit_mapping, tag_position, velocity_threshold=0.1)
            if not success:
                rospy.logerr("Failed to move to corrected tag position")
                break

            # Start a new tag capture session
            new_tag_request = NewTagRequest()
            new_tag_request.tag_id = tag_id
            self.srv_apriltag_mapping_new_tag(new_tag_request)

            # Capture images
            capture_failed = False
            for offset in self.mapping_imaging_offsets:
                offset_position = tag_position + offset
                success = self.move_and_wait(self.should_exit_mapping, offset_position, velocity_threshold=0.1)
                if not success:
                    rospy.logerr("Failed to move to image offset position")
                    capture_failed = True
                    break
                self.srv_apriltag_mapping_capture_img()
                
            if capture_failed:
                break

            # Process the tag
            self.src_apriltag_mapping_process_tag()

            tag_idx += 1

        # Fly home
        success = self.move_and_wait(self.should_exit_mapping, self.home_position, velocity_threshold=0.1)
        if not success:
            rospy.logerr("Failed to move home")
        self.set_comms_mode(CommsMode.IDLE)

    def begin_inspecting(self):
        """
        The steps for inspecting are as follows:
        0. Instruct the vicon_set_position node to start using AprilTags for localization
        1. Load the newest tag map
        2. Plan a path of close to minimum distance to visit all the tags
        3. For each tag:
            a. Fly to the tag position
            b. Hover for a few seconds
            c. TODO: Capture an image (We don't have a node for this yet)
        4. Repeat until
        """
        # Instruct the vicon_set_position node to start using AprilTags for localization
        localization_mode_request = SetLocalizationModeRequest()
        localization_mode_request.mode = "APRILTAG"
        res = self.srv_set_localization_mode(localization_mode_request)
        if not res.success:
            rospy.logerr("Failed to set localization mode")
            return

        # Load the newest tag map
        tag_map = load_current_flight_tag_map()

        # Find the closest tag
        current_pose = pose_stamped_to_matrix(self.current_pose_VICON)
        closest_tag_id = get_closest_tag(current_pose, tag_map, height_m=1.0)

        tag_ids, positions, total_distance = plan_path(
            tag_map,
            height_m=1.0,
            start_tag_id=closest_tag_id
        )
        rospy.loginfo(f"Planned path: {tag_ids}. Total distance: {total_distance}")

        tag_idx = 0
        while tag_idx < len(tag_ids) and not self.should_exit_inspecting():
            tag_id = tag_ids[tag_idx]
            flight_position = positions[tag_idx]
            success = self.move_and_wait(self.should_exit_inspecting, flight_position, velocity_threshold=0.1)
            if not success:
                rospy.logerr("Failed to move to tag position")
                break
            rospy.sleep(3)

    ##### Utility Functions #####
    def is_at_position(self, x: float, y: float, z: float, tolerance: float = 0.1):
        """
        Returns True if the drone is within the given tolerance of the given position
        """
        if self.current_pose_VICON is None:
            return False
        pose = self.current_pose_VICON.pose.position
        x_dist = abs(pose.x - x)
        y_dist = abs(pose.y - y)
        z_dist = abs(pose.z - z)
        print(f"Distances: {(x_dist, y_dist, z_dist)}. {tolerance} - {((x_dist < tolerance), (y_dist < tolerance), (z_dist < tolerance))}")
        return (x_dist < tolerance) and (y_dist < tolerance) and (z_dist < tolerance)

    def is_velocity_below_threshold(self, threshold: float = 0.1):
        """
        Returns True if the drone's velocity is below the given threshold
        """
        if self.current_velocity_VICON is None:
            return False
        velocity = self.current_velocity_VICON.twist.linear
        print(f"Velocities: {(abs(velocity.x), abs(velocity.y), abs(velocity.z))}. {threshold} - {((abs(velocity.x) < threshold), (abs(velocity.y) < threshold), (abs(velocity.z) < threshold))}")
        return (abs(velocity.x) < threshold) and (abs(velocity.y) < threshold) and (abs(velocity.z) < threshold)

    def move_and_wait(
        self,
        should_exit_function,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float] = None,
        position_tolerance: float = 0.1,  # Must be closer than this to the target position
        velocity_threshold: float = None,  # Must be moving slower than this to be considered at the target position
        settle_time: float = None  # Wait time after reaching the target position and velocity is below threshold
    ):
        """
        Begins moving to the given position and waits until the drone is at the position or should exit

        Returns True if the drone reached the position, False if the function should exit
        """
        print(f"Moving to {position}")
        x, y, z = position
        if orientation is None:
            orientation = (0, 0, 0, 1)
        qx, qy, qz, qw = orientation
        self.move_to(x, y, z, qx=qx, qy=qy, qz=qz, qw=qw)
        is_at_postion = self.wait_until_at_position(x, y, z, position_tolerance, should_exit_function, settle_time)
        is_below_velocity = velocity_threshold is None or self.is_velocity_below_threshold(velocity_threshold)
        print("Waiting to get to position")
        while is_at_postion and is_below_velocity and not should_exit_function() and not rospy.is_shutdown() and self.in_offboard():
            rospy.sleep(0.1)
            is_at_postion = self.is_at_position(x, y, z, position_tolerance)
            is_below_velocity = velocity_threshold is None or self.is_velocity_below_threshold(velocity_threshold)
        if not is_at_postion:
            rospy.logerr("Failed to reach target position")
            return False
        if not is_below_velocity:
            rospy.logerr("Failed to reach target velocity")
            return False
        if settle_time is not None:
            rospy.sleep(settle_time)
        print("Got to position")
        return True

    def is_in_mode(self, mode):
        return self.current_state.mode == mode

    ##### High Level Control Functions #####
    def launch(self):
        self.set_comms_mode(CommsMode.BUSY)

        print("Launching")
        self.move_to(*self.home_position)  # Start publishing early so we don't drop out of offboard mode
        rospy.sleep(1)
        offboard = self.start_offboard_mode()
        if not offboard:
            rospy.logerr('Drone failed to enter offboard mode')
            self.cancel_move()
            self.set_comms_mode(CommsMode.IDLE)
            return False
        rospy.loginfo("Set position")
        rospy.sleep(2)  # Sleep to allow for some position messages to be sent
        armed = self.arm()
        if not armed:
            rospy.logerr('Drone failed to arm')
            self.cancel_move()
            self.set_comms_mode(CommsMode.IDLE)
            return False

        # Wait until we are at the home position and not moving
        rospy.loginfo("Waiting for drone to reach home position")
        is_at_position = self.is_at_position(*self.home_position, tolerance=0.2)
        is_at_rest = self.is_velocity_below_threshold(threshold=10)
        while (not is_at_position or not is_at_rest) and not rospy.is_shutdown():
            rospy.sleep(1)
            is_at_position = self.is_at_position(*self.home_position, tolerance=0.2)
            is_at_rest = self.is_velocity_below_threshold(threshold=10)
            print(f"As at position {is_at_position}. Is at rest {is_at_position}")
        print("Home!")

        rospy.loginfo("Drone is ready to take commands")
        self.set_comms_mode(CommsMode.IDLE)
        return True

    def land(self):
        """
        Lands the drone
        """
        if self.offboard_thread is None:
            rospy.logwarn("Tried to land when not in control")
            return False
        self.set_comms_mode(CommsMode.BUSY)

        self.stop_offboard_mode()
        landing = self.set_mode("AUTO.LAND")
        if not landing:
            rospy.logerr('Drone failed to land')
            return False

        # No need to set comms mode because stop_offboard_mode will do that
        return True

    def abort(self):
        """
        Aborts the current operation and lands the drone
        """
        self.comms_mode = CommsMode.ABORT
        self.disarm()
        self.stop_offboard_mode()

    ##### Low Level Control Functions #####
    def set_mode(self, custom_mode):
        """
        Changes the mode of the drone
        """
        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = custom_mode
        mode_set = self.set_mode_client.call(offb_set_mode)
        if not mode_set:
            rospy.logerr(f'Failed to enter {custom_mode} mode')
            return False
        return True

    def publish_position_loop(self):
        rate = rospy.Rate(self.pose_update_rate)
        
        while self._should_run_pose_thread and not rospy.is_shutdown():
            if self.current_requested_position is not None:
                # if not self.is_in_mode("OFFBOARD"):
                    # rospy.loginfo("Publishing pose")
                pose = self.current_requested_position
                pose.header.stamp = rospy.Time.now()
                self.local_pos_pub.publish(pose)
            else:
                rospy.logdebug('No position requested')
            rate.sleep()

    def start_offboard_mode(self):
        """

        """
        mode_set = self.set_mode("OFFBOARD")
        # mode_set = self.set_mode("POSITION")  # Testing to make sure arm is subsequently rejected
        if not mode_set:
            rospy.logerr("Failed to start offboard control")
            return False
        self._should_run_pose_thread = True
        self.offboard_thread = threading.Thread(target=self.publish_position_loop)
        self.offboard_thread.start()
        return True

    def stop_offboard_mode(self):
        self.cancel_move()
        self._should_run_pose_thread = False
        if self.offboard_thread is not None:
            self.offboard_thread.join()
        self.offboard_thread = None
        self.set_comms_mode(CommsMode.NON_OFFBOARD)

    def arm(self):
        """
        Arms the drone
        """
        # Check if the drone is in offboard mode
        rospy.loginfo(f"Current Mode: {self.current_state.mode}")
        if not self.is_in_mode("OFFBOARD"):
            rospy.logerr("CANNOT ARM! NOT IN OFFBOARD MODE!")
            return False
        arm_request = CommandBoolRequest()
        arm_request.value = True
        return self.arming_client.call(arm_request).success

    def disarm(self):
        """
        Disarms the drone
        """
        arm_request = CommandBoolRequest()
        arm_request.value = False
        return self.arming_client.call(arm_request).success

    def cancel_move(self):
        """
        Cancels the current move request
        """
        self.current_requested_position = None

    def move_to(self, x, y, z, qx = 0, qy = 0, qz = 0, qw = 1):
        """
        Moves the drone to the specified position
        """
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        self.current_requested_position = pose


if __name__ == "__main__":
    node = CommsNode(
        group_id=6,
        takeoff_altitude_m=1,
        archive_previous_flight_data=False
    )
    print("Spinning")
    rospy.spin()