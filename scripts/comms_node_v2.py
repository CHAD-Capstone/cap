#!/usr/bin/python3

"""
Main node for the capdrone. This node keeps a state machine that controls the drone's behavior.
It also coordiantes the other nodes in the system.
MUST BE STARTED BEFORE ANY OTHER NODES.
"""

from cap.data_lib import archive_existing_current_flight_data, load_current_flight_tag_map

# Service messages
from cap_srvs.srv import IsReady, IsReadyResponse
from cap_srvs.srv import SetLocalizationMode, SetLocalizationModeResponse
from cap_srvs.srv import NewTag, NewTagResponse
from cap_srvs.srv import TagPoses, TagPoses
from std_srvs.srv import Empty, EmptyResponse
from mavros_msgs.srv import CommandBool, SetMode

# Message types
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped, TwistStamped

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
        node_name = f'rob498_drone_{group_number:02d}'
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

        # Wait for other nodes to be ready
        self.ready_services = [
            "/capdrone/vicon_set_position/ready",  # The vicon_set_position node
            "/capdrone/apriltag_mapping/ready",  # The apriltag_mapping node
            "/capdrone/apriltag_localization/ready",  # The apriltag_localization node
        ]
        for service_name in self.ready_services:
            self.wait_for_service_ready(service_name)

        self.comms_mode = CommsMode.NON_OFFBOARD

        ##### Service Servers #####
        self.srv_ping = rospy.Service(node_name + '/comm/ping', Empty, self.callback_ping)
        self.srv_launch = rospy.Service(node_name + '/comm/launch', Empty, self.callback_launch)
        self.srv_land = rospy.Service(node_name + '/comm/land', Empty, self.callback_land)
        self.srv_abort = rospy.Service(node_name + '/comm/abort', Empty, self.callback_abort)

        # Service to begin mapping
        self.src_begin_mapping = rospy.Service(node_name + '/comm/begin_mapping', TagPoses, self.callback_begin_mapping)
        # Service to begin inspecting
        self.src_begin_inspecting = rospy.Service(node_name + '/comm/begin_inspecting', Empty, self.callback_begin_inspecting)

        #### Service Clients ####
        ## VICON Set Position Services
        # Sets the drone to use a different estimate for its global position
        self.srv_set_localization_mode = rospy.ServiceProxy("/capdrone/set_localization_mode", SetLocalizationMode)

        ## AprilTag Mapping Services
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
        response = service(IsReadyRequest())
        while not response.ready:
            print(f"Service {service_name} not ready: {response.message}")
            rospy.sleep(1)
            response = service(IsReadyRequest())
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
        return EmptyResponse()

    def callback_launch(self, request):
        if self.comms_mode == CommsMode.NON_OFFBOARD:
            self.launch()
        else:
            # We are already in a mode where we can't launch
            rospy.logwarn(f"Launch requested while already flying. Current mode: {self.comms_mode}")
            return EmptyResponse()

    def callback_land(self, request):
        # We are allowed to land in any mode
        self.land()

    def callback_abort(self, request):
        # We are allowed to abort in any mode
        self.abort()

    def callback_begin_mapping(self, request):
        # We can only begin mapping in IDLE mode
        if self.comms_mode == CommsMode.IDLE:
            self.begin_mapping(request)
        else:
            rospy.logwarn(f"Begin mapping requested while not in IDLE mode. Current mode: {self.comms_mode}")
            return ApproximateTagPosesResponse()

    def callback_begin_inspecting(self, request):
        # We can only begin inspecting in IDLE mode
        if self.comms_mode == CommsMode.IDLE:
            self.begin_inspecting()
        else:
            rospy.logwarn(f"Begin inspecting requested while not in IDLE mode. Current mode: {self.comms_mode}")
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

    ##### Utility Functions #####
    def is_at_position(self, x: float, y: float, z: float, tolerance: float = 0.1):
        """
        Returns True if the drone is within the given tolerance of the given position
        """
        if self.current_pose_VICON is None:
            return False
        pose = self.current_pose_VICON.pose.position
        return (abs(pose.x - x) < tolerance) and (abs(pose.y - y) < tolerance) and (abs(pose.z - z) < tolerance)

    def is_velocity_below_threshold(self, threshold: float = 0.1):
        """
        Returns True if the drone's velocity is below the given threshold
        """
        if self.current_velocity_VICON is None:
            return False
        velocity = self.current_velocity_VICON.twist.linear
        return (abs(velocity.x) < threshold) and (abs(velocity.y) < threshold) and (abs(velocity.z) < threshold)

    ##### High Level Control Functions #####
    def launch(self):
        self.set_comms_mode(CommsMode.BUSY)

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
        while (not self.is_at_position(*self.home_position) or not self.is_velocity_below_threshold()) and not rospy.is_shutdown():
            rospy.sleep(1)

        rospy.loginfo("Drone is ready to take commands")
        self.set_comms_mode(CommsMode.IDLE)
        return True

    def land(self):
        """
        Lands the drone
        """
        self.set_comms_mode(CommsMode.BUSY)

        self.stop_offboard_mode()
        landing = self.set_mode("AUTO.LAND")
        if not landing:
            rospy.logerr('Drone failed to land')
            return False

        # No need to set comms mode because stop_offboard_mode will do that
        return True

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