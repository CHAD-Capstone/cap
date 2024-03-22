#!/usr/bin/python3

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from geometry_msgs.msg import PoseArray
from std_srvs.srv import EmptyResponse, Empty
import threading
from enum import Enum
import numpy as np

class CommsMode(Enum):
    NON_OFFBOARD = 0
    IDLE = 1
    WAYPOINT_FOLLOWING = 2
    ABORT = 3


class DroneComms:
    def __init__(self, group_number: int, takeoff_altitude: float = 2.0, pose_update_rate: float = 20.0):
        node_name = f'rob498_drone_{group_number:02d}'
        rospy.init_node(node_name) 

        rospy.wait_for_service("/mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)

        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        self.local_pos_pub = rospy.Publisher("/capdrone/setpoint_position/local", PoseStamped, queue_size=10)

        # State Subscribers
        self.current_state = State()
        state_sub = rospy.Subscriber("/mavros/state", State, callback = self.state_cb)

        self.last_position_timestamp = None
        self.current_position = None  # (x, y, z) numpy array
        self.current_orientation = None  # (x, y, z, w) quaternion numpy array
        self.last_velocity_timestamp = None
        self.current_linear_velocity = None  # (x, y, z) numpy array
        self.current_angular_velocity = None  # (x, y, z) numpy array
        pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, callback = self.pose_cb)
        velocity_sub = rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, callback = self.velocity_cb) 

        # Service topics
        self.srv_ping = rospy.Service(node_name + '/comm/ping', Empty, self.callback_ping)
        self.srv_launch = rospy.Service(node_name + '/comm/launch', Empty, self.callback_launch)
        self.srv_test = rospy.Service(node_name + '/comm/test', Empty, self.callback_test)
        self.srv_land = rospy.Service(node_name + '/comm/land', Empty, self.callback_land)
        self.srv_abort = rospy.Service(node_name + '/comm/abort', Empty, self.callback_abort)

        # Object state
        self.takeoff_altitude = takeoff_altitude
        self.pose_update_rate = pose_update_rate

        self.home_position = (0, 0, takeoff_altitude)
        self.current_requested_position = None
        self.should_run_pose_thread = True
        self.offboard_thread = None

        self.flight_test = 3

        # Comms Modes
        self.comms_mode: CommsMode = CommsMode.NON_OFFBOARD

        # Waypoint following:
        self.waypoints_sub = rospy.Subscriber('/rob498_drone_06/comm/waypoints', PoseArray, self.waypoints_cb)
        self.waypoint_follow_loop_thread = None
        self.current_waypoints = None
        self.current_waypoint_index = -1
        self.waypoint_tol = 0.2

        # Wait for startup
        rate = rospy.Rate(20.0)
        while(not rospy.is_shutdown() and not self.current_state.connected):
            rate.sleep()
        rospy.loginfo('Drone is connected')

        rospy.spin()

    ### Callbacks
    def state_cb(self, msg):
        """
        Callback for the state subscriber
        """
        self.current_state = msg

    def pose_cb(self, msg):
        """
        Callback for the pose subscriber
        """
        self.current_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.current_orientation = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        self.last_position_timestamp = rospy.Time.now()

    def velocity_cb(self, msg):
        """
        Callback for the velocity subscriber
        """
        self.current_linear_velocity = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.current_angular_velocity = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])
        self.last_velocity_timestamp = rospy.Time.now()

    def waypoints_cb(self, msg):
        """
        Callback for the waypoints subscriber
        """
        if self.current_waypoints is None:
            rospy.loginfo('Received new waypoints')
            rospy.loginfo(f'Waypoints: {msg}')
            self.current_waypoints = msg
        else:
            rospy.logwarn('Received new waypoints before previous waypoints were completed')

    ### Service callbacks
    def callback_ping(self, request):
        rospy.loginfo("Got ping!")
        return EmptyResponse()

    def callback_launch(self, request):
        self.handle_launch()
        return EmptyResponse()

    def callback_test(self, request):
        self.handle_test()
        return EmptyResponse()

    def callback_land(self, request):
        self.handle_land()
        return EmptyResponse()

    def callback_abort(self, request):
        self.handle_abort()
        return EmptyResponse()
        
    ### Handlers
    def handle_launch(self):
        print('Launch Requested. Your drone should take off.')
        took_off = self.arm_and_takeoff()
        if not took_off:
            rospy.logerr('Drone failed to take off')
            return
        self.comms_mode = CommsMode.IDLE

    def handle_test(self):
        print('Test Requested. Your drone should perform the required tasks. Recording starts now.')
        if self.flight_test == 3:
            # Flight tets 3 is to fly a series of waypoints
            rospy.loginfo(f"Starting flight test {self.flight_test}")
            self.start_waypoint_following()
        else:
            rospy.logerror(f"Flight test {self.flight_test} not implemented")
            raise NotImplementedError(f"Flight test {self.flight_test} not implemented")

    def handle_land(self):
        print('Land Requested. Your drone should land.')
        self.land()

    def handle_abort(self):
        print('Abort Requested. Your drone should land immediately due to safety considerations')
        self.abort()

    ### Commands
    def arm_and_takeoff(self):
        """
        Sends an arm service call to the drone and then takes off to the target altitude
        """
        self.move_to(*self.home_position)  # Start publishing early so we don't drop out of offboard mode
        rospy.sleep(1)
        offboard = self.start_offboard_mode()
        if not offboard:
            rospy.logerr('Drone failed to enter offboard mode')
            self.cancel_move()
            return False
        rospy.loginfo("Set position")
        rospy.sleep(2)  # Sleep to allow for some position messages to be sent
        armed = self.arm()
        if not armed:
            rospy.logerr('Drone failed to arm')
            self.cancel_move()
            return False
        return True

    def go_home(self):
        """
        Moves the drone to the home position
        """
        if not self.is_in_mode("OFFBOARD"):
            rospy.logerr("CANNOT GO HOME! NOT IN OFFBOARD MODE!")
            return False
        if self.comms_mode != CommsMode.IDLE:
            rospy.logerr(f'Cannot go home while not in IDLE mode. Current mode: {self.comms_mode}')
            return False
        self.move_to(*self.home_position)

    def abort(self):
        """
        Aborts the current operation and lands the drone
        """
        self.comms_mode = CommsMode.ABORT
        self.disarm()
        self.stop_offboard_mode()

    def start_offboard_mode(self):
        """

        """
        mode_set = self.set_mode("OFFBOARD")
        # mode_set = self.set_mode("POSITION")  # Testing to make sure arm is subsequently rejected
        if not mode_set:
            rospy.logerr("Failed to start offboard control")
            return False
        self.should_run_pose_thread = True
        self.offboard_thread = threading.Thread(target=self.publish_position_loop)
        self.offboard_thread.start()
        return True

    def start_waypoint_following(self):
        """
        Starts a new thread that sets the drone to follow a series of waypoints
        """
        if self.waypoint_follow_loop_thread is not None:
            rospy.logerr('Waypoint following thread already running')
            return False

        if self.comms_mode != CommsMode.IDLE:
            rospy.logerr(f'Cannot start waypoint following while not in IDLE mode. Current mode: {self.comms_mode}')
            return False

        def is_at_position(x, y, z, tol=0.1):
            current_pose = self.current_position
            if current_pose is None:
                return False
            print("Checking if at position")
            print(f"\tCurrent x: {current_pose[0]}, Desired x: {x}, Diff: {current_pose[0] - x}")
            print(f"\tCurrent y: {current_pose[1]}, Desired y: {y}, Diff: {current_pose[1] - y}")
            print(f"\tCurrent z: {current_pose[2]}, Desired z: {z}, Diff: {current_pose[2] - z}")
            return (abs(current_pose[0] - x) < tol and
                    abs(current_pose[1] - y) < tol and
                    abs(current_pose[2] - z) < tol)
        
        def waypoint_follow_loop():
            self.comms_mode = CommsMode.WAYPOINT_FOLLOWING
            rate = rospy.Rate(self.pose_update_rate)
            self.current_waypoint_index = 0
            while self.comms_mode == CommsMode.WAYPOINT_FOLLOWING and not rospy.is_shutdown() and self.is_in_mode("OFFBOARD"):
                if self.current_waypoints is not None:
                    if self.current_waypoint_index < len(self.current_waypoints.poses):
                        rospy.loginfo(f"Moving to waypoint {self.current_waypoint_index} - {self.current_waypoints.poses[self.current_waypoint_index]}")
                        waypoint = self.current_waypoints.poses[self.current_waypoint_index]
                        self.move_to(waypoint.position.x, waypoint.position.y, waypoint.position.z)
                    else:
                        rospy.loginfo('Waypoints complete')
                        if self.comms_mode == CommsMode.WAYPOINT_FOLLOWING:
                            # Exit the waypoint mode if we are still in it
                            self.comms_mode = CommsMode.IDLE
                            self.move_to(*self.home_position)
                        break

                    if is_at_position(waypoint.position.x, waypoint.position.y, waypoint.position.z, tol=self.waypoint_tol):
                        # Sleep for a bit to allow the drone to reach the waypoint
                        rospy.loginfo(f"Reached waypoint {self.current_waypoint_index}")
                        rospy.sleep(2)
                        self.current_waypoint_index += 1
                rate.sleep()
            self.waypoint_follow_loop_thread = None
            self.current_waypoint_index = -1
            

        self.waypoint_follow_loop_thread = threading.Thread(target=waypoint_follow_loop)
        self.waypoint_follow_loop_thread.start()
        return True

    def publish_position_loop(self):
        rate = rospy.Rate(self.pose_update_rate)
        
        while self.should_run_pose_thread and not rospy.is_shutdown():
            if self.current_requested_position is not None:
                # if not self.is_in_mode("OFFBOARD"):
                    # rospy.loginfo("Publishing pose")
                pose = self.current_requested_position
                pose.header.stamp = rospy.Time.now()
                self.local_pos_pub.publish(pose)
            else:
                rospy.logdebug('No position requested')
            rate.sleep()

    def stop_offboard_mode(self):
        self.cancel_move()
        self.should_run_pose_thread = False
        if self.offboard_thread is not None:
            self.offboard_thread.join()
        self.offboard_thread = None
        self.comms_mode = CommsMode.NON_OFFBOARD

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

    def land(self):
        """
        Lands the drone
        """
        self.stop_offboard_mode()
        landing = self.set_mode("AUTO.LAND")
        if not landing:
            rospy.logerr('Drone failed to land')
            return False
        self.comms_mode = CommsMode.NON_OFFBOARD
        return True

    def is_in_mode(self, mode):
        return self.current_state.mode == mode

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


# Main communication node for ground control
def comm_node():
    # print('This is a dummy drone node to test communication with the ground control')
    # print('node_name should be rob498_drone_TeamID. Service topics should follow the name convention below')
    # print('The TAs will test these service calls prior to flight')
    # print('Your own code should be integrated into this node')

    # Your code goes below
    drone_comms = DroneComms(6, takeoff_altitude=1.0 - 0.18)

if __name__ == "__main__":
    comm_node()
