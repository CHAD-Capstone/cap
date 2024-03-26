#!/usr/bin/python3
"""
This node takes input from the user to control the drone.
It interacts with the comms node via service calls to instruct the drone to takeoff, land, etc.
"""

from threading import Thread
import json

import rospy
from std_srvs.srv import Empty
from cap.srv import TagPoses
from cap.srv import SetPosition
from cap.msg import TagTransform
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion

from cap.transformation_lib import matrix_to_params
from cap.data_lib import load_approximate_tag_poses
from cap.data_lib import FLIGHT_INPUTS_DIR

class GroundStationNode:
    def __init__(self, group_id: int = 6):
        node_name = f'ground_station_{group_id:02d}'
        rospy.init_node(node_name)

        comms_node_name = f'rob498_drone_{group_id:02d}'

        # Define the service clients
        rospy.wait_for_service(f'{comms_node_name}/comm/ping')
        self.ping_client = rospy.ServiceProxy(f'{comms_node_name}/comm/ping', Empty)
        rospy.wait_for_service(f'{comms_node_name}/comm/launch')
        self.launch_client = rospy.ServiceProxy(f'{comms_node_name}/comm/launch', Empty)
        rospy.wait_for_service(f'{comms_node_name}/comm/land')
        self.land_client = rospy.ServiceProxy(f'{comms_node_name}/comm/land', Empty)
        rospy.wait_for_service(f'{comms_node_name}/comm/abort')
        self.abort_client = rospy.ServiceProxy(f'{comms_node_name}/comm/abort', Empty)
        rospy.wait_for_service(f'{comms_node_name}/comm/set_position')
        self.set_position_client = rospy.ServiceProxy(f'{comms_node_name}/comm/set_position', SetPosition)

        rospy.wait_for_service(f'{comms_node_name}/comm/begin_mapping')
        self.begin_mapping_client = rospy.ServiceProxy(f'{comms_node_name}/comm/begin_mapping', TagPoses)
        rospy.wait_for_service(f'{comms_node_name}/comm/begin_inspecting')
        self.begin_inspecting_client = rospy.ServiceProxy(f'{comms_node_name}/comm/begin_inspecting', Empty)

        self.begin_input_loop()

    def begin_input_loop(self):
        """
        The input loop takes single character commands from the user and sends them to the drone.
        The execution of the command is done in a separate thread so that the input loop can continue to accept commands
        while the service call is ongoing. This is important to allow abort commands to be sent at any time.
        """
        def execute_command(command: str, *args, **kwargs):
            if command == 't':
                self.ping()
            elif command == 'l':
                self.launch()
            elif command == 'd':
                self.land()
            elif command == 'a':
                self.abort()
            elif command == 's':
                self.set_position(*args)
            elif command == 'p':
                self.run_position_test()
            elif command == 'm':
                self.begin_mapping()
            elif command == 'i':
                self.begin_inspecting()
            else:
                print(f"Invalid command: {command}")
        
        while True:
            command = input("Enter command (t=ping, l=launch, d=land, a=abort, s=set position, m=begin mapping, i=begin inspecting, q=quit): ")
            if command == 'q':
                break
            if command == 's':
                position = input("Enter position (x y z): ")
                position = [float(coord) for coord in position.split()]
            else:
                position = None
            Thread(target=execute_command, args=(command, position), daemon=True).start()

    def ping(self):
        self.ping_client()
    
    def launch(self):
        self.launch_client()

    def land(self):
        self.land_client()

    def abort(self):
        self.abort_client()

    def set_position(self, position: list):
        # Ensure that the position has 3 coordinates and is not too near the ground
        if len(position) != 3 or position[2] < 0.5:
            print("Invalid position")
            return
        position = Vector3(*position)
        orientation = Quaternion(*(0, 0, 0, 1))
        res = self.set_position_client(position, orientation)
        suc = res.success
        msg = res.message
        if suc:
            print("Successfully set position")
        else:
            print(f"Failed to set position: {msg}")

    def run_position_test(self):
        # Load the json file 
        test_flight_data_file = FLIGHT_INPUTS_DIR / "test_flight.json"
        with open(test_flight_data_file, 'r') as f:
            test_flight_data = json.load(f)
        for pose in test_flight_data:
            position = pose["position"]
            msg = SetPosition()
            msg.position = position
            res = self.set_position_client(msg)
            suc = res.success
            msg = res.message
            if suc:
                print(f"Successfully set position: {position}")
            else:
                print(f"Failed to set position: {position}")
                return False
            rospy.sleep(1)
        return True

    def begin_mapping(self):
        approximate_tag_map = load_approximate_tag_poses()
        tags = []
        for tag_id in approximate_tag_map.tag_ids():
            tag_params = approximate_tag_map.get_pose(tag_id)
            tag = TransformStamped()
            tag.header.frame_id = "map"
            tag.child_frame_id = f"tag_{tag_id}"
            
            position = tag_params[:3]
            rotation = tag_params[3:]
            tag.transform.translation.x = position[0]
            tag.transform.translation.y = position[1]
            tag.transform.translation.z = position[2]
            tag.transform.rotation.x = rotation[0]
            tag.transform.rotation.y = rotation[1]
            tag.transform.rotation.z = rotation[2]
            tag.transform.rotation.w = rotation[3]

            tag_transform = TagTransform()
            tag_transform.tag_id = tag_id
            tag_transform.transform = tag

            tags.append(tag_transform)
        print(f"Sending mapping message: {tags}")
        self.begin_mapping_client(tags)

    def begin_inspecting(self):
        self.begin_inspecting_client()

if __name__ == '__main__':
    GroundStationNode()
    rospy.spin()
