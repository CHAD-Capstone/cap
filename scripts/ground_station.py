#!/usr/bin/python3
"""
This node takes input from the user to control the drone.
It interacts with the comms node via service calls to instruct the drone to takeoff, land, etc.
"""

from threading import Thread
import rospy
from std_srvs.srv import Empty
from cap_srvs.srv import TagPoses
from cap_srvs.srv import SetPosition
from cap_msgs.msg import TagTransform
from geometry_msgs.msg import TransformStamped

from cap.data_lib import load_approximate_tag_poses

class GroundStationNode:
    def __init__(self, group_id: int = 6):
        node_name = f'ground_station_{group_number:02d}'
        rospy.init_node(node_name)

        comms_node_name = f'rob498_drone_{group_number:02d}'

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
        res = self.set_position_client(position)
        suc = res.success
        msg = res.message
        if suc:
            print("Successfully set position")
        else:
            print(f"Failed to set position: {msg}")

    def begin_mapping(self):
        approximate_tag_map = load_approximate_tag_poses()
        tag_poses = TagPoses()
        tags = []
        for tag_id in approximate_tag_map.tag_ids():
            tag_pose = approximate_tag_map.get_tag_pose(tag_id)
            tag = TransformStamped()
            tag.header.frame_id = "map"
            tag.child_frame_id = f"tag_{tag_id}"
            
            tag_params = matrix_to_params(tag_pose, type='quaternion')
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
        tag_poses.tags = tags
        self.begin_mapping_client(tag_poses)

    def begin_inspecting(self):
        self.begin_inspecting_client()

if __name__ == '__main__':
    GroundStationNode()
    rospy.spin()
