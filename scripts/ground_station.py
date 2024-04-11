#!/usr/bin/python3
"""
This node takes input from the user to control the drone.
It interacts with the comms node via service calls to instruct the drone to takeoff, land, etc.
"""

from threading import Thread
import json

import rospy
from std_srvs.srv import Empty
from std_msgs.msg import Int64
from cap.srv import TagPoses
from cap.srv import SetPosition
from cap.msg import TagTransform
from cap.srv import FindTag, FindTagResponse
from cap.srv import NewTag, NewTagResponse
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion

from cap.transformation_lib import matrix_to_params, transform_stamped_to_matrix
from cap.data_lib import load_approximate_tag_poses
from cap.data_lib import FLIGHT_INPUTS_DIR

class GroundStationNode:
    def __init__(self, group_id: int = 6):
        print("Starting Ground Station")
        node_name = f'ground_station_{group_id:02d}'
        rospy.init_node(node_name)

        comms_node_name = f'rob498_drone_{group_id:02d}'

        # Define the service clients
        print("Waiting for comms node")
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
        print("Comms node up")

        # Test mapping clients
        ## AprilTag Mapping Services
        # Instruct the mapping node to find the location of a tag in the world based on a single image
        self.srv_apriltag_mapping_find_tag = rospy.ServiceProxy("/capdrone/apriltag_mapping/find_tag", FindTag)
        # Instruct the mapping node to expect a new tag id
        self.srv_apriltag_mapping_new_tag = rospy.ServiceProxy("/capdrone/apriltag_mapping/new_tag", NewTag)
        # Instruct the mapping node to capture an image
        self.srv_apriltag_mapping_capture_img = rospy.ServiceProxy("/capdrone/apriltag_mapping/capture_img", Empty)
        # Instruct the mapping node to process the tag
        # Once this returns, we will have a new tag map in the flight data directory
        self.src_apriltag_mapping_process_tag = rospy.ServiceProxy("/capdrone/apriltag_mapping/process_tag", Empty)

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
            if command == 'm':
                self.begin_mapping()
            else:
                position = None
                Thread(target=execute_command, args=(command, position), daemon=True).start()
            
            if command == 'test_map':
                print("Entering mapping testing loop")
                while True:
                    try:
                        command = input("Enter Mapping Command (f=find_tag, n=new_tag, c=take_image, p=process_tag): ")
                        if command == 'f':
                            tag_id = int(input("Enter tag id: "))
                            self.find_tag(tag_id)
                        elif command == 'n':
                            tag_id = int(input("Enter tag id: "))
                            self.new_tag(tag_id)
                        elif command == 'c':
                            self.take_img()
                        elif command == 'p':
                            self.process_tag()
                        elif command == 'q':
                            break
                    except Exception as e:
                        print(e)
        rospy.loginfo("Exiting ground station")

    def find_tag(self, tag_id: int):
        tag_id_param = Int64(tag_id)
        res = self.srv_apriltag_mapping_find_tag(tag_id_param)
        found = res.found.data
        transform = res.transform
        T = transform_stamped_to_matrix(transform)
        if found:
            print(f"Found tag with transform:\n{transform}")
            print(T)
            return matrix_to_params(T)
        else:
            print(f"Tag not found")
            return None

    def new_tag(self, tag_id: int):
        tag_id_param = Int64(tag_id)
        self.srv_apriltag_mapping_new_tag(tag_id_param)

    def take_img(self):
        self.srv_apriltag_mapping_capture_img()

    def process_tag(self):
        self.src_apriltag_mapping_process_tag()

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
            return True
        else:
            print(f"Failed to set position: {msg}")
            return False

    def run_position_test(self):
        # Load the json file 
        test_flight_data_file = FLIGHT_INPUTS_DIR / "test_flight.json"
        with open(test_flight_data_file, 'r') as f:
            test_flight_data = json.load(f)
        for pose in test_flight_data:
            position = pose["position"]
            rospy.loginfo(f"Flying to position: {position}")
            position = Vector3(*position)
            orientation = Quaternion(*(0, 0, 0, 1))
            res = self.set_position_client(position, orientation)
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
        rospy.loginfo(f"Starting mapping: {approximate_tag_map}")
        for tag_id in approximate_tag_map.tag_ids():
            tag_params = approximate_tag_map.get_pose(tag_id)
            tag_position = tag_params[:3]
            tag_position[2] += 1
            rospy.loginfo(f"Mapping {tag_id} at pose {tag_params}. Flying to {tag_position}")
            inp = input(f"About to fly to {tag_position}. Type n to exit: ")
            if inp == 'n':
                print("BREAKING")
                return
            self.set_position(tag_position)

            # See if the tag is in view
            res = self.find_tag(tag_id)
            if res is None:
                rospy.loginfo(f"Tag {tag_id} not found. Moving to next tag.")
                continue

            # If it is, update the tag position
            refined_tag_position = res[:3]
            rospy.loginfo(f"Found tag at {refined_tag_position}")
            refined_tag_position[2] += 1
            inp = input(f"About to fly to {refined_tag_position}. Type n to exit: ")
            if inp == 'n':
                print("BREAKING")
                return
            self.set_position(refined_tag_position)

            # If the tag is in view, update the tag map
            rospy.loginfo(f"Mapping tag {tag_id}")
            self.new_tag(tag_id)
            self.take_image()
            for x_offset in [-0.3, 0.3]:
                for y_offset in [-0.3, 0.3]:
                    self.set_position([refined_tag_position[0] + x_offset, refined_tag_position[1] + y_offset, refined_tag_position[2]])
                    rospy.sleep(1)
                    self.take_image()
            self.process_tag()

        # Fly back to home
        self.set_position([0, 0, 1])
            

            
        # tags = []
        # for tag_id in approximate_tag_map.tag_ids():
        #     tag_params = approximate_tag_map.get_pose(tag_id)
        #     tag = TransformStamped()
        #     tag.header.frame_id = "map"
        #     tag.child_frame_id = f"tag_{tag_id}"
            
        #     position = tag_params[:3]
        #     rotation = tag_params[3:]
        #     tag.transform.translation.x = position[0]
        #     tag.transform.translation.y = position[1]
        #     tag.transform.translation.z = position[2]
        #     tag.transform.rotation.x = rotation[0]
        #     tag.transform.rotation.y = rotation[1]
        #     tag.transform.rotation.z = rotation[2]
        #     tag.transform.rotation.w = rotation[3]

        #     tag_transform = TagTransform()
        #     tag_transform.tag_id = tag_id
        #     tag_transform.transform = tag

        #     tags.append(tag_transform)
        # print(f"Sending mapping message: {tags}")
        # self.begin_mapping_client(tags)

    def begin_inspecting(self):
        self.begin_inspecting_client()

if __name__ == '__main__':
    GroundStationNode()
    rospy.spin()
