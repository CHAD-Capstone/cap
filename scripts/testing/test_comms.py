#!/usr/bin/python3
"""
Waypoints sample
---
header:
  seq: 21
  stamp:
    secs: 1710520594
    nsecs: 948517084
  frame_id: "/vicon/world"
poses:
  -
    position:
      x: -2.6071505825
      y: 2.62710934148
      z: 1.68726454553
    orientation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 0.0
  -
    position:
      x: 3.30590306681
      y: -0.149833106578
      z: 2.21343780336
    orientation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 0.0
  -
    position:
      x: -0.0989495410287
      y: -2.7614141138
      z: 1.34134053549
    orientation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 0.0
  -
    position:
      x: -2.97149312775
      y: -3.05240989713
      z: 1.74142065204
    orientation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 0.0
---
"""

import rospy
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose, Point, Quaternion
import threading

should_publish_waypoints = False

def call_service(service_name):
    global should_publish_waypoints
    rospy.wait_for_service(service_name)
    if "test" in service_name:
        print("Starting test. Will publish waypoints.")
        start_waypoint_publisher()
    else:
        should_publish_waypoints = False
    try:
        service = rospy.ServiceProxy(service_name, Empty)
        resp = service()
        print(f"Service {service_name} called successfully")
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")


def start_waypoint_publisher():
    pub = rospy.Publisher('/rob498_drone_06/comm/waypoints', PoseArray, queue_size=10)
    def publish_waypoints():
        global should_publish_waypoints
        print("Publishing waypoints")
        should_publish_waypoints = True
        rate = rospy.Rate(1)
        while not rospy.is_shutdown() and should_publish_waypoints:
            msg = PoseArray()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "/vicon/world"
            msg.poses = [
                Pose(
                    position=Point(x=-2.6071505825, y=2.62710934148, z=1.68726454553),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=0.0)
                ),
                Pose(
                    position=Point(x=3.30590306681, y=-0.149833106578, z=2.21343780336),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=0.0)
                ),
                Pose(
                    position=Point(x=-0.0989495410287, y=-2.7614141138, z=1.34134053549),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=0.0)
                ),
                Pose(
                    position=Point(x=-2.97149312775, y=-3.05240989713, z=1.74142065204),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=0.0)
                )
            ]
            # msg.poses = [
            #     Pose(
            #         position=Point(x=0, y=2, z=0.5),
            #         orientation=Quaternion(x=0, y=0, z=0, w=0)
            #     ),
            #     Pose(
            #         position=Point(x=2, y=2, z=1.5),
            #         orientation=Quaternion(x=0, y=0, z=0, w=0)
            #     ),
            #     Pose(
            #         position=Point(x=2, y=0, z=0.5),
            #         orientation=Quaternion(x=0, y=0, z=0, w=0)
            #     ),
            # ]
            pub.publish(msg)
            rate.sleep()
        print("Stopped publishing waypoints")
        
    threading.Thread(target=publish_waypoints, daemon=True).start()


def main():
    rospy.init_node('drone_test_script')
    group_number = input("Enter the drone group number: ")
    base_service_name = f"rob498_drone_{int(group_number):02d}/comm/"

    print("Press 'l' to launch, 't' to test, 'a' to abort, 'd' to land, or 'q' to quit.")

    while not rospy.is_shutdown():
        key = input("Enter command: ")
        if key.lower() == 'l':
            call_service(base_service_name + "launch")
        elif key.lower() == 't':
            call_service(base_service_name + "test")
        elif key.lower() == 'a':
            call_service(base_service_name + "abort")
        elif key.lower() == 'd':
            call_service(base_service_name + "land")
        elif key.lower() == 'p':
            call_service(base_service_name + "ping")
        elif key.lower() == 'q':
            break
        else:
            print("Invalid input. Please try again.")

if __name__ == '__main__':
    main()