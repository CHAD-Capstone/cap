#!/usr/bin/python
# In order to use the tf2 library we need to use python 2. It sucks. There is a way to recompile for python3 but it's not worth it right now
# since this is just a simple node
# https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/

"""
A node that allows for setting positions in the VICON frame even when the local position estimate is coming from realsense
We do this by continuously updating our local position and VICON position
Then when we get a position request on the /capdrone/set_position/local topic
we pass that through a transformation from the VICON frame to the local frame and then publish
on the /mavros/setpoint_position/local topic

In order to get the drone's local position estimate we subscribe to the /mavros/local_position/pose topic which returns a PoseStamped
In order to get the VICON position we subscribe to /vicon/ROB498_Drone/ROB498_Drone which returns a TransformStamped

Or if a configuration is set to not use VICON we just pass the position request through to the /mavros/setpoint_position/local topic
"""


"""
Example Vicon Message:
header:
  seq: 54563
  stamp:
    secs: 1710347918
    nsecs: 918024047
  frame_id: "/vicon/world"
child_frame_id: "vicon/ROB498_Drone/ROB498_Drone"
transform:
  translation:
    x: 2.34793242537
    y: 3.75589067817
    z: 1.33607517665
  rotation:
    x: 0.971905385926
    y: 0.0122890208308
    z: -0.169273762041
    w: 0.163080637294

Example Local Position Message:
header:
  seq: 46
  stamp:
    secs: 1710254631
    nsecs: 171372019
  frame_id: "map"
pose:
  position:
    x: -0.097338013351
    y: -0.0493664368987
    z: -0.00404835911468
  orientation:
    x: -0.0164427111554
    y: 0.0027288563173
    z: 0.000668491164231
    w: -0.999860940432
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped
# import tf2_ros
import tf.transformations as tr

from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Transform
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3

from cap.transformation_lib import transform_stamped_to_matrix, pose_stamped_to_matrix

class ViconPositionSetNode:
    def __init__(self, group_number, use_vicon):
        node_name = 'vicon_recorder_{:02d}'.format(group_number)
        rospy.init_node(node_name)
        print("Starting VICON Node with name", node_name)

        self.use_vicon = use_vicon

        self.vicon_sub = rospy.Subscriber('/vicon/ROB498_Drone/ROB498_Drone', TransformStamped, callback=self.vicon_cb)
        self.local_position_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, callback=self.local_position_cb)

        self.capdrone_set_position_local_sub = rospy.Subscriber('/capstone/setpoint_position/local', PoseStamped, callback=self.set_position_local_cb)
        self.local_position_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)

        # self.tfBuffer = tf2_ros.Buffer()
        # self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.current_vicon_transform = None
        self.last_vicon_update_time = -1
        # self.vicon_transform_publisher = tf2_ros.TransformBroadcaster()
        self.current_local_position = None
        self.last_local_position_update_time = -1

        self.safety_bounds_horizontal = 3
        self.safety_bounds_vertical = 3

        self.ready = False

        if self.use_vicon:
          print("Waiting for VICON...")
          while self.current_vicon_transform is None and not rospy.is_shutdown():
              rospy.sleep(0.1)

        print("Waiting for local position...")
        while self.current_local_position is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        print("Ready to set positions")
        rospy.spin()

    def vicon_cb(self, msg):
        self.current_vicon_transform = msg
        self.last_vicon_update_time = rospy.get_time()

    def local_position_cb(self, msg):
        self.current_local_position = msg
        self.last_local_position_update_time = rospy.get_time()

    def set_position_local_cb(self, msg):
        if not self.use_vicon:
            rospy.loginfo("Sending position command")
            self.local_position_pub.publish(msg)
            return

        print('\n\n')
        
        # # Otherwise we need to transform the position from the VICON frame to the local frame
        # transform = self.tfBuffer.lookup_transform('base_link', 'vicon/ROB498_Drone/ROB498_Drone', rospy.Time())
        # # Now we transform the position from the VICON frame to the local frame
        # pose_transformed = tf2_geometry_msgs.do_transform_pose(msg, transform)
        # self.local_position_pub.publish(pose_transformed)

        # Ok, I don't know how to set that up properly so we are just going to use a transformation matrix
        # Get the transform from the vicon drone frame to the vicon world frame
        T_V_D = transform_stamped_to_matrix(self.current_vicon_transform)
        # Sanity check, set T_V_D to identity with position 1, 0, 0
        #T_V_D = np.eye(4)
        #T_V_D[0, 3] = 1
        T_D_V = np.linalg.inv(T_V_D)
        # Get the xyz euler angles from the transform
        print("VICON euler angles (VICON to Drone)", tr.euler_from_matrix(T_D_V))
        print("Vicon Transform Stamped", self.current_vicon_transform.transform)
        print("VICON World to Drone SE3", T_D_V)
        # Get the transform from the realsense drone frame to the realsense world frame
        T_R_D = pose_stamped_to_matrix(self.current_local_position)
        # Sanity check, set T_R_D to identity
        #T_R_D = np.eye(4)
        print("Realsense euler angles (Drone to Realsense)", tr.euler_from_matrix(T_R_D))
        print("Realsense Local Position", self.current_local_position.pose)
        print("Drone to Realsense World SE3", T_R_D)
        # We can then get the transform from the vicon world frame to the realsense world frame
        # T_R_V = T_R @ np.linalg.inv(T_V)
        # Not allowed in python2
        T_R_V = np.dot(T_R_D, T_D_V)
        print("VICON to realsense euler angles", tr.euler_from_matrix(T_R_V))
        print("VICON to Realsense Transform SE3", T_R_V)

        # Now we get the transformation matrix representing the position we want to set in the VICON world frame
        T_V_D = pose_stamped_to_matrix(msg)

        # And then transform it to the realsense world frame
        # T_R_D = T_R_V @ T_V_D
        T_R_D = np.dot(T_R_V, T_V_D)

        # We can then extract the position and orientation from this transformation matrix
        p = T_R_D[0:3, -1]
        q = tr.quaternion_from_matrix(T_R_D)

        # If the position exceeds the safety bounds we set it to the safety bounds
        # For the horizontal, if if either abs(x) or abs(y) is greater than the safety bounds we set it to the safety bounds
        # For the vertical, if z < 0 or if z > safety bounds we set it to 0 or the safety bounds
        if self.safety_bounds_horizontal is not None:
            print("WARNING: Safety bounds are set to", self.safety_bounds_horizontal)
            p[0] = np.clip(p[0], -self.safety_bounds_horizontal, self.safety_bounds_horizontal)
            p[1] = np.clip(p[1], -self.safety_bounds_horizontal, self.safety_bounds_horizontal)
        
        if self.safety_bounds_vertical is not None:
            print("WARNING: Safety bounds are set to", self.safety_bounds_vertical)
            p[2] = np.clip(p[2], 0, self.safety_bounds_vertical)

        # And then create a new PoseStamped message to publish
        new_pose = PoseStamped()
        new_pose.header.stamp = rospy.Time.now()
        new_pose.header.frame_id = 'map'
        new_pose.pose.position.x = p[0]
        new_pose.pose.position.y = p[1]
        new_pose.pose.position.z = p[2]
        new_pose.pose.orientation.x = q[0]
        new_pose.pose.orientation.y = q[1]
        new_pose.pose.orientation.z = q[2]
        new_pose.pose.orientation.w = q[3]

        print("PoseStamped in Realsense World Frame:", new_pose)

        # self.local_position_pub.publish(new_pose)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start the VICON position set node")
    parser.add_argument("group_number", type=int, help="The group number of the drone")
    parser.add_argument("--use_vicon", action="store_true", help="Whether to use VICON or not")
    args = parser.parse_args()

    vicon_position_set_node = ViconPositionSetNode(args.group_number, args.use_vicon)
