#!/usr/bin/python
# In order to use the tf2 library we need to use python 2. It sucks. There is a way to recompile for python3 but it's not worth it right now
# since this is just a simple node
# https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/

"""
Subscribes to the VICON topic and transforms into the drone local frame

We have a known transformation matrix from the tracker frame to the drone local frame
We receive the pose in the tracker frame and transform it into the local frame and then publish it on the /mavros/vision_pose/pose
topic to update the drone's position

Sample Odometry Message:
header:
  seq: 52
  stamp:
    secs: 1709852024
    nsecs: 922752142
  frame_id: "camera_odom_frame"
child_frame_id: "camera_pose_frame"
pose:
  pose:
    position:
      x: -0.00908094551414
      y: -0.00197364669293
      z: 0.00299437879585
    orientation:
      x: -0.00822026468813
      y: 0.704043865204
      z: -0.00792602729052
      w: 0.710064709187
  covariance: [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001]
twist:
  twist:
    linear:
      x: -0.0308093444474
      y: 0.0142328086198
      z: -0.00703251670542
    angular:
      x: 0.00189428603112
      y: 0.000682949090887
      z: 0.00048497095954
  covariance: [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001]
"""

import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped, Point, Quaternion
from nav_msgs.msg import Odometry
import threading
import tf.transformations as tf_trans

dx = 0.0  # Given in VICON frame. Vector from VICON to local frame
dy = 0.0
dz = 0.0
roll = 0.0  # From VICON to local frame
pitch = 0.0
yaw = 0.0

trans_quaternion = tf_trans.quaternion_from_euler(roll, pitch, yaw)

def transform_vicon_to_local(vicon_translation, vicon_quaternion):
    """
    Transforms the VICON pose into the local frame

    Returns an Odometry object
    """
    point = Point()
    point.x = vicon_translation.x + dx
    point.y = vicon_translation.y + dy
    point.z = vicon_translation.z + dz

    # Mulitply the quaternions
    vicon_quaternion = [vicon_quaternion.x, vicon_quaternion.y, vicon_quaternion.z, vicon_quaternion.w]
    local_quaternion = tf_trans.quaternion_multiply(vicon_quaternion, trans_quaternion)

    pose = Odometry()
    pose.pose.pose.position = point
    pose.pose.pose.orientation = Quaternion(*local_quaternion)
    pose.header.frame_id = "odom"
    pose.child_frame_id = "base_link"

    pose.pose.covariance = [0] * 36
    pose.twist.covariance = [0] * 36

    pose.twist.twist.linear.x = 0
    pose.twist.twist.linear.y = 0
    pose.twist.twist.linear.z = 0
    pose.twist.twist.angular.x = 0
    pose.twist.twist.angular.y = 0
    pose.twist.twist.angular.z = 0

    return pose

    # odom = Odometry()
    # odom.pose.pose.position = point
    # odom.pose.pose.orientation = Quaternion(*local_quaternion)
    # odom.child_frame_id = "base_link"
    # odom.header.frame_id = "odom"

    # return odom
    
    

class ViconNode:
    def __init__(self, group_number=6, verbose=False):
        """
        Constructor for the ViconNode class
        """
        ### Initialize the node
        node_name = 'vicon_node_{:02d}'.format(group_number)
        rospy.init_node(node_name)
        print("Starting VICON Node with name", node_name)

        self.vicon_sub = rospy.Subscriber('/vicon/ROB498_Drone/ROB498_Drone', TransformStamped, callback=self.vicon_cb)
        self.local_position_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, callback=self.local_position_cb)

        self.current_stamp = None
        self.started = False

        # Spin until we have a timestamp
        print("Waiting for timestampe")
        while self.current_stamp is None and not rospy.is_shutdown():
          rospy.sleep(0.1)
        print("Got timestamp")
        self.started = True

        self.local_pose_pub = rospy.Publisher("/mavros/odometry/out", Odometry, queue_size=10)
        # self.local_pose_pub = rospy.Publisher("/mavros/vision_pose/pose", PoseStamped, queue_size=10)
        """
        Is this the right topic?
        Apparently realsense publishes /mavros/odometry/out
        This person is talking about /mavros/local_position/pose (https://discuss.ardupilot.org/t/vision-position-estimate-not-appearing-in-qgroundcontrol/23978)

        It looks like a few things get shoved together into the visual odometry uORB topics https://docs.px4.io/main/en/ros/external_position_estimation.html#px4-mavlink-integration
        The EKF2 is subscribing to the vehicle_visual_odometry uORB topic so whatever mavros topic we use needs to be published to either 
        VISION_POSITION_ESTIMATE or ODOMETRY.
        """

        self.verbose = verbose

        rospy.spin()

    def vicon_cb(self, msg):
        """
        Callback for the VICON subscriber
        """
        if not self.started:
          return
          
        transform = msg.transform
        translation = transform.translation
        rotation = transform.rotation
        stamp = msg.header.stamp

        if self.verbose:
            print(msg)
            # Print out the translation and rotation
            print('VICON Translation: (x={}, y={}, z={})'.format(translation.x, translation.y, translation.z))
            euler = tf_trans.euler_from_quaternion([rotation.x, rotation.y, rotation.z, rotation.w])
            # print('VICON Rotation: (x={}, y={}, z={}, w={})'.format(rotation.x, rotation.y, rotation.z, rotation.w))
            print('VICON Rotation: (x={}, y={}, z={})'.format(euler[0], euler[1], euler[2]))
            print('VICON Stamp: {}'.format(stamp))

        local_pose = transform_vicon_to_local(translation, rotation)
        # local_pose.header.stamp = stamp
        # local_pose.header.stamp = rospy.Time.now()
        local_pose.header.stamp = self.current_stamp

        if self.verbose and False:
            position = local_pose.pose.pose.position
            oreientation = local_pose.pose.pose.orientation
            print('Local Translation: (x={}, y={}, z={})'.format(position.x, position.y, position.z))
            print('Local Rotation: (x={}, y={}, z={}, w={})'.format(oreientation.x, oreientation.y, oreientation.z, oreientation.w))

        print("\n\n")
        print("pub")
        self.local_pose_pub.publish(local_pose)

    def local_position_cb(self, msg):
      self.current_stamp = msg.header.stamp


if __name__ == '__main__':
    vicon_node = ViconNode(
        group_number=6,
        verbose=True
    )

