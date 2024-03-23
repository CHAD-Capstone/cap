#!/usr/bin/python3
"""
A simple node that records VICON data and saves it to a file.
"""

import rospy
import numpy as np
from geometry_msgs.msg import TransformStamped
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pathlib import Path

class ViconRecorder:
    def __init__(self, save_path: Path, group_number: int = 6):
        ### Initialize the node
        node_name = 'vicon_recorder_{:02d}'.format(group_number)
        rospy.init_node(node_name)
        print("Starting VICON Node with name", node_name)

        self.recorded_positions = []
        self.recorded_quaternions = []
        self.vicon_sub = rospy.Subscriber('/vicon/ROB498_Drone/ROB498_Drone', TransformStamped, callback=self.vicon_cb)
        self.recorded_flag = False
        self.recording_flag = False

        # Wait until we have received a message from the VICON system
        print("Waiting for VICON data...")
        while not self.recorded_flag and not rospy.is_shutdown():
            rospy.sleep(0.1)

        # Wait for the user to press enter to start recording
        input("Press enter to start recording...")
        self.recording_flag = True

        # Wait for the user to press enter to stop recording
        input("Press enter to stop recording...")
        self.recording_flag = False

        # Save the recorded data to a file
        self.recorded_positions = np.array(self.recorded_positions)
        self.recorded_quaternions = np.array(self.recorded_quaternions)
        np.save(save_path / 'recorded_positions.npy', self.recorded_positions)
        print("Saved recorded positions to", save_path / 'recorded_positions.npy')
        np.save(save_path / 'recorded_quaternions.npy', self.recorded_quaternions)
        print("Saved recorded quaternions to", save_path / 'recorded_quaternions.npy')

        # Print some statistics (mean and std of position)
        print("Mean position:", np.mean(self.recorded_positions, axis=0))
        print("Std position:", np.std(self.recorded_positions, axis=0))

        # Distance traveled
        distance_travelled = np.sum(np.linalg.norm(np.diff(self.recorded_positions, axis=0), axis=1))
        print("Distance travelled:", distance_travelled)

        # Now we produce a 3d plot of the recorded positions and save it with the data
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.recorded_positions[:,0], self.recorded_positions[:,1], self.recorded_positions[:,2], c='k')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Recorded VICON Positions')
        plt.savefig(str(save_path / 'recorded_positions.png'))


    def vicon_cb(self, msg):
        if self.recording_flag:
            self.recorded_positions.append([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z])
            self.recorded_quaternions.append([msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w])

        self.recorded_flag = True


if __name__ == "__main__":
    save_path = Path(__file__).parent / 'recorded_data'
    save_path.mkdir(exist_ok=True)
    ViconRecorder(save_path, group_number=6)
    rospy.spin()