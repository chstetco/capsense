# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:01:10 2021

@author: Simulation
"""
import time

import gym, assistive_gym
import pybullet as p
import util_shared
import os
import numpy as np
import scipy.io as spio
import cape
import rospy
from std_msgs.msg import Float64MultiArray

arm_filename = 'arm_touchread_sensitive_ikea_6sensorsnewimproved_pi8_15cm_iteration_%d_movement_%s.pkl'
leg_filename = 'leg_touchread_sensitive_ikea_6sensorsnewimproved_pi8_15cm_iteration_%d_movement_%s.pkl'
directory = 'trainingdata'

def rot_z(angle):
    Rz = np.array([[np.cos(angle), -np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0],
                   [0, 0, 1]])
    return Rz

def rot_y(angle):
    Ry = np.array([[np.cos(angle), 0, np.sin(angle)],
                   [0, 1, 0],
                   [-np.sin(angle), 0, np.cos(angle)]])
    return Ry

def rot_x(angle):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle), np.cos(angle)]])
    return Rx

def load_data(participants=range(2, 14), arm=True, limb_segment='wrist', trajectory=None):
    # Load collected capacitance data and ground truth pose
    seg = {'wrist': 0, 'forearm': 1, 'upperarm': 2, 'ankle': 0, 'shin': 1, 'knee': 2}[limb_segment]
    files = [os.path.join('participant_%d' % p, (arm_filename if arm else leg_filename) % (seg, wildcard)) for p in participants for wildcard in (['?', '??', '???'] if trajectory is None else [str(trajectory)])]
    capacitance, pos_orient, times = util_shared.load_data_from_file(files=files, directory=directory)
    print(np.shape(capacitance), np.shape(pos_orient))
    pos_y, pos_z, angle_y, angle_z = pos_orient[:, 0], pos_orient[:, 1], pos_orient[:, 2], pos_orient[:, 3]
    # return capacitance, pos_y, pos_z, angle_y, angle_z
    return capacitance, pos_orient

c_real, po = load_data(participants=[12], arm=True, limb_segment='forearm', trajectory=8)


# position on the human lib where the robot touches the forarm (startposition for measurements)
init_pos =  [-0.38, -0.84, 1.10]
init_orient = [0, np.pi/2, 0]
positions = np.insert(po, 0, [init_pos[0], init_pos[2], 0, 0], 0)
nr_points = positions.shape[0]

## create ros node for depth image transfer
pub = rospy.Publisher('/depth_image', Float64MultiArray, queue_size=10)
rospy.init_node('depth_image_publisher', anonymous=True)
rate = rospy.Rate(100)
msg = Float64MultiArray()

## create assistive gym envrionment for validation
env = gym.make('ValidationSawyerMesh-v1')
env.render()
env._max_episode_steps = 10000

observation = env.reset()
frames = []

# Simulate 50 time steps
done = 0
counter = 0
nr_pixels = 64
while counter < nr_points:
    print("Go to next position\n")
    done = 0
    pos, orient = env.tool.get_base_pos_orient()
    if counter > 0:
        target_pos = [init_pos[0], init_pos[1]+positions[counter, 0], init_pos[2]+positions[counter, 1]]
        target_orient = [np.pi/2, np.pi/2-positions[counter,2], positions[counter,3]]
        #target_orient = [np.pi / 2, np.pi / 2 + np.pi/2 , 0]
    else:
        target_pos = [init_pos[0], init_pos[1], 0.04+positions[counter, 1]]
        target_orient = [np.pi/2, np.pi/2, 0]

    target_joint_angles = env.robot.ik(env.robot.right_end_effector, target_pos, target_orient,
                                       env.robot.right_arm_ik_indices, max_iterations=1000, use_current_as_rest=True)

    while done == 0:
        # Step the simulation forward. Have the robot take a random action.
        current_joint_angles = env.robot.get_joint_angles(env.robot.right_arm_joint_indices)

        pos, orient = env.tool.get_base_pos_orient()
        rot_mat = p.getMatrixFromQuaternion(orient)

        eye_pos = [pos[0], pos[1], pos[2]-0.02]
        #view_vector = np.dot(np.reshape(np.array(rot_mat), (3, 3)), eye_pos)
        #env.setup_cap_camera(camera_eye=eye_pos, fov=50, camera_width=nr_pixels, camera_height=nr_pixels)

        euler = env.get_euler(orient)

        Rz = rot_z(euler[2])#positions[counter, 3])
        Rx = rot_y(-euler[0])#positions[counter, 2])

        init_view_vec = [0, 0, -1]

        t1 = np.dot(Rz, init_view_vec)
        t2 = np.dot(Rx, t1)

        env.setup_cap_camera(camera_eye=eye_pos, camera_target=[eye_pos[0]+t2[0], eye_pos[1]+t2[1], eye_pos[2]+t2[2]],
                                                 fov=150, camera_width=nr_pixels, camera_height=nr_pixels)

        img, depth = env.get_camera_image_depth()

        msg.data = np.reshape(depth, (nr_pixels*nr_pixels, 1))

        #if counter >= 1:
        pub.publish(msg)

        action = (target_joint_angles - current_joint_angles)*2

        observation, reward, done, info = env.step(action)

        if np.linalg.norm(action) < 0.1:
            counter = counter + 1
            done = 1

env.disconnect()