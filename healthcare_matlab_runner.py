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
import matplotlib.pyplot as plt
import scipy.io as spio
import cape
import socket

arm_filename = 'arm_touchread_sensitive_ikea_6sensorsnewimproved_pi8_15cm_iteration_%d_movement_%s.pkl'
leg_filename = 'leg_touchread_sensitive_ikea_6sensorsnewimproved_pi8_15cm_iteration_%d_movement_%s.pkl'
directory = 'trainingdata'

def load_data(participants=range(2, 14), arm=True, limb_segment='wrist', trajectory=None):
    # Load collected capacitance data and ground truth pose
    seg = {'wrist': 0, 'forearm': 1, 'upperarm': 2, 'ankle': 0, 'shin': 1, 'knee': 2}[limb_segment]
    files = [os.path.join('participant_%d' % p, (arm_filename if arm else leg_filename) % (seg, wildcard)) for p in participants for wildcard in (['?', '??', '???'] if trajectory is None else [str(trajectory)])]
    capacitance, pos_orient, times = util_shared.load_data_from_file(files=files, directory=directory)
    print(np.shape(capacitance), np.shape(pos_orient))
    pos_y, pos_z, angle_y, angle_z = pos_orient[:, 0], pos_orient[:, 1], pos_orient[:, 2], pos_orient[:, 3]
    # return capacitance, pos_y, pos_z, angle_y, angle_z
    return capacitance, pos_orient

c_real, po = load_data(participants=[8], arm=True, limb_segment='forearm', trajectory=27)


# position on the human lib where the robot touches the forarm (startposition for measurements)
init_pos =  [-0.38, -0.84, 1.10]
init_orient = [0, np.pi/2, 0]

positions = np.insert(po, 0, [init_pos[0], init_pos[2], 0, 0], 0)

nr_points = positions.shape[0]

env = gym.make('ValidationSawyer-v1')
env.render()

env._max_episode_steps = 10000

observation = env.reset()
frames = []

# Simulate 50 time steps
done = 0
counter = 0
loop_counter=0
while counter < nr_points:
    print("Go to next position\n")
    done = 0
    pos, orient = env.tool.get_base_pos_orient()#env.robot.get_pos_orient(env.robot.right_end_effector)
    if counter > 0:
        target_pos = [init_pos[0], init_pos[1]+positions[counter, 0], init_pos[2]+positions[counter, 1]]
        target_orient = [np.pi/2, np.pi/2-positions[counter,2], positions[counter,3]]
    else:
        target_pos = [init_pos[0], init_pos[1], 0.05+positions[counter, 1]] #0.02 accounts for sensor offset at tcp
        target_orient = [np.pi/2, np.pi/2, 0]

    target_joint_angles = env.robot.ik(env.robot.right_end_effector, target_pos, target_orient,
                                       env.robot.right_arm_ik_indices, max_iterations=1000, use_current_as_rest=True)

    while done == 0:
        loop_counter = loop_counter + 1

        # Step the simulation forward. Have the robot take a random action.
        current_joint_angles = env.robot.get_joint_angles(env.robot.right_arm_joint_indices)
        print(current_joint_angles)

        pos, orient = env.tool.get_base_pos_orient()#env.robot.get_pos_orient(env.robot.right_end_effector)
        rot_mat = p.getMatrixFromQuaternion(orient)

        eye_pos = [pos[0],pos[1],pos[2]-0.03]
        view_vector = np.dot(np.reshape(np.array(rot_mat),(3,3)),eye_pos)
        env.setup_cap_camera(camera_eye=eye_pos, camera_target=[-view_vector[0], -view_vector[1], view_vector[2]*np.cos(np.pi)],
                                                  fov=100, camera_width=64, camera_height=64)

        img, depth = env.get_camera_image_depth()

        action = (target_joint_angles - current_joint_angles)*2
        observation, reward, done, info = env.step(action)

        if np.linalg.norm(action) < 0.1:
            counter = counter + 1
            done = 1
