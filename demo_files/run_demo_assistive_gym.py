# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:01:10 2021

@author: Simulation
"""

import os, gym
import numpy as np
import assistive_gym
from numpngw import write_png, write_apng
#from IPython.display import display, Image
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as spio
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
#from IPython.display import clear_output
import cape

# load mesh data for capacitive simulation
data = spio.loadmat('mesh_data_3stripe_electrodes.mat', squeeze_me=True)
nr_elements = data['NrElements']
nr_nodes = data['NrNodes']
inzid = data['inzid']
nodes = data['node']
roi_nodes = data['AllNodes']
electrode_nodes = data['elecNodes']
centroids = data['s']
nr_electrodes = 4
max_distance = 50
max_length = 50
nr_pixels = 64

# initialize capacitive simulation
cap = cape.CAPE(nr_electrodes, electrode_nodes, nr_elements, nr_nodes,
                roi_nodes, max_distance, max_length,
               nr_pixels, nodes, inzid, centroids)

# initialiize FEM matrices
cap.assembleSystem()

# assign boundary conditions to the problem -> first electrode
bnd_roi = np.zeros((roi_nodes.size, 1))
bnd_electrode = np.ones((electrode_nodes[0].size, 1))
bnd_vector = np.concatenate((roi_nodes, electrode_nodes[0]))
bnd_vals = np.concatenate((bnd_roi, bnd_electrode))

# compute boundary vector and matrix
K1, B1 = cap.generateBoundaryMatrices(bnd_vector, bnd_vals)
cap.K_full = cap.K_full + K1

# compute clusters based on mesh
cap.computeClusters()


#np.set_printoptions(suppress=True, precision=3)

# Make a feeding assistance environment with the PR2 robot.
env = gym.make('ScratchItchPR2-v1')

# Setup a global camera in the environment for scene capturing
env.setup_camera(camera_eye=[-0.6, -0.4, 2], camera_target=[0.2, 0.2, 0], fov=50, camera_width=512, camera_height=512)

nr_runs = 100
observation = env.reset()
frames = []
rgb_global = [None]*nr_runs
depth_cap = [None]*nr_runs
rgb_cap = [None]*nr_runs

plt.ion()

#fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

# Simulate time steps
for i in range(nr_runs):
    # Step the simulation forward. Have the robot take a random action.
    observation, reward, done, info = env.step(env.action_space.sample())

    pos, orient = env.robot.get_pos_orient(env.robot.left_end_effector)
    pos_real, orient_real = env.robot.convert_to_realworld(pos, orient)

    # render image from global camera
    global_img, _ = env.get_camera_image_depth()
    rgb_global[i] = np.array(global_img)

    # Setup local camera for capacitive sensor
    rgb_cap_img, depth_img = env.setup_cap_sensor(camera_eye=[pos[0]+0.05, pos[1], pos[2]-0.05], camera_target=[pos[0], pos[1], -pos[2]])
    depth_cap[i] = np.array(depth_img)
    rgb_cap[i] = np.array(rgb_cap_img)

    cap.depth_data = depth_cap[i]
    cap.rgb_data = 50 * np.ones((64, 64, 4))
    cap.meshMasking()
    cap.solveSystem(K1, B1)
    print("Capacitance: ", cap.cap_vector[i])

    #ax1.imshow(rgb_global[i])
    #ax2.imshow(rgb_cap[i])
    #ax3.imshow(depth_cap[i])
    #ax4.plot(cap.cap_vector[0:i])
    #ax1.title.set_text('Scene View')
    #ax2.title.set_text('RGB Image Cap. Sensor')
    #ax3.title.set_text('Depth Image Cap. Sensor')
    #ax4.title.set_text('Cap. Values (F)')
    #plt.pause(0.001)
    #plt.show()