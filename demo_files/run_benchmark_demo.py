# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:01:10 2021

@author: Simulation
"""

import cape
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import sim 
import time
import csv

joint_states = [None]*415019
jointHandler = [None]*6
row_count = 0

with open('jointAngles_sphere_14_09.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        joint_states[row_count] = row[2::1]
        row_count = row_count + 1

joint_states = np.array(joint_states[1::1]).astype(float)
joint_states[:,0] = joint_states[:,0] - np.pi/2
joint_states[:,1] = joint_states[:,1] + np.pi/2
joint_states[:,3] = joint_states[:,3] + np.pi/2

# connect to CoppeliaSim
sim.simxFinish(-1)
clientID1 = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
#clientID1 = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

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
nr_pixels = 32


# Set the number of threads in a block
threadsperblock = 32 

# Calculate the number of thread blocks in the grid
blockspergrid = (nr_nodes + (threadsperblock - 1)) // threadsperblock



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

# compute clusters based on mesh
cap.computeClusters()

joint_counter = 1
nr_points=400000
init_joint_pos = 0.8#joint_states[1,0]

if clientID1 != -1:
    print('Connected to CoppeliaSim')
    
    _, visionHandler = sim.simxGetObjectHandle(clientID1, 'cap', sim.simx_opmode_blocking)
    
    for i in range(6): 
        string = "UR10_joint%d" % (i)
        _, jointHandler[i] = sim.simxGetObjectHandle(clientID1, string, sim.simx_opmode_blocking)
    
    sim.simxSetJointTargetPosition(clientID1, jointHandler[0], init_joint_pos, sim.simx_opmode_oneshot)
    
    while 1:
      _, _, depthImage = sim.simxGetVisionSensorDepthBuffer(clientID1, visionHandler, sim.simx_opmode_blocking)
      _, _, rgbImage = sim.simxGetVisionSensorImage(clientID1, visionHandler, 0, sim.simx_opmode_blocking)
      
      depthImage = np.array(depthImage)
      depthImage = depthImage * max_distance   
      rgbImage = np.array(rgbImage[0::3])  # extract r-channel of RGB image
      
      cap.depth_data = depthImage
      cap.rgb_data = rgbImage
      
      for j in range(nr_points):
          #for i in range(2):  
          #sim.simxSetJointTargetPosition(clientID1, jointHandler[0], init_joint_pos+j*1e-6, sim.simx_opmode_oneshot)
          _, jointpos=sim.simxGetJointPosition(clientID1, jointHandler[0], sim.simx_opmode_oneshot)
          
         # if jointpos >= 0.5:
          #sim.simxSetJointTargetPosition(clientID1, jointHandler[0], init_joint_pos-j*1e-5, sim.simx_opmode_oneshot)
          
          if j >= nr_points/2:
              sim.simxSetJointTargetPosition(clientID1, jointHandler[0], (init_joint_pos-(0.5e-5*nr_points/2))+(j-nr_points/2)*0.5e-5, sim.simx_opmode_oneshot)
          else:
              sim.simxSetJointTargetPosition(clientID1, jointHandler[0], init_joint_pos-j*0.5e-5, sim.simx_opmode_oneshot)
             # if (jointpos >= -0.5) and (jointpos < 0.5):
              #    sim.simxSetJointTargetPosition(clientID1, jointHandler[0], init_joint_pos+j*1e-6, sim.simx_opmode_oneshot)
          #sim.simxSetJointTargetPosition(clientID, jointHandler[1], float(joint_states[j+1][0]), sim.simx_opmode_oneshot)
          #time.sleep(0.05)
      #joint_counter = joint_counter + 1
      #joint_counter = joint_counter + 1
          #cap.meshMasking()            
          #cap.solveSystem(K1, B1)
      
      #plt.plot(cap.cap_vector)
      #plt.show()
      #time.sleep(0.01)
      #fig = plt.figure()
      #ax = fig.add_subplot(projection='3d')
      #ax.scatter(cap.centroids[cap.elements,0], cap.centroids[cap.elements,1], cap.centroids[cap.elements,2])

      #ax.set_xlim(0,50)
      #ax.set_ylim(0,50)
      #ax.set_zlim(0,50)
      #plt.set_xlabel('Time')
      #plt.set_ylabel('Capacitance (F)')
      #ax.set_zlabel('Z Label')
      #plt.show()