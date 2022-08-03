# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:01:10 2021

@author: Simulation
"""

import cape
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import sim 
import time


# connect to CoppeliaSim
sim.simxFinish(-1)
clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

data = spio.loadmat('mesh_data_stripe_electrodes.mat', squeeze_me=True)

nr_elements = data['NrElements']
nr_nodes = data['NrNodes']
inzid = data['inzid']
nodes = data['node']
roi_nodes = data['AllNodes']
electrode_nodes = data['elecNodes']
centroids = data['s']
nr_electrodes = 4
max_distance = 500
max_length = 500
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
bnd_electrode = np.ones((electrode_nodes[1].size, 1))
bnd_vector = np.concatenate((roi_nodes, electrode_nodes[1]))
bnd_vals = np.concatenate((bnd_roi, bnd_electrode))

# compute boundary vector and matrix
K1, B1 = cap.generateBoundaryMatrices(bnd_vector, bnd_vals)


# compute clusters based on mesh
cap.computeClusters()

if clientID != -1:
    print('Connected to CoppeliaSim')
    
    _, visionHandler = sim.simxGetObjectHandle(clientID, 'cap', sim.simx_opmode_blocking)
    
    for kk in range(100000):

      _, _, depthImage = sim.simxGetVisionSensorDepthBuffer(clientID, visionHandler, sim.simx_opmode_blocking)
      _, _, rgbImage = sim.simxGetVisionSensorImage(clientID, visionHandler, 0, sim.simx_opmode_blocking)
      
      depthImage = np.array(depthImage)
      depthImage = depthImage * max_distance
      rgbImage = np.array(rgbImage[0::3])  # extract r-channel of RGB image
      
      cap.depth_data = depthImage

      cap.rgb_data = rgbImage

      tic = time.time()
      cap.meshMasking()            
      cap.solveSystem(K1, B1)
      toc = time.time()