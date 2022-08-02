# -*- coding: utf-8 -*-
"""
CAPE - Real-time Capacitive Sensor Simulation Engine for Robotics

This code contains CAPE, a real-time 3D FEM-based capacitive sensor 
simulator aimed for robotic applications such as assitive living, 
healthcare robotics and general human-robot interaction.


Author: Christian Schöffmann
Affiliation: University of Klagenfurt, Insitute of Smart System Technologies
Latest Version: 15.06.2021
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import time


class CAPE:
    def __init__(self, nr_electrodes, electrode_nodes, nr_elements, nr_nodes, roi_nodes,
                 max_distance, max_length, nr_pixels, nodes, inzid, centroids):
        self.nr_elements = nr_elements
        self.nr_electrodes = nr_electrodes
        self.nr_nodes = nr_nodes
        self.nr_pixels = nr_pixels
        self.max_distance = max_distance
        self.max_length = max_length
        self.roi_nodes = roi_nodes
        self.electrode_nodes = electrode_nodes
        self.matval = np.ones((1, self.nr_elements))
        self.centroids = centroids
        self.nodes = nodes
        self.inzid = inzid - 1
        self.K_local = np.zeros((4, 4, self.nr_elements))
        self.K_init = np.zeros((self.nr_nodes, self.nr_nodes))
        self.B_init = np.zeros((1, self.nr_nodes))
        self.K = np.zeros((self.nr_nodes, self.nr_nodes))
        self.B = np.zeros((1, self.nr_nodes))
        self.clipping = 1e6
        self.matval_old = np.ones((1, self.nr_elements))
        self.delta_matval = np.ones((1, self.nr_elements))
        self.permittivity_air = 1
        self.pixel_idx = np.ones((1, self.nr_elements))
        self.depth_data = np.zeros((self.nr_pixels, self.nr_pixels))
        self.rgb_data = np.zeros((self.nr_pixels, self.nr_pixels))
        self.delta_elements = 0
        self.counter = 0
        self.K_bnd = np.zeros((self.nr_nodes, self.nr_nodes))
        self.K_full = np.zeros((self.nr_nodes, self.nr_nodes))
        self.diag_idx = 0
        self.elements_idx = [None] * self.nr_pixels * self.nr_pixels
        self.u = np.zeros((1, self.nr_nodes))
        self.cap_vector = [None] * 1000000

    ###########################################################
    # Function which assembles and initialized 
    # the local and global stiffnes matrices given from 
    # mesh input data
    ###########################################################
    def assembleSystem(self):
        scaler = 1e-3  # scale to meters
        self.nodes = self.nodes.T

        for k in range(self.nr_elements - 1):
            y1 = self.nodes[1, self.inzid[k, 0]] * scaler
            y2 = self.nodes[1, self.inzid[k, 1]] * scaler
            y3 = self.nodes[1, self.inzid[k, 2]] * scaler
            y4 = self.nodes[1, self.inzid[k, 3]] * scaler

            x1 = self.nodes[0, self.inzid[k, 0]] * scaler
            x2 = self.nodes[0, self.inzid[k, 1]] * scaler
            x3 = self.nodes[0, self.inzid[k, 2]] * scaler
            x4 = self.nodes[0, self.inzid[k, 3]] * scaler

            z1 = self.nodes[2, self.inzid[k, 0]] * scaler
            z2 = self.nodes[2, self.inzid[k, 1]] * scaler
            z3 = self.nodes[2, self.inzid[k, 2]] * scaler
            z4 = self.nodes[2, self.inzid[k, 3]] * scaler

            b = np.array([-y2 * z4 + y4 * z2 + y2 * z3 - y4 * z3 - y3 * z2 + y3 * z4,
                          -y1 * z3 + y3 * z1 - y4 * z1 + y1 * z4 - y3 * z4 + y4 * z3,
                          -y4 * z2 - z1 * y2 + y4 * z1 - y1 * z4 + y1 * z2 + y2 * z4,
                          +z1 * y2 - y1 * z2 + y1 * z3 - y2 * z3 + y3 * z2 - y3 * z1])

            c = -np.array([x4 * z2 + x3 * z4 - x2 * z4 - x3 * z2 - x4 * z3 + x2 * z3,
                           x1 * z4 - x1 * z3 - x4 * z1 - x3 * z4 + x4 * z3 + x3 * z1,
                           -x1 * z4 + x1 * z2 + x4 * z1 + x2 * z4 - x4 * z2 - x2 * z1,
                           -x1 * z2 + x1 * z3 - x2 * z3 + x3 * z2 + x2 * z1 - x3 * z1])

            d = -np.array([-y2 * x4 + y2 * x3 - y4 * x3 + y3 * x4 - y3 * x2 + y4 * x2,
                           y4 * x3 - x3 * y1 - y3 * x4 + x4 * y1 + x1 * y3 - x1 * y4,
                           -x1 * y2 + x2 * y1 + x1 * y4 + y2 * x4 - y4 * x2 - x4 * y1,
                           -x2 * y1 - y2 * x3 + y3 * x2 - x1 * y3 + x3 * y1 + x1 * y2])

            A = np.array([[1, x1, y1, z1],
                          [1, x2, y2, z2],
                          [1, x3, y3, z3],
                          [1, x4, y4, z4]])

            volume = (1 / 6) * abs(np.linalg.det(A))

            self.K_local[:, :, k] = (1 / (36 * volume)) * (np.outer(b, b) + np.outer(c, c) + np.outer(d, d))

        for k in range(self.nr_elements - 1):
            self.K_init[np.ix_(self.inzid[k, :], self.inzid[k, :])] = self.K_init[np.ix_(self.inzid[k, :], self.inzid[k, :])] + self.matval[:, k] * self.K_local[:, :, k]

        mat_size = self.K_init.shape
        self.diag_idx = np.arange(1, mat_size[0]*mat_size[1], mat_size[0]+1)
        self.K_full = self.K_init

    ###########################################################
    # Function which precomputes boundary vector and matrices
    # for given electrode structure
    #
    # output: assembled stiffness matrix and vector used
    # for the inverse solver
    ###########################################################
    def generateBoundaryMatrices(self, bnd_vector, bnd_values):
        K_temp = np.eye(self.nr_nodes)
        B_temp = np.zeros((1, self.nr_nodes))
        nodes = np.arange(1., self.nr_nodes)
        nodes[bnd_vector] = 0

        resnodes = np.nonzero(nodes)
        K_temp[resnodes, :] = 0
        B_temp[:, bnd_vector] = bnd_values.T

        return K_temp, B_temp

    ###########################################################
    # Function which precomputes mesh element clusters based
    # on the depth image given from the robot simulator
    ###########################################################
    def computeClusters(self):
        xx = np.linspace(0, self.max_length, self.nr_pixels)
        yy = np.linspace(0, self.max_length, self.nr_pixels)
        B = np.zeros((self.nr_pixels, self.nr_pixels))
        xc_old = xx[0]
        xc = xx[1]
        counter = 0

        for k in range(self.nr_elements - 1):
            B = np.zeros((self.nr_pixels, self.nr_pixels))
            sx = self.centroids[k, 1]
            sy = self.centroids[k, 0]

            idx_x = np.argmin(abs(xx - sx))
            idx_y = np.argmin(abs(yy - sy))

            B[idx_x, idx_y] = 1
            b = B.T.flatten()
            self.pixel_idx[:, k] = int(np.where(b == 1)[0][:])

        for j in range(self.nr_pixels - 1):
            yc_old = yy[j]
            yc = yy[j + 1]

            idx_x = np.where((self.centroids[:, 0] > xc_old) & (self.centroids[:, 0] < xc))[0][:]
            idx_y = np.where((self.centroids[:, 1] > yc_old) & (self.centroids[:, 1] < yc))[0][:]
            self.elements_idx[counter] = np.intersect1d(idx_x, idx_y)

            counter = counter + 1

        for k in range(self.nr_pixels - 1):
            counter = counter + 1
            xc_old = xx[k]
            xc = xx[k + 1]

            for i in range(self.nr_pixels - 1):
                yc_old = yy[i]
                yc = yy[i + 1]

                idx_x = np.where((self.centroids[:, 0] > xc_old) & (self.centroids[:, 0] < xc))[0][:]
                idx_y = np.where((self.centroids[:, 1] > yc_old) & (self.centroids[:, 1] < yc))[0][:]
                self.elements_idx[counter] = np.intersect1d(idx_x, idx_y)

                counter = counter + 1

    ###########################################################
    # Function which sets the boundary conditions in the respective
    # matrices and vectors (i.e. ground potential, drive potential, ...)
    ###########################################################
    def setBoundaries(self, bnd_vector, bnd_values):
        K_temp = np.eye((self.nr_nodes, self.nr_nodes))
        nodes = np.arange(1., self.nr_nodes)
        nodes[bnd_vector] = 0

        resnodes = np.nonzero(nodes)
        K_temp[resnodes, :] = 0
        self.K[bnd_vector, :] = 0
        self.K = sp.csr_matrix(self.K + K_temp)
        self.B[bnd_vector] = bnd_values

    ###########################################################
    # Function which executes mesh masking procedure
    ###########################################################
    def meshMasking(self):
        Z = self.depth_data
        RGB = self.rgb_data
        sidx = []

        idx = np.where(Z >= self.max_distance)[0][:]
        Z[idx] = 0

        Z = np.reshape(Z, (self.nr_pixels, self.nr_pixels))
        U = Z > 0
        UZ = Z * U
        uz = UZ.flatten()
        rgb = RGB.flatten()
        idx = np.where(U.flatten() == 1)[0][:]

        for i in idx:
            if self.elements_idx[i] is not None:
                sidx.append(np.ndarray.tolist(self.elements_idx[i]))
        # sidx = [np.ndarray.tolist(self.elements_idx[i]) for i in idx]
        # sidx = np.concatenate(sidx[:])
        # sidx = np.array(sidx)
        sidx = np.array([np.array(xi) for xi in sidx])
        suidx = np.where(sidx != None)[0][:]

        if (len(suidx) == 0):
            self.matval_old[:] = self.matval[:]
            self.matval[:] = 1
        else:
            sidx = np.concatenate(sidx[:])
            sidx = np.unique(sidx)

            zvals = uz[self.pixel_idx[:, sidx.astype(int)].astype(int)]

            zero_idx = np.where(zvals == 0.0)
            zvals[zero_idx] = np.min(uz[idx])
            z_idx = np.where(self.centroids[sidx.astype(int), 2] < zvals)
            sidx = np.delete(sidx, z_idx)

            rvals = rgb[self.pixel_idx[:, sidx.astype(int)].astype(int)]

            self.elements = sidx
            self.matval_old[:] = self.matval[:]
            self.matval[:] = 1
            self.matval[:, sidx.astype(int)] = rvals[:]

    ###########################################################
    # Function which updates stiffness matrix based on local changes
    # in the obtained region of interest
    ###########################################################
    def femUpdate(self):
        de = self.delta_elements
        for k in range(de[1].size):
            self.K_full[np.ix_(self.inzid[de[1][k], :], self.inzid[de[1][k], :])] = self.K_full[np.ix_(self.inzid[de[1][k], :], self.inzid[de[1][k], :])] + self.delta_matval[:, de[1][k]] * self.K_local[:, :, de[1][k]]

    ##########################################################
    # Function which computes capacitances from electrodes (nodes) of interest
    ###########################################################
    def computeCapacitance(self, electrode_nodes):
        Q = np.sum(self.K_init[electrode_nodes] * np.array(self.u[0]).T)
        self.C = Q * 8.8541878128e-12

    ##########################################################
    # Function which solves the sparse linear system of equations
    ###########################################################    
    def solveSystem(self, K, B):
        self.delta_matval = self.matval - self.matval_old
        self.delta_elements = np.where(self.delta_matval != 0)

        print("Changed Elements", len(self.delta_elements[0]))
        if (len(self.delta_elements[0]) != 0) or (self.counter == 0):
            tic = time.time()
            self.femUpdate()
            toc = time.time()
            #print("Update Time", toc - tic)

            self.K_bnd = self.K_full + K

            tic = time.time()
            self.u = splinalg.cg(sp.csr_matrix(self.K_bnd), B.T)
            toc = time.time()
            #print("stabilized BiCG method:", toc - tic)

            self.computeCapacitance(self.electrode_nodes[0])

            self.cap_vector[self.counter] = self.C

            self.counter = self.counter + 1

        self.cap_vector[self.counter] = self.C
        self.counter = self.counter + 1
