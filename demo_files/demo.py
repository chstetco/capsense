#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 16:59:42 2021

@author: chstetco
"""

import cape as sim
import time

c = sim.CAPE()

c.buildMesh()         

c.assembleFEM()

tic = time.time()
c.updateStiffnessMatrix()
toc = time.time()

print(toc-tic)