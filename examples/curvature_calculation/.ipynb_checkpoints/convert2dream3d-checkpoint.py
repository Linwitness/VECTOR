#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:21:12 2022

@author: Lin
"""

import os
current_path = os.getcwd()+'/'
import sys
sys.path.append(current_path+'../../')
import numpy as np
import math
import myInput


nx, ny, nz = 200, 200, 200
r = 2

P0,R=myInput.Circle_IC3d(nx,ny,nz,r)
path = current_path + f"sphere_domain{nx}x{ny}x{nz}_r{r}.hdf5"

myInput.output_dream3d(P0, path)