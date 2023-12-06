#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 14:55:28 2021

@author: lin.yang
"""

import os
current_path = os.getcwd()
import sys
sys.path.append(current_path)
sys.path.append('./../../.')
import numpy as np
import math
from itertools import repeat
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

import myInput
import PACKAGE_MP_Linear as smooth     #2D Bilinear smooth algorithm

if __name__ == '__main__':

    # %% 2D initial conditions
    
    input_name = "input/phasefiled_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1)_kt(0.66)_cut(0).h5"
    output_name = "output/phasefiled_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1)_kt(0.66)_cut(0)_inclination_"
    f = h5py.File(input_name, 'r')
    
    for simu in f.keys():
        DataContainer = f.get(simu)
        for dataset in DataContainer.keys():
            tmpdata = DataContainer[dataset]
            dataset = dataset.replace(' ','_')
            globals()[dataset] = tmpdata
    
    steps, nz, nx, ny = ims_id.shape
    ng = len(euler_angles)
    
    main_matrix = np.zeros((steps,3,nx,ny))
    for i in tqdm([300,1600]):
        microstructure = ims_id[i,:]
        microstructure = np.squeeze(microstructure)
        R = np.zeros((nx,ny,2))
        
    
    #%% Start the algorithms
    
        cores = 8
        loop_times = 5
        test1 = smooth.linear_class(nx, ny, ng, cores, loop_times, microstructure, R, 0, False)
        test1.linear_main('inclination')
        P = test1.get_P()
        
            
        #%% Running time
          
        # print('loop_times = ' + str(test1.loop_times))
        # print('running_time = %.2f' % test1.running_time)
        # print('running_core time = %.2f' % test1.running_coreTime)
        # print()
            
    #%% Output the inclination data
    
        P_final = np.array(P)
        P_final[1] = -P[2]
        P_final[2] = P[1]
        P_final[1:] = P_final[1:] / (P_final[1]**2+P_final[2]**2)**0.5
        P_final = np.nan_to_num(P_final)
        main_matrix[i] = P_final
    
        np.save(output_name + f"step{i}", P_final)
    # main_matrix[time_step, inclination_axis, x, y] is the matrix saved all inclination. 
    # The first index is the time step index, 
    # the second index is the incliination axis (0 is microstructrue, 1 is the inclination vector in x-axis, 2 is inclination vector in y-axis),
    # the third index is the x-axis index
    # the fourth index is the y-axis index