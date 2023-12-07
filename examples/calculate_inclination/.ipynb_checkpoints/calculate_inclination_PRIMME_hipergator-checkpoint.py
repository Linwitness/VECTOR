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
sys.path.append('../../.')
import numpy as np
import math
from itertools import repeat
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

import myInput
import PACKAGE_MP_Linear as smooth     #2D Bilinear smooth algorithm

def calculate_inclination_data(input_path, output_path, step_list):
    # Calculate and save inclination_data

    # Extract all variable from h5 files
    f = h5py.File(input_path+".h5", 'r')
    for simu in f.keys():
        DataContainer = f.get(simu)
        for dataset in DataContainer.keys():
            tmpdata = DataContainer[dataset]
            dataset = dataset.replace(' ','_')
            globals()[dataset] = tmpdata
    # Get necessary data
    steps, nz, nx, ny = ims_id.shape
    ng = len(euler_angles)

    for i in step_list:
        microstructure = ims_id[i,:]
        microstructure = np.squeeze(microstructure)
        R = np.zeros((nx,ny,2))

        # Build smoothing algorithm class and get P
        cores = 8
        loop_times = 5
        test1 = smooth.linear_class(nx, ny, ng, cores, loop_times, microstructure, R, 0, False)
        test1.linear_main('inclination')
        P = test1.get_P()
        #Running time
        # print('For ' + input_path.spilt('/')[-1])
        # print('loop_times = ' + str(test1.loop_times))
        # print('running_time = %.2f' % test1.running_time)
        # print('running_core time = %.2f' % test1.running_coreTime)
        # print()

        # Output the inclination data
        P_final = np.array(P)
        P_final[1] = -P[2]
        P_final[2] = P[1]
        P_final[1:] = P_final[1:] / (P_final[1]**2+P_final[2]**2)**0.5
        P_final = np.nan_to_num(P_final)
        np.save(output_path + f"step{i}", P_final)


if __name__ == '__main__':

    # 2D initial conditions
    input_folder = "/blue/michael.tonks/share/PRIMME_Inclination/"
    output_folder = "/blue/michael.tonks/share/PRIMME_Inclination_npy_files/"

    input_name = ["Case2AS_T3_tstep_300_600",
                  "Case2BF_T8_tstep_300_1600",
                  "Case2BS_T1_tstep_300_1600",
                  "Case2CF_T4_tstep_300_400",
                  "Case2DF_T1_tstep_300_1600",
                  "Case2DS_T5_tstep_300_800",
                  "Case2DS_T8_tstep_300_1600",
                  "Case3AF_T9_tstep_300_600",
                  "Case3AS_T6_tstep_300_1600",
                  "Case3BF_T7_tstep_300_600",
                  "Case3BS_T10_step_300_1600",
                  "Case3BS_T6_tstep_300_1600",
                  "Case3CS_T7_tstep_300_400",
                  "spparks_s1_tstep_300_400_600_800_1600"]

    step_lists = [[300,600],
                  [300,1600],
                  [300,1600],
                  [300,400],
                  [300,1600],
                  [300,800],
                  [300,1600],
                  [300,600],
                  [300,1600],
                  [300,600],
                  [300,1600],
                  [300,1600],
                  [300,400],
                  [300,400,600,800,1600]]

    for i in range(len(input_name)):
      output_name = input_name[i] + "_inclination_"
      calculate_inclination_data(input_folder + input_name[i], output_folder + output_name, step_lists[i])


