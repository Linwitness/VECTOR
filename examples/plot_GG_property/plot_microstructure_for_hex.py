#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:33:57 2023

@author: Lin
"""

import os
current_path = os.getcwd()
import numpy as np
from numpy import seterr
seterr(all='raise')
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
import sys
sys.path.append(current_path)
sys.path.append(current_path+'/../../')
import myInput
import PACKAGE_MP_Linear as linear2d
sys.path.append(current_path+'/../calculate_tangent/')

def plot_structure_figure(step, structure_figure, figure_path):

    plt.close()
    fig, ax = plt.subplots()

    cv_initial = np.squeeze(structure_figure[0])
    cv0 = np.squeeze(structure_figure[step])
    cv0 = np.rot90(cv0,1)
    im = ax.imshow(cv0,vmin=np.min(cv_initial),vmax=np.max(cv_initial),cmap='rainbow',interpolation='none') #jet rainbow plasma
    # cb = fig.colorbar(im)
    # cb.ax.tick_params(labelsize=20)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.tick_params(which = 'both', size = 0, labelsize = 0)

    plt.savefig(figure_path + f"_ts{step*30}.png", dpi=400,bbox_inches='tight')


if __name__ == '__main__':
    # File name
    npy_file_folder = "/Users/lin/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_hex_for_TJE/results/"
    TJ_energy_type_cases = "ave"


    npy_file_name = f"h_ori_ave_{TJ_energy_type_cases}E_hex_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_angle.npy"

    # Initial data
    npy_file_hex = np.load(npy_file_folder + npy_file_name)
    
    print(f"The 000 data size is: {npy_file_hex.shape}")
    print("READING DATA DONE")

    # Initial container
    initial_grain_num = 48
    step_num = npy_file_hex.shape[0]

    bin_width = 0.16 # Grain size distribution

    special_step_distribution_hex = 0 

    # Start microstructure figure
    figure_path = current_path + "/figures/microstructure_hex"
    plot_structure_figure(special_step_distribution_hex, npy_file_hex[:,:,:,0], figure_path + "_initial")
    









