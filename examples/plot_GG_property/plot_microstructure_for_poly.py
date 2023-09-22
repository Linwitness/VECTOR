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

    cv0 = np.squeeze(structure_figure[step])
    cv0 = np.rot90(cv0,1)
    im = ax.imshow(cv0,vmin=np.min(cv0),vmax=np.max(cv0),cmap='rainbow',interpolation='none') #jet rainbow plasma
    cb = fig.colorbar(im)
    plt.setp(ax.spines.values(), alpha = 0)
    ax.tick_params(which = 'both', size = 0, labelsize = 0)

    plt.savefig(figure_path + f"_ts{step}.png", dpi=400,bbox_inches='tight')


if __name__ == '__main__':
    # File name
    npy_file_folder = "/Users/lin.yang/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_poly_multiCoreCompare/results/"
    circle_energy_000 = "0.0"
    circle_energy_020 = "0.2"
    circle_energy_040 = "0.4"
    circle_energy_060 = "0.6"
    circle_energy_080 = "0.8"
    circle_energy_095 = "0.95"


    npy_file_name_aniso_000 = f"p_ori_ave_aveE_512_multiCore16_delta{circle_energy_000}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_020 = f"p_ori_ave_aveE_512_multiCore16_delta{circle_energy_020}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_040 = f"p_ori_ave_aveE_512_multiCore16_delta{circle_energy_040}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_060 = f"p_ori_ave_aveE_512_multiCore8_delta{circle_energy_060}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_080 = f"p_ori_ave_aveE_512_multiCore16_delta{circle_energy_080}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_095 = f"p_ori_ave_aveE_512_multiCore16_delta{circle_energy_095}_m2_J1_refer_1_0_0_seed56689_kt066.npy"

    # Initial data
    npy_file_aniso_000 = np.load(npy_file_folder + npy_file_name_aniso_000)
    npy_file_aniso_020 = np.load(npy_file_folder + npy_file_name_aniso_020)
    npy_file_aniso_040 = np.load(npy_file_folder + npy_file_name_aniso_040)
    npy_file_aniso_060 = np.load(npy_file_folder + npy_file_name_aniso_060)
    npy_file_aniso_080 = np.load(npy_file_folder + npy_file_name_aniso_080)
    npy_file_aniso_095 = np.load(npy_file_folder + npy_file_name_aniso_095)
    print(f"The 000 data size is: {npy_file_aniso_000.shape}")
    print(f"The 020 data size is: {npy_file_aniso_020.shape}")
    print(f"The 040 data size is: {npy_file_aniso_040.shape}")
    print(f"The 060 data size is: {npy_file_aniso_060.shape}")
    print(f"The 080 data size is: {npy_file_aniso_080.shape}")
    print(f"The 095 data size is: {npy_file_aniso_095.shape}")
    print("READING DATA DONE")

    # Initial container
    initial_grain_num = 512
    step_num = npy_file_aniso_000.shape[0]

    bin_width = 0.16 # Grain size distribution
    x_limit = [-0.5, 3.5]
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)

    special_step_distribution_000 = 89 # 2670/30 - 10 grains
    special_step_distribution_020 = 75 # 2250/30 - 10 grains
    special_step_distribution_040 = 116 # 3480/30 - 10 grains
    special_step_distribution_060 = 106 # 3180/30 - 10 grains
    special_step_distribution_080 = 105 # 3150/30 - 10 grains
    special_step_distribution_095 = 64 # 1920/30 - 10 grains


    # Start microstructure figure
    figure_path = current_path + "/figures/microstructure_poly"
    plot_structure_figure(special_step_distribution_000, npy_file_aniso_000[:,:,:,0], figure_path + "_000")
    plot_structure_figure(special_step_distribution_020, npy_file_aniso_020[:,:,:,0], figure_path + "_020")
    plot_structure_figure(special_step_distribution_040, npy_file_aniso_040[:,:,:,0], figure_path + "_040")
    plot_structure_figure(special_step_distribution_060, npy_file_aniso_060[:,:,:,0], figure_path + "_060")
    plot_structure_figure(special_step_distribution_080, npy_file_aniso_080[:,:,:,0], figure_path + "_080")
    plot_structure_figure(special_step_distribution_095, npy_file_aniso_095[:,:,:,0], figure_path + "_095")










