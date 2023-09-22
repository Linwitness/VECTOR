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
    im = ax.imshow(cv0,vmin=np.min(cv_initial),vmax=np.max(cv_initial),cmap='gray_r',interpolation='none') #jet rainbow plasma
    cb = fig.colorbar(im)
    cb.ax.tick_params(labelsize=14)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.tick_params(which = 'both', size = 0, labelsize = 0)

    plt.savefig(figure_path + f"_ts{step*30}.png", dpi=400,bbox_inches='tight')


if __name__ == '__main__':
    # File name
    npy_file_folder = "/Users/lin.yang/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_circle_multiCoreCompare/results/"
    circle_energy_000 = "0.0"
    circle_energy_020 = "0.2"
    circle_energy_040 = "0.4"
    circle_energy_060 = "0.6"
    circle_energy_080 = "0.8"
    circle_energy_095 = "0.95"
    circle_energy_080_087 = "_0.87_0.5_0"
    circle_energy_080_071 = "_0.71_0.71_0"
    circle_energy_080_050 = "_0.5_0.87_0"
    circle_energy_080_100 = "_0_1_0"

    circle_energy_095_m4 = "4"
    circle_energy_095_m6 = "6"



    npy_file_name_aniso_000 = f"c_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_000}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_020 = f"c_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_020}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_040 = f"c_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_040}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_060 = f"c_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_060}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_080 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_080}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_095 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_095}_m2_refer_1_0_0.npy"

    npy_file_name_aniso_080_087 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_095}_m2_refer{circle_energy_080_087}.npy"
    npy_file_name_aniso_080_071 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_095}_m2_refer{circle_energy_080_071}.npy"
    npy_file_name_aniso_080_050 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_095}_m2_refer{circle_energy_080_050}.npy"
    npy_file_name_aniso_080_100 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_095}_m2_refer{circle_energy_080_100}.npy"

    npy_file_name_aniso_095_m4 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_095}_m{circle_energy_095_m4}_refer_1_0_0.npy"
    npy_file_name_aniso_095_m6 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_095}_m{circle_energy_095_m6}_refer_1_0_0.npy"

    # Initial data
    npy_file_aniso_000 = np.load(npy_file_folder + npy_file_name_aniso_000)
    npy_file_aniso_020 = np.load(npy_file_folder + npy_file_name_aniso_020)
    npy_file_aniso_040 = np.load(npy_file_folder + npy_file_name_aniso_040)
    npy_file_aniso_060 = np.load(npy_file_folder + npy_file_name_aniso_060)
    npy_file_aniso_080 = np.load(npy_file_folder + npy_file_name_aniso_080)
    npy_file_aniso_095 = np.load(npy_file_folder + npy_file_name_aniso_095)
    npy_file_aniso_080_087 = np.load(npy_file_folder + npy_file_name_aniso_080_087)
    npy_file_aniso_080_071 = np.load(npy_file_folder + npy_file_name_aniso_080_071)
    npy_file_aniso_080_050 = np.load(npy_file_folder + npy_file_name_aniso_080_050)
    npy_file_aniso_080_100 = np.load(npy_file_folder + npy_file_name_aniso_080_100)
    npy_file_aniso_095_m4 = np.load(npy_file_folder + npy_file_name_aniso_095_m4)
    npy_file_aniso_095_m6 = np.load(npy_file_folder + npy_file_name_aniso_095_m6)
    print(f"The 000 data size is: {npy_file_aniso_000.shape}")
    print(f"The 020 data size is: {npy_file_aniso_020.shape}")
    print(f"The 040 data size is: {npy_file_aniso_040.shape}")
    print(f"The 060 data size is: {npy_file_aniso_060.shape}")
    print(f"The 080 data size is: {npy_file_aniso_080.shape}")
    print(f"The 095 data size is: {npy_file_aniso_095.shape}")
    print("READING DATA DONE")

    # Initial container
    initial_grain_num = 2
    step_num = npy_file_aniso_000.shape[0]

    special_step_distribution_000 = 30#4
    special_step_distribution_020 = 30#4
    special_step_distribution_040 = 30#4
    special_step_distribution_060 = 30#4
    special_step_distribution_080 = 30#4
    special_step_distribution_095 = 30#4

    special_step_distribution_080_000 = 28#4
    special_step_distribution_080_087 = 28#4
    special_step_distribution_080_071 = 28#4
    special_step_distribution_080_050 = 28#4
    special_step_distribution_080_100 = 28#4

    special_step_distribution_095_m2 = 14
    special_step_distribution_095_m4 = 14
    special_step_distribution_095_m6 = 14


    # Start microstructure figure
    figure_path = current_path + "/figures/microstructure_circle"
    plot_structure_figure(special_step_distribution_000, npy_file_aniso_000[:,:,:,0], figure_path + "_000")
    plot_structure_figure(special_step_distribution_020, npy_file_aniso_020[:,:,:,0], figure_path + "_020")
    plot_structure_figure(special_step_distribution_040, npy_file_aniso_040[:,:,:,0], figure_path + "_040")
    plot_structure_figure(special_step_distribution_060, npy_file_aniso_060[:,:,:,0], figure_path + "_060")
    plot_structure_figure(special_step_distribution_080, npy_file_aniso_080[:,:,:,0], figure_path + "_080")
    plot_structure_figure(special_step_distribution_095, npy_file_aniso_095[:,:,:,0], figure_path + "_095")

    plot_structure_figure(special_step_distribution_080_000, npy_file_aniso_095[:,:,:,0], figure_path + "_095_000")
    plot_structure_figure(special_step_distribution_080_087, npy_file_aniso_080_087[:,:,:,0], figure_path + "_095_087")
    plot_structure_figure(special_step_distribution_080_071, npy_file_aniso_080_071[:,:,:,0], figure_path + "_095_071")
    plot_structure_figure(special_step_distribution_080_050, npy_file_aniso_080_050[:,:,:,0], figure_path + "_095_050")
    plot_structure_figure(special_step_distribution_080_100, npy_file_aniso_080_100[:,:,:,0], figure_path + "_095_100")

    plot_structure_figure(special_step_distribution_095_m2, npy_file_aniso_095[:,:,:,0], figure_path + "_095_m2")
    plot_structure_figure(special_step_distribution_095_m4, npy_file_aniso_095_m4[:,:,:,0], figure_path + "_095_m4")
    plot_structure_figure(special_step_distribution_095_m6, npy_file_aniso_095_m6[:,:,:,0], figure_path + "_095_m6")












