#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Grain Boundary Normal Vector Distribution Analysis

Analyzes the orientation distribution of grain boundary normal vectors
in 3D polycrystalline microstructures across different projection planes
(XY, XZ, YZ) and energy formulations.

Created on Mon Jul 31 14:33:57 2023
@author: Lin
"""

import os
current_path = os.getcwd()
import math
import sys

import numpy as np
from numpy import seterr
seterr(all='raise')

import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(current_path)
sys.path.append(current_path+'/../../')
sys.path.append(current_path+'/../calculate_tangent/')

import myInput
import post_processing

if __name__ == '__main__':
    # -----------------------------------------------------------------------
    # DATA SOURCE CONFIGURATION
    # -----------------------------------------------------------------------

    npy_file_folder = "/Users/lin/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/3d_poly_for_GG/results/"

    TJ_energy_type_ave = "ave"
    TJ_energy_type_min = "min"
    TJ_energy_type_sum = "sum"

    npy_file_name_aniso_ave = f"p_ori_ave_{TJ_energy_type_ave}E_100_20k_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_min = f"p_ori_ave_{TJ_energy_type_min}E_100_20k_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_sum = f"p_ori_ave_{TJ_energy_type_sum}E_100_20k_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_iso = "p_ori_ave_aveE_100_20k_multiCore64_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"

    # Load data
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)
    npy_file_aniso_min = np.load(npy_file_folder + npy_file_name_aniso_min)
    npy_file_aniso_sum = np.load(npy_file_folder + npy_file_name_aniso_sum)
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)

    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    print(f"The min data size is: {npy_file_aniso_min.shape}")
    print(f"The sum data size is: {npy_file_aniso_sum.shape}")
    print(f"The iso data size is: {npy_file_iso.shape}")
    print("READING DATA DONE")

    # -----------------------------------------------------------------------
    # GRAIN EVOLUTION ANALYSIS SETUP
    # -----------------------------------------------------------------------

    initial_grain_num = 20000
    step_num = npy_file_aniso_ave.shape[0]

    grain_num_aniso_ave = np.zeros(step_num)
    grain_num_aniso_min = np.zeros(step_num)
    grain_num_aniso_sum = np.zeros(step_num)
    grain_num_iso = np.zeros(step_num)

    for i in range(step_num):
        grain_num_aniso_ave[i] = len(set(npy_file_aniso_ave[i,:].flatten()))
        grain_num_aniso_min[i] = len(set(npy_file_aniso_min[i,:].flatten()))
        grain_num_aniso_sum[i] = len(set(npy_file_aniso_sum[i,:].flatten()))
        grain_num_iso[i] = len(set(npy_file_iso[i,:].flatten()))

    bin_width = 0.16
    x_limit = [-0.5, 3.5]
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)

    grain_size_distribution_ave = np.zeros(bin_num)
    special_step_distribution_ave = 2
    grain_size_distribution_min = np.zeros(bin_num)
    special_step_distribution_min = 2
    grain_size_distribution_sum = np.zeros(bin_num)
    special_step_distribution_sum = 2
    grain_size_distribution_iso = np.zeros(bin_num)
    special_step_distribution_iso = 2

    cache_dir = os.path.join(current_path, 'normal_distribution_data')

    # ===============================================================================
    # XY PLANE POLAR VISUALIZATION
    # ===============================================================================

    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=20.0, fontsize=14)

    # Ave case
    newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_ave', special_step_distribution_ave, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave case")

    # Min case
    newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_min', special_step_distribution_min, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_min, "min case")

    # Sum case
    newplace = np.rot90(npy_file_aniso_sum[special_step_distribution_sum,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_sum', special_step_distribution_sum, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_sum, "Sum case")

    # Iso case
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_iso', special_step_distribution_iso, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso case")

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "x", fontsize=14)
    plt.text(np.pi/2, 0.0095, "y", fontsize=14)
    plt.savefig(current_path + "/figures/normal_distribution_3d_xy.png", dpi=400,bbox_inches='tight')

    # ===============================================================================
    # XZ PLANE POLAR VISUALIZATION
    # ===============================================================================

    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=20.0, fontsize=14)

    # Ave case
    newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_ave', special_step_distribution_ave, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave case", 1)

    # Min case
    newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_min', special_step_distribution_min, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_min, "min case", 1)

    # Sum case
    newplace = np.rot90(npy_file_aniso_sum[special_step_distribution_sum,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_sum', special_step_distribution_sum, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_sum, "Sum case", 1)

    # Iso case
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_iso', special_step_distribution_iso, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso case", 1)

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "x", fontsize=14)
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)
    plt.savefig(current_path + "/figures/normal_distribution_3d_xz.png", dpi=400,bbox_inches='tight')

    # ===============================================================================
    # YZ PLANE POLAR VISUALIZATION
    # ===============================================================================

    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=20.0, fontsize=14)

    # Ave case
    newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_ave', special_step_distribution_ave, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave case", 2)

    # Min case
    newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_min', special_step_distribution_min, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_min, "min case", 2)

    # Sum case
    newplace = np.rot90(npy_file_aniso_sum[special_step_distribution_sum,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_sum', special_step_distribution_sum, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_sum, "Sum case", 2)

    # Iso case
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_iso', special_step_distribution_iso, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso case", 2)

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "y", fontsize=14)
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)
    plt.savefig(current_path + "/figures/normal_distribution_3d_yz.png", dpi=400,bbox_inches='tight')
