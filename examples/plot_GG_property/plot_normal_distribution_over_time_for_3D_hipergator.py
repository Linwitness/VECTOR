#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Grain Boundary Normal Vector Distribution Analysis for HiPerGator

Analyzes the distribution of grain boundary normal vectors in 3D
microstructures using data from SPPARKS Monte Carlo simulations run on the
University of Florida HiPerGator supercomputing cluster.

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
from tqdm import tqdm
import sys

sys.path.append(current_path)
sys.path.append(current_path+'/../../')
sys.path.append(current_path+'/../calculate_tangent/')

import myInput
import post_processing

if __name__ == '__main__':
    # ========================================================================
    # DATA LOADING: HiPerGator Simulation Results
    # ========================================================================

    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_poly/results/"

    TJ_energy_type_ave = "ave"
    TJ_energy_type_min = "min"
    TJ_energy_type_max = "max"

    npy_file_name_aniso_ave = f"p2_ori_ave_{TJ_energy_type_ave}E_264_5k_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt1.95.npy"
    npy_file_name_aniso_min = f"p_ori_ave_{TJ_energy_type_min}E_264_5k_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt1.95.npy"
    npy_file_name_aniso_max = f"p_ori_ave_{TJ_energy_type_max}E_264_5k_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt1.95.npy"
    npy_file_name_iso = "p_ori_ave_aveE_264_5k_multiCore64_delta0.0_m2_J1_refer_1_0_0_seed56689_kt1.95.npy"

    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)
    npy_file_aniso_min = np.load(npy_file_folder + npy_file_name_aniso_min)
    npy_file_aniso_max = np.load(npy_file_folder + npy_file_name_aniso_max)
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)

    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    print(f"The min data size is: {npy_file_aniso_min.shape}")
    print(f"The max data size is: {npy_file_aniso_max.shape}")
    print(f"The iso data size is: {npy_file_iso.shape}")
    print("READING DATA DONE")

    # ========================================================================
    # GRAIN EVOLUTION ANALYSIS
    # ========================================================================

    initial_grain_num = 5000
    step_num = npy_file_aniso_min.shape[0]

    grain_num_aniso_ave = np.zeros(step_num)
    grain_num_aniso_min = np.zeros(step_num)
    grain_num_aniso_max = np.zeros(step_num)
    grain_num_iso = np.zeros(step_num)

    for i in range(step_num):
        grain_num_aniso_ave[i] = len(set(npy_file_aniso_ave[i,:].flatten()))
        grain_num_aniso_min[i] = len(set(npy_file_aniso_min[i,:].flatten()))
        grain_num_aniso_max[i] = len(set(npy_file_aniso_max[i,:].flatten()))
        grain_num_iso[i] = len(set(npy_file_iso[i,:].flatten()))

    # ========================================================================
    # TARGET GRAIN SELECTION
    # ========================================================================

    expected_grain_num = 200

    special_step_distribution_ave = int(np.argmin(abs(grain_num_aniso_ave - expected_grain_num)))
    special_step_distribution_min = int(np.argmin(abs(grain_num_aniso_min - expected_grain_num)))
    special_step_distribution_max = int(np.argmin(abs(grain_num_aniso_max - expected_grain_num)))
    special_step_distribution_iso = int(np.argmin(abs(grain_num_iso - expected_grain_num)))

    cache_dir = os.path.join(current_path, '3D_normal_distribution_data')

    # ========================================================================
    # XY PLANE ANALYSIS
    # ========================================================================

    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=20.0, fontsize=14)

    # Iso case
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_iso', special_step_distribution_iso, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso")

    # Compute XY bias
    slope_list_bias = post_processing.compute_bias(slope_list)

    # Ave case
    newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_ave', special_step_distribution_ave, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave")

    # Min case
    newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_min', special_step_distribution_min, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_min, "Min")

    # Max case
    newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_max', special_step_distribution_max, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_max, "Max")

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=3)
    plt.text(0.0, 0.0095, "x", fontsize=14)
    plt.text(np.pi/2, 0.0095, "y", fontsize=14)
    plt.savefig(current_path + f"/figures/normal_distribution_3d_xy_{expected_grain_num}grains.png",
                dpi=400, bbox_inches='tight')

    # ========================================================================
    # XZ PLANE ANALYSIS
    # ========================================================================

    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=20.0, fontsize=14)

    # Iso case
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_iso', special_step_distribution_iso, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 1)

    # Compute XZ bias
    slope_list_bias_1 = post_processing.compute_bias(slope_list)

    # Ave case
    newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_ave', special_step_distribution_ave, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 1)

    # Min case
    newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_min', special_step_distribution_min, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_min, "Min", 1)

    # Max case
    newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_max', special_step_distribution_max, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_max, "Max", 1)

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=3)
    plt.text(0.0, 0.0095, "x", fontsize=14)
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)
    plt.savefig(current_path + f"/figures/normal_distribution_3d_xz_{expected_grain_num}grains.png",
                dpi=400, bbox_inches='tight')

    # ========================================================================
    # YZ PLANE ANALYSIS
    # ========================================================================

    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=20.0, fontsize=14)

    # Iso case
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_iso', special_step_distribution_iso, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 2)

    # Compute YZ bias
    slope_list_bias_2 = post_processing.compute_bias(slope_list)

    # Ave case
    newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_ave', special_step_distribution_ave, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 2)

    # Min case
    newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_min', special_step_distribution_min, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_min, "Min", 2)

    # Max case
    newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_max', special_step_distribution_max, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_max, "Max", 2)

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=3)
    plt.text(0.0, 0.0095, "y", fontsize=14)
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)
    plt.savefig(current_path + f"/figures/normal_distribution_3d_yz_{expected_grain_num}grains.png",
                dpi=400, bbox_inches='tight')

    # ========================================================================
    # BIAS-CORRECTED ANALYSIS: XY Plane
    # ========================================================================

    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=20.0, fontsize=14)

    # Iso case
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_iso', special_step_distribution_iso, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 0, slope_list_bias)

    # Ave case
    newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_ave', special_step_distribution_ave, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 0, slope_list_bias)

    # Min case
    newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_min', special_step_distribution_min, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_min, "Min", 0, slope_list_bias)

    # Max case
    newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_max', special_step_distribution_max, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_max, "Max", 0, slope_list_bias)

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=3)
    plt.text(0.0, 0.0095, "x", fontsize=14)
    plt.text(np.pi/2, 0.0095, "y", fontsize=14)
    plt.savefig(current_path + f"/figures/normal_distribution_3d_xy_{expected_grain_num}grains_after_removing_bias.png",
                dpi=400, bbox_inches='tight')

    # ========================================================================
    # BIAS-CORRECTED ANALYSIS: XZ Plane
    # ========================================================================

    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=20.0, fontsize=14)

    # Iso case
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_iso', special_step_distribution_iso, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 1, slope_list_bias_1)

    # Ave case
    newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_ave', special_step_distribution_ave, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 1, slope_list_bias_1)

    # Min case
    newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_min', special_step_distribution_min, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_min, "Min", 1, slope_list_bias_1)

    # Max case
    newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_max', special_step_distribution_max, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_max, "Max", 1, slope_list_bias_1)

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=3)
    plt.text(0.0, 0.0095, "x", fontsize=14)
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)
    plt.savefig(current_path + f"/figures/normal_distribution_3d_xz_{expected_grain_num}grains_after_removing_bias.png",
                dpi=400, bbox_inches='tight')

    # ========================================================================
    # BIAS-CORRECTED ANALYSIS: YZ Plane
    # ========================================================================

    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=20.0, fontsize=14)

    # Iso case
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_iso', special_step_distribution_iso, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 2, slope_list_bias_2)

    # Ave case
    newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_ave', special_step_distribution_ave, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 2, slope_list_bias_2)

    # Min case
    newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_min', special_step_distribution_min, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_min, "Min", 2, slope_list_bias_2)

    # Max case
    newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3D_max', special_step_distribution_max, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_max, "Max", 2, slope_list_bias_2)

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=3)
    plt.text(0.0, 0.0095, "y", fontsize=14)
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)
    plt.savefig(current_path + f"/figures/normal_distribution_3d_yz_{expected_grain_num}grains_after_removing_bias.png",
                dpi=400, bbox_inches='tight')
