#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Spherical Grain Boundary Normal Vector Distribution Analysis using HiPerGator

Analyzes the orientation distribution of grain boundary normal vectors
in 3D spherical microstructures with bias correction methodology.

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
    # HIPERGATOR DATA SOURCE CONFIGURATION
    # -----------------------------------------------------------------------

    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_sphere/results/"

    TJ_energy_type_ave = "ave"
    TJ_energy_type_iso = "ave_delta000"

    npy_file_name_aniso_ave = f"p_ori_ave_{TJ_energy_type_ave}E_150_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt1.95.npy"
    npy_file_name_iso = "p_ori_ave_aveE_150_multiCore64_delta0.0_m2_J1_refer_1_0_0_seed56689_kt1.95.npy"

    # Load data
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)

    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    print(f"The iso data size is: {npy_file_iso.shape}")
    print("READING DATA DONE")

    # -----------------------------------------------------------------------
    # SPHERICAL GRAIN EVOLUTION ANALYSIS SETUP
    # -----------------------------------------------------------------------

    initial_grain_num = 2
    step_num = npy_file_iso.shape[0]

    grain_num_aniso_ave = np.zeros(step_num)
    grain_num_iso = np.zeros(step_num)

    for i in range(step_num):
        grain_num_aniso_ave[i] = len(set(npy_file_aniso_ave[i,:].flatten()))
        grain_num_iso[i] = len(set(npy_file_iso[i,:].flatten()))

    special_step_distribution_ave = 10
    special_step_distribution_iso = 10

    cache_dir = os.path.join(current_path, 'normal_distribution_data')

    # ===============================================================================
    # BIAS CALCULATION FOR SPHERICAL GEOMETRY CORRECTION
    # ===============================================================================

    # Isotropic reference case processing for bias determination
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3Dsphere_iso', special_step_distribution_iso, post_processing.get_normal_vector_3d, newplace)

    # XY plane bias (angle_index=0)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso")
    slope_list_bias_0 = post_processing.compute_bias(slope_list)

    # XZ plane bias (angle_index=1)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 1)
    slope_list_bias_1 = post_processing.compute_bias(slope_list)

    # YZ plane bias (angle_index=2)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 2)
    slope_list_bias_2 = post_processing.compute_bias(slope_list)

    # ===============================================================================
    # PART I: UNCORRECTED POLAR VISUALIZATIONS
    # ===============================================================================

    # -----------------------------------------------------------------------
    # XY PLANE POLAR PLOT
    # -----------------------------------------------------------------------

    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=20.0, fontsize=14)

    # Anisotropic case
    newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3Dsphere_ave', special_step_distribution_ave, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave")

    # Isotropic reference
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3Dsphere_iso', special_step_distribution_iso, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso")

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "x", fontsize=14)
    plt.text(np.pi/2, 0.0095, "y", fontsize=14)
    plt.savefig(current_path + f"/figures/normal_distribution_3dsphere_xy_{special_step_distribution_iso}steps.png", dpi=400,bbox_inches='tight')

    # -----------------------------------------------------------------------
    # XZ PLANE POLAR PLOT
    # -----------------------------------------------------------------------

    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=20.0, fontsize=14)

    # Anisotropic case
    newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3Dsphere_ave', special_step_distribution_ave, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 1)

    # Isotropic reference
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3Dsphere_iso', special_step_distribution_iso, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 1)

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "x", fontsize=14)
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)
    plt.savefig(current_path + f"/figures/normal_distribution_3dsphere_xz_{special_step_distribution_iso}steps.png", dpi=400,bbox_inches='tight')

    # -----------------------------------------------------------------------
    # YZ PLANE POLAR PLOT
    # -----------------------------------------------------------------------

    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=20.0, fontsize=14)

    # Anisotropic case
    newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3Dsphere_ave', special_step_distribution_ave, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 2)

    # Isotropic reference
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3Dsphere_iso', special_step_distribution_iso, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 2)

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "y", fontsize=14)
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)
    plt.savefig(current_path + f"/figures/normal_distribution_3dsphere_yz_{special_step_distribution_iso}steps.png", dpi=400,bbox_inches='tight')

    # ===============================================================================
    # PART II: BIAS-CORRECTED POLAR VISUALIZATIONS
    # ===============================================================================

    # -----------------------------------------------------------------------
    # XY PLANE BIAS-CORRECTED
    # -----------------------------------------------------------------------

    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=20.0, fontsize=14)

    # Anisotropic case with bias correction
    newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3Dsphere_ave', special_step_distribution_ave, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 0, slope_list_bias_0)

    # Isotropic reference with bias correction
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3Dsphere_iso', special_step_distribution_iso, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 0, slope_list_bias_0)

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "x", fontsize=14)
    plt.text(np.pi/2, 0.0095, "y", fontsize=14)
    plt.savefig(current_path + f"/figures/normal_distribution_3dsphere_xy_{special_step_distribution_iso}steps_after_removing_bias.png", dpi=400,bbox_inches='tight')

    # -----------------------------------------------------------------------
    # XZ PLANE BIAS-CORRECTED
    # -----------------------------------------------------------------------

    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=20.0, fontsize=14)

    # Anisotropic case with bias correction
    newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3Dsphere_ave', special_step_distribution_ave, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 1, slope_list_bias_1)

    # Isotropic reference with bias correction
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3Dsphere_iso', special_step_distribution_iso, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 1, slope_list_bias_1)

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "x", fontsize=14)
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)
    plt.savefig(current_path + f"/figures/normal_distribution_3dsphere_xz_{special_step_distribution_iso}steps_after_removing_bias.png", dpi=400,bbox_inches='tight')

    # -----------------------------------------------------------------------
    # YZ PLANE BIAS-CORRECTED
    # -----------------------------------------------------------------------

    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=20.0, fontsize=14)

    # Anisotropic case with bias correction
    newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3Dsphere_ave', special_step_distribution_ave, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 2, slope_list_bias_2)

    # Isotropic reference with bias correction
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '3Dsphere_iso', special_step_distribution_iso, post_processing.get_normal_vector_3d, newplace)
    slope_list = post_processing.get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 2, slope_list_bias_2)

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "y", fontsize=14)
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)
    plt.savefig(current_path + f"/figures/normal_distribution_3dsphere_yz_{special_step_distribution_iso}steps_after_removing_bias.png", dpi=400,bbox_inches='tight')

    print("3D spherical grain boundary analysis completed")
