#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grain Boundary Normal Vector Distribution Analysis: Temporal Evolution and Energy Method Comparison

Analyzes grain boundary normal vector distributions and their temporal evolution
under different energy calculation methods.

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
    # =============================================================================
    # Local Data Source Configuration
    # =============================================================================
    npy_file_folder = "/Users/lin/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_poly_for_GG/results/"

    TJ_energy_type_ave = "ave"
    TJ_energy_type_consMin = "consMin"
    TJ_energy_type_sum = "sum"
    TJ_energy_type_min = "min"
    TJ_energy_type_max = "max"
    TJ_energy_type_consMax = "consMax"

    npy_file_name_aniso_ave = f"p_ori_ave_{TJ_energy_type_ave}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMin = f"p_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_sum = f"p_ori_ave_{TJ_energy_type_sum}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_min = f"p_ori_ave_{TJ_energy_type_min}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_max = f"p_ori_ave_{TJ_energy_type_max}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMax = f"p_ori_ave_{TJ_energy_type_consMax}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_iso = "p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"

    grain_size_data_name_ave = f"grain_size_p_ori_ave_{TJ_energy_type_ave}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_consMin = f"grain_size_p_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_sum = f"grain_size_p_ori_ave_{TJ_energy_type_sum}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_min = f"grain_size_p_ori_ave_{TJ_energy_type_min}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_max = f"grain_size_p_ori_ave_{TJ_energy_type_max}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_consMax = f"grain_size_p_ori_ave_{TJ_energy_type_consMax}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_iso = "grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"

    # Load data
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)
    npy_file_aniso_consMin = np.load(npy_file_folder + npy_file_name_aniso_consMin)
    npy_file_aniso_sum = np.load(npy_file_folder + npy_file_name_aniso_sum)
    npy_file_aniso_min = np.load(npy_file_folder + npy_file_name_aniso_min)
    npy_file_aniso_max = np.load(npy_file_folder + npy_file_name_aniso_max)
    npy_file_aniso_consMax = np.load(npy_file_folder + npy_file_name_aniso_consMax)
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)

    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    print(f"The consMin data size is: {npy_file_aniso_consMin.shape}")
    print(f"The sum data size is: {npy_file_aniso_sum.shape}")
    print(f"The min data size is: {npy_file_aniso_min.shape}")
    print(f"The max data size is: {npy_file_aniso_max.shape}")
    print(f"The consMax data size is: {npy_file_aniso_consMax.shape}")
    print(f"The iso data size is: {npy_file_iso.shape}")
    print("READING DATA DONE")

    initial_grain_num = 20000
    step_num = npy_file_aniso_ave.shape[0]

    grain_num_ave = np.zeros(step_num)
    grain_area_ave = np.zeros((step_num,initial_grain_num))
    grain_size_ave = np.zeros((step_num,initial_grain_num))
    grain_ave_size_ave = np.zeros(step_num)

    grain_num_consMin = np.zeros(step_num)
    grain_area_consMin = np.zeros((step_num,initial_grain_num))
    grain_size_consMin = np.zeros((step_num,initial_grain_num))
    grain_ave_size_consMin = np.zeros(step_num)

    grain_num_sum = np.zeros(step_num)
    grain_area_sum = np.zeros((step_num,initial_grain_num))
    grain_size_sum = np.zeros((step_num,initial_grain_num))
    grain_ave_size_sum = np.zeros(step_num)

    grain_num_min = np.zeros(step_num)
    grain_area_min = np.zeros((step_num,initial_grain_num))
    grain_size_min = np.zeros((step_num,initial_grain_num))
    grain_ave_size_min = np.zeros(step_num)

    grain_num_max = np.zeros(step_num)
    grain_area_max = np.zeros((step_num,initial_grain_num))
    grain_size_max = np.zeros((step_num,initial_grain_num))
    grain_ave_size_max = np.zeros(step_num)

    grain_num_consMax = np.zeros(step_num)
    grain_area_consMax = np.zeros((step_num,initial_grain_num))
    grain_size_consMax = np.zeros((step_num,initial_grain_num))
    grain_ave_size_consMax = np.zeros(step_num)

    grain_num_iso = np.zeros(step_num)
    grain_area_iso = np.zeros((step_num,initial_grain_num))
    grain_size_iso = np.zeros((step_num,initial_grain_num))
    grain_ave_size_iso = np.zeros(step_num)

    bin_width = 0.16
    x_limit = [-0.5, 3.5]
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)
    grain_size_distribution_ave = np.zeros(bin_num)
    special_step_distribution_ave = 11
    grain_size_distribution_consMin = np.zeros(bin_num)
    special_step_distribution_consMin = 11
    grain_size_distribution_sum = np.zeros(bin_num)
    special_step_distribution_sum = 11
    grain_size_distribution_iso = np.zeros(bin_num)
    grain_size_distribution_min = np.zeros(bin_num)
    special_step_distribution_min = 11
    grain_size_distribution_max = np.zeros(bin_num)
    special_step_distribution_max = 11
    grain_size_distribution_consMax = np.zeros(bin_num)
    special_step_distribution_consMax = 11
    special_step_distribution_iso = 10

    # Start polar figure
    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=20.0, fontsize=14)

    cache_dir = os.path.join(current_path, 'normal_distribution_data')

    for i in tqdm(range(9,12)):

        # Aniso - min
        if i == special_step_distribution_min:
            newplace = np.rot90(npy_file_aniso_min[i,:,:,:], 1, (0,1))
            P, sites = post_processing.load_or_compute_normal_vectors(
                cache_dir, 'min', i, post_processing.get_normal_vector, newplace)
            slope_list = post_processing.get_normal_vector_slope(P, sites, i, "Min case")

        # Aniso - max
        if i == special_step_distribution_max:
            newplace = np.rot90(npy_file_aniso_max[i,:,:,:], 1, (0,1))
            P, sites = post_processing.load_or_compute_normal_vectors(
                cache_dir, 'max', i, post_processing.get_normal_vector, newplace)
            slope_list = post_processing.get_normal_vector_slope(P, sites, i, "Ave case")

        # Aniso - ave
        if i == special_step_distribution_ave:
            newplace = np.rot90(npy_file_aniso_ave[i,:,:,:], 1, (0,1))
            P, sites = post_processing.load_or_compute_normal_vectors(
                cache_dir, 'ave', i, post_processing.get_normal_vector, newplace)
            slope_list = post_processing.get_normal_vector_slope(P, sites, i, "Ave case")

        # Aniso - sum
        if i == special_step_distribution_sum:
            newplace = np.rot90(npy_file_aniso_sum[i,:,:,:], 1, (0,1))
            P, sites = post_processing.load_or_compute_normal_vectors(
                cache_dir, 'sum', i, post_processing.get_normal_vector, newplace)
            slope_list = post_processing.get_normal_vector_slope(P, sites, i, "Sum case")

        # Aniso - consMin
        if i == special_step_distribution_consMin:
            newplace = np.rot90(npy_file_aniso_consMin[i,:,:,:], 1, (0,1))
            P, sites = post_processing.load_or_compute_normal_vectors(
                cache_dir, 'consMin', i, post_processing.get_normal_vector, newplace)
            slope_list = post_processing.get_normal_vector_slope(P, sites, i, "ConsMin case")

        # Aniso - consMax
        if i == special_step_distribution_consMax:
            newplace = np.rot90(npy_file_aniso_consMax[i,:,:,:], 1, (0,1))
            P, sites = post_processing.load_or_compute_normal_vectors(
                cache_dir, 'consMax', i, post_processing.get_normal_vector, newplace)
            slope_list = post_processing.get_normal_vector_slope(P, sites, i, "ConsMax case")

        # Aniso - iso
        if i == special_step_distribution_iso:
            newplace = np.rot90(npy_file_iso[i,:,:,:], 1, (0,1))
            P, sites = post_processing.load_or_compute_normal_vectors(
                cache_dir, 'iso', i, post_processing.get_normal_vector, newplace)
            slope_list = post_processing.get_normal_vector_slope(P, sites, i, "Iso case")

    plt.legend(loc=(0.22,-0.1),fontsize=14)
    plt.savefig(current_path + "/figures/normal_distribution.png", dpi=400,bbox_inches='tight')
