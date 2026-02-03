#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VECTOR Framework: Grain Boundary Energy Function Comparison Analysis

Comprehensive energy function comparison for large-scale polycrystalline
microstructures with different energy averaging methods.

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
import myInput
import post_processing
sys.path.append(current_path+'/../calculate_tangent/')

if __name__ == '__main__':
    # File Configuration: HiPerGator 64-Core Energy Function Data
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/"

    TJ_energy_type_ave = "ave"
    TJ_energy_type_consMin = "consMin"
    TJ_energy_type_sum = "sum"
    TJ_energy_type_min = "min"
    TJ_energy_type_max = "max"
    TJ_energy_type_consMax = "consMax"

    npy_file_name_aniso_ave = f"p_ori_ave_{TJ_energy_type_ave}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMin = f"p_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_sum = f"p_ori_ave_{TJ_energy_type_sum}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
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

    # Data Loading
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

    # Analysis Configuration
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

    # Grain Size Distribution Configuration
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
    grain_size_distribution_min = np.zeros(bin_num)
    special_step_distribution_min = 30
    grain_size_distribution_max = np.zeros(bin_num)
    special_step_distribution_max = 15
    grain_size_distribution_consMax = np.zeros(bin_num)
    special_step_distribution_consMax = 11
    grain_size_distribution_iso = np.zeros(bin_num)
    special_step_distribution_iso = 10

    # Polar visualization setup
    fig, ax = post_processing.setup_polar_figure(r_max=0.01, r_tick=0.004, theta_tick=45.0, fontsize=16)

    # Bias correction data
    special_step_distribution_T066_bias = 10
    data_file_name_bias = f'/normal_distribution_data/normal_distribution_T066_bias_sites_step{special_step_distribution_T066_bias}.npy'
    slope_list_bias = np.load(current_path + data_file_name_bias)

    # Min energy function
    cache_dir = os.path.join(current_path, 'normal_distribution_data')
    newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, 'min', special_step_distribution_min, post_processing.get_normal_vector, newplace)
    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_min, "Min", slope_list_bias)

    # Max energy function
    data_file_name_P = f'/normal_distribution_data/normal_distribution_max_P_step{special_step_distribution_max}.npy'
    data_file_name_P = f'/normal_distribution_data/normal_distribution_max_P_step{special_step_distribution_max}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_max_sites_step{special_step_distribution_max}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
        P, sites, sites_list = post_processing.get_normal_vector(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)
    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_max, "Max", slope_list_bias)

    # Ave energy function
    cache_dir = os.path.join(current_path, 'normal_distribution_data')
    newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, 'ave', special_step_distribution_ave, post_processing.get_normal_vector, newplace)
    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_ave, "Ave", slope_list_bias)

    # Sum energy function
    cache_dir = os.path.join(current_path, 'normal_distribution_data')
    newplace = np.rot90(npy_file_aniso_sum[special_step_distribution_sum,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, 'sum', special_step_distribution_sum, post_processing.get_normal_vector, newplace)
    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_sum, "Sum", slope_list_bias)

    # ConsMin energy function
    data_file_name_P = f'/normal_distribution_data/normal_distribution_consMin_P_step{special_step_distribution_consMin}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_consMin_P_sites_step{special_step_distribution_consMin}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_consMin[special_step_distribution_consMin,:,:,:], 1, (0,1))
        P, sites, sites_list = post_processing.get_normal_vector(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)
    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_consMin, "CMin", slope_list_bias)

    # ConsMax energy function
    cache_dir = os.path.join(current_path, 'normal_distribution_data')
    newplace = np.rot90(npy_file_aniso_consMax[special_step_distribution_consMax,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, 'consMax', special_step_distribution_consMax, post_processing.get_normal_vector, newplace)
    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_consMax, "CMax", slope_list_bias)

    # Isotropic baseline
    data_file_name_P = f'/normal_distribution_data/normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites, sites_list = post_processing.get_normal_vector(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)
    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_iso, "Iso", slope_list_bias)

    plt.legend(loc=(-0.12,-0.35),fontsize=16,ncol=3)
    plt.savefig(current_path + "/figures/poly_aniso_iso_compare_vector_distribution_after_removing_bias.png",
               dpi=400, bbox_inches='tight')

    print("=" * 100)
    print("COMPREHENSIVE ENERGY FUNCTION COMPARISON ANALYSIS COMPLETED")
    print("=" * 100)
