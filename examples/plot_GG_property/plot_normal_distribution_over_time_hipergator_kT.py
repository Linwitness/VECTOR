#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VECTOR Framework: Thermal Fluctuation (kT) Analysis for Grain Boundary Orientation

Comprehensive thermal fluctuation analysis for large-scale polycrystalline
microstructures using Monte Carlo temperature scaling (kT).

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
    # File Configuration: HiPerGator Multi-Core Thermal Data
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/"

    TJ_energy_type_T000 = "T000"
    TJ_energy_type_T025 = "T025"
    TJ_energy_type_T050 = "T050"
    TJ_energy_type_T066 = "T066"
    TJ_energy_type_T095 = "T095"

    npy_file_name_aniso_T000 = f"p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt000.npy"
    npy_file_name_aniso_T025 = f"p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt025.npy"
    npy_file_name_aniso_T050 = f"p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt050.npy"
    npy_file_name_aniso_T066 = f"p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_T095 = f"p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt095.npy"

    grain_size_data_name_T000 = f"grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt000.npy"
    grain_size_data_name_T025 = f"grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt025.npy"
    grain_size_data_name_T050 = f"grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt050.npy"
    grain_size_data_name_T066 = f"grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_T095 = f"grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt095.npy"

    # Data Loading
    npy_file_aniso_T000 = np.load(npy_file_folder + npy_file_name_aniso_T000)
    npy_file_aniso_T025 = np.load(npy_file_folder + npy_file_name_aniso_T025)
    npy_file_aniso_T050 = np.load(npy_file_folder + npy_file_name_aniso_T050)
    npy_file_aniso_T066 = np.load(npy_file_folder + npy_file_name_aniso_T066)
    npy_file_aniso_T095 = np.load(npy_file_folder + npy_file_name_aniso_T095)

    print(f"The T000 data size is: {npy_file_aniso_T000.shape}")
    print(f"The T025 data size is: {npy_file_aniso_T025.shape}")
    print(f"The T050 data size is: {npy_file_aniso_T050.shape}")
    print(f"The T066 data size is: {npy_file_aniso_T066.shape}")
    print(f"The T095 data size is: {npy_file_aniso_T095.shape}")
    print("READING DATA DONE")

    # Thermal Analysis Configuration
    initial_grain_num = 20000

    step_num = npy_file_aniso_T000.shape[0]
    grain_num_T000 = np.zeros(step_num)
    grain_area_T000 = np.zeros((step_num,initial_grain_num))
    grain_size_T000 = np.zeros((step_num,initial_grain_num))
    grain_ave_size_T000 = np.zeros(step_num)

    step_num = npy_file_aniso_T025.shape[0]
    grain_num_T025 = np.zeros(step_num)
    grain_area_T025 = np.zeros((step_num,initial_grain_num))
    grain_size_T025 = np.zeros((step_num,initial_grain_num))
    grain_ave_size_T025 = np.zeros(step_num)

    grain_num_T050 = np.zeros(step_num)
    grain_area_T050 = np.zeros((step_num,initial_grain_num))
    grain_size_T050 = np.zeros((step_num,initial_grain_num))
    grain_ave_size_T050 = np.zeros(step_num)

    grain_num_T066 = np.zeros(step_num)
    grain_area_T066 = np.zeros((step_num,initial_grain_num))
    grain_size_T066 = np.zeros((step_num,initial_grain_num))
    grain_ave_size_T066 = np.zeros(step_num)

    grain_num_T095 = np.zeros(step_num)
    grain_area_T095 = np.zeros((step_num,initial_grain_num))
    grain_size_T095 = np.zeros((step_num,initial_grain_num))
    grain_ave_size_T095 = np.zeros(step_num)

    # Grain Size Distribution Configuration
    bin_width = 0.16
    x_limit = [-0.5, 3.5]
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)

    grain_size_distribution_T000 = np.zeros(bin_num)
    special_step_distribution_T000 = 10
    grain_size_distribution_T025 = np.zeros(bin_num)
    special_step_distribution_T025 = 10
    grain_size_distribution_T050 = np.zeros(bin_num)
    special_step_distribution_T050 = 10
    grain_size_distribution_T066 = np.zeros(bin_num)
    special_step_distribution_T066 = 10
    grain_size_distribution_T095 = np.zeros(bin_num)
    special_step_distribution_T095 = 11

    # Polar visualization setup
    fig, ax = post_processing.setup_polar_figure(r_max=0.01, r_tick=0.004, theta_tick=20.0, fontsize=14)

    # Systematic Thermal Analysis
    for i in tqdm(range(9,12)):

        # kT = 0.00 Analysis
        if i == special_step_distribution_T000:
            cache_dir = os.path.join(current_path, 'normal_distribution_data')
            newplace = np.rot90(npy_file_aniso_T000[i,:,:,:], 1, (0,1))
            P, sites = post_processing.load_or_compute_normal_vectors(
                cache_dir, 'T000', i, post_processing.get_normal_vector, newplace)
            slope_list = post_processing.get_normal_vector_slope(P, sites, i, r"$kT=0.00$")

        # kT = 0.25 Analysis
        if i == special_step_distribution_T025:
            cache_dir = os.path.join(current_path, 'normal_distribution_data')
            newplace = np.rot90(npy_file_aniso_T025[i,:,:,:], 1, (0,1))
            P, sites = post_processing.load_or_compute_normal_vectors(
                cache_dir, 'T025', i, post_processing.get_normal_vector, newplace)
            slope_list = post_processing.get_normal_vector_slope(P, sites, i, r"$kT=0.25$")

        # kT = 0.50 Analysis
        if i == special_step_distribution_T050:
            data_file_name_P = f'/normal_distribution_data/normal_distribution_T050_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_T050_P_sites_step{i}.npy'
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                newplace = np.rot90(npy_file_aniso_T050[i,:,:,:], 1, (0,1))
                P, sites, sites_list = post_processing.get_normal_vector(newplace)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)
            slope_list = post_processing.get_normal_vector_slope(P, sites, i, r"$kT=0.50$")

        # kT = 0.66 Analysis
        if i == special_step_distribution_T066:
            cache_dir = os.path.join(current_path, 'normal_distribution_data')
            newplace = np.rot90(npy_file_aniso_T066[i,:,:,:], 1, (0,1))
            P, sites = post_processing.load_or_compute_normal_vectors(
                cache_dir, 'T066', i, post_processing.get_normal_vector, newplace)
            slope_list = post_processing.get_normal_vector_slope(P, sites, i, r"$kT=0.66$")

        # kT = 0.95 Analysis
        if i == special_step_distribution_T095:
            cache_dir = os.path.join(current_path, 'normal_distribution_data')
            newplace = np.rot90(npy_file_aniso_T095[i,:,:,:], 1, (0,1))
            P, sites = post_processing.load_or_compute_normal_vectors(
                cache_dir, 'T095', i, post_processing.get_normal_vector, newplace)
            slope_list = post_processing.get_normal_vector_slope(P, sites, i, r"$kT=0.95$")

        # Bias Calculation: Circular Reference for kT = 0.66
        if i == special_step_distribution_T066:
            xLim = [0, 360]
            binValue = 10.01
            binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
            xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)
            freqArray_circle = np.ones(binNum)
            freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)

            data_file_name_P = f'/normal_distribution_data/normal_distribution_T066_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_T066_sites_step{i}.npy'
            data_file_name_bias = f'/normal_distribution_data/normal_distribution_T066_bias_sites_step{i}.npy'

            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                newplace = np.rot90(npy_file_aniso_T066[i,:,:,:], 1, (0,1))
                P, sites, sites_list = post_processing.get_normal_vector(newplace)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)

            #slope_list = post_processing.get_normal_vector_slope(P, sites, i, "T066 case")
            #bias = freqArray_circle - slope_list
            #np.save(current_path + data_file_name_bias, bias)
            #print(bias)

    # Publication-Quality Output
    plt.legend(loc=(-0.25,-0.3),fontsize=14,ncol=3)
    plt.savefig(current_path + "/figures/normal_distribution_kT.png", dpi=400,bbox_inches='tight')
