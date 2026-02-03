#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot normal distribution over time for 20k-grain polycrystalline systems
with isotropic grain boundaries (isoGBs) on HiPerGator.

Compares six triple junction energy approaches (Min, Max, Ave, Sum, CMin, CMax)
plus an isotropic reference, with bias correction from kT=0.66.

Created: Mon Jul 31 14:33:57 2023
Author: Lin
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
    # HiPerGator Blue storage paths
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/"

    # Triple junction energy formulation identifiers
    TJ_energy_type_ave = "ave"
    TJ_energy_type_consMin = "consMin"
    TJ_energy_type_sum = "sum"
    TJ_energy_type_min = "min"
    TJ_energy_type_max = "max"
    TJ_energy_type_consMax = "consMax"

    # File names
    npy_file_name_iso = "p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_ave = f"p_ori_ave_{TJ_energy_type_ave}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_isoGBs.npy"
    npy_file_name_aniso_consMin = f"p_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_isoGBs.npy"
    npy_file_name_aniso_sum = f"p_ori_ave_{TJ_energy_type_sum}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_isoGBs.npy"
    npy_file_name_aniso_min = f"p_ori_ave_{TJ_energy_type_min}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_isoGBs.npy"
    npy_file_name_aniso_max = f"p_ori_ave_{TJ_energy_type_max}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_isoGBs.npy"
    npy_file_name_aniso_consMax = f"p_ori_ave_{TJ_energy_type_consMax}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_isoGBs.npy"

    # Load datasets
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)
    npy_file_aniso_consMin = np.load(npy_file_folder + npy_file_name_aniso_consMin)
    npy_file_aniso_sum = np.load(npy_file_folder + npy_file_name_aniso_sum)
    npy_file_aniso_min = np.load(npy_file_folder + npy_file_name_aniso_min)
    npy_file_aniso_max = np.load(npy_file_folder + npy_file_name_aniso_max)
    npy_file_aniso_consMax = np.load(npy_file_folder + npy_file_name_aniso_consMax)

    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    print(f"The consMin data size is: {npy_file_aniso_consMin.shape}")
    print(f"The sum data size is: {npy_file_aniso_sum.shape}")
    print(f"The min data size is: {npy_file_aniso_min.shape}")
    print(f"The max data size is: {npy_file_aniso_max.shape}")
    print(f"The consMax data size is: {npy_file_aniso_consMax.shape}")
    print(f"The iso data size is: {npy_file_iso.shape}")
    print("READING DATA DONE")

    # System parameters
    initial_grain_num = 20000
    step_num = npy_file_aniso_ave.shape[0]

    # Grain size distribution parameters
    bin_width = 0.16
    x_limit = [-0.5, 3.5]
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)

    grain_size_distribution_iso = np.zeros(bin_num)
    special_step_distribution_iso = 10
    grain_size_distribution_ave = np.zeros(bin_num)
    special_step_distribution_ave = 11
    grain_size_distribution_consMin = np.zeros(bin_num)
    special_step_distribution_consMin = 11
    grain_size_distribution_sum = np.zeros(bin_num)
    special_step_distribution_sum = 11
    grain_size_distribution_iso = np.zeros(bin_num)
    grain_size_distribution_min = np.zeros(bin_num)
    special_step_distribution_min = 30
    grain_size_distribution_max = np.zeros(bin_num)
    special_step_distribution_max = 15
    grain_size_distribution_consMax = np.zeros(bin_num)
    special_step_distribution_consMax = 11

    # Polar figure setup
    fig, ax = post_processing.setup_polar_figure(r_max=0.01, r_tick=0.004, theta_tick=45.0, fontsize=16)
    ax.set_yticklabels(['0', '4e-3', '8e-3'],fontsize=16)

    # Load bias correction data
    special_step_distribution_T066_bias = 10
    data_file_name_bias = f'/normal_distribution_data/normal_distribution_T066_bias_sites_step{special_step_distribution_T066_bias}.npy'
    slope_list_bias = np.load(current_path + data_file_name_bias)

    # Initialize anisotropy magnitude arrays
    aniso_mag = np.zeros(6)
    aniso_mag_stand = np.zeros(6)

    # MIN
    cache_dir = os.path.join(current_path, 'normal_distribution_data')
    newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, "min_P_20k_isoGBs", special_step_distribution_min,
        post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_min, "Min", slope_list_bias)
    aniso_mag[0], aniso_mag_stand[0] = post_processing.simple_magnitude(slope_list)

    # MAX
    newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, "max_P_20k_isoGBs", special_step_distribution_max,
        post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_max, "Max", slope_list_bias)
    aniso_mag[1], aniso_mag_stand[1] = post_processing.simple_magnitude(slope_list)

    # AVE
    newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, "ave_P_20k_isoGBs", special_step_distribution_ave,
        post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_ave, "Ave", slope_list_bias)
    aniso_mag[2], aniso_mag_stand[2] = post_processing.simple_magnitude(slope_list)

    # SUM
    newplace = np.rot90(npy_file_aniso_sum[special_step_distribution_sum,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, "sum_P_20k_isoGBs", special_step_distribution_sum,
        post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_sum, "Sum", slope_list_bias)
    aniso_mag[3], aniso_mag_stand[3] = post_processing.simple_magnitude(slope_list)

    # CMIN
    newplace = np.rot90(npy_file_aniso_consMin[special_step_distribution_consMin,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, "consMin_P_20k_isoGBs", special_step_distribution_consMin,
        post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_consMin, "CMin", slope_list_bias)
    aniso_mag[4], aniso_mag_stand[4] = post_processing.simple_magnitude(slope_list)

    # CMAX
    newplace = np.rot90(npy_file_aniso_consMax[special_step_distribution_consMax,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, "consMax_P_20k_isoGBs", special_step_distribution_consMax,
        post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_consMax, "CMax", slope_list_bias)
    aniso_mag[5], aniso_mag_stand[5] = post_processing.simple_magnitude(slope_list)

    # ISO (reference)
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, "iso", special_step_distribution_iso,
        post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_iso, "Iso", slope_list_bias)

    # Save polar plot
    plt.legend(loc=(-0.12,-0.35),fontsize=16,ncol=3)
    plt.savefig(current_path + "/figures/normal_distribution_poly_20k_isoGBs_after_removing_bias.png",
                dpi=400,bbox_inches='tight')

    # Magnitude comparison plot
    plt.close()
    fig = plt.figure(figsize=(5, 5))

    label_list = ["Min", "Max", "Ave", "Sum", "CMin", "CMax"]

    plt.plot(np.linspace(0,len(label_list)-1,len(label_list)), aniso_mag, '.-',
             markersize=8, label='around 2000 grains', linewidth=2)

    plt.xlabel("TJ energy approach", fontsize=16)
    plt.ylabel("Anisotropic Magnitude", fontsize=16)
    plt.xticks([0,1,2,3,4,5],label_list)
    plt.ylim([-0.05,1.0])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig(current_path + "/figures/anisotropic_poly_20k_isoGBs_magnitude_polar_ave.png",
                dpi=400,bbox_inches='tight')
