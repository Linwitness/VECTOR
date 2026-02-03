#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polycrystalline Normal Vector Distribution Analysis: Advanced Statistical Characterization

This script provides comprehensive statistical analysis of normal vector distributions in
polycrystalline systems with enhanced focus on magnitude analysis, ellipse fitting,
and statistical deviation characterization.

Created on Mon Jul 31 14:33:57 2023
@author: Lin
"""

# Core scientific computing libraries
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

# Add VECTOR framework paths
sys.path.append(current_path)
sys.path.append(current_path+'/../../')
import myInput
import post_processing
sys.path.append(current_path+'/../calculate_tangent/')

def fit_ellipse_for_poly(micro_matrix, sites_list, step):
    """
    Advanced ellipse fitting analysis for polycrystalline grain shape characterization.

    Note: This function is NOT in post_processing.py, so it is kept inline.
    """
    grains_num = len(sites_list)

    sites_num_list = np.zeros(grains_num)
    # Calculate the area
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            grain_id = int(micro_matrix[step,i,j,0]-1)
            sites_num_list[grain_id] += 1
    center_list,_ = post_processing.get_poly_center(micro_matrix, step)

    a_square_list = np.ones(grains_num)
    b_square_list = np.ones(grains_num)
    unphysic_result = 0
    grains_num_real = 0.001
    for i in range(grains_num):
        array = np.array(sites_list[i])
        grain_center = center_list[i]

        # Avoid the really small grains
        rest_site_num = 10
        if len(array) < rest_site_num or (center_list[i,0] < 0.1 and center_list[i,1] < 0.1):
            a_square_list[i] = 1
            b_square_list[i] = 1
            continue

        my_list = []
        prefered_angles = np.linspace(0,2*np.pi,rest_site_num+1)[:rest_site_num]
        max_angles = np.ones(rest_site_num)*2*np.pi
        predered_sites = np.zeros((rest_site_num,2))
        for n in range(len(array)):
            current_site_angle = math.atan2(array[n,0] - grain_center[0], array[n,1] - grain_center[1]) + np.pi
            my_list.append(current_site_angle)
            min_angle = np.min(abs(prefered_angles - current_site_angle))
            min_angle_index = np.argmin(abs(prefered_angles - current_site_angle))
            if min_angle < max_angles[min_angle_index]:
                max_angles[min_angle_index] = min_angle
                predered_sites[min_angle_index] = array[n]

        array = predered_sites
        grains_num_real += 1
        # Get the self-variable
        X = array[:,0]
        Y = array[:,1]

        # Calculation Kernel
        K_mat = np.array([X**2, X*Y, Y**2, X, Y]).T
        Y_mat = -np.ones_like(X)
        X_mat = np.linalg.lstsq(K_mat, Y_mat, rcond=None)[0].squeeze()

        # Calculate the long and short axis
        center_base = 4 * X_mat[0] * X_mat[2] - X_mat[1] * X_mat[1]
        center_x = (X_mat[1] * X_mat[4] - 2 * X_mat[2]* X_mat[3]) / center_base
        center_y = (X_mat[1] * X_mat[3] - 2 * X_mat[0]* X_mat[4]) / center_base
        axis_square_root = np.sqrt((X_mat[0] - X_mat[2])**2 + X_mat[1]**2)
        a_square = 2*(X_mat[0]*center_x*center_x + X_mat[2]*center_y*center_y + X_mat[1]*center_x*center_y - 1) / (X_mat[0] + X_mat[2] + axis_square_root)
        b_square = 2*(X_mat[0]*center_x*center_x + X_mat[2]*center_y*center_y + X_mat[1]*center_x*center_y - 1) / (X_mat[0] + XMat[2] - axis_square_root)

        #  Avoid the grains with strange shape
        if a_square < 0 or b_square < 0:
            a_square_list[i] = 1
            b_square_list[i] = 1
            unphysic_result += 1
            continue
        a_square_list[i] = a_square
        b_square_list[i] = b_square
    print(f"The unphysical result is {round(unphysic_result/grains_num_real*100,3)}%")

    return np.sum(b_square_list * sites_num_list) / np.sum(a_square_list * sites_num_list)

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
    grain_size_data_name_000 = f"grain_size_p_ori_aveE_512_multiCore32_delta{circle_energy_000}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_020 = f"grain_size_p_ori_aveE_512_multiCore32_delta{circle_energy_020}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_040 = f"grain_size_p_ori_aveE_512_multiCore32_delta{circle_energy_040}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_060 = f"grain_size_p_ori_aveE_512_multiCore32_delta{circle_energy_060}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_080 = f"grain_size_p_ori_aveE_512_multiCore32_delta{circle_energy_080}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_095 = f"grain_size_p_ori_aveE_512_multiCore32_delta{circle_energy_095}_m2_J1_refer_1_0_0_seed56689_kt066.npy"

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

    # Start polar figure
    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=45.0, fontsize=16)

    # Aniso - 000
    cache_dir = os.path.join(current_path, 'normal_distribution_data')
    newplace = np.rot90(npy_file_aniso_000[special_step_distribution_000,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, 'poly_000', special_step_distribution_000, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_000, r"$\sigma=0.00$")
    # For bias
    slope_list_bias = post_processing.compute_bias(slope_list)

    # Aniso - 020
    data_file_name_P = f'/normal_distribution_data/normal_distribution_poly_020_P_step{special_step_distribution_020}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_poly_020_P_sites_step{special_step_distribution_020}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_020[special_step_distribution_020,:,:,:], 1, (0,1))
        P, sites, sites_list = post_processing.get_normal_vector(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_020, r"$\sigma=0.20$")

    # Aniso - 040
    cache_dir = os.path.join(current_path, 'normal_distribution_data')
    newplace = np.rot90(npy_file_aniso_040[special_step_distribution_040,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, 'poly_040', special_step_distribution_040, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_040, r"$\sigma=0.40$")

    # Aniso - 060
    cache_dir = os.path.join(current_path, 'normal_distribution_data')
    newplace = np.rot90(npy_file_aniso_060[special_step_distribution_060,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, 'poly_060', special_step_distribution_060, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_060, r"$\sigma=0.60$")

    # Aniso - 080
    cache_dir = os.path.join(current_path, 'normal_distribution_data')
    newplace = np.rot90(npy_file_aniso_080[special_step_distribution_080,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, 'poly_080', special_step_distribution_080, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_080, r"$\sigma=0.80$")

    # Aniso - 095
    cache_dir = os.path.join(current_path, 'normal_distribution_data')
    newplace = np.rot90(npy_file_aniso_095[special_step_distribution_095,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, 'poly_095', special_step_distribution_095, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_095, r"$\sigma=0.95$")

    plt.legend(loc=(-0.24,-0.3),fontsize=16,ncol=3)
    plt.savefig(current_path + "/figures/normal_distribution_poly.png", dpi=400,bbox_inches='tight')

    # For figure after bias
    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=45.0, fontsize=16)

    aniso_mag = np.zeros(6)
    aniso_mag_stand = np.zeros(6)
    # Aniso - 000
    cache_dir = os.path.join(current_path, 'normal_distribution_data')
    newplace = np.rot90(npy_file_aniso_000[special_step_distribution_000,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, 'poly_000', special_step_distribution_000, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_000, r"$\sigma=0.00$", slope_list_bias)
    aniso_mag[0], aniso_mag_stand[0] = post_processing.simple_magnitude(slope_list)

    # Aniso - 020
    data_file_name_P = f'/normal_distribution_data/normal_distribution_poly_020_P_step{special_step_distribution_020}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_poly_020_P_sites_step{special_step_distribution_020}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_020[special_step_distribution_020,:,:,:], 1, (0,1))
        P, sites, sites_list = post_processing.get_normal_vector(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_020, r"$\sigma=0.20$", slope_list_bias)
    aniso_mag[1], aniso_mag_stand[1] = post_processing.simple_magnitude(slope_list)

    # Aniso - 040
    cache_dir = os.path.join(current_path, 'normal_distribution_data')
    newplace = np.rot90(npy_file_aniso_040[special_step_distribution_040,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, 'poly_040', special_step_distribution_040, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_040, r"$\sigma=0.40$", slope_list_bias)
    aniso_mag[2], aniso_mag_stand[2] = post_processing.simple_magnitude(slope_list)

    # Aniso - 060
    cache_dir = os.path.join(current_path, 'normal_distribution_data')
    newplace = np.rot90(npy_file_aniso_060[special_step_distribution_060,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, 'poly_060', special_step_distribution_060, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_060, r"$\sigma=0.60$", slope_list_bias)
    aniso_mag[3], aniso_mag_stand[3] = post_processing.simple_magnitude(slope_list)

    # Aniso - 080
    cache_dir = os.path.join(current_path, 'normal_distribution_data')
    newplace = np.rot90(npy_file_aniso_080[special_step_distribution_080,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, 'poly_080', special_step_distribution_080, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_080, r"$\sigma=0.80$", slope_list_bias)
    aniso_mag[4], aniso_mag_stand[4] = post_processing.simple_magnitude(slope_list)

    # Aniso - 095
    cache_dir = os.path.join(current_path, 'normal_distribution_data')
    newplace = np.rot90(npy_file_aniso_095[special_step_distribution_095,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, 'poly_095', special_step_distribution_095, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_095, r"$\sigma=0.95$", slope_list_bias)
    aniso_mag[5], aniso_mag_stand[5] = post_processing.simple_magnitude(slope_list)

    plt.legend(loc=(-0.24,-0.3),fontsize=16,ncol=3)
    plt.savefig(current_path + "/figures/normal_distribution_poly_after_removing_bias.png", dpi=400,bbox_inches='tight')
    print("Polar figure done.")

    plt.close()
    fig = plt.figure(figsize=(5, 5))
    delta_value = np.array([0.0,0.2,0.4,0.6,0.8,0.95])
    plt.plot(delta_value, aniso_mag, '.-', markersize=8, label='10 grains', linewidth=2)

    plt.xlabel(r"$\sigma$", fontsize=16)
    plt.ylabel("Anisotropic Magnitude", fontsize=16)
    plt.ylim([-0.05,0.7])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(current_path + "/figures/anisotropic_magnitude_poly_polar_ave.png", dpi=400,bbox_inches='tight')
