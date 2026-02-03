#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Circular Grain Boundary Normal Distribution Analysis

Analyzes grain boundary normal vector distributions in 2D circular
microstructures with anisotropic grain boundary energy.

Created on Mon Jul 31 14:33:57 2023
@author: Lin
"""

import os
current_path = os.getcwd()
import sys
sys.path.append(current_path)
sys.path.append(current_path+'/../../')
sys.path.append(current_path+'/../calculate_tangent/')

import numpy as np
from numpy import seterr
seterr(all='raise')
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

import myInput
import post_processing
import PACKAGE_MP_Linear as linear2d

# ===============================================================================
# STATISTICAL ANALYSIS FUNCTIONS FOR CIRCULAR GRAIN BOUNDARIES
# (These functions are specific to circular analysis and not in post_processing)
# ===============================================================================

def fit_ellipse_for_circle(sites_list):
    """Elliptical fitting analysis for circular grain shape characterization."""
    grain_num = len(sites_list)
    if grain_num < 2: return 1

    a_square_list = np.ones(grain_num)
    b_square_list = np.ones(grain_num)

    for i in range(grain_num):
        array = np.array(sites_list[i])
        X = array[:,0]
        Y = array[:,1]
        K_mat = np.array([X**2, X*Y, Y**2, X, Y]).T
        Y_mat = -np.ones_like(X)
        X_mat = np.linalg.lstsq(K_mat, Y_mat, rcond=None)[0].squeeze()

        center_base = 4 * X_mat[0] * X_mat[2] - X_mat[1] * X_mat[1]
        center_x = (X_mat[1] * X_mat[4] - 2 * X_mat[2]* X_mat[3]) / center_base
        center_y = (X_mat[1] * X_mat[3] - 2 * X_mat[0]* X_mat[4]) / center_base
        axis_square_root = np.sqrt((X_mat[0] - X_mat[2])**2 + X_mat[1]**2)
        a_square = 2*(X_mat[0]*center_x*center_x + X_mat[2]*center_y*center_y + X_mat[1]*center_x*center_y - 1) / (X_mat[0] + X_mat[2] + axis_square_root)
        b_square = 2*(X_mat[0]*center_x*center_x + X_mat[2]*center_y*center_y + X_mat[1]*center_x*center_y - 1) / (X_mat[0] + X_mat[2] - axis_square_root)

        print(f"a: {np.sqrt(a_square)}, b: {np.sqrt(b_square)}")
        a_square_list[i] = a_square
        b_square_list[i] = b_square

    return np.average(np.sqrt(b_square_list) / np.sqrt(a_square_list))

def get_circle_center(micro_matrix, step):
    """Calculate geometric centers and statistical radii for circular grains."""
    num_grains = int(np.max(micro_matrix[0,:]))
    center_list = np.zeros((num_grains,2))
    sites_num_list = np.zeros(num_grains)
    ave_radius_list = np.zeros(num_grains)

    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i,j] = i
            coord_refer_j[i,j] = j

    table = micro_matrix[step,:,:,0]

    for i in range(num_grains):
        sites_num_list[i] = np.sum(table == i+1)

        if sites_num_list[i] == 0:
          center_list[i, 0] = 0
          center_list[i, 1] = 0
        else:
          center_list[i, 0] = np.sum(coord_refer_i[table == i+1]) / sites_num_list[i]
          center_list[i, 1] = np.sum(coord_refer_j[table == i+1]) / sites_num_list[i]

    ave_radius_list = np.sqrt(sites_num_list / np.pi)

    return center_list, ave_radius_list

def get_circle_statistical_radius(micro_matrix, sites_list, step):
    """Statistical analysis of radial deviations in circular grain morphology."""
    center_list, ave_radius_list = get_circle_center(micro_matrix, step)
    center = center_list[1]
    ave_radius = ave_radius_list[1]

    if len(sites_list) < 2:
        sites = []
    else:
        sites = sites_list[1]

    max_radius_offset = 0
    ave_radius_offset_list = np.zeros(len(sites))

    for index, sitei in enumerate(sites):
        [i,j] = sitei
        current_radius = np.sqrt((i - center[0])**2 + (j - center[1])**2)
        radius_offset = abs(current_radius - ave_radius)
        ave_radius_offset_list[index] = radius_offset
        if radius_offset > max_radius_offset: max_radius_offset = radius_offset

    if ave_radius == 0:
        max_radius_offset = 0
        ave_radius_offset = 0
    else:
        max_radius_offset = max_radius_offset / ave_radius
        ave_radius_offset = np.average(ave_radius_offset_list) / ave_radius
        magnitude_stan = np.sqrt(np.sum((ave_radius_offset_list/ave_radius - ave_radius_offset)**2)/len(sites))

    return ave_radius_offset, magnitude_stan

def get_circle_statistical_ar(micro_matrix, step):
    """Calculate aspect ratio statistics for circular grain morphology analysis."""
    num_grains = int(np.max(micro_matrix[step,:]))
    sites_num_list = np.zeros(num_grains)

    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i,j] = i
            coord_refer_j[i,j] = j

    aspect_ratio_i = np.zeros(2)
    aspect_ratio_j = np.zeros(2)
    aspect_ratio = 0
    table = micro_matrix[step,:,:,0]

    for i in [1]:
        sites_num_list = np.sum(table == i+1)
        aspect_ratio_i[0] = len(list(set(coord_refer_i[table == i+1])))
        aspect_ratio_j[1] = len(list(set(coord_refer_j[table == i+1])))
        if aspect_ratio_j[1] == 0:
            aspect_ratio = 1
        else:
            aspect_ratio = aspect_ratio_i[0] / aspect_ratio_j[1]

    return aspect_ratio

# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

if __name__ == '__main__':
    # Data source configuration
    npy_file_folder = "/Users/lin.yang/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_circle_multiCoreCompare/results/"

    circle_energy_000 = "0.0"
    circle_energy_020 = "0.2"
    circle_energy_040 = "0.4"
    circle_energy_060 = "0.6"
    circle_energy_080 = "0.8"
    circle_energy_095 = "0.95"

    npy_file_name_aniso_000 = f"c_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_000}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_020 = f"c_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_020}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_040 = f"c_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_040}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_060 = f"c_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_060}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_080 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_080}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_095 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_095}_m2_refer_1_0_0.npy"

    grain_size_data_name_000 = f"grain_size_c_ori_aveE_000_000_multiCore32_delta{circle_energy_000}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_020 = f"grain_size_c_ori_aveE_000_000_multiCore32_delta{circle_energy_020}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_040 = f"grain_size_c_ori_aveE_000_000_multiCore32_delta{circle_energy_040}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_060 = f"grain_size_c_ori_aveE_000_000_multiCore32_delta{circle_energy_060}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_080 = f"grain_size_c_ori_aveE_000_000_multiCore32_delta{circle_energy_080}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_095 = f"grain_size_c_ori_aveE_000_000_multiCore32_delta{circle_energy_095}_m2_J1_refer_1_0_0_seed56689_kt066.npy"

    # Load data
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

    # Analysis parameters
    initial_grain_num = 2
    step_num = npy_file_aniso_000.shape[0]

    bin_width = 0.16
    x_limit = [-0.5, 3.5]
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)

    special_step_distribution_000 = 30
    special_step_distribution_020 = 30
    special_step_distribution_040 = 30
    special_step_distribution_060 = 30
    special_step_distribution_080 = 30
    special_step_distribution_095 = 30

    # ==========================================================================
    # PART I: RAW POLAR DISTRIBUTION ANALYSIS (WITHOUT BIAS CORRECTION)
    # ==========================================================================

    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=45.0, fontsize=16)

    # sigma = 0.00
    cache_dir = os.path.join(current_path, 'normal_distribution_data')
    newplace = np.rot90(npy_file_aniso_000[special_step_distribution_000,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '000', special_step_distribution_000, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_000, r"$\sigma=0.00$")

    # Bias calculation
    slope_list_bias = post_processing.compute_bias(slope_list)

    # sigma = 0.20
    newplace = np.rot90(npy_file_aniso_020[special_step_distribution_020,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '020', special_step_distribution_020, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_020, r"$\sigma=0.20$")

    # sigma = 0.40
    newplace = np.rot90(npy_file_aniso_040[special_step_distribution_040,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '040', special_step_distribution_040, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_040, r"$\sigma=0.40$")

    # sigma = 0.60
    newplace = np.rot90(npy_file_aniso_060[special_step_distribution_060,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '060', special_step_distribution_060, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_060, r"$\sigma=0.60$")

    # sigma = 0.80
    newplace = np.rot90(npy_file_aniso_080[special_step_distribution_080,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '080', special_step_distribution_080, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_080, r"$\sigma=0.80$")

    # sigma = 0.95
    newplace = np.rot90(npy_file_aniso_095[special_step_distribution_095,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '095', special_step_distribution_095, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_095, r"$\sigma=0.95$")

    plt.legend(loc=(-0.24,-0.3),fontsize=16,ncol=3)
    plt.savefig(current_path + "/figures/normal_distribution_circle.png", dpi=400,bbox_inches='tight')

    # ==========================================================================
    # PART II: BIAS-CORRECTED POLAR DISTRIBUTION ANALYSIS
    # ==========================================================================

    fig, ax = post_processing.setup_polar_figure(r_max=0.008, r_tick=0.004, theta_tick=45.0, fontsize=16)

    aniso_mag = np.zeros(6)
    aniso_mag_stand = np.zeros(6)

    # sigma = 0.00
    newplace = np.rot90(npy_file_aniso_000[special_step_distribution_000,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '000', special_step_distribution_000, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_000, r"$\sigma=0.00$", slope_list_bias)
    aniso_mag[0], aniso_mag_stand[0] = post_processing.simple_magnitude(slope_list)

    # sigma = 0.20
    newplace = np.rot90(npy_file_aniso_020[special_step_distribution_020,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '020', special_step_distribution_020, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_020, r"$\sigma=0.20$", slope_list_bias)
    aniso_mag[1], aniso_mag_stand[1] = post_processing.simple_magnitude(slope_list)

    # sigma = 0.40
    newplace = np.rot90(npy_file_aniso_040[special_step_distribution_040,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '040', special_step_distribution_040, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_040, r"$\sigma=0.40$", slope_list_bias)
    aniso_mag[2], aniso_mag_stand[2] = post_processing.simple_magnitude(slope_list)

    # sigma = 0.60
    newplace = np.rot90(npy_file_aniso_060[special_step_distribution_060,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '060', special_step_distribution_060, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_060, r"$\sigma=0.60$", slope_list_bias)
    aniso_mag[3], aniso_mag_stand[3] = post_processing.simple_magnitude(slope_list)

    # sigma = 0.80
    newplace = np.rot90(npy_file_aniso_080[special_step_distribution_080,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '080', special_step_distribution_080, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_080, r"$\sigma=0.80$", slope_list_bias)
    aniso_mag[4], aniso_mag_stand[4] = post_processing.simple_magnitude(slope_list)

    # sigma = 0.95
    newplace = np.rot90(npy_file_aniso_095[special_step_distribution_095,:,:,:], 1, (0,1))
    P, sites = post_processing.load_or_compute_normal_vectors(
        cache_dir, '095', special_step_distribution_095, post_processing.get_normal_vector, newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_095, r"$\sigma=0.95$", slope_list_bias)
    aniso_mag[5], aniso_mag_stand[5] = post_processing.simple_magnitude(slope_list)

    plt.legend(loc=(-0.24,-0.3),fontsize=16,ncol=3)
    plt.savefig(current_path + "/figures/normal_distribution_circle_after_removing_bias.png", dpi=400,bbox_inches='tight')
    print("Polar figure done.")

    # ==========================================================================
    # PART III: MORPHOLOGICAL ANISOTROPY ANALYSIS (RADIUS-BASED METRICS)
    # ==========================================================================

    num_step_magni = 30

    aniso_mag2 = np.zeros(6)
    aniso_mag_stand2 = np.zeros(6)

    cores = 16
    loop_times = 5

    for i in [num_step_magni]:

        # sigma = 0.00
        newplace = npy_file_aniso_000[i,:,:,:]
        nx = newplace.shape[0]
        ny = newplace.shape[1]
        ng = np.max(newplace)
        R = np.zeros((nx,ny,2))
        P0 = newplace
        smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
        sites_list = smooth_class.get_all_gb_list()
        aniso_mag2[0], aniso_mag_stand2[0] = get_circle_statistical_radius(npy_file_aniso_000, sites_list, i)

        # sigma = 0.20
        newplace = npy_file_aniso_020[i,:,:,:]
        nx = newplace.shape[0]
        ny = newplace.shape[1]
        ng = np.max(newplace)
        R = np.zeros((nx,ny,2))
        P0 = newplace
        smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
        sites_list = smooth_class.get_all_gb_list()
        aniso_mag2[1], aniso_mag_stand2[1] = get_circle_statistical_radius(npy_file_aniso_000, sites_list, i)

        # sigma = 0.40
        newplace = npy_file_aniso_040[i,:,:,:]
        nx = newplace.shape[0]
        ny = newplace.shape[1]
        ng = np.max(newplace)
        R = np.zeros((nx,ny,2))
        P0 = newplace
        smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
        sites_list = smooth_class.get_all_gb_list()
        aniso_mag2[2], aniso_mag_stand2[2] = get_circle_statistical_radius(npy_file_aniso_000, sites_list, i)

        # sigma = 0.60
        newplace = npy_file_aniso_060[i,:,:,:]
        nx = newplace.shape[0]
        ny = newplace.shape[1]
        ng = np.max(newplace)
        R = np.zeros((nx,ny,2))
        P0 = newplace
        smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
        sites_list = smooth_class.get_all_gb_list()
        aniso_mag2[3], aniso_mag_stand2[3] = get_circle_statistical_radius(npy_file_aniso_000, sites_list, i)

        # sigma = 0.80
        newplace = npy_file_aniso_080[i,:,:,:]
        nx = newplace.shape[0]
        ny = newplace.shape[1]
        ng = np.max(newplace)
        R = np.zeros((nx,ny,2))
        P0 = newplace
        smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
        sites_list = smooth_class.get_all_gb_list()
        aniso_mag2[4], aniso_mag_stand2[4] = get_circle_statistical_radius(npy_file_aniso_000, sites_list, i)

        # sigma = 0.95
        newplace = npy_file_aniso_095[i,:,:,:]
        nx = newplace.shape[0]
        ny = newplace.shape[1]
        ng = np.max(newplace)
        R = np.zeros((nx,ny,2))
        P0 = newplace
        smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
        sites_list = smooth_class.get_all_gb_list()
        aniso_mag2[5], aniso_mag_stand2[5] = get_circle_statistical_radius(npy_file_aniso_000, sites_list, i)

    # ==========================================================================
    # PART IV: COMPARATIVE ANISOTROPY MAGNITUDE VISUALIZATION
    # ==========================================================================

    plt.close()
    fig = plt.figure(figsize=(5, 5))

    delta_value = np.array([0.0,0.2,0.4,0.6,0.8,0.95])

    plt.errorbar(delta_value, aniso_mag, yerr=aniso_mag_stand,
                linestyle='None', marker='None',color='black',linewidth=1, capsize=2)
    plt.plot(delta_value, aniso_mag, '.-', markersize=8, label='time step = 900', linewidth=2)

    plt.xlabel(r"$\sigma$", fontsize=16)
    plt.ylabel("Anisotropic Magnitude", fontsize=16)
    plt.legend(fontsize=16)
    plt.ylim([-0.05,1.1])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(current_path + "/figures/anisotropic_magnitude_circle_polar_ave.png", dpi=400,bbox_inches='tight')

    plt.close()
    fig = plt.figure(figsize=(5, 5))
    delta_value = np.array([0.0,0.2,0.4,0.6,0.8,0.95])
    plt.plot(delta_value, aniso_mag2, '.-', markersize=8, label='time step = 900', linewidth=2)

    plt.xlabel(r"$\sigma$", fontsize=16)
    plt.ylabel("Anisotropic Magnitude", fontsize=16)
    plt.ylim([-0.05,0.7])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(current_path + "/figures/anisotropic_magnitude_circle_radius.png", dpi=400,bbox_inches='tight')

    print("2D CIRCULAR GRAIN BOUNDARY ANALYSIS COMPLETED")
