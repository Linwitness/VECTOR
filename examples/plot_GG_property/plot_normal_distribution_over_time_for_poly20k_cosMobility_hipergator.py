#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grain Boundary Normal Vector Distribution Analysis with Cosine Mobility
for 20K Polycrystalline Systems on HiPerGator

Analyzes the influence of crystallographic orientation-dependent mobility
on grain growth kinetics and texture evolution in large statistical ensembles.

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

def find_fittingEllipse2(array):
    """Fit elliptical parameters to grain boundary points using least-squares method."""
    K_mat = []
    Y_mat = []
    X = array[:,0]
    Y = array[:,1]
    K_mat = np.hstack([X**2, X*Y, Y**2, X, Y])
    Y_mat = np.ones_like(X)
    X_mat = np.linalg.lstsq(K_mat, Y_mat)[0].squeeze()
    print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(
          X_mat[0], X_mat[1], X_mat[2], X_mat[3], X_mat[4]))
    print(X_mat)
    return X_mat

def find_fittingEllipse3(array):
    """Fit elliptical parameters using OpenCV computer vision algorithms."""
    import cv2
    ellipse = cv2.fitEllipse(array)
    return ellipse

if __name__ == '__main__':
    # HiPerGator Data Source Configuration
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_wellEnergy/results/"

    TJ_energy_type_070 = "0.7"
    TJ_energy_type_080 = "0.8"
    TJ_energy_type_090 = "0.9"

    energy_function = "CosMax1Mobility"

    npy_file_name_iso = "p_aveE_20000_Cos_delta0.0_J1_refer_1_0_0_seed56689_kt0.66.npy"
    npy_file_name_aniso_070 = f"p_aveE_20000_{energy_function}_delta{TJ_energy_type_070}_J1_refer_1_0_0_seed56689_kt0.66.npy"
    npy_file_name_aniso_080 = f"p_aveE_20000_{energy_function}_delta{TJ_energy_type_080}_J1_refer_1_0_0_seed56689_kt0.66.npy"
    npy_file_name_aniso_090 = f"p_aveE_20000_{energy_function}_delta{TJ_energy_type_090}_J1_refer_1_0_0_seed56689_kt0.66.npy"

    # Data Loading
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)
    npy_file_aniso_070 = np.load(npy_file_folder + npy_file_name_aniso_070)
    npy_file_aniso_080 = np.load(npy_file_folder + npy_file_name_aniso_080)
    npy_file_aniso_090 = np.load(npy_file_folder + npy_file_name_aniso_090)

    print(f"The 0.7 data size is: {npy_file_aniso_070.shape}")
    print(f"The 0.8 data size is: {npy_file_aniso_080.shape}")
    print(f"The 0.90 data size is: {npy_file_aniso_090.shape}")
    print(f"The iso data size is: {npy_file_iso.shape}")
    print("READING DATA DONE")

    # Grain Count Analysis
    initial_grain_num = 20000
    step_num = npy_file_iso.shape[0]

    grain_num_aniso_070 = np.zeros(step_num)
    grain_num_aniso_080 = np.zeros(step_num)
    grain_num_aniso_090 = np.zeros(step_num)
    grain_num_iso = np.zeros(step_num)

    for i in range(step_num):
        grain_num_aniso_070[i] = len(np.unique(npy_file_aniso_070[i,:].flatten()))
        grain_num_aniso_080[i] = len(np.unique(npy_file_aniso_080[i,:].flatten()))
        grain_num_aniso_090[i] = len(np.unique(npy_file_aniso_090[i,:].flatten()))
        grain_num_iso[i] = len(np.unique(npy_file_iso[i,:].flatten()))

    # Target Grain Count Identification
    expected_grain_num = 1000
    special_step_distribution_070 = int(np.argmin(abs(grain_num_aniso_070 - expected_grain_num)))
    special_step_distribution_080 = int(np.argmin(abs(grain_num_aniso_080 - expected_grain_num)))
    special_step_distribution_090 = int(np.argmin(abs(grain_num_aniso_090 - expected_grain_num)))
    special_step_distribution_iso = int(np.argmin(abs(grain_num_iso - expected_grain_num)))
    print("Found time steps")

    # Isotropic Reference Processing for Bias Calculation
    data_file_name_P = f'/well_normal_data/normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/well_normal_data/normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites, sites_list = post_processing.get_normal_vector(newplace)
    np.save(current_path + data_file_name_P, P)
    np.save(current_path + data_file_name_sites, sites)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_iso, "Iso")
    # For bias
    slope_list_bias = post_processing.compute_bias(slope_list)

    # Start polar figure
    fig, ax = post_processing.setup_polar_figure(r_max=0.01, r_tick=0.004, theta_tick=45.0, fontsize=16)

    label_list = ["0.0", "0.7", "0.8", "0.9"]
    aniso_mag = np.zeros(len(label_list))
    aniso_mag_stand = np.zeros(len(label_list))
    aniso_rs = np.zeros(len(label_list))

    # Iso
    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_iso, "Iso",slope_list_bias)
    as_list = []
    for n in range(len(sites_list)):
        if len(sites_list[n]) < 10: continue
        ellipse = find_fittingEllipse3(np.array(sites_list[n]))
        as_list.append(ellipse[1][0]/ellipse[1][1])
    aniso_rs[0] = np.average(as_list)
    print("iso done")

    # Aniso - 070
    data_file_name_P = f'/well_normal_data/normal_distribution_070_P_{energy_function}_step{special_step_distribution_070}.npy'
    data_file_name_sites = f'/well_normal_data/normal_distribution_070_sites_{energy_function}_step{special_step_distribution_070}.npy'
    newplace = np.rot90(npy_file_aniso_070[special_step_distribution_070,:,:,:], 1, (0,1))
    P, sites, sites_list = post_processing.get_normal_vector(newplace)
    np.save(current_path + data_file_name_P, P)
    np.save(current_path + data_file_name_sites, sites)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_070, r"$\sigma$=0.7",slope_list_bias)
    as_list = []
    for n in range(len(sites_list)):
        if len(sites_list[n]) < 10: continue
        ellipse = find_fittingEllipse3(np.array(sites_list[n]))
        as_list.append(ellipse[1][0]/ellipse[1][1])
    aniso_rs[1] = np.average(as_list)
    print("070 done")

    # Aniso - 0.8
    data_file_name_P = f'/well_normal_data/normal_distribution_080_P_{energy_function}_step{special_step_distribution_080}.npy'
    data_file_name_sites = f'/well_normal_data/normal_distribution_080_sites_{energy_function}_step{special_step_distribution_080}.npy'
    newplace = np.rot90(npy_file_aniso_080[special_step_distribution_080,:,:,:], 1, (0,1))
    P, sites, sites_list = post_processing.get_normal_vector(newplace)
    np.save(current_path + data_file_name_P, P)
    np.save(current_path + data_file_name_sites, sites)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_080, r"$\sigma$=0.8",slope_list_bias)
    as_list = []
    for n in range(len(sites_list)):
        if len(sites_list[n]) < 10: continue
        ellipse = find_fittingEllipse3(np.array(sites_list[n]))
        as_list.append(ellipse[1][0]/ellipse[1][1])
    aniso_rs[2] = np.average(as_list)
    print("080 done")

    # Aniso - 090
    data_file_name_P = f'/well_normal_data/normal_distribution_090_P_{energy_function}_step{special_step_distribution_090}.npy'
    data_file_name_sites = f'/well_normal_data/normal_distribution_090_sites_{energy_function}_step{special_step_distribution_090}.npy'
    newplace = np.rot90(npy_file_aniso_090[special_step_distribution_090,:,:,:], 1, (0,1))
    P, sites, sites_list = post_processing.get_normal_vector(newplace)
    np.save(current_path + data_file_name_P, P)
    np.save(current_path + data_file_name_sites, sites)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_090, r"$\sigma$=0.9",slope_list_bias)
    as_list = []
    for n in range(len(sites_list)):
        if len(sites_list[n]) < 10: continue
        ellipse = find_fittingEllipse3(np.array(sites_list[n]))
        as_list.append(ellipse[1][0]/ellipse[1][1])
    aniso_rs[3] = np.average(as_list)
    print("090 done")

    plt.legend(loc=(-0.12,-0.35),fontsize=16,ncol=3)
    plt.savefig(current_path + f"/figures/normal_distribution_poly_20k_after_removing_bias_{energy_function}_{expected_grain_num}grains.png", dpi=400,bbox_inches='tight')

    plt.close()
    fig = plt.figure(figsize=(5, 5))
    plt.plot(np.linspace(0,len(label_list)-1,len(label_list)), 1/aniso_rs, '.-', markersize=8, linewidth=2)
    plt.xlabel(r"$\sigma$", fontsize=16)
    plt.ylabel("Aspect Ratio", fontsize=16)
    plt.xticks(np.linspace(0,len(label_list)-1,len(label_list)),label_list)
    plt.ylim([-0.05,1.0])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(current_path + f"/figures/anisotropic_poly_20k_aspect_ratio_{energy_function}_{expected_grain_num}grains.png", dpi=400,bbox_inches='tight')
