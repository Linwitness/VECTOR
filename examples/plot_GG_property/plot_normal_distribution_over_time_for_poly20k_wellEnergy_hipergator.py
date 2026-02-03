#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot normal distribution over time for 20k-grain polycrystalline systems
with well energy functions on HiPerGator.

Compares parametric anisotropy strengths (sigma = 0.7, 0.8, 0.9) plus
an isotropic reference, with bias correction and OpenCV ellipse fitting
for aspect ratio analysis.

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


def find_fittingEllipse3(array):
    """Fit an ellipse to grain boundary points using OpenCV.

    Args:
        array (numpy.ndarray): Array of [x, y] boundary site coordinates (n_points >= 5).

    Returns:
        tuple: OpenCV ellipse ((center_x, center_y), (major_axis, minor_axis), angle).
    """
    import cv2
    ellipse = cv2.fitEllipse(array)
    return ellipse


if __name__ == '__main__':
    # HiPerGator Blue storage path
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_wellEnergy/results/"

    # Parametric anisotropy strength values
    TJ_energy_type_070 = "0.7"
    TJ_energy_type_080 = "0.8"
    TJ_energy_type_090 = "0.9"

    energy_function = "Cos"

    # File names
    npy_file_name_iso = "p_aveE_20000_Cos_delta0.0_J1_refer_1_0_0_seed56689_kt0.66.npy"
    npy_file_name_aniso_070 = f"p_aveE_20000_{energy_function}_delta{TJ_energy_type_070}_J1_refer_1_0_0_seed56689_kt0.66.npy"
    npy_file_name_aniso_080 = f"p_aveE_20000_{energy_function}_delta{TJ_energy_type_080}_J1_refer_1_0_0_seed56689_kt0.66.npy"
    npy_file_name_aniso_090 = f"p_aveE_20000_{energy_function}_delta{TJ_energy_type_090}_J1_refer_1_0_0_seed56689_kt0.66.npy"

    # Load datasets
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)
    npy_file_aniso_070 = np.load(npy_file_folder + npy_file_name_aniso_070)
    npy_file_aniso_080 = np.load(npy_file_folder + npy_file_name_aniso_080)
    npy_file_aniso_090 = np.load(npy_file_folder + npy_file_name_aniso_090)

    print(f"The 0.7 data size is: {npy_file_aniso_070.shape}")
    print(f"The 0.8 data size is: {npy_file_aniso_080.shape}")
    print(f"The 0.9 data size is: {npy_file_aniso_090.shape}")
    print(f"The iso data size is: {npy_file_iso.shape}")
    print("READING DATA DONE")

    # Analysis parameters
    initial_grain_num = 20000
    step_num = npy_file_aniso_070.shape[0]

    # Grain count arrays for temporal evolution
    grain_num_aniso_070 = np.zeros(npy_file_aniso_070.shape[0])
    grain_num_aniso_080 = np.zeros(npy_file_aniso_080.shape[0])
    grain_num_aniso_090 = np.zeros(npy_file_aniso_090.shape[0])
    grain_num_iso = np.zeros(npy_file_iso.shape[0])

    # Calculate grain counts across temporal evolution
    for i in range(npy_file_aniso_070.shape[0]):
        grain_num_aniso_070[i] = len(np.unique(npy_file_aniso_070[i,:].flatten()))
        grain_num_aniso_080[i] = len(np.unique(npy_file_aniso_080[i,:].flatten()))
        grain_num_iso[i] = len(np.unique(npy_file_iso[i,:].flatten()))

    for i in range(npy_file_aniso_090.shape[0]):
        grain_num_aniso_090[i] = len(np.unique(npy_file_aniso_090[i,:].flatten()))

    # Find time steps for target grain count
    expected_grain_num = 1000
    special_step_distribution_070 = int(np.argmin(abs(grain_num_aniso_070 - expected_grain_num)))
    special_step_distribution_080 = int(np.argmin(abs(grain_num_aniso_080 - expected_grain_num)))
    special_step_distribution_090 = int(np.argmin(abs(grain_num_aniso_090 - expected_grain_num)))
    special_step_distribution_iso = int(np.argmin(abs(grain_num_iso - expected_grain_num)))
    print("Found time steps for 1000-grain analysis")

    # Isotropic reference: compute normal vectors
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites, sites_list = post_processing.get_normal_vector(newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_iso, "Iso")

    # Bias correction from isotropic reference
    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    freqArray_circle = np.ones(binNum)
    freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)
    slope_list_bias = freqArray_circle - slope_list

    # Polar figure setup
    fig, ax = post_processing.setup_polar_figure(r_max=0.015, r_tick=0.004, theta_tick=45.0, fontsize=16)
    ax.set_yticklabels(['0', '4e-3', '8e-3', '1.2e-3'],fontsize=16)

    # Analysis containers
    label_list = ["0.0", "0.7", "0.8", "0.9"]
    aniso_mag = np.zeros(len(label_list))
    aniso_mag_stand = np.zeros(len(label_list))
    aniso_rs = np.zeros(len(label_list))

    # Isotropic reference with bias correction
    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_iso, "Iso", slope_list_bias)

    # Ellipse fitting for isotropic reference
    as_list = []
    for n in range(len(sites_list)):
        if len(sites_list[n]) < 10: continue
        ellipse = find_fittingEllipse3(np.array(sites_list[n]))
        as_list.append(ellipse[1][0]/ellipse[1][1])
    aniso_rs[0] = np.average(as_list)
    print("iso done")

    # sigma = 0.7
    newplace = np.rot90(npy_file_aniso_070[special_step_distribution_070,:,:,:], 1, (0,1))
    P, sites, sites_list = post_processing.get_normal_vector(newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_070, r"$\sigma$=0.7", slope_list_bias)

    as_list = []
    for n in range(len(sites_list)):
        if len(sites_list[n]) < 10: continue
        ellipse = find_fittingEllipse3(np.array(sites_list[n]))
        as_list.append(ellipse[1][0]/ellipse[1][1])
    aniso_rs[1] = np.average(as_list)
    print("070 done")

    # sigma = 0.8
    newplace = np.rot90(npy_file_aniso_080[special_step_distribution_080,:,:,:], 1, (0,1))
    P, sites, sites_list = post_processing.get_normal_vector(newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_080, r"$\sigma$=0.8", slope_list_bias)

    as_list = []
    for n in range(len(sites_list)):
        if len(sites_list[n]) < 10: continue
        ellipse = find_fittingEllipse3(np.array(sites_list[n]))
        as_list.append(ellipse[1][0]/ellipse[1][1])
    aniso_rs[2] = np.average(as_list)
    print("080 done")

    # sigma = 0.9
    newplace = np.rot90(npy_file_aniso_090[special_step_distribution_090,:,:,:], 1, (0,1))
    P, sites, sites_list = post_processing.get_normal_vector(newplace)

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_090, r"$\sigma$=0.9", slope_list_bias)

    as_list = []
    for n in range(len(sites_list)):
        if len(sites_list[n]) < 10: continue
        ellipse = find_fittingEllipse3(np.array(sites_list[n]))
        as_list.append(ellipse[1][0]/ellipse[1][1])
    aniso_rs[3] = np.average(as_list)
    print("090 done")

    # Save polar plot
    plt.legend(loc=(-0.12,-0.35),fontsize=16,ncol=3)
    plt.savefig(current_path + f"/figures/normal_distribution_poly_20k_after_removing_bias_{energy_function}_{expected_grain_num}grains.png",
                dpi=400,bbox_inches='tight')

    # Aspect ratio plot
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    plt.plot(np.linspace(0,len(label_list)-1,len(label_list)), aniso_rs, '.-',
             markersize=8, linewidth=2)

    plt.xlabel(r"$\sigma$", fontsize=16)
    plt.ylabel("Aspect Ratio", fontsize=16)
    plt.xticks(np.linspace(0,len(label_list)-1,len(label_list)),label_list)
    plt.ylim([-0.05,1.0])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig(current_path + f"/figures/anisotropic_poly_p_20k_aspect_ratio_{energy_function}_{expected_grain_num}grains.png",
                dpi=400,bbox_inches='tight')
