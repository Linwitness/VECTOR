#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normal Vector Distribution Analysis Example for Grain Boundary Orientations

Example script for analyzing normal vector distributions in polycrystalline
grain boundaries using the post_processing module.

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
    # DATA LOADING AND CONFIGURATION
    # ========================================================================

    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/"

    TJ_energy_type_ave = "ave"

    npy_file_name_aniso_ave = f"pT_ori_ave_{TJ_energy_type_ave}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"

    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)
    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    print("READING DATA DONE")

    # ========================================================================
    # GRAIN COUNT ANALYSIS AND TEMPORAL STEP SELECTION
    # ========================================================================

    expected_grains = 200

    special_step_distribution_ave, _ = post_processing.calculate_expected_step([npy_file_name_aniso_ave], expected_grains)

    # ========================================================================
    # POLAR COORDINATE VISUALIZATION
    # ========================================================================

    fig, ax = post_processing.setup_polar_figure(r_max=0.01, r_tick=0.004, theta_tick=45.0, fontsize=16)

    # ========================================================================
    # NORMAL VECTOR EXTRACTION AND PROCESSING
    # ========================================================================

    data_file_name = f'/normal_distribution_data/normal_distribution_ave_step{special_step_distribution_ave}.npz'

    if os.path.exists(current_path + data_file_name):
        inclination_npz_data = np.load(current_path + data_file_name)
        P = inclination_npz_data["P"]
        sites = inclination_npz_data["sites"]
    else:
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites, sites_list = post_processing.get_normal_vector(newplace)
        np.savez(current_path + data_file_name, P=P, sites=sites)

    # ========================================================================
    # STATISTICAL ANALYSIS AND VISUALIZATION
    # ========================================================================

    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_ave, "Ave")

    plt.legend(loc=(-0.12,-0.35),fontsize=16,ncol=3)
    plt.savefig(current_path + "/figures/normal_distribution_poly_20k_after_removing_bias.png", dpi=400,bbox_inches='tight')
