#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
import PACKAGE_MP_Linear as linear2d
import post_processing
sys.path.append(current_path+'/../calculate_tangent/')

if __name__ == '__main__':
    # File name
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/" # .npy folder
    TJ_energy_type_ave = "ave"
    npy_file_name_aniso_ave = f"pT_ori_ave_{TJ_energy_type_ave}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy" # .npy name

    # Initial data
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)
    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    print("READING DATA DONE")

    # Initial container
    expected_grains = 200
    special_step_distribution_ave, _ = post_processing.calculate_expected_step([npy_file_name_aniso_ave], expected_grains) # get steps for 200 grains

    # Start polar figure
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    ax.set_thetagrids(np.arange(0.0, 360.0, 45.0),fontsize=16)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)

    ax.set_rgrids(np.arange(0, 0.01, 0.004))
    ax.set_rlabel_position(0.0)
    ax.set_rlim(0.0, 0.01)
    ax.set_yticklabels(['0', '4e-3', '8e-3'],fontsize=16)

    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    # Aniso - ave
    # Save temporary data for future plotting
    data_file_name = f'/normal_distribution_data/normal_distribution_ave_step{special_step_distribution_ave}.npz' # tmp data file name
    if os.path.exists(current_path + data_file_name):
        inclination_npz_data = np.load(current_path + data_file_name)
        P = inclination_npz_data["P"]
        sites = inclination_npz_data["sites"]
    else:
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites, sites_list = post_processing.get_normal_vector(newplace)
        np.savez(current_path + data_file_name, P=P, sites=sites)
    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_ave, "Ave") # plot inclinaitn distribution
    # save inclination distribution figure
    plt.legend(loc=(-0.12,-0.35),fontsize=16,ncol=3)
    plt.savefig(current_path + "/figures/normal_distribution_poly_20k_after_removing_bias.png", dpi=400,bbox_inches='tight')












