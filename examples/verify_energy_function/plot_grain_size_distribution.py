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
sys.path.append(current_path+'/../../')
import myInput
import post_processing as inclination_processing
import PACKAGE_MP_3DLinear as linear3d


if __name__ == '__main__':
    # File name
    # case_name = "264_5kMab"
    case_name = "264_5k"
    init_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_poly/IC/"
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_poly_fully/results/"
    npy_file_name_aniso = f"p_ori_fully5d_fz_aveE_f1.0_t1.0_264_5k_multiCore64_J1_refer_1_0_0_seed56689_kt1.95.npy"
    npy_file_name_anisoab = f"p_ori_fully5d_fzab_aveE_f1.0_t1.0_264_5k_multiCore64_J1_refer_1_0_0_seed56689_kt1.95.npy"
    npy_file_name_iso = f"p_iso_264_5k_multiCore64_J1_refer_1_0_0_seed56689_kt1.95.npy"
    input_npy_data = [npy_file_folder+npy_file_name_aniso, npy_file_folder+npy_file_name_anisoab, npy_file_folder+npy_file_name_iso]
    compare_label = ["Aniso", "Aniso_Abnormal", "Iso"]

    # Get time step with expected grain num
    expected_grain_num = 1000
    special_step_distribution, microstructure_list = inclination_processing.calculate_expected_step(input_npy_data, expected_grain_num)
    print(f"Steps for {compare_label} are {list(map(int, special_step_distribution))}")
    
    # Get grain size distribution
    grain_size_list_norm_list = []
    for i in range(len(input_npy_data)):
        size_data_name = f"/size_data/grain_size_data_{case_name}_{compare_label[i]}_step{special_step_distribution[i]}.npz"
        if os.path.exists(current_path + size_data_name):
            grain_size_npz_file = np.load(current_path + size_data_name)
            grain_size_list_norm = grain_size_npz_file["grain_size_list_norm"]
        else:
            current_microstructure = microstructure_list[i]
            grain_id_list = np.unique(current_microstructure)
            grain_area_list = np.zeros(len(grain_id_list))
            for k in range(len(grain_id_list)):
                grain_area_list[k] = np.sum(current_microstructure==grain_id_list[k])

            grain_size_list = (grain_area_list*3/4/np.pi)**(1/3)
            grain_size_ave = np.sum(grain_size_list)/len(grain_size_list)
            grain_size_list_norm = grain_size_list/grain_size_ave
            grain_size_list_norm_log = np.log10(grain_size_list_norm)
            np.savez(current_path + size_data_name,grain_size_list_norm=grain_size_list_norm)
        grain_size_list_norm_list.append(grain_size_list_norm)


    # plot Normalized Grain Size Distribution figure [-2.5,1.5]
    xLim = [-0.5, 4.0]
    binValue = 0.02
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)
    case_num = len(input_npy_data)
    freqArray = np.zeros((case_num, binNum))
    for i in range(case_num):
        for k in range(len(grain_size_list_norm_list[i])):
            freqArray[i, int((grain_size_list_norm_list[i][k]-xLim[0])/binValue)] += 1
        freqArray[i] = freqArray[i] / sum(freqArray[i]*binValue)
    plt.figure()
    for i in range(case_num):
        plt.plot(xCor,freqArray[i], linewidth=2, label=compare_label[i])
    plt.xlabel(r"R/$\langle$R$\rangle$", fontsize=18)
    plt.ylabel("frequence", fontsize=18)
    plt.title(f"grain num: {expected_grain_num}", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim([0,2.1])
    plt.xlim(xLim)
    plt.savefig(f'./size_figures/normalized_grain_size_distribution_{case_name}_compare.png',dpi=400,bbox_inches='tight')

