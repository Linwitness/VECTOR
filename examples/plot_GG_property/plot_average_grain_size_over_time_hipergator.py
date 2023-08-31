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
sys.path.append(current_path+'/../calculate_tangent/')


if __name__ == '__main__':
    # File name
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

    # Initial data
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

    # Initial container
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

    bin_width = 0.16 # Grain size distribution
    x_limit = [-0.5, 3.5]
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)
    grain_size_distribution_ave = np.zeros(bin_num)
    special_step_distribution_ave = 11#4
    grain_size_distribution_consMin = np.zeros(bin_num)
    special_step_distribution_consMin = 11#4
    grain_size_distribution_sum = np.zeros(bin_num)
    special_step_distribution_sum = 11#4
    grain_size_distribution_iso = np.zeros(bin_num)
    grain_size_distribution_min = np.zeros(bin_num)
    special_step_distribution_min = 11#4
    grain_size_distribution_max = np.zeros(bin_num)
    special_step_distribution_max = 11#4
    grain_size_distribution_consMax = np.zeros(bin_num)
    special_step_distribution_consMax = 11#4
    grain_size_distribution_iso = np.zeros(bin_num)
    special_step_distribution_iso = 10#4


    # Start grain size initialing
    if os.path.exists(npy_file_folder + grain_size_data_name_iso):
        grain_area_iso = np.load(npy_file_folder + grain_size_data_name_iso)
    else:
        for i in tqdm(range(step_num)):
            for j in range(npy_file_iso.shape[1]):
                for k in range(npy_file_iso.shape[2]):
                    grain_id = int(npy_file_iso[i,j,k,0])
                    grain_area_iso[i,grain_id-1] += 1
        np.save(npy_file_folder + grain_size_data_name_iso, grain_area_iso)
    if os.path.exists(npy_file_folder + grain_size_data_name_ave):
        grain_area_ave = np.load(npy_file_folder + grain_size_data_name_ave)
    else:
        for i in tqdm(range(step_num)):
            for j in range(npy_file_aniso_ave.shape[1]):
                for k in range(npy_file_aniso_ave.shape[2]):
                    grain_id = int(npy_file_aniso_ave[i,j,k,0])
                    grain_area_ave[i,grain_id-1] += 1
        np.save(npy_file_folder + grain_size_data_name_ave, grain_area_ave)
    if os.path.exists(npy_file_folder + grain_size_data_name_consMin):
        grain_area_consMin = np.load(npy_file_folder + grain_size_data_name_consMin)
    else:
        for i in tqdm(range(step_num)):
            for j in range(npy_file_aniso_consMin.shape[1]):
                for k in range(npy_file_aniso_consMin.shape[2]):
                    grain_id = int(npy_file_aniso_consMin[i,j,k,0])
                    grain_area_consMin[i,grain_id-1] += 1
        np.save(npy_file_folder + grain_size_data_name_consMin, grain_area_consMin)
    if os.path.exists(npy_file_folder + grain_size_data_name_min):
        grain_area_min = np.load(npy_file_folder + grain_size_data_name_min)
    else:
        for i in tqdm(range(step_num)):
            for j in range(npy_file_aniso_min.shape[1]):
                for k in range(npy_file_aniso_min.shape[2]):
                    grain_id = int(npy_file_aniso_min[i,j,k,0])
                    grain_area_min[i,grain_id-1] += 1
        np.save(npy_file_folder + grain_size_data_name_min, grain_area_min)
    if os.path.exists(npy_file_folder + grain_size_data_name_sum):
        grain_area_sum = np.load(npy_file_folder + grain_size_data_name_sum)
    else:
        for i in tqdm(range(step_num)):
            for j in range(npy_file_aniso_sum.shape[1]):
                for k in range(npy_file_aniso_sum.shape[2]):
                    grain_id = int(npy_file_aniso_sum[i,j,k,0])
                    grain_area_sum[i,grain_id-1] += 1
        np.save(npy_file_folder + grain_size_data_name_sum, grain_area_sum)
    if os.path.exists(npy_file_folder + grain_size_data_name_consMax):
        grain_area_consMax = np.load(npy_file_folder + grain_size_data_name_consMax)
    else:
        for i in tqdm(range(step_num)):
            for j in range(npy_file_aniso_consMax.shape[1]):
                for k in range(npy_file_aniso_consMax.shape[2]):
                    grain_id = int(npy_file_aniso_consMax[i,j,k,0])
                    grain_area_consMax[i,grain_id-1] += 1
        np.save(npy_file_folder + grain_size_data_name_consMax, grain_area_consMax)
    if os.path.exists(npy_file_folder + grain_size_data_name_max):
        grain_area_max = np.load(npy_file_folder + grain_size_data_name_max)
    else:
        for i in tqdm(range(step_num)):
            for j in range(npy_file_aniso_max.shape[1]):
                for k in range(npy_file_aniso_max.shape[2]):
                    grain_id = int(npy_file_aniso_max[i,j,k,0])
                    grain_area_max[i,grain_id-1] += 1
        np.save(npy_file_folder + grain_size_data_name_max, grain_area_max)
    print("GRAIN SIZE INITIALING DONE")

    # Start grain size analysis
    for i in tqdm(range(step_num)):
        grain_num_ave[i] = np.sum(grain_area_ave[i,:] != 0) # grain num
        grain_num_consMin[i] = np.sum(grain_area_consMin[i,:] != 0) # grain num
        grain_num_sum[i] = np.sum(grain_area_sum[i,:] != 0) # grain num
        grain_num_min[i] = np.sum(grain_area_min[i,:] != 0) # grain num
        grain_num_consMax[i] = np.sum(grain_area_consMax[i,:] != 0) # grain num
        grain_num_max[i] = np.sum(grain_area_max[i,:] != 0) # grain num
        grain_num_iso[i] = np.sum(grain_area_iso[i,:] != 0) # grain num

        grain_size_ave[i] = (grain_area_ave[i] / np.pi)**0.5
        grain_ave_size_ave[i] = np.sum(grain_size_ave[i]) / grain_num_ave[i] # average grain size
        grain_size_consMin[i] = (grain_area_consMin[i] / np.pi)**0.5
        grain_ave_size_consMin[i] = np.sum(grain_size_consMin[i]) / grain_num_consMin[i] # average grain size
        grain_size_sum[i] = (grain_area_sum[i] / np.pi)**0.5
        grain_ave_size_sum[i] = np.sum(grain_size_sum[i]) / grain_num_sum[i] # average grain size
        grain_size_min[i] = (grain_area_min[i] / np.pi)**0.5
        grain_ave_size_min[i] = np.sum(grain_size_min[i]) / grain_num_min[i] # average grain size
        grain_size_consMax[i] = (grain_area_consMax[i] / np.pi)**0.5
        grain_ave_size_consMax[i] = np.sum(grain_size_consMax[i]) / grain_num_consMax[i] # average grain size
        grain_size_max[i] = (grain_area_max[i] / np.pi)**0.5
        grain_ave_size_max[i] = np.sum(grain_size_max[i]) / grain_num_max[i] # average grain size
        grain_size_iso[i] = (grain_area_iso[i] / np.pi)**0.5
        grain_ave_size_iso[i] = np.sum(grain_size_iso[i]) / grain_num_iso[i] # average grain size

        if i == special_step_distribution_ave:
            # Aniso
            special_size_ave = grain_size_ave[i][grain_size_ave[i] != 0] # remove zero grain size
            special_size_ave = special_size_ave/grain_ave_size_ave[i] # normalize grain size
            for j in range(len(special_size_ave)):
                grain_size_distribution_ave[int((special_size_ave[j]-x_limit[0])/bin_width)] += 1 # Get frequency
        if i == special_step_distribution_consMin:
            # Aniso
            special_size_consMin = grain_size_consMin[i][grain_size_consMin[i] != 0] # remove zero grain size
            special_size_consMin = special_size_consMin/grain_ave_size_consMin[i] # normalize grain size
            for j in range(len(special_size_consMin)):
                grain_size_distribution_consMin[int((special_size_consMin[j]-x_limit[0])/bin_width)] += 1 # Get frequency
        if i == special_step_distribution_sum:
            # Aniso
            special_size_sum = grain_size_sum[i][grain_size_sum[i] != 0] # remove zero grain size
            special_size_sum = special_size_sum/grain_ave_size_sum[i] # normalize grain size
            for j in range(len(special_size_sum)):
                grain_size_distribution_sum[int((special_size_sum[j]-x_limit[0])/bin_width)] += 1 # Get frequency
        if i == special_step_distribution_min:
            # Aniso
            special_size_min = grain_size_min[i][grain_size_min[i] != 0] # remove zero grain size
            special_size_min = special_size_min/grain_ave_size_min[i] # normalize grain size
            for j in range(len(special_size_min)):
                grain_size_distribution_min[int((special_size_min[j]-x_limit[0])/bin_width)] += 1 # Get frequency
        if i == special_step_distribution_consMax:
            # Aniso
            special_size_consMax = grain_size_consMax[i][grain_size_consMax[i] != 0] # remove zero grain size
            special_size_consMax = special_size_consMax/grain_ave_size_consMax[i] # normalize grain size
            for j in range(len(special_size_consMax)):
                grain_size_distribution_consMax[int((special_size_consMax[j]-x_limit[0])/bin_width)] += 1 # Get frequency
        if i == special_step_distribution_max:
            # Aniso
            special_size_max = grain_size_max[i][grain_size_max[i] != 0] # remove zero grain size
            special_size_max = special_size_max/grain_ave_size_max[i] # normalize grain size
            for j in range(len(special_size_max)):
                grain_size_distribution_max[int((special_size_max[j]-x_limit[0])/bin_width)] += 1 # Get frequency
        if i == special_step_distribution_iso:
            # Iso
            special_size_iso = grain_size_iso[i][grain_size_iso[i] != 0] # remove zero grain size
            special_size_iso = special_size_iso/grain_ave_size_iso[i] # normalize grain size
            for j in range(len(special_size_iso)):
                grain_size_distribution_iso[int((special_size_iso[j]-x_limit[0])/bin_width)] += 1 # Get frequency
    grain_size_distribution_ave = grain_size_distribution_ave/np.sum(grain_size_distribution_ave*bin_width) # normalize frequency
    grain_size_distribution_consMin = grain_size_distribution_consMin/np.sum(grain_size_distribution_consMin*bin_width) # normalize frequency
    grain_size_distribution_sum = grain_size_distribution_sum/np.sum(grain_size_distribution_sum*bin_width) # normalize frequency
    grain_size_distribution_min = grain_size_distribution_min/np.sum(grain_size_distribution_min*bin_width) # normalize frequency
    grain_size_distribution_consMax = grain_size_distribution_consMax/np.sum(grain_size_distribution_consMax*bin_width) # normalize frequency
    grain_size_distribution_max = grain_size_distribution_max/np.sum(grain_size_distribution_max*bin_width) # normalize frequency
    grain_size_distribution_iso = grain_size_distribution_iso/np.sum(grain_size_distribution_iso*bin_width) # normalize frequency
    print("GRAIN SIZE ANALYSIS DONE")

    # Start plotting
    plt.clf()
    plt.plot(list(range(step_num)), grain_ave_size_iso, label="Iso case", linewidth=2)
    plt.plot(list(range(step_num)), grain_ave_size_ave, label="Ave case", linewidth=2)
    plt.plot(list(range(step_num)), grain_ave_size_consMin, label="ConsMin case", linewidth=2)
    plt.plot(list(range(step_num)), grain_ave_size_min, label="Min case", linewidth=2)
    plt.plot(list(range(step_num)), grain_ave_size_sum, label="Sum case", linewidth=2)
    plt.plot(list(range(step_num)), grain_ave_size_consMax, label="ConsMax case", linewidth=2)
    plt.plot(list(range(step_num)), grain_ave_size_max, label="Max case", linewidth=2)
    plt.xlabel("Time step", fontsize=14)
    plt.ylabel("Grain Size", fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(current_path + "/figures/ave_grain_size_over_time.png", dpi=400,bbox_inches='tight')

    plt.clf()
    plt.plot(size_coordination, grain_size_distribution_iso, label="Iso case", linewidth=2)
    plt.plot(size_coordination, grain_size_distribution_ave, label="Ave case", linewidth=2)
    plt.plot(size_coordination, grain_size_distribution_consMin, label="ConsMin case", linewidth=2)
    plt.plot(size_coordination, grain_size_distribution_min, label="Min case", linewidth=2)
    plt.plot(size_coordination, grain_size_distribution_sum, label="Sum case", linewidth=2)
    plt.plot(size_coordination, grain_size_distribution_consMax, label="ConsMax case", linewidth=2)
    plt.plot(size_coordination, grain_size_distribution_max, label="Max case", linewidth=2)
    plt.xlabel("R/$\langle$R$\rangle$", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.legend(fontsize=14)
    plt.title(f"Grain num is around 2000", fontsize=14)
    plt.savefig(current_path + "/figures/normalized_size_distribution.png", dpi=400,bbox_inches='tight')


    # # Start find the steady state during grain growth
    # for i in tqdm(range(200)):
    #     grain_num_ave[i] = np.sum(grain_area_ave[i,:] != 0) # grain num
    #     grain_num_consMin[i] = np.sum(grain_area_consMin[i,:] != 0) # grain num
    #     grain_num_sum[i] = np.sum(grain_area_sum[i,:] != 0) # grain num
    #     grain_num_iso[i] = np.sum(grain_area_iso[i,:] != 0) # grain num

    #     grain_size_ave[i] = (grain_area_ave[i] / np.pi)**0.5
    #     grain_ave_size_ave[i] = np.sum(grain_size_ave[i]) / grain_num_ave[i] # average grain size
    #     grain_size_consMin[i] = (grain_area_consMin[i] / np.pi)**0.5
    #     grain_ave_size_consMin[i] = np.sum(grain_size_consMin[i]) / grain_num_consMin[i] # average grain size
    #     grain_size_sum[i] = (grain_area_sum[i] / np.pi)**0.5
    #     grain_ave_size_sum[i] = np.sum(grain_size_sum[i]) / grain_num_sum[i] # average grain size
    #     grain_size_iso[i] = (grain_area_iso[i] / np.pi)**0.5
    #     grain_ave_size_iso[i] = np.sum(grain_size_iso[i]) / grain_num_iso[i] # average grain size

    #     # Aniso
    #     special_size_ave = grain_size_ave[i][grain_size_ave[i] != 0] # remove zero grain size
    #     special_size_ave = special_size_ave/grain_ave_size_ave[i] # normalize grain size
    #     for j in range(len(special_size_ave)):
    #         grain_size_distribution_ave[int((special_size_ave[j]-x_limit[0])/bin_width)] += 1 # Get frequency

    #     # Aniso
    #     special_size_consMin = grain_size_consMin[i][grain_size_consMin[i] != 0] # remove zero grain size
    #     special_size_consMin = special_size_consMin/grain_ave_size_consMin[i] # normalize grain size
    #     for j in range(len(special_size_consMin)):
    #         grain_size_distribution_consMin[int((special_size_consMin[j]-x_limit[0])/bin_width)] += 1 # Get frequency

    #     # Aniso
    #     special_size_sum = grain_size_sum[i][grain_size_sum[i] != 0] # remove zero grain size
    #     special_size_sum = special_size_sum/grain_ave_size_sum[i] # normalize grain size
    #     for j in range(len(special_size_sum)):
    #         grain_size_distribution_sum[int((special_size_sum[j]-x_limit[0])/bin_width)] += 1 # Get frequency

    #     # Iso
    #     special_size_iso = grain_size_iso[i][grain_size_iso[i] != 0] # remove zero grain size
    #     special_size_iso = special_size_iso/grain_ave_size_iso[i] # normalize grain size
    #     for j in range(len(special_size_iso)):
    #         grain_size_distribution_iso[int((special_size_iso[j]-x_limit[0])/bin_width)] += 1 # Get frequency
    #     grain_size_distribution_ave = grain_size_distribution_ave/np.sum(grain_size_distribution_ave*bin_width) # normalize frequency
    #     grain_size_distribution_consMin = grain_size_distribution_consMin/np.sum(grain_size_distribution_consMin*bin_width) # normalize frequency
    #     grain_size_distribution_sum = grain_size_distribution_sum/np.sum(grain_size_distribution_sum*bin_width) # normalize frequency
    #     grain_size_distribution_iso = grain_size_distribution_iso/np.sum(grain_size_distribution_iso*bin_width) # normalize frequency

    #     # Start plotting
    #     plt.clf()
    #     plt.plot(size_coordination, grain_size_distribution_ave, label="Ave case", linewidth=2)
    #     plt.xlabel(r"R/$\langle$R$\rangle$", fontsize=14)
    #     plt.ylabel("Frequency", fontsize=14)
    #     plt.legend(fontsize=14)
    #     plt.ylim([0,1.2])
    #     plt.title(f"Grain num is {grain_num_ave[i]}", fontsize=14)
    #     plt.savefig(current_path + f"/NGSD_ave/normalized_size_distribution_{i}.png", dpi=400,bbox_inches='tight')
    #     plt.clf()
    #     plt.plot(size_coordination, grain_size_distribution_consMin, label="ConsMin case", linewidth=2)
    #     plt.xlabel(r"R/$\langle$R$\rangle$", fontsize=14)
    #     plt.ylabel("Frequency", fontsize=14)
    #     plt.legend(fontsize=14)
    #     plt.ylim([0,1.2])
    #     plt.title(f"Grain num is {grain_num_consMin[i]}", fontsize=14)
    #     plt.savefig(current_path + f"/NGSD_consMin/normalized_size_distribution_{i}.png", dpi=400,bbox_inches='tight')
    #     plt.clf()
    #     plt.plot(size_coordination, grain_size_distribution_sum, label="Sum case", linewidth=2)
    #     plt.xlabel(r"R/$\langle$R$\rangle$", fontsize=14)
    #     plt.ylabel("Frequency", fontsize=14)
    #     plt.legend(fontsize=14)
    #     plt.ylim([0,1.2])
    #     plt.title(f"Grain num is {grain_num_sum[i]}", fontsize=14)
    #     plt.savefig(current_path + f"/NGSD_sum/normalized_size_distribution_{i}.png", dpi=400,bbox_inches='tight')
    #     plt.clf()
    #     plt.plot(size_coordination, grain_size_distribution_iso, label="Iso case", linewidth=2)
    #     plt.xlabel(r"R/$\langle$R$\rangle$", fontsize=14)
    #     plt.ylabel("Frequency", fontsize=14)
    #     plt.legend(fontsize=14)
    #     plt.ylim([0,1.2])
    #     plt.title(f"Grain num is {grain_num_iso[i]}", fontsize=14)
    #     plt.savefig(current_path + f"/NGSD_iso/normalized_size_distribution_{i}.png", dpi=400,bbox_inches='tight')
    # print("NORMALIZED GRAIN SIZE DISTRIBUTION DONE")







