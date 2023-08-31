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

def get_normal_vector(grain_structure_figure_one, grain_num):
    nx = grain_structure_figure_one.shape[0]
    ny = grain_structure_figure_one.shape[1]
    ng = np.max(grain_structure_figure_one)
    cores = 8
    loop_times = 5
    P0 = grain_structure_figure_one
    R = np.zeros((nx,ny,2))
    smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)

    smooth_class.linear_main("inclination")
    P = smooth_class.get_P()
    # sites = smooth_class.get_gb_list(1)
    # print(len(sites))
    # for id in range(2,grain_num+1): sites += smooth_class.get_gb_list(id)
    # print(len(sites))
    sites = smooth_class.get_all_gb_list()
    sites_together = []
    for id in range(len(sites)): sites_together += sites[id]
    print("Total num of GB sites: " + str(len(sites_together)))

    return P, sites_together

def get_normal_vector_slope(P, sites, step, para_name):
    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)

    freqArray = np.zeros(binNum)
    degree = []
    for sitei in sites:
        [i,j] = sitei
        dx,dy = myInput.get_grad(P,i,j)
        degree.append(math.atan2(-dy, dx) + math.pi)
        # if dx == 0:
        #     degree.append(math.pi/2)
        # elif dy >= 0:
        #     degree.append(abs(math.atan(-dy/dx)))
        # elif dy < 0:
        #     degree.append(abs(math.atan(dy/dx)))
    for i in range(len(degree)):
        freqArray[int((degree[i]/math.pi*180-xLim[0])/binValue)] += 1
    freqArray = freqArray/sum(freqArray*binValue)

    # Plot
    # plt.close()
    # fig = plt.figure(figsize=(5, 5))
    # ax = plt.gca(projection='polar')

    # ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    # ax.set_thetamin(0.0)
    # ax.set_thetamax(360.0)

    # ax.set_rgrids(np.arange(0, 0.008, 0.004))
    # ax.set_rlabel_position(0.0)  # 标签显示在0°
    # ax.set_rlim(0.0, 0.008)  # 标签范围为[0, 5000)
    # ax.set_yticklabels(['0', '0.004'],fontsize=14)

    # ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    # ax.set_axisbelow('True')
    plt.plot(np.append(xCor,xCor[0])/180*math.pi, np.append(freqArray,freqArray[0]), linewidth=2, label=para_name)

    # fitting
    fit_coeff = np.polyfit(xCor, freqArray, 1)
    return fit_coeff[0]

if __name__ == '__main__':
    # File name
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/"
    TJ_energy_type_T025 = "T025"
    TJ_energy_type_T050 = "T050"
    TJ_energy_type_T066 = "T066"
    TJ_energy_type_T095 = "T095"



    npy_file_name_aniso_T025 = f"p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt025.npy"
    npy_file_name_aniso_T050 = f"p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt050.npy"
    npy_file_name_aniso_T066 = f"p_ori_ave_aveE_20000_multiCore64_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_T095 = f"p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt095.npy"

    grain_size_data_name_T025 = f"grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt025.npy"
    grain_size_data_name_T050 = f"grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt050.npy"
    grain_size_data_name_T066 = f"grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_T095 = f"grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt095.npy"

    # Initial data
    npy_file_aniso_T025 = np.load(npy_file_folder + npy_file_name_aniso_T025)
    npy_file_aniso_T050 = np.load(npy_file_folder + npy_file_name_aniso_T050)
    npy_file_aniso_T066 = np.load(npy_file_folder + npy_file_name_aniso_T066)
    npy_file_aniso_T095 = np.load(npy_file_folder + npy_file_name_aniso_T095)
    print(f"The T025 data size is: {npy_file_aniso_T025.shape}")
    print(f"The T050 data size is: {npy_file_aniso_T050.shape}")
    print(f"The T066 data size is: {npy_file_aniso_T066.shape}")
    print(f"The T095 data size is: {npy_file_aniso_T095.shape}")
    print("READING DATA DONE")

    # Initial container
    initial_grain_num = 20000
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

    bin_width = 0.16 # Grain size distribution
    x_limit = [-0.5, 3.5]
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)
    grain_size_distribution_T025 = np.zeros(bin_num)
    special_step_distribution_T025 = 11#4
    grain_size_distribution_T050 = np.zeros(bin_num)
    special_step_distribution_T050 = 11#4
    grain_size_distribution_T066 = np.zeros(bin_num)
    special_step_distribution_T066 = 11#4
    grain_size_distribution_T095 = np.zeros(bin_num)
    special_step_distribution_T095 = 11#4


    # Start polar figure
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)

    ax.set_rgrids(np.arange(0, 0.01, 0.004))
    ax.set_rlabel_position(0.0)  # 标签显示在0°
    ax.set_rlim(0.0, 0.01)  # 标签范围为[0, 5000)
    ax.set_yticklabels(['0', '0.004', '0.008'],fontsize=14)

    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    for i in tqdm(range(9,12)):

        # Aniso - T095
        if i == special_step_distribution_T095:
            data_file_name_P = f'/normal_distribution_data/normal_distribution_T095_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_T095_sites_step{i}.npy'
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                newplace = np.rot90(npy_file_aniso_T095[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)

            slope_list = get_normal_vector_slope(P, sites, i, "T095 case")

        # Aniso - T025
        if i == special_step_distribution_T025:
            data_file_name_P = f'/normal_distribution_data/normal_distribution_T025_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_T025_sites_step{i}.npy'
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                newplace = np.rot90(npy_file_aniso_T025[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)

            slope_list = get_normal_vector_slope(P, sites, i, "T025 case")

        # Aniso - T066
        if i == special_step_distribution_T066:
            data_file_name_P = f'/normal_distribution_data/normal_distribution_T066_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_T066_sites_step{i}.npy'
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                newplace = np.rot90(npy_file_aniso_T066[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)

            slope_list = get_normal_vector_slope(P, sites, i, "T066 case")

        # Aniso - T050
        if i == special_step_distribution_T050:
            data_file_name_P = f'/normal_distribution_data/normal_distribution_T050_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_T050_P_sites_step{i}.npy'
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                newplace = np.rot90(npy_file_aniso_T050[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)

            slope_list = get_normal_vector_slope(P, sites, i, "T050 case")


    plt.legend(loc=(-0.25,-0.3),fontsize=14,ncol=3)
    plt.savefig(current_path + "/figures/normal_distribution_kT.png", dpi=400,bbox_inches='tight')











