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
    special_step_distribution_ave = 11 #to get 2000 grains
    grain_size_distribution_consMin = np.zeros(bin_num)
    special_step_distribution_consMin = 11#to get 2000 grains
    grain_size_distribution_sum = np.zeros(bin_num)
    special_step_distribution_sum = 11#to get 2000 grains
    grain_size_distribution_iso = np.zeros(bin_num)
    grain_size_distribution_min = np.zeros(bin_num)
    special_step_distribution_min = 30#to get 2000 grains
    grain_size_distribution_max = np.zeros(bin_num)
    special_step_distribution_max = 15#to get 2000 grains
    grain_size_distribution_consMax = np.zeros(bin_num)
    special_step_distribution_consMax = 11#to get 2000 grains
    grain_size_distribution_iso = np.zeros(bin_num)
    special_step_distribution_iso = 10#to get 2000 grains


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

        # Aniso - min
        if i == special_step_distribution_min:
            data_file_name_P = f'/normal_distribution_data/normal_distribution_min_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_min_sites_step{i}.npy'
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                newplace = np.rot90(npy_file_aniso_min[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)

            slope_list = get_normal_vector_slope(P, sites, i, "Min case")

        # Aniso - max
        if i == special_step_distribution_max:
            data_file_name_P = f'/normal_distribution_data/normal_distribution_max_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_max_sites_step{i}.npy'
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                newplace = np.rot90(npy_file_aniso_max[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)

            slope_list = get_normal_vector_slope(P, sites, i, "Max case")

        # Aniso - ave
        if i == special_step_distribution_ave:
            data_file_name_P = f'/normal_distribution_data/normal_distribution_ave_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_ave_sites_step{i}.npy'
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                newplace = np.rot90(npy_file_aniso_ave[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)

            slope_list = get_normal_vector_slope(P, sites, i, "Ave case")

        # Aniso - sum
        if i == special_step_distribution_sum:
            data_file_name_P = f'/normal_distribution_data/normal_distribution_sum_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_sum_sites_step{i}.npy'
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                newplace = np.rot90(npy_file_aniso_sum[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)

            slope_list = get_normal_vector_slope(P, sites, i, "Sum case")

        # Aniso - consMin
        if i == special_step_distribution_consMin:
            data_file_name_P = f'/normal_distribution_data/normal_distribution_consMin_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_consMin_P_sites_step{i}.npy'
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                newplace = np.rot90(npy_file_aniso_consMin[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)

            slope_list = get_normal_vector_slope(P, sites, i, "ConsMin case")

        # Aniso - consMax
        if i == special_step_distribution_consMax:
            data_file_name_P = f'/normal_distribution_data/normal_distribution_consMax_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_consMax_sites_step{i}.npy'
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                newplace = np.rot90(npy_file_aniso_consMax[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)

            slope_list = get_normal_vector_slope(P, sites, i, "ConsMax case")

        # Aniso - iso
        if i == special_step_distribution_iso:
            data_file_name_P = f'/normal_distribution_data/normal_distribution_iso_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_iso_sites_step{i}.npy'
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                newplace = np.rot90(npy_file_iso[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)

            slope_list = get_normal_vector_slope(P, sites, i, "Iso case")

    plt.legend(loc=(-0.25,-0.3),fontsize=14,ncol=3)
    plt.savefig(current_path + "/figures/normal_distribution.png", dpi=400,bbox_inches='tight')











