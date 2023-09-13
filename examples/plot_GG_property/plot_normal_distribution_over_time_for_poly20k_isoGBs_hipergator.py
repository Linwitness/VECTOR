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

def get_poly_center(micro_matrix, step):
    # Get the center of all non-periodic grains in matrix
    num_grains = int(np.max(micro_matrix[step,:]))
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

        if (sites_num_list[i] == 0) or \
           (np.max(coord_refer_i[table == i+1]) - np.min(coord_refer_i[table == i+1]) == micro_matrix.shape[1]) or \
           (np.max(coord_refer_j[table == i+1]) - np.min(coord_refer_j[table == i+1]) == micro_matrix.shape[2]): # grains on bc are ignored
          center_list[i, 0] = 0
          center_list[i, 1] = 0
          sites_num_list[i] == 0
        else:
          center_list[i, 0] = np.sum(coord_refer_i[table == i+1]) / sites_num_list[i]
          center_list[i, 1] = np.sum(coord_refer_j[table == i+1]) / sites_num_list[i]
    ave_radius_list = np.sqrt(sites_num_list / np.pi)

    return center_list, ave_radius_list

def get_poly_statistical_radius(micro_matrix, sites_list, step):
    # Get the max offset of average radius and real radius
    center_list, ave_radius_list = get_poly_center(micro_matrix, step)
    num_grains = int(np.max(micro_matrix[step,:]))

    max_radius_offset_list = np.zeros(num_grains)
    for n in range(num_grains):
        center = center_list[n]
        ave_radius = ave_radius_list[n]
        sites = sites_list[n]

        if ave_radius != 0:
          for sitei in sites:
              [i,j] = sitei
              current_radius = np.sqrt((i - center[0])**2 + (j - center[1])**2)
              radius_offset = abs(current_radius - ave_radius)
              if radius_offset > max_radius_offset_list[n]: max_radius_offset_list[n] = radius_offset

          max_radius_offset_list[n] = max_radius_offset_list[n] / ave_radius

    max_radius_offset = np.average(max_radius_offset_list[max_radius_offset_list!=0])
    return max_radius_offset

def get_normal_vector(grain_structure_figure_one, grain_num):
    nx = grain_structure_figure_one.shape[0]
    ny = grain_structure_figure_one.shape[1]
    ng = np.max(grain_structure_figure_one)
    cores = 16
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

    return P, sites_together, sites

def get_normal_vector_slope(P, sites, step, para_name, bias=None):
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

    if bias is not None:
        freqArray = freqArray + bias
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


    npy_file_name_aniso_ave = f"p_ori_ave_{TJ_energy_type_ave}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_isoGBs.npy"
    npy_file_name_aniso_consMin = f"p_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_isoGBs.npy"
    npy_file_name_aniso_sum = f"p_ori_ave_{TJ_energy_type_sum}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_isoGBs.npy"
    npy_file_name_aniso_min = f"p_ori_ave_{TJ_energy_type_min}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_isoGBs.npy"
    npy_file_name_aniso_max = f"p_ori_ave_{TJ_energy_type_max}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_isoGBs.npy"
    npy_file_name_aniso_consMax = f"p_ori_ave_{TJ_energy_type_consMax}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_isoGBs.npy"

    # Initial data
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)
    npy_file_aniso_consMin = np.load(npy_file_folder + npy_file_name_aniso_consMin)
    npy_file_aniso_sum = np.load(npy_file_folder + npy_file_name_aniso_sum)
    npy_file_aniso_min = np.load(npy_file_folder + npy_file_name_aniso_min)
    npy_file_aniso_max = np.load(npy_file_folder + npy_file_name_aniso_max)
    npy_file_aniso_consMax = np.load(npy_file_folder + npy_file_name_aniso_consMax)
    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    print(f"The consMin data size is: {npy_file_aniso_consMin.shape}")
    print(f"The sum data size is: {npy_file_aniso_sum.shape}")
    print(f"The min data size is: {npy_file_aniso_min.shape}")
    print(f"The max data size is: {npy_file_aniso_max.shape}")
    print(f"The consMax data size is: {npy_file_aniso_consMax.shape}")
    print("READING DATA DONE")

    # Initial container
    initial_grain_num = 20000
    step_num = npy_file_aniso_ave.shape[0]

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

    # Aniso - min
    data_file_name_P = f'/normal_distribution_data/normal_distribution_min_P_20k_isoGBs_step{special_step_distribution_min}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_min_sites_20k_isoGBs_step{special_step_distribution_min}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_min, "Min case")

    # Aniso - max
    data_file_name_P = f'/normal_distribution_data/normal_distribution_max_P_20k_isoGBs_step{special_step_distribution_max}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_max_sites_20k_isoGBs_step{special_step_distribution_max}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_max, "Max case")

    # Aniso - ave
    data_file_name_P = f'/normal_distribution_data/normal_distribution_ave_P_20k_isoGBs_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_ave_sites_20k_isoGBs_step{special_step_distribution_ave}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_ave, "Ave case")

    # Aniso - sum
    data_file_name_P = f'/normal_distribution_data/normal_distribution_sum_P_20k_isoGBs_step{special_step_distribution_sum}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_sum_sites_20k_isoGBs_step{special_step_distribution_sum}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_sum[special_step_distribution_sum,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_sum, "Sum case")

    # Aniso - consMin
    data_file_name_P = f'/normal_distribution_data/normal_distribution_consMin_P_20k_isoGBs_step{special_step_distribution_consMin}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_consMin_P_sites_20k_isoGBs_step{special_step_distribution_consMin}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_consMin[special_step_distribution_consMin,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_consMin, "ConsMin case")

    # Aniso - consMax
    data_file_name_P = f'/normal_distribution_data/normal_distribution_consMax_P_20k_isoGBs_step{special_step_distribution_consMax}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_consMax_sites_20k_isoGBs_step{special_step_distribution_consMax}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_consMax[special_step_distribution_consMax,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_consMax, "ConsMax case")

    plt.legend(loc=(-0.25,-0.3),fontsize=14,ncol=3)
    plt.savefig(current_path + "/figures/normal_distribution_poly_20k_isoGBs.png", dpi=400,bbox_inches='tight')

    # PLot magnitude of anisotropy
    data_file_name_aniso_mag = f'/normal_distribution_data/aniso_magnitude_poly_20k_isoGBs_energy_type.npz'
    if os.path.exists(current_path + data_file_name_aniso_mag):
        data_file_aniso_mag = np.load(current_path + data_file_name_aniso_mag)
        aniso_mag_min=data_file_aniso_mag['aniso_mag_min']
        aniso_mag_max=data_file_aniso_mag['aniso_mag_max']
        aniso_mag_ave=data_file_aniso_mag['aniso_mag_ave']
        aniso_mag_sum=data_file_aniso_mag['aniso_mag_sum']
        aniso_mag_consMin=data_file_aniso_mag['aniso_mag_consMin']
        aniso_mag_consMax=data_file_aniso_mag['aniso_mag_consMax']
    else:
        aniso_mag_min = np.zeros(step_num)
        aniso_mag_max = np.zeros(step_num)
        aniso_mag_ave = np.zeros(step_num)
        aniso_mag_sum = np.zeros(step_num)
        aniso_mag_consMin = np.zeros(step_num)
        aniso_mag_consMax = np.zeros(step_num)
        cores = 16
        loop_times = 5
        for i in tqdm(range(step_num)):
            # newplace = np.rot90(npy_file_aniso_min[i,:,:,:], 1, (0,1))
            newplace = npy_file_aniso_min[i,:,:,:]
            nx = newplace.shape[0]
            ny = newplace.shape[1]
            ng = np.max(newplace)
            R = np.zeros((nx,ny,2))
            P0 = newplace
            smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
            sites_list = smooth_class.get_all_gb_list()
            # P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
            aniso_mag_min[i] = get_poly_statistical_radius(npy_file_aniso_min, sites_list, i)

            # newplace = np.rot90(npy_file_aniso_max[i,:,:,:], 1, (0,1))
            newplace = npy_file_aniso_max[i,:,:,:]
            nx = newplace.shape[0]
            ny = newplace.shape[1]
            ng = np.max(newplace)
            R = np.zeros((nx,ny,2))
            P0 = newplace
            smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
            sites_list = smooth_class.get_all_gb_list()
            # P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
            aniso_mag_max[i] = get_poly_statistical_radius(npy_file_aniso_max, sites_list, i)

            # newplace = np.rot90(npy_file_aniso_ave[i,:,:,:], 1, (0,1))
            newplace = npy_file_aniso_ave[i,:,:,:]
            nx = newplace.shape[0]
            ny = newplace.shape[1]
            ng = np.max(newplace)
            R = np.zeros((nx,ny,2))
            P0 = newplace
            smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
            sites_list = smooth_class.get_all_gb_list()
            # P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
            aniso_mag_ave[i] = get_poly_statistical_radius(npy_file_aniso_ave, sites_list, i)

            # newplace = np.rot90(npy_file_aniso_sum[i,:,:,:], 1, (0,1))
            newplace = npy_file_aniso_sum[i,:,:,:]
            nx = newplace.shape[0]
            ny = newplace.shape[1]
            ng = np.max(newplace)
            R = np.zeros((nx,ny,2))
            P0 = newplace
            smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
            sites_list = smooth_class.get_all_gb_list()
            # P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
            aniso_mag_sum[i] = get_poly_statistical_radius(npy_file_aniso_sum, sites_list, i)

            # newplace = np.rot90(npy_file_aniso_consMin[i,:,:,:], 1, (0,1))
            newplace = npy_file_aniso_consMin[i,:,:,:]
            nx = newplace.shape[0]
            ny = newplace.shape[1]
            ng = np.max(newplace)
            R = np.zeros((nx,ny,2))
            P0 = newplace
            smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
            sites_list = smooth_class.get_all_gb_list()
            # P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
            aniso_mag_consMin[i] = get_poly_statistical_radius(npy_file_aniso_consMin, sites_list, i)

            # newplace = np.rot90(npy_file_aniso_consMax[i,:,:,:], 1, (0,1))
            newplace = npy_file_aniso_consMax[i,:,:,:]
            nx = newplace.shape[0]
            ny = newplace.shape[1]
            ng = np.max(newplace)
            R = np.zeros((nx,ny,2))
            P0 = newplace
            smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
            sites_list = smooth_class.get_all_gb_list()
            # P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
            aniso_mag_consMax[i] = get_poly_statistical_radius(npy_file_aniso_consMax, sites_list, i)
        np.savez(current_path + data_file_name_aniso_mag, aniso_mag_min=aniso_mag_min,
                                                          aniso_mag_max=aniso_mag_max,
                                                          aniso_mag_ave=aniso_mag_ave,
                                                          aniso_mag_sum=aniso_mag_sum,
                                                          aniso_mag_consMin=aniso_mag_consMin,
                                                          aniso_mag_consMax=aniso_mag_consMax)
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    plt.plot(np.linspace(0,step_num)*30, aniso_mag_min, label='Min case', linewidth=2)
    plt.plot(np.linspace(0,step_num)*30, aniso_mag_max, label='Max case', linewidth=2)
    plt.plot(np.linspace(0,step_num)*30, aniso_mag_ave, label='Ave case', linewidth=2)
    plt.plot(np.linspace(0,step_num)*30, aniso_mag_sum, label='Sum case', linewidth=2)
    plt.plot(np.linspace(0,step_num)*30, aniso_mag_consMin, label='ConsMin case', linewidth=2)
    plt.plot(np.linspace(0,step_num)*30, aniso_mag_consMax, label='ConsMax case', linewidth=2)
    plt.xlabel("Time step", fontsize=14)
    plt.ylabel(r"$r_{offset}/r_{ave}$", fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(current_path + "/figures/anisotropic_poly_20k_isoGBs_magnitude.png", dpi=400,bbox_inches='tight')











