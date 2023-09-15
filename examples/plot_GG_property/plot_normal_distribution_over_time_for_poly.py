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
import random
from tqdm import tqdm
import sys
sys.path.append(current_path)
sys.path.append(current_path+'/../../')
import myInput
import PACKAGE_MP_Linear as linear2d
sys.path.append(current_path+'/../calculate_tangent/')

def simple_magnitude(freqArray):
    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)
    
    coeff_high = abs(np.cos((xCor-90)/180*np.pi))
    coeff_low = abs(np.cos((xCor)/180*np.pi))
    
    if np.sum(freqArray * coeff_high) > np.sum(freqArray * coeff_low):
        return np.sum(freqArray * coeff_high)/np.sum(freqArray * coeff_low)
    else:
        return np.sum(freqArray * coeff_low)/np.sum(freqArray * coeff_high)
    
    
    

def fit_ellipse_for_poly(micro_matrix, sites_list, step): #failure
    
    # For circle, we only care about the circular grain
    grains_num = len(sites_list)
    # grains_num_real = np.sum([len(sites_list[i])>0 for i in range(len(sites_list))])
    sites_num_list = np.zeros(grains_num)
    # Calculate the area
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            grain_id = int(micro_matrix[step,i,j,0]-1)
            sites_num_list[grain_id] += 1
    
    a_square_list = np.ones(grains_num)
    b_square_list = np.ones(grains_num)
    unphysic_result = 0
    grains_num_real = 0.001
    for i in range(grains_num):
        array = np.array(sites_list[i])
        
        # Avoid the really small grains
        # if sites_num_list[i] < 2100: continue
        if len(array) < 50: 
            a_square_list[i] = 1
            b_square_list[i] = 1
            continue
        array = array[random.sample(list(np.linspace(0,len(array)-1,len(array)).astype(int)), 50)]
        grains_num_real += 1
        # Get the self-variable
        X = array[:,0]
        Y = array[:,1]
        
        # Calculation Kernel
        K_mat = np.array([X**2, X*Y, Y**2, X, Y]).T
        Y_mat = -np.ones_like(X)
        X_mat = np.linalg.lstsq(K_mat, Y_mat, rcond=None)[0].squeeze()
        
        # Calculate the long and short axis
        center_base = 4 * X_mat[0] * X_mat[2] - X_mat[1] * X_mat[1]
        center_x = (X_mat[1] * X_mat[4] - 2 * X_mat[2]* X_mat[3]) / center_base
        center_y = (X_mat[1] * X_mat[3] - 2 * X_mat[0]* X_mat[4]) / center_base
        axis_square_root = np.sqrt((X_mat[0] - X_mat[2])**2 + X_mat[1]**2)
        a_square = 2*(X_mat[0]*center_x*center_x + X_mat[2]*center_y*center_y + X_mat[1]*center_x*center_y - 1) / (X_mat[0] + X_mat[2] + axis_square_root)
        b_square = 2*(X_mat[0]*center_x*center_x + X_mat[2]*center_y*center_y + X_mat[1]*center_x*center_y - 1) / (X_mat[0] + X_mat[2] - axis_square_root)
        
        #  Avoid the grains with strange shape
        if a_square < 0 or b_square < 0:
            # matrix = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
            # for s in range(len(array)): 
            #     matrix[int(array[s,0]),int(array[s,1])] = 1
            # plt.close()
            # plt.imshow(matrix)
            a_square_list[i] = 1
            b_square_list[i] = 1
            unphysic_result += 1#sites_num_list[i]
            continue
        # print(f"a: {np.sqrt(a_square)}, b: {np.sqrt(b_square)}")
        a_square_list[i] = a_square
        b_square_list[i] = b_square
    print(f"The unphysical result is {round(unphysic_result/grains_num_real*100,3)}%")
        
    return np.sum(b_square_list * sites_num_list) / np.sum(a_square_list * sites_num_list)

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

        if (sites_num_list[i] < 2100) or \
           (np.max(coord_refer_i[table == i+1]) - np.min(coord_refer_i[table == i+1]) == micro_matrix.shape[1]) or \
           (np.max(coord_refer_j[table == i+1]) - np.min(coord_refer_j[table == i+1]) == micro_matrix.shape[2]): # grains on bc are ignored
          center_list[i, 0] = 0
          center_list[i, 1] = 0
          sites_num_list[i] = 0
        else:
          center_list[i, 0] = (np.max(coord_refer_i[table == i+1]) + np.min(coord_refer_i[table == i+1])) / 2
          center_list[i, 1] = (np.max(coord_refer_j[table == i+1]) + np.min(coord_refer_j[table == i+1])) / 2
    ave_radius_list = np.sqrt(sites_num_list / np.pi)
    # print(np.max(sites_num_list))

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

    # max_radius_offset = np.average(max_radius_offset_list[max_radius_offset_list!=0])
    area_list = np.pi*ave_radius_list*ave_radius_list
    if np.sum(area_list) == 0: max_radius_offset = 0
    else: max_radius_offset = np.sum(max_radius_offset_list * area_list) / np.sum(area_list)

    return max_radius_offset

def get_poly_statistical_ar(micro_matrix, step):
    # Get the average aspect ratio
    num_grains = int(np.max(micro_matrix[step,:]))
    sites_num_list = np.zeros(num_grains)
    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i,j] = i
            coord_refer_j[i,j] = j

    aspect_ratio_i = np.zeros((num_grains,2))
    aspect_ratio_j = np.zeros((num_grains,2))
    aspect_ratio = np.zeros(num_grains)
    table = micro_matrix[step,:,:,0]
    for i in range(num_grains):
        sites_num_list[i] = np.sum(table == i+1)
        aspect_ratio_i[i, 0] = len(list(set(coord_refer_i[table == i+1])))
        aspect_ratio_j[i, 1] = len(list(set(coord_refer_j[table == i+1])))
        if aspect_ratio_j[i, 1] == 0: aspect_ratio[i] = 0
        else: aspect_ratio[i] = aspect_ratio_i[i, 0] / aspect_ratio_j[i, 1]



    # aspect_ratio = np.average(aspect_ratio[aspect_ratio!=0])
    aspect_ratio = np.sum(aspect_ratio * sites_num_list) / np.sum(sites_num_list)

    return aspect_ratio

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
    return freqArray

if __name__ == '__main__':
    # File name
    npy_file_folder = "/Users/lin.yang/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_poly_multiCoreCompare/results/"
    circle_energy_000 = "0.0"
    circle_energy_020 = "0.2"
    circle_energy_040 = "0.4"
    circle_energy_060 = "0.6"
    circle_energy_080 = "0.8"
    circle_energy_095 = "0.95"


    npy_file_name_aniso_000 = f"p_ori_ave_aveE_512_multiCore16_delta{circle_energy_000}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_020 = f"p_ori_ave_aveE_512_multiCore16_delta{circle_energy_020}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_040 = f"p_ori_ave_aveE_512_multiCore16_delta{circle_energy_040}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_060 = f"p_ori_ave_aveE_512_multiCore8_delta{circle_energy_060}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_080 = f"p_ori_ave_aveE_512_multiCore16_delta{circle_energy_080}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_095 = f"p_ori_ave_aveE_512_multiCore16_delta{circle_energy_095}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_000 = f"grain_size_p_ori_aveE_512_multiCore32_delta{circle_energy_000}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_020 = f"grain_size_p_ori_aveE_512_multiCore32_delta{circle_energy_020}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_040 = f"grain_size_p_ori_aveE_512_multiCore32_delta{circle_energy_040}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_060 = f"grain_size_p_ori_aveE_512_multiCore32_delta{circle_energy_060}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_080 = f"grain_size_p_ori_aveE_512_multiCore32_delta{circle_energy_080}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_095 = f"grain_size_p_ori_aveE_512_multiCore32_delta{circle_energy_095}_m2_J1_refer_1_0_0_seed56689_kt066.npy"

    # Initial data
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

    # Initial container
    initial_grain_num = 512
    step_num = npy_file_aniso_000.shape[0]

    bin_width = 0.16 # Grain size distribution
    x_limit = [-0.5, 3.5]
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)

    special_step_distribution_000 = 100#4
    special_step_distribution_020 = 100#4
    special_step_distribution_040 = 100#4
    special_step_distribution_060 = 100#4
    special_step_distribution_080 = 100#4
    special_step_distribution_095 = 100#4


    # Start polar figure
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)

    ax.set_rgrids(np.arange(0, 0.008, 0.004))
    ax.set_rlabel_position(0.0)  # 标签显示在0°
    ax.set_rlim(0.0, 0.008)  # 标签范围为[0, 5000)
    ax.set_yticklabels(['0', '0.004'],fontsize=14)

    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    # Aniso - 000
    data_file_name_P = f'/normal_distribution_data/normal_distribution_poly_000_P_step{special_step_distribution_000}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_poly_000_sites_step{special_step_distribution_000}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_000[special_step_distribution_000,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_000, r"$\delta=000$")
    # For bias
    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    freqArray_circle = np.ones(binNum)
    freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)
    slope_list_bias = freqArray_circle - slope_list

    # Aniso - 020
    data_file_name_P = f'/normal_distribution_data/normal_distribution_poly_020_P_step{special_step_distribution_020}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_poly_020_P_sites_step{special_step_distribution_020}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_020[special_step_distribution_020,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_020, r"$\delta=020$")

    # Aniso - 040
    data_file_name_P = f'/normal_distribution_data/normal_distribution_poly_040_P_step{special_step_distribution_040}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_poly_040_sites_step{special_step_distribution_040}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_040[special_step_distribution_040,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_040, r"$\delta=040$")

    # Aniso - 060
    data_file_name_P = f'/normal_distribution_data/normal_distribution_poly_060_P_step{special_step_distribution_060}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_poly_060_sites_step{special_step_distribution_060}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_060[special_step_distribution_060,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_060, r"$\delta=060$")

    # Aniso - 080
    data_file_name_P = f'/normal_distribution_data/normal_distribution_poly_080_P_step{special_step_distribution_080}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_poly_080_sites_step{special_step_distribution_080}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_080[special_step_distribution_080,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_080, r"$\delta=080$")

    # Aniso - 095
    data_file_name_P = f'/normal_distribution_data/normal_distribution_poly_095_P_step{special_step_distribution_095}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_poly_095_sites_step{special_step_distribution_095}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_095[special_step_distribution_095,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_095, r"$\delta=095$")

    plt.legend(loc=(-0.14,-0.3),fontsize=14,ncol=3)
    plt.savefig(current_path + "/figures/normal_distribution_poly.png", dpi=400,bbox_inches='tight')

    # For figure after bias
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)

    ax.set_rgrids(np.arange(0, 0.008, 0.004))
    ax.set_rlabel_position(0.0)  # 标签显示在0°
    ax.set_rlim(0.0, 0.008)  # 标签范围为[0, 5000)
    ax.set_yticklabels(['0', '0.004'],fontsize=14)

    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    aniso_mag = np.zeros(6)
    # Aniso - 000
    data_file_name_P = f'/normal_distribution_data/normal_distribution_poly_000_P_step{special_step_distribution_000}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_poly_000_sites_step{special_step_distribution_000}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_000[special_step_distribution_000,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_000, r"$\delta=0.00$", slope_list_bias)
    aniso_mag[0] = simple_magnitude(slope_list)

    # Aniso - 020
    data_file_name_P = f'/normal_distribution_data/normal_distribution_poly_020_P_step{special_step_distribution_020}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_poly_020_P_sites_step{special_step_distribution_020}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_020[special_step_distribution_020,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_020, r"$\delta=0.20$", slope_list_bias)
    aniso_mag[1] = simple_magnitude(slope_list)

    # Aniso - 040
    data_file_name_P = f'/normal_distribution_data/normal_distribution_poly_040_P_step{special_step_distribution_040}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_poly_040_sites_step{special_step_distribution_040}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_040[special_step_distribution_040,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_040, r"$\delta=0.40$", slope_list_bias)
    aniso_mag[2] = simple_magnitude(slope_list)

    # Aniso - 060
    data_file_name_P = f'/normal_distribution_data/normal_distribution_poly_060_P_step{special_step_distribution_060}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_poly_060_sites_step{special_step_distribution_060}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_060[special_step_distribution_060,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_060, r"$\delta=0.60$", slope_list_bias)
    aniso_mag[3] = simple_magnitude(slope_list)

    # Aniso - 080
    data_file_name_P = f'/normal_distribution_data/normal_distribution_poly_080_P_step{special_step_distribution_080}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_poly_080_sites_step{special_step_distribution_080}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_080[special_step_distribution_080,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_080, r"$\delta=0.80$", slope_list_bias)
    aniso_mag[4] = simple_magnitude(slope_list)

    # Aniso - 095
    data_file_name_P = f'/normal_distribution_data/normal_distribution_poly_095_P_step{special_step_distribution_095}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_poly_095_sites_step{special_step_distribution_095}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_095[special_step_distribution_095,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_095, r"$\delta=0.95$", slope_list_bias)
    aniso_mag[5] = simple_magnitude(slope_list)

    plt.legend(loc=(-0.14,-0.3),fontsize=14,ncol=3)
    plt.savefig(current_path + "/figures/normal_distribution_poly_after_removing_bias.png", dpi=400,bbox_inches='tight')
    print("Polar figure done.")

    # PLot magnitude of anisotropy
    # num_step_magni = 30
    # data_file_name_aniso_mag = f'/normal_distribution_data/aniso_magnitude_poly_delta.npz'
    # if os.path.exists(current_path + data_file_name_aniso_mag):
    #     data_file_aniso_mag = np.load(current_path + data_file_name_aniso_mag)
    #     # aniso_mag_000=data_file_aniso_mag['aniso_mag_000']
    #     # aniso_mag_020=data_file_aniso_mag['aniso_mag_020']
    #     # aniso_mag_040=data_file_aniso_mag['aniso_mag_040']
    #     # aniso_mag_060=data_file_aniso_mag['aniso_mag_060']
    #     # aniso_mag_080=data_file_aniso_mag['aniso_mag_080']
    #     # aniso_mag_095=data_file_aniso_mag['aniso_mag_095']
    #     aniso_mag = data_file_aniso_mag['aniso_mag']
    # else:
    #     # aniso_mag_000 = np.zeros(step_num)
    #     # aniso_mag_020 = np.zeros(step_num)
    #     # aniso_mag_040 = np.zeros(step_num)
    #     # aniso_mag_060 = np.zeros(step_num)
    #     # aniso_mag_080 = np.zeros(step_num)
    #     # aniso_mag_095 = np.zeros(step_num)
    #     # aniso_mag = np.zeros(6)
    #     cores = 16
    #     loop_times = 5
    #     for i in [num_step_magni]:#tqdm(range(step_num)):
    #         # newplace = np.rot90(npy_file_aniso_000[i,:,:,:], 1, (0,1))
    #         newplace = npy_file_aniso_000[i,:,:,:]
    #         nx = newplace.shape[0]
    #         ny = newplace.shape[1]
    #         ng = np.max(newplace)
    #         R = np.zeros((nx,ny,2))
    #         P0 = newplace
    #         smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
    #         sites_list = smooth_class.get_all_gb_list()
    #         # aniso_mag[0] = fit_ellipse_for_poly(npy_file_aniso_000, sites_list, i)

    #         # newplace = np.rot90(npy_file_aniso_020[i,:,:,:], 1, (0,1))
    #         newplace = npy_file_aniso_020[i,:,:,:]
    #         nx = newplace.shape[0]
    #         ny = newplace.shape[1]
    #         ng = np.max(newplace)
    #         R = np.zeros((nx,ny,2))
    #         P0 = newplace
    #         smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
    #         sites_list = smooth_class.get_all_gb_list()
    #         # aniso_mag[1] = fit_ellipse_for_poly(npy_file_aniso_020, sites_list, i)

    #         # newplace = np.rot90(npy_file_aniso_040[i,:,:,:], 1, (0,1))
    #         newplace = npy_file_aniso_040[i,:,:,:]
    #         nx = newplace.shape[0]
    #         ny = newplace.shape[1]
    #         ng = np.max(newplace)
    #         R = np.zeros((nx,ny,2))
    #         P0 = newplace
    #         smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
    #         sites_list = smooth_class.get_all_gb_list()
    #         # aniso_mag[2] = fit_ellipse_for_poly(npy_file_aniso_040, sites_list, i)

    #         # newplace = np.rot90(npy_file_aniso_060[i,:,:,:], 1, (0,1))
    #         newplace = npy_file_aniso_060[i,:,:,:]
    #         nx = newplace.shape[0]
    #         ny = newplace.shape[1]
    #         ng = np.max(newplace)
    #         R = np.zeros((nx,ny,2))
    #         P0 = newplace
    #         smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
    #         sites_list = smooth_class.get_all_gb_list()
    #         # aniso_mag[3] = fit_ellipse_for_poly(npy_file_aniso_060, sites_list, i)

    #         # newplace = np.rot90(npy_file_aniso_080[i,:,:,:], 1, (0,1))
    #         newplace = npy_file_aniso_080[i,:,:,:]
    #         nx = newplace.shape[0]
    #         ny = newplace.shape[1]
    #         ng = np.max(newplace)
    #         R = np.zeros((nx,ny,2))
    #         P0 = newplace
    #         smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
    #         sites_list = smooth_class.get_all_gb_list()
    #         # aniso_mag[4] = fit_ellipse_for_poly(npy_file_aniso_080, sites_list, i)

    #         # newplace = np.rot90(npy_file_aniso_095[i,:,:,:], 1, (0,1))
    #         newplace = npy_file_aniso_095[i,:,:,:]
    #         nx = newplace.shape[0]
    #         ny = newplace.shape[1]
    #         ng = np.max(newplace)
    #         R = np.zeros((nx,ny,2))
    #         P0 = newplace
    #         smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
    #         sites_list = smooth_class.get_all_gb_list()
    #         # aniso_mag[5] = fit_ellipse_for_poly(npy_file_aniso_095, sites_list, i)
            
    #     # np.savez(current_path + data_file_name_aniso_mag, aniso_mag_000=aniso_mag_000,
    #     #                                                   aniso_mag_020=aniso_mag_020,
    #     #                                                   aniso_mag_040=aniso_mag_040,
    #     #                                                   aniso_mag_060=aniso_mag_060,
    #     #                                                   aniso_mag_080=aniso_mag_080,
    #     #                                                   aniso_mag_095=aniso_mag_095)
    #     np.savez(current_path + data_file_name_aniso_mag, aniso_mag=aniso_mag)
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_000, label=r'$\delta=0.00$', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_020, label=r'$\delta=0.20$', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_040, label=r'$\delta=0.40$', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_060, label=r'$\delta=0.60$', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_080, label=r'$\delta=0.80$', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_095, label=r'$\delta=0.95$', linewidth=2)
    delta_value = np.array([0.0,0.2,0.4,0.6,0.8,0.95])
    plt.plot(delta_value, aniso_mag, '.-', markersize=8, label='time step = 3000', linewidth=2)
    
    plt.xlabel(r"$\delta$", fontsize=14)
    plt.ylabel("Magnitude", fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(current_path + "/figures/anisotropic_magnitude_poly_aspect_ratio_fit.png", dpi=400,bbox_inches='tight')










