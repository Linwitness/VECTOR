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
import PACKAGE_MP_3DLinear as linear3d
sys.path.append(current_path+'/../calculate_tangent/')

def get_normal_vector(grain_structure_figure_one):
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

def get_normal_vector_3d(grain_structure_figure_one):
    nx = grain_structure_figure_one.shape[0]
    ny = grain_structure_figure_one.shape[1]
    nz = grain_structure_figure_one.shape[2]
    ng = np.max(grain_structure_figure_one)
    cores = 8
    loop_times = 5
    P0 = grain_structure_figure_one[:,:,:,np.newaxis]
    R = np.zeros((nx,ny,nz,3))
    smooth_class = linear3d.linear3d_class(nx,ny,nz,ng,cores,loop_times,P0,R,'np')

    smooth_class.linear3d_main("inclination")
    P = smooth_class.get_P()
    sites = smooth_class.get_all_gb_list()
    sites_together = []
    for id in range(len(sites)): sites_together += sites[id]
    print(f"Total num of GB sites: {len(sites_together)}")

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
    plt.plot(xCor/180*math.pi, freqArray, linewidth=2, label=para_name)

    return 0

def get_normal_vector_slope_3d(P, sites, step, para_name, angle_index=0, bias=None):
    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)

    freqArray = np.zeros(binNum)
    degree = []
    # degree_shadow = []
    for sitei in sites:
        [i,j,k] = sitei
        dx,dy,dz = myInput.get_grad3d(P,i,j,k)
        # dy_fake = math.sqrt(dy**2 + dz**2)
        if angle_index == 0:
            dx_fake = dx
            dy_fake = dy
        elif angle_index == 1:
            dx_fake = dx
            dy_fake = dz
        elif angle_index == 2:
            dx_fake = dy
            dy_fake = dz

        # Normalize
        if math.sqrt(dy_fake**2+dx_fake**2) < 1e-5: continue
        dy_fake_norm = dy_fake / math.sqrt(dy_fake**2+dx_fake**2)
        dx_fake_norm = dx_fake / math.sqrt(dy_fake**2+dx_fake**2)

        degree.append(math.atan2(-dy_fake_norm, dx_fake_norm) + math.pi)
        # degree_shadow.append([i,j,k,dz])
    for n in range(len(degree)):
        freqArray[int((degree[n]/math.pi*180-xLim[0])/binValue)] += 1
        # if int((degree[n]/math.pi*180-xLim[0])/binValue) == 0:
        #     print(f"loc: {degree_shadow[n][0]},{degree_shadow[n][1]},{degree_shadow[n][2]} : {degree[n]/np.pi*180} and {degree_shadow[n][3]}")
    freqArray = freqArray/sum(freqArray*binValue)

    if bias is not None:
        freqArray = freqArray + bias
        freqArray = freqArray/sum(freqArray*binValue)

    # Plot
    plt.plot(np.append(xCor,xCor[0])/180*math.pi, np.append(freqArray,freqArray[0]),linewidth=2,label=para_name)

    return freqArray

if __name__ == '__main__':
    # File name
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_sphere/results/"
    TJ_energy_type_ave = "ave"
    TJ_energy_type_iso = "ave_delta000"

    npy_file_name_aniso_ave = f"p_ori_ave_{TJ_energy_type_ave}E_150_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt1.95.npy"
    npy_file_name_iso = "p_ori_ave_aveE_150_multiCore64_delta0.0_m2_J1_refer_1_0_0_seed56689_kt1.95.npy"

    # Initial data
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)
    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    print(f"The iso data size is: {npy_file_iso.shape}")
    print("READING DATA DONE")

    # Initial container
    initial_grain_num = 2
    step_num = npy_file_iso.shape[0]
    grain_num_aniso_ave = np.zeros(step_num)
    grain_num_iso = np.zeros(step_num)

    # Calculate the number of grains
    for i in range(step_num):
        grain_num_aniso_ave[i] = len(set(npy_file_aniso_ave[i,:].flatten()))
        grain_num_iso[i] = len(set(npy_file_iso[i,:].flatten()))

    # expected_grain_num = 500
    # special_step_distribution_ave = int(np.argmin(abs(grain_num_aniso_ave - expected_grain_num)))
    # special_step_distribution_min = int(np.argmin(abs(grain_num_aniso_min - expected_grain_num)))
    # special_step_distribution_min195D264 = int(np.argmin(abs(grain_num_aniso_min195D264 - expected_grain_num)))
    # special_step_distribution_iso = int(np.argmin(abs(grain_num_iso - expected_grain_num)))
    special_step_distribution_ave = 10
    special_step_distribution_iso = 10

    # iso - 000
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso")
    # For bias
    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    freqArray_circle = np.ones(binNum)
    freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)
    slope_list_bias_0 = freqArray_circle - slope_list

    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso",1)
    slope_list_bias_1 = freqArray_circle - slope_list

    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso",2)
    slope_list_bias_2 = freqArray_circle - slope_list


    # Start polar figure xy
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)

    ax.set_rgrids(np.arange(0, 0.008, 0.004))
    ax.set_rlabel_position(0.0)  # start from 0 degree
    ax.set_rlim(0.0, 0.008)  # radiu lim range is from 0 to 0.008
    ax.set_yticklabels(['0', '0.004'],fontsize=14)

    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    # Aniso - ave
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave")

    # iso - 000
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso")

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "x", fontsize=14)
    plt.text(np.pi/2, 0.0095, "y", fontsize=14)
    plt.savefig(current_path + f"/figures/normal_distribution_3dsphere_xy_{special_step_distribution_iso}steps.png", dpi=400,bbox_inches='tight')

    # Start polar figure xz
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)

    ax.set_rgrids(np.arange(0, 0.008, 0.004))
    ax.set_rlabel_position(0.0)  # start from 0 degree
    ax.set_rlim(0.0, 0.008)  # radiu lim range is from 0 to 0.008
    ax.set_yticklabels(['0', '0.004'],fontsize=14)

    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    # Aniso - ave
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 1)

    # iso - 000
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 1)

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "x", fontsize=14)
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)
    plt.savefig(current_path + f"/figures/normal_distribution_3dsphere_xz_{special_step_distribution_iso}steps.png", dpi=400,bbox_inches='tight')


    # Start polar figure yz
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)

    ax.set_rgrids(np.arange(0, 0.008, 0.004))
    ax.set_rlabel_position(0.0)  # start from 0 degree
    ax.set_rlim(0.0, 0.008)  # radiu lim range is from 0 to 0.008
    ax.set_yticklabels(['0', '0.004'],fontsize=14)

    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    # Aniso - ave
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 2)

    # iso - 000
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 2)

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "y", fontsize=14)
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)
    plt.savefig(current_path + f"/figures/normal_distribution_3dsphere_yz_{special_step_distribution_iso}steps.png", dpi=400,bbox_inches='tight')



    # After removing bias
    # Start polar figure xy
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)

    ax.set_rgrids(np.arange(0, 0.008, 0.004))
    ax.set_rlabel_position(0.0)  # start from 0 degree
    ax.set_rlim(0.0, 0.008)  # radiu lim range is from 0 to 0.008
    ax.set_yticklabels(['0', '0.004'],fontsize=14)

    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    # Aniso - ave
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 0, slope_list_bias_0)

    # iso - 000
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 0, slope_list_bias_0)

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "x", fontsize=14)
    plt.text(np.pi/2, 0.0095, "y", fontsize=14)
    plt.savefig(current_path + f"/figures/normal_distribution_3dsphere_xy_{special_step_distribution_iso}steps_after_removing_bias.png", dpi=400,bbox_inches='tight')

    # Start polar figure xz
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)

    ax.set_rgrids(np.arange(0, 0.008, 0.004))
    ax.set_rlabel_position(0.0)  # start from 0 degree
    ax.set_rlim(0.0, 0.008)  # radiu lim range is from 0 to 0.008
    ax.set_yticklabels(['0', '0.004'],fontsize=14)

    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    # Aniso - ave
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 1, slope_list_bias_1)

    # iso - 000
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 1, slope_list_bias_1)

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "x", fontsize=14)
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)
    plt.savefig(current_path + f"/figures/normal_distribution_3dsphere_xz_{special_step_distribution_iso}steps_after_removing_bias.png", dpi=400,bbox_inches='tight')


    # Start polar figure yz
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)

    ax.set_rgrids(np.arange(0, 0.008, 0.004))
    ax.set_rlabel_position(0.0)  # start from 0 degree
    ax.set_rlim(0.0, 0.008)  # radiu lim range is from 0 to 0.008
    ax.set_yticklabels(['0', '0.004'],fontsize=14)

    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    # Aniso - ave
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 2, slope_list_bias_2)

    # iso - 000
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 2, slope_list_bias_2)

    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "y", fontsize=14)
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)
    plt.savefig(current_path + f"/figures/normal_distribution_3dsphere_yz_{special_step_distribution_iso}steps_after_removing_bias.png", dpi=400,bbox_inches='tight')






