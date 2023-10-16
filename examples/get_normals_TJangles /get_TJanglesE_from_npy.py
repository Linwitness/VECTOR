#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:48:29 2023

@author: Lin
"""

import os
current_path = os.getcwd()
import numpy as np
from numpy import seterr
seterr(all='raise')
import matplotlib.pyplot as plt
import math
from itertools import repeat
import sys
sys.path.append(current_path)
sys.path.append(current_path+'/../../')
import myInput
import PACKAGE_MP_Linear as linear2d
sys.path.append(current_path+'/../calculate_tangent/')
import output_tangent

from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a * np.exp(-x / b) + c

def get_gb_sites(P,grain_num):
    _,nx,ny=np.shape(P)
    timestep=5
    ggn_gbsites = [[] for i in repeat(None, grain_num)]
    for i in range(timestep,nx-timestep):
        for j in range(timestep,ny-timestep):
            ip,im,jp,jm = myInput.periodic_bc(nx,ny,i,j)
            if ( ((P[0,ip,j]-P[0,i,j])!=0) or ((P[0,im,j]-P[0,i,j])!=0) or\
                 ((P[0,i,jp]-P[0,i,j])!=0) or ((P[0,i,jm]-P[0,i,j])!=0) ) and\
                P[0,i,j] <= grain_num:
                ggn_gbsites[int(P[0,i,j]-1)].append([i,j])
    return ggn_gbsites

def norm_list(grain_num, P_matrix):
    # get the norm list
    # grain_num -= 1
    boundary_site = get_gb_sites(P_matrix, grain_num)
    norm_list = [np.zeros(( len(boundary_site[i]), 2 )) for i in range(grain_num)]
    for grain_i in range(grain_num):
        print(f"finish {grain_i}")
        
        for site in range(len(boundary_site[grain_i])):
            norm = myInput.get_grad(P_matrix, boundary_site[grain_i][site][0], boundary_site[grain_i][site][1])
            norm_list[grain_i][site,:] = list(norm)
        
    return norm_list, boundary_site

def get_orientation(grain_num, init_name ):
    # read the input euler angle from *.init
    eulerAngle = np.ones((grain_num,3))*-10
    with open(init_name, 'r', encoding = 'utf-8') as f:
        for line in f:
            eachline = line.split()
    
            if len(eachline) == 5 and eachline[0] != '#':
                lineN = int(eachline[1])-1
                if eulerAngle[lineN,0] == -10:
                    eulerAngle[lineN,:] = [float(eachline[2]), float(eachline[3]), float(eachline[4])]
    return eulerAngle[:ng-1]

def output_inclination(output_name, norm_list, site_list, orientation_list=0):
    
    file = open(output_name,'w')
    for i in range(len(norm_list)):
        if orientation_list != 0:
            file.writelines(['Grain ' + str(i+1) + ' Orientation: ' + str(orientation_list[i]) + ' centroid: ' + '\n'])
        else:
            file.writelines(['Grain ' + str(i+1) + ' Orientation: empty centroid: ' + '\n'])
        
        for j in range(len(norm_list[i])):
            file.writelines([str(site_list[i][j][0]) + ', ' + str(site_list[i][j][1]) + ', ' + str(norm_list[i][j][0]) + ', ' + str(norm_list[i][j][1]) + '\n'])
            
        file.writelines(['\n'])
    
    file.close()
    return

def output_dihedral_angle(output_name, triple_coord, triple_angle, triple_grain):
    file = open(output_name,'w')
    file.writelines(['triple_index triple_coordination grain_id0:dihedral0 grain_id1:dihedral1 grain_id2:dihedral2 angle_sum\n'])
    for i in range(len(triple_coord)):
        file.writelines([str(i+1) + ', ' + str(triple_coord[i][0]) + ' ' + str(triple_coord[i][1]) + ', ' + \
                         str(int(triple_grain[i][0])) + ':' + str(round(triple_angle[i][0],2)) + ' ' + \
                         str(int(triple_grain[i][1])) + ':' + str(round(triple_angle[i][1],2)) + ' ' + \
                         str(int(triple_grain[i][2])) + ':' + str(round(triple_angle[i][2],2)) + ' ' + \
                         str(round(np.sum(triple_angle[i]),2)) + '\n'])
        
    file.close()
    return

def find_window(P,i,j,iteration,refer_id):
    # Find the windows around the voxel i,j, the size depend on iteration
    nx,ny=P.shape
    tableL=2*(iteration+1)+1
    fw_len = tableL
    fw_half = int((fw_len-1)/2)
    window = np.zeros((fw_len,fw_len))
    
    for wi in range(fw_len):
        for wj in range(fw_len):
            global_x = (i-fw_half+wi)%nx
            global_y = (j-fw_half+wj)%ny
            if P[global_x,global_y] == refer_id:
                window[wi,wj] = 1
            else:
                window[wi,wj] = 0
    
    return window

def data_smooth(data_array, smooth_level=2):
    
    data_array_smoothed = np.zeros(len(data_array))
    for i in range(len(data_array)):
        if i < smooth_level: data_array_smoothed[i] = np.sum(data_array[0:i+smooth_level+1])/(i+smooth_level+1)
        elif (len(data_array) - 1 - i) < smooth_level: data_array_smoothed[i] = np.sum(data_array[i-smooth_level:])/(len(data_array)-i+smooth_level)
        else: data_array_smoothed[i] = np.sum(data_array[i-smooth_level:i+smooth_level+1])/(smooth_level*2+1)
        # print(data_array_smoothed[i]) 
        
    return data_array_smoothed

if __name__ == '__main__':
    
    npy_file_folder = "/Users/lin/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_hex_for_TJE/results/"
    TJ_energy_type_cases = ["ave", "sum", "consMin", "consMax", "consTest"]
    step_equalibrium_end = int(8000/100)
    
    
    average_TJtype_energy = np.zeros(len(TJ_energy_type_cases))
    average_TJtype_dihedral_angle = np.zeros(len(TJ_energy_type_cases))
    for energy_type_index, energy_type in enumerate(TJ_energy_type_cases):
        print(f"\nStart {energy_type} energy type:")
        npy_file_name = f"h_ori_ave_{energy_type}E_hex_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_angle.npy"
        energy_npy_file_name = f"h_ori_ave_{energy_type}E_hex_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_angle_energy.npy"
        
        base_name = f"dihedral_results/hex_{energy_type}_"
        energy_base_name = f"energy_results/hex_{energy_type}_"
        dihedral_over_time_data_name = energy_base_name + "energy_data.npy"

        
        # If the data file exist, just read the data file instead of recalculate it
        if os.path.exists(npy_file_folder + dihedral_over_time_data_name):
            average_TJ_energy = np.load(npy_file_folder + dihedral_over_time_data_name)
            print("Energy data readed")
        else:
            average_TJ_energy = np.zeros(step_equalibrium_end)
            for timestep in range(step_equalibrium_end):
                print(f"\nCalculation for time step {timestep}")
                # Initial setting
                P0_list = np.load(npy_file_folder + npy_file_name)
                P0_energy_list = np.load(npy_file_folder + energy_npy_file_name)
                print("IC is read as matrix")
                P0 = P0_list[timestep,:,:,0]
                P0_energy = P0_energy_list[timestep,:,:,0]
                
                # Get IC (2400, 2400)
                nx, ny = P0.shape
                ng = 50
                cores = 8
                loop_times = 5
                R = np.zeros((nx,ny,2))

                allTJ_ave_energy = 0
                num_TJ = 0
                for i in range(nx-1):
                    for j in range(ny-1):
                        nei = np.zeros((2,2))
                        nei = P0[i:i+2,j:j+2]
                        energy_nei = P0_energy[i:i+2,j:j+2]
                        nei_flat = nei.flatten()
                        energy_nei_flat = energy_nei.flatten()
                        if len(set(nei_flat)) == 3 and 0 not in nei_flat:
                            
                            # # Average site energy based on grain num
                            # oneTJ_ave_energy = 0
                            # for k in list(set(nei_flat)):
                            #     index = int(np.where(nei_flat==k)[0][0])
                            #     oneTJ_ave_energy += energy_nei_flat[index]
                            # oneTJ_ave_energy = oneTJ_ave_energy/len(list(set(nei_flat)))
                            
                            # # Average site energy based on site num
                            # oneTJ_ave_energy = np.average(energy_nei_flat)
                            
                            # allTJ_ave_energy += oneTJ_ave_energy
                            # num_TJ += 1
                        
                            # energy per site
                            oneTJ_ave_energy = 0
                            for k in [[i,j],[i,j+1],[i+1,j],[i+1,j+1]]:
                                window_matrix = find_window(P0, k[0], k[1], 0, P0[k[0],k[1]])
                                window_matrix_flat = window_matrix.flatten()
                                # np.delete(window_matrix_flat, 4)
                                nei_sites_num = np.sum(window_matrix_flat==0)
                                oneTJ_ave_energy += P0_energy[k[0],k[1]] / nei_sites_num
                            oneTJ_ave_energy = oneTJ_ave_energy / 4
                            allTJ_ave_energy += oneTJ_ave_energy
                            num_TJ += 1
                        
                allTJ_ave_energy = allTJ_ave_energy / num_TJ
                average_TJ_energy[timestep] = allTJ_ave_energy
                
            
        # Save the data
        np.save(npy_file_folder + dihedral_over_time_data_name, average_TJ_energy)
        average_TJtype_energy[energy_type_index] = np.average(average_TJ_energy)
        
        # Get the average dihedral angle
        dihedral_over_time = np.load(npy_file_folder + base_name + "data.npy")
        average_TJtype_dihedral_angle[energy_type_index] = np.average(dihedral_over_time[:step_equalibrium_end])
        
    dihedral_siteEnergy_cases_figure_name = "energy_results/hex_aveDihedral_aveEnergy_" + "figure.png"
    plt.clf()
    plt.plot(average_TJtype_energy, average_TJtype_dihedral_angle, 'o', markersize=4, label = "average angle in energy types")
    
    # Fitting
    a = max(average_TJtype_dihedral_angle)-min(average_TJtype_dihedral_angle)
    b = average_TJtype_dihedral_angle[round(len(average_TJtype_dihedral_angle)/2)]
    c = min(average_TJtype_dihedral_angle)
    p0 = [a,b,c]
    popt, pcov = curve_fit(func, average_TJtype_energy, average_TJtype_dihedral_angle,p0=p0)
    print(f"The equation to dit the relationship is {round(popt[0],2)} * exp(-x * {round(popt[1],2)}) + {round(popt[2],2)}")
    y_fit = [func(i,popt[0], popt[1], popt[2]) for i in np.linspace(0, 4, 50)]
    plt.plot(np.linspace(0, 4, 50), y_fit, '-', linewidth=2, label = "fitting results")
    # Find the exact result
    exact_list = np.linspace(0.2, 1.0, 101)
    min_level = 10
    expect_site_energy = 0
    for m in exact_list: 
        if min_level > abs(func(m, popt[0], popt[1], popt[2]) - 145.46):
            min_level = abs(func(m, popt[0], popt[1], popt[2]) - 145.46)
            expect_site_energy = m
    print(f"The expected average TJ site energy is {expect_site_energy}")
    
    plt.plot(np.linspace(0,4,24), [145.46]*24, '--', linewidth=2, label = "expected angle results") # Max-100
    plt.ylim([120,160])
    plt.xlim([0,4])
    plt.legend(fontsize=14, loc='lower center')
    plt.xlabel("Coupled energy", fontsize=14)
    plt.ylabel(r"Angle ($^\circ$)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.savefig(npy_file_folder + dihedral_siteEnergy_cases_figure_name, bbox_inches='tight', format='png', dpi=400)
        
        