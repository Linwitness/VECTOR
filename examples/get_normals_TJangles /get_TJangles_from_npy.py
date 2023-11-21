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
from tqdm import tqdm



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

def dihedral_angle_from_Joseph(case_path, num_steps):
    # Get data from Joseph method
    triple_results_joseph = np.load(case_path)
    
    max_dihedral_list_joseph = np.zeros(num_steps)
    
    for i in tqdm(range(num_steps)):
        # From Joseph algorithm
        triple_results_step_joseph = triple_results_joseph[i,3:6]

        # triple_results_step_joseph = triple_results_step_joseph[:,~np.isnan(triple_results_step_joseph[0,:])]
        # print(f"The number in current case is {len(triple_results_step_joseph[0,:])}")
        max_dihedral_list_joseph[i] = triple_results_step_joseph[2]
        
    return max_dihedral_list_joseph
    
def dihedral_angle_from_Lin(npy_file_folder, base_name, energy_type, num_steps):
    # Get data from Lin method
    dihedral_over_time_data_name = base_name + "data.npy"
    max_dihedral_list_lin = np.zeros(num_steps)
    npy_file_name = f"t_{energy_type}_512x512_delta0.6_m2_refer_1_0_0_seed56689_kt066.npy"
    P0_list = np.load(npy_file_folder + npy_file_name)
    print("IC is read as matrix")
    
    # If the data file exist , just read the data dile instead of recalculate it
    if os.path.exists(npy_file_folder + dihedral_over_time_data_name):
        max_dihedral_list_lin = np.load(npy_file_folder + dihedral_over_time_data_name)
        print("Dihedral angle readed")
        
    else:
        for timestep in range(num_steps):
            print(f"\nCalculation for time step {timestep}")
            # Initial setting
            P0 = P0_list[timestep,:,:,:]
            output_inclination_name = base_name + "inclination.txt"
            output_dihedral_name = base_name + "dihedral.txt"
            nx, ny, _ = P0.shape   # Get IC (512, 512)
            ng = 3
            cores = 8
            loop_times = 5
            R = np.zeros((nx,ny,2))

            # Get inclination
            test1 = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
            test1.linear_main("inclination")
            P = test1.get_P()
            print('loop_times = ' + str(test1.loop_times))
            print('running_time = %.2f' % test1.running_time)
            print('running_core time = %.2f' % test1.running_coreTime)
            print('total_errors = %.2f' % test1.errors)
            print('per_errors = %.3f' % test1.errors_per_site)
            print("Inclinatin calculation done")

            # Calculate the tanget for corresponding triple junction
            triple_coord, triple_angle, triple_grain = output_tangent.calculate_tangent(P0[:,:,0], loop_times)
            print("tanget calculation done")
            # The triple_coord is the coordination of the triple junction(left-upper voxel), axis 0 is the index of triple, axis 1 is the coordination (i,j,k)
            # The triple_angle saves the three dihedral angle for corresponding triple junction,
            #   axis 0 is the index of triple,
            #   axis 1 is the three dihedral angle
            # The triple_grain saves the sequence of three grains for corresponding triple point
            #   axis 0 is the index of the triple,
            #   axis 1 is the three grains
            print(triple_grain)
            print(triple_angle)

            output_dihedral_angle(npy_file_folder + output_dihedral_name, triple_coord, triple_angle, triple_grain)
            print("Dihedral angle outputted")

            # Find the max/min dihedral and average them
            sum_grain_dihedral = 0
            sum_dihedral_num = 0
            specific_grain = 1
            # For max dihedral
            for i in range(len(triple_angle)):
                if (np.sum(triple_angle[i]) - 360) > 5: continue
                # if np.max(triple_angle[i]) > 180: continue
                
                # Get the angle in grain 1 and ignore angle not in grain 1
                print("specific angle: " + str(triple_angle[i][int(np.argwhere(triple_grain[i]==specific_grain))]))
                try:
                    sum_grain_dihedral += triple_angle[i][int(np.argwhere(triple_grain[i]==specific_grain))]
                    sum_dihedral_num += 1
                except:
                    continue
            if sum_dihedral_num == 0: average_max_dihedral = 0
            else: average_max_dihedral = sum_grain_dihedral / sum_dihedral_num
            print(f"The average dihedral angle on grain {specific_grain} is {average_max_dihedral}")
            print("Average dihedral angle obtained")
            max_dihedral_list_lin[timestep] = average_max_dihedral        
    # Save the data
    np.save(npy_file_folder + dihedral_over_time_data_name, max_dihedral_list_lin)
    
    return max_dihedral_list_lin
    

if __name__ == '__main__':

    npy_file_folder = "/Users/lin/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_triple_for_TJE/results/"
    TJ_energy_type_cases = ["ave", "sum", "consMin", "consMax","consTest"]

    # Joseph method
    file_path_joseph = "/Users/lin/Dropbox (UFL)/UFdata/Dihedral_angle/TJ_IC_11152023/Results/"
    TJ_energy_type_cases_joseph = ["t_dihedrals_3.npy", "t_dihedrals_0.npy", "t_dihedrals_5.npy", "t_dihedrals_2.npy", "t_dihedrals_1.npy"]


    for index, energy_type in enumerate(TJ_energy_type_cases):

        base_name = f"dihedral_results/triple_{energy_type}_"
        dihedral_over_time_figure_name = "triple_dihedral_over_time_" + energy_type + ".png"
        num_steps = 61

        # Get data from Joseph method
        dihedral_over_time = dihedral_angle_from_Joseph(file_path_joseph + TJ_energy_type_cases_joseph[index], num_steps)
        dihedral_over_time[np.isnan(dihedral_over_time[:])]=120
        dihedral_over_time_smooth = data_smooth(dihedral_over_time, 10)
        # dihedral_over_time_smooth = np.ones(num_steps)*np.average(dihedral_over_time_update) # just average it

        # Get data from Lin method
        # dihedral_over_time = dihedral_angle_from_Lin(npy_file_folder, base_name, energy_type, num_steps)
        # dihedral_over_time_smooth = data_smooth(dihedral_over_time, 10)
        # dihedral_over_time_smooth = np.ones(num_steps)*np.average(dihedral_over_time) # just average it
        

        
        plt.clf()
        plt.plot(np.linspace(0,(num_steps-1)*25,num_steps), dihedral_over_time, '.', markersize=4, label = "average angle")
        plt.plot(np.linspace(0,(num_steps-1)*25,num_steps), dihedral_over_time_smooth, '-', linewidth=2, label = "fit")
        # plt.plot(np.linspace(0,(num_steps-1)*100,num_steps), [145.46]*num_steps, '--', linewidth=2, label = "equilibrium from GB area") # Max-100
        # plt.plot(np.linspace(0,160*100,161), [45.95]*161, '--', linewidth=2, label = "expected angle results") # Min-010
        plt.ylim([80,140])
        plt.xlim([0,1500])
        plt.legend(fontsize=20, loc='upper right')
        plt.xlabel("Timestep (MCS)", fontsize=20)
        plt.ylabel(r"Angle ($\degree$)", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xticks([0, 300, 600, 900, 1200, 1500])
        plt.savefig(npy_file_folder + dihedral_over_time_figure_name, bbox_inches='tight', format='png', dpi=400)

