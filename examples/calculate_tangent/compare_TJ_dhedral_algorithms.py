#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 16:51:24 2022

@author: lin.yang
"""

import numpy as np
from numpy import seterr
seterr(all='raise')
import math
import myInput
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def func(x, a, b, c):
    return a * np.exp(-x / b) + c

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

def find_normal_structure(P,i,j,iteration,refer_id):
    smoothed_vector_i, smoothed_vector_j = myInput.output_linear_vector_matrix(iteration)

    a = (np.sum(find_window(P,i,j,iteration,refer_id) * smoothed_vector_i) +
        np.sum(find_window(P,i+1,j,iteration,refer_id) * smoothed_vector_i) +
        np.sum(find_window(P,i,j+1,iteration,refer_id) * smoothed_vector_i) +
        np.sum(find_window(P,i+1,j+1,iteration,refer_id) * smoothed_vector_i)) / 4
    b = (np.sum(find_window(P,i,j,iteration,refer_id) * smoothed_vector_j) +
        np.sum(find_window(P,i+1,j,iteration,refer_id) * smoothed_vector_j) +
        np.sum(find_window(P,i,j+1,iteration,refer_id) * smoothed_vector_j) +
        np.sum(find_window(P,i+1,j+1,iteration,refer_id) * smoothed_vector_j)) / 4
    return a, b

def find_normal(P,i,j,nei_flat,iteration):
    # Calculate the nomals for all the four voxels in the triple junction
    nx,ny=P.shape
    tri_norm = np.zeros((4,2))
    tri_grains = np.zeros(3)

    if nei_flat[0] == nei_flat[1]:

        tri_norm[0,0], tri_norm[0,1] = find_normal_structure(P,i,j,iteration,P[i,j])
        tri_norm[1,0], tri_norm[1,1] = find_normal_structure(P,i,j,iteration,P[i,j+1])
        tri_norm[2,0], tri_norm[2,1] = find_normal_structure(P,i,j,iteration,P[i+1,j+1])
        tri_norm[3,0], tri_norm[3,1] = find_normal_structure(P,i,j,iteration,P[i+1,j])
        tri_grains = np.array([P[i,j+1], P[i+1,j+1], P[i+1,j]])


    elif nei_flat[0] == nei_flat[2]:

        tri_norm[0,0], tri_norm[0,1] = find_normal_structure(P,i,j,iteration,P[i+1,j])
        tri_norm[1,0], tri_norm[1,1] = find_normal_structure(P,i,j,iteration,P[i,j])
        tri_norm[2,0], tri_norm[2,1] = find_normal_structure(P,i,j,iteration,P[i,j+1])
        tri_norm[3,0], tri_norm[3,1] = find_normal_structure(P,i,j,iteration,P[i+1,j+1])
        tri_grains = np.array([P[i,j], P[i,j+1], P[i+1,j+1]])

    elif nei_flat[2] == nei_flat[3]:

        tri_norm[0,0], tri_norm[0,1] = find_normal_structure(P,i,j,iteration,P[i+1,j+1])
        tri_norm[1,0], tri_norm[1,1] = find_normal_structure(P,i,j,iteration,P[i+1,j])
        tri_norm[2,0], tri_norm[2,1] = find_normal_structure(P,i,j,iteration,P[i,j])
        tri_norm[3,0], tri_norm[3,1] = find_normal_structure(P,i,j,iteration,P[i,j+1])
        tri_grains = np.array([P[i+1,j], P[i,j], P[i,j+1]])

    elif nei_flat[1] == nei_flat[3]:

        tri_norm[0,0], tri_norm[0,1] = find_normal_structure(P,i,j,iteration,P[i,j+1])
        tri_norm[1,0], tri_norm[1,1] = find_normal_structure(P,i,j,iteration,P[i+1,j+1])
        tri_norm[2,0], tri_norm[2,1] = find_normal_structure(P,i,j,iteration,P[i+1,j])
        tri_norm[3,0], tri_norm[3,1] = find_normal_structure(P,i,j,iteration,P[i,j])
        tri_grains = np.array([P[i+1,j+1], P[i+1,j], P[i,j]])

    else:
        print("ERROR: This is not a triple junction!")
        return 0, 0

    for ni in range(4):
        tri_norm[ni] = tri_norm[ni]/np.linalg.norm(tri_norm[ni])


    return tri_norm, tri_grains

def find_angle_tan(each_normal):
    # Find the three tangent depend on the four normals from four voxels
    tri_tang = np.zeros((3,2))
    tri_angle = np.zeros(3)
    clock90 = np.array([[0,-1],[1,0]])
    anti_clock90 = np.array([[0,1],[-1,0]])

    tri_tang[0] = each_normal[0]@clock90
    tri_tang[1] = each_normal[1]@anti_clock90
    tri_tang[2] = -(each_normal[2]+each_normal[3])/np.linalg.norm(each_normal[2]+each_normal[3])

    tri_angle[0] = 180 / np.pi * math.acos(np.dot(tri_tang[0], tri_tang[2]))
    tri_angle[1] = 180 / np.pi * math.acos(np.dot(tri_tang[1], tri_tang[2]))
    tri_angle[2] = 180 / np.pi * math.acos(round(np.dot(tri_tang[0], tri_tang[1]),5))
    if abs(sum(tri_angle) - 360) > 5:
        tri_angle[2] = 360 - tri_angle[2]

    return tri_angle

def find_angle(each_normal):
    tri_angle = np.zeros(3)

    third_normal = (each_normal[0]+each_normal[1])/np.linalg.norm(each_normal[0]+each_normal[1])
    tri_angle[0] = 180 / np.pi * (2 * np.pi - 2 * math.acos(np.dot(each_normal[3], third_normal)))
    tri_angle[1] = 180 / np.pi * (2 * np.pi - 2 * math.acos(np.dot(each_normal[2], third_normal)))

    tri_angle[2] = 180 / np.pi * (2 * np.pi - 2 * math.acos(np.dot(each_normal[2], each_normal[3])))


    # A way to ignore the issue of some special triple angle
    # if abs(sum(tri_angle)-360) > 5:
        # print()
        # print(sum(tri_angle))
        # print(tri_angle)
        # print(each_normal)
        # tri_angle[2] = 360 - tri_angle[0] - tri_angle[1]

    return tri_angle

def read_2d_input(filename,nx,ny):
    # Keep the 2d input macrostructure

    triple_map = np.zeros((nx,ny))
    f = open(filename)
    line = f.readline()
    line = f.readline()
    # print(line)
    while line:
        each_element = line.split()
        i = int(each_element[0])-1
        j = int(each_element[1])-1
        triple_map[i,j] = int(each_element[6])

        line = f.readline()
    f.close()

    return triple_map

def calculate_tangent(triple_map,iteration=5):
    nx, ny = triple_map.shape
    num = 0
    issue_num = 0
    triple_grain = []
    triple_coord = []
    triple_normal = []
    triple_angle = []
    triple_angle_tang = []
    for i in range(nx-1):
        for j in range(ny-1):
            nei = np.zeros((2,2))
            nei = triple_map[i:i+2,j:j+2]
            nei_flat = nei.flatten()
            if len(set(nei_flat)) == 3 and 0 not in nei_flat:

                # print(str(i)+" "+str(j))

                each_normal, grain_sequence = find_normal(triple_map,i,j,nei_flat,iteration) # Get basic normals and grain id sequence
                if isinstance(each_normal,(int, float)): continue
                triple_normal.append(each_normal) # Save the normals
                triple_coord.append(np.array([i,j])) # Save the coordinate of the triple point
                triple_grain.append(grain_sequence) # Save the grain id sequence
                triple_angle.append(find_angle(each_normal)) # Save the 3 dihedral angles

                num += 1 # Get the num of all triple point
                if abs(sum(find_angle(each_normal))-360) > 5: issue_num += 1

    print(f"The number of useful triple junction is {num}")
    if num==0: print("The issue propotion is 0%")
    else: print(f"The issue propotion is {issue_num/num*100}%")

    return np.array(triple_coord), np.array(triple_angle), np.array(triple_grain)



if __name__ == '__main__':

    average_coupled_energy = np.array([0.99858999, 3.05656703, 0.4, 1.6, 0.093]) #0.55020987
    # Joseph resuults
    file_path_joseph = "/Users/lin.yang/Dropbox (UFL)/UFdata/Dihedral_angle/output/"
    npy_file_name_joseph_ave = "hex_dihedrals1.npy" # (161, 6, 96) grain ID's involved (first 3) and the dihedral angles (last 3)
    npy_file_name_joseph_sum = "hex_dihedrals5.npy" # (161, 6, 96) grain ID's involved (first 3) and the dihedral angles (last 3)
    npy_file_name_joseph_consmin = "hex_dihedrals3.npy" # (161, 6, 96) grain ID's involved (first 3) and the dihedral angles (last 3)
    npy_file_name_joseph_consmax = "hex_dihedrals2.npy" # (161, 6, 96) grain ID's involved (first 3) and the dihedral angles (last 3)
    npy_file_name_joseph_constest = "hex_dihedrals_consTest2E.npy" # (161, 6, 96) grain ID's involved (first 3) and the dihedral angles (last 3)
    triple_results_ave = np.load(file_path_joseph + npy_file_name_joseph_ave)
    triple_results_sum = np.load(file_path_joseph + npy_file_name_joseph_sum)
    triple_results_consmin = np.load(file_path_joseph + npy_file_name_joseph_consmin)
    triple_results_consmax = np.load(file_path_joseph + npy_file_name_joseph_consmax)
    triple_results_constest = np.load(file_path_joseph + npy_file_name_joseph_constest)

    # Necessary parameters
    num_grain_initial = 48

    num_steps = 80
    max_dihedral_ave_list = np.zeros(num_steps)
    max_dihedral_sum_list = np.zeros(num_steps)
    max_dihedral_consmin_list = np.zeros(num_steps)
    max_dihedral_consmax_list = np.zeros(num_steps)
    max_dihedral_constest_list = np.zeros(num_steps)
    for i in tqdm(range(num_steps)):
        # From Joseph algorithm
        triple_results_step_ave = triple_results_ave[i,3:6,:]
        triple_results_step_sum = triple_results_sum[i,3:6,:]
        triple_results_step_consmin = triple_results_consmin[i,3:6,:]
        triple_results_step_consmax = triple_results_consmax[i,3:6,:]
        triple_results_step_constest = triple_results_constest[i,3:6,:]

        triple_results_step_ave = triple_results_step_ave[:,~np.isnan(triple_results_step_ave[0,:])]
        triple_results_step_sum = triple_results_step_sum[:,~np.isnan(triple_results_step_sum[0,:])]
        triple_results_step_consmin = triple_results_step_consmin[:,~np.isnan(triple_results_step_consmin[0,:])]
        triple_results_step_consmax = triple_results_step_consmax[:,~np.isnan(triple_results_step_consmax[0,:])]
        triple_results_step_constest = triple_results_step_constest[:,~np.isnan(triple_results_step_constest[0,:])]
        print(f"The number in ave is {len(triple_results_step_ave[0,:])}")
        print(f"The number in sum is {len(triple_results_step_sum[0,:])}")
        print(f"The number in consmin is {len(triple_results_step_consmin[0,:])}")
        print(f"The number in consmax is {len(triple_results_step_consmax[0,:])}")
        print(f"The number in constest is {len(triple_results_step_constest[0,:])}")
        max_dihedral_ave_list[i] = np.mean(np.max(triple_results_step_ave,0))
        max_dihedral_sum_list[i]= np.mean(np.max(triple_results_step_sum,0))
        max_dihedral_consmin_list[i] = np.mean(np.max(triple_results_step_consmin,0))
        max_dihedral_consmax_list[i] = np.mean(np.max(triple_results_step_consmax,0))
        max_dihedral_constest_list[i] = np.mean(np.max(triple_results_step_constest,0))

    # average the max dihedral angle for all time steps
    max_dihedral_ave = np.mean(max_dihedral_ave_list)
    max_dihedral_sum = np.mean(max_dihedral_sum_list)
    max_dihedral_consmin = np.mean(max_dihedral_consmin_list)
    max_dihedral_consmax = np.mean(max_dihedral_consmax_list)
    max_dihedral_constest = np.mean(max_dihedral_constest_list)

    max_dihedral_list = np.array([max_dihedral_ave, max_dihedral_sum, max_dihedral_consmin, max_dihedral_consmax, max_dihedral_constest])


    dihedral_siteEnergy_cases_figure_name = "energy_results/hex_aveDihedral_aveEnergy_" + "figure.png"
    plt.clf()
    plt.plot(average_coupled_energy, max_dihedral_list, 'o', markersize=4, label = "average angle")

    # Fitting
    a = max(max_dihedral_list)-min(max_dihedral_list)
    b = max_dihedral_list[round(len(max_dihedral_list)/2)]
    c = min(max_dihedral_list)
    p0 = [a,b,c]
    popt, pcov = curve_fit(func, average_coupled_energy, max_dihedral_list,p0=p0)
    print(f"The equation to fit the relationship is {round(popt[0],2)} * exp(-x * {round(popt[1],2)}) + {round(popt[2],2)}")
    y_fit = [func(i,popt[0], popt[1], popt[2]) for i in np.linspace(0, 4, 50)]
    plt.plot(np.linspace(0, 4, 50), y_fit, '-', linewidth=2, label = "fitting")
    # Find the root mean square error
    y_fit_rmse = [func(i,popt[0], popt[1], popt[2]) for i in average_coupled_energy]
    data_rmse = (np.mean((y_fit_rmse - max_dihedral_list)**2))**0.5
    data_r2 = r2_score(max_dihedral_list, y_fit_rmse)
    print(rf"The RMSE is {round(data_rmse,3)}$^\circ$, the r$^2$ is {round(data_r2,4)}")
    # Find the exact result
    exact_list = np.linspace(0.01, 0.5, 101)
    min_level = 10
    expect_site_energy = 0
    for m in exact_list:
        if min_level > abs(func(m, popt[0], popt[1], popt[2]) - 145.46):
            min_level = abs(func(m, popt[0], popt[1], popt[2]) - 145.46)
            expect_site_energy = m
    print(f"The expected average TJ site energy is {expect_site_energy}")

    plt.plot(np.linspace(0,4,24), [145.46]*24, '--', linewidth=2, label = "energy equilibrium on GB area") # Max-100

    # My algorithm
    npy_file_folder = "/Users/lin/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_hex_for_TJE/results/"

    # Get the average dihedral angle
    # cases=5
    # cases_name = ["ave", "sum", "consMin", "consMax", "consTest"]
    # max_dihedral_angle_lin = np.zeros(cases)
    # for i in range(cases):
    #     energy_type = cases_name[i]
    #     base_name = f"dihedral_results/hex_{energy_type}_"
    #     dihedral_over_time = np.load(npy_file_folder + base_name + "data.npy")
    #     max_dihedral_angle_lin[i] = np.average(dihedral_over_time[:num_steps])
    # plt.plot(average_coupled_energy, max_dihedral_angle_lin, 'o', markersize=4, label = "average angle (Lin)")
    plt.ylim([110,155])
    plt.xlim([0,4])
    plt.legend(fontsize=14, loc='lower center')
    plt.xlabel("Average TJ energy (J/MCU)", fontsize=14)
    plt.ylabel(r"Angle ($^\circ$)", fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # plt.savefig(dihedral_siteEnergy_cases_figure_name, bbox_inches='tight', format='png', dpi=400)







