#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:44:12 2023

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
import csv

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
    
    csv_file_name_3705 = "grain_size_distribution_3705.csv"
    csv_file_name_hillert = "grain_size_distribution_Hillert.csv"
    
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
    
    # Get PF grain num
    csv_file_3705_r = []
    csv_file_3705_frequency = []
    # npy_file_folder = "/Users/lin/Downloads/"
    with open(npy_file_folder + csv_file_name_3705, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            csv_file_3705_r.append(float(row[0]))
            csv_file_3705_frequency.append(float(row[1]))
            
    csv_file_hillert_r = []
    csv_file_hillert_frequency = []
    with open(npy_file_folder + csv_file_name_hillert, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            csv_file_hillert_r.append(float(row[0]))
            csv_file_hillert_frequency.append(float(row[1]))
            
    csv_file_3705_r = np.array(csv_file_3705_r)
    csv_file_3705_frequency = np.array(csv_file_3705_frequency)
    csv_file_hillert_r = np.array(csv_file_hillert_r)
    csv_file_hillert_frequency = np.array(csv_file_hillert_frequency)
    
    # # Normalized distribution
    # PF_bin_width = csv_file_3705_r - np.hstack((np.array(0),csv_file_3705_r[:-1]))
    # PF_area = np.sum(csv_file_3705_frequency*PF_bin_width)
    # Hillert_bin_width = csv_file_hillert_r - np.hstack((np.array(0),csv_file_hillert_r[:-1]))
    # Hillert_area = np.sum(csv_file_hillert_frequency*Hillert_bin_width)
    
        
    
    # Get MCP grain num
    step_num = npy_file_aniso_ave.shape[0]
    
    grain_num_MCP_iso = np.zeros(step_num)
    grain_num_MCP_ave = np.zeros(step_num)
    for i in range(step_num):
        grain_num_MCP_iso[i] = len(list(set(npy_file_iso[i].reshape(-1))))
        grain_num_MCP_ave[i] = len(list(set(npy_file_aniso_ave[i].reshape(-1))))
        
    special_time_step = np.argmin(abs(grain_num_MCP_ave - 3705))
    special_time_step_grain_num = grain_num_MCP_ave[special_time_step]
    special_time_step_iso = np.argmin(abs(grain_num_MCP_iso - 3705))
    special_time_step_grain_num_iso = grain_num_MCP_iso[special_time_step_iso]
    
    
    # Start grain size initialing
    if os.path.exists(npy_file_folder + grain_size_data_name_ave):
        grain_area_ave = np.load(npy_file_folder + grain_size_data_name_ave)
    if os.path.exists(npy_file_folder + grain_size_data_name_iso):
        grain_area_iso = np.load(npy_file_folder + grain_size_data_name_iso)
        
    # Start grain size analysis
    grain_size_ave = (grain_area_ave[special_time_step] / np.pi)**0.5
    grain_ave_size_ave = np.sum(grain_size_ave) / special_time_step_grain_num # average grain size
    grain_size_iso = (grain_area_iso[special_time_step_iso] / np.pi)**0.5
    grain_ave_size_iso = np.sum(grain_size_iso) / special_time_step_grain_num_iso # average grain size
    
    # Get final size dsitribution
    bin_width = 0.16 # Grain size distribution
    x_limit = [-0.5, 3.5]
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)
    grain_size_distribution_ave = np.zeros(bin_num)
    grain_size_distribution_iso = np.zeros(bin_num)
    
    special_size_ave = grain_size_ave[grain_size_ave != 0] # remove zero grain size
    special_size_ave = special_size_ave/grain_ave_size_ave # normalize grain size
    for j in range(len(special_size_ave)):
        grain_size_distribution_ave[int((special_size_ave[j]-x_limit[0])/bin_width)] += 1 # Get frequency
    special_size_iso = grain_size_iso[grain_size_iso != 0] # remove zero grain size
    special_size_iso = special_size_iso/grain_ave_size_iso # normalize grain size
    for j in range(len(special_size_iso)):
        grain_size_distribution_iso[int((special_size_iso[j]-x_limit[0])/bin_width)] += 1 # Get frequency
    
    
        
    # Plot
    plt.clf()
    plt.plot(csv_file_3705_r, csv_file_3705_frequency, label="3D aniso PF - 3705 grains", linewidth=2)
    plt.plot(csv_file_hillert_r, csv_file_hillert_frequency, label="3D Hillert", linewidth=2)
    plt.plot(size_coordination, grain_size_distribution_ave, label=f"2D aniso MCP - {int(special_time_step_grain_num)} grains", linewidth=2)
    plt.plot(size_coordination, grain_size_distribution_iso, label=f"2D iso MCP - {int(special_time_step_grain_num_iso)} grains", linewidth=2)
    
    plt.xlabel(r"R/$\langle$R$\rangle$", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.legend(fontsize=20)
    plt.ylim([0,1.5])
    plt.xlim(x_limit)
    plt.savefig(npy_file_folder + "/size_figure/grain_size_distribution_MCP_PF.png", dpi=400,bbox_inches='tight')
    
    
    
    
    