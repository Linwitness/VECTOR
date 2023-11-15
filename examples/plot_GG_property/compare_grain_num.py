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
    
    csv_file_name_iso = "grain_num_iso.csv"
    csv_file_name_aniso = "grain_num_aniso.csv"
    
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
    csv_file_iso_step = []
    csv_file_iso_grain_num = []
    # npy_file_folder = "/Users/lin/Downloads/"
    with open(npy_file_folder + csv_file_name_iso, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            csv_file_iso_step.append(float(row[0]))
            csv_file_iso_grain_num.append(float(row[1])*1e4)
            
    csv_file_aniso_step = []
    csv_file_aniso_grain_num = []
    with open(npy_file_folder + csv_file_name_aniso, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            csv_file_aniso_step.append(float(row[0]))
            csv_file_aniso_grain_num.append(float(row[1])*1e4)
            
    csv_file_iso_step = np.array(csv_file_iso_step)
    csv_file_iso_grain_num = np.array(csv_file_iso_grain_num)
    csv_file_aniso_step = np.array(csv_file_aniso_step)
    csv_file_aniso_grain_num = np.array(csv_file_aniso_grain_num)
        
    
    # Get MCP grain num
    initial_grain_num = 20000
    step_num = npy_file_aniso_ave.shape[0]
    
    grain_num_MCP_iso = np.zeros(step_num)
    grain_num_MCP_ave = np.zeros(step_num)
    for i in range(step_num):
        grain_num_MCP_iso[i] = len(list(set(npy_file_iso[i].reshape(-1))))
        grain_num_MCP_ave[i] = len(list(set(npy_file_aniso_ave[i].reshape(-1))))
        
    # Plot
    scaling_parameter = 0.4
    plt.clf()
    plt.plot(csv_file_iso_step*scaling_parameter, csv_file_iso_grain_num, label="Iso - PF", linewidth=2)
    plt.plot(csv_file_aniso_step*scaling_parameter, csv_file_aniso_grain_num, label="Aniso - PF", linewidth=2)
    plt.plot(np.linspace(0,(step_num-1)*30,step_num), grain_num_MCP_iso, label="Iso - MCP", linewidth=2)
    plt.plot(np.linspace(0,(step_num-1)*30,step_num), grain_num_MCP_ave, label="Aniso - MCP", linewidth=2)
    
    plt.xlabel("Time step (MCS)", fontsize=20)
    plt.ylabel("Grain number (-)", fontsize=20)
    plt.legend(fontsize=20)
    plt.ylim([10,20000])
    plt.xlim([0,4000])
    plt.yscale('log')
    plt.savefig(npy_file_folder + "/size_figure/grain_num_MCP_PF.png", dpi=400,bbox_inches='tight')
    
    
    
    
    