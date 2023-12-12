#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 14:55:28 2021

@author: lin.yang
"""

import os
current_path = os.getcwd()
import sys
sys.path.append(current_path)
sys.path.append('../../.')
import numpy as np
import math
from itertools import repeat
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

import myInput
import PACKAGE_MP_Linear as linear2d     #2D Bilinear smooth algorithm


def get_all_gb_list(P0):
    nx, ny = P0.shape
    gagn_gbsites = []
    for i in range(0,nx):
        for j in range(0,ny):
            ip,im,jp,jm = myInput.periodic_bc(nx,ny,i,j)
            if ( ((P0[ip,j]-P0[i,j])!=0) or
                 ((P0[im,j]-P0[i,j])!=0) or
                 ((P0[i,jp]-P0[i,j])!=0) or
                 ((P0[i,jm]-P0[i,j])!=0) ):
                gagn_gbsites.append([i,j])
    return gagn_gbsites

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

def get_normal_vector_slope(P, sites, para_name, bias=None):
    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)

    freqArray = np.zeros(binNum)
    degree = []
    for sitei in sites:
        [i,j] = sitei
        dx,dy = P[1:,i,j]
        degree.append(math.atan2(-dy, dx) + math.pi)

    for i in range(len(degree)):
        freqArray[int((degree[i]/math.pi*180-xLim[0])/binValue)] += 1
    freqArray = freqArray/sum(freqArray*binValue)

    if bias is not None:
        freqArray = freqArray + bias
        freqArray = freqArray/sum(freqArray*binValue)
    # Plot
    plt.plot(np.append(xCor,xCor[0])/180*math.pi, np.append(freqArray,freqArray[0]), linewidth=2, label=para_name)

    return freqArray

def compare_inclination_dsitribution(primme_data, spparks_data, step_list, input_path, output_path):

    for i in step_list:
        npy_file_s20k = np.load(input_path+spparks_data+f"{i}.npy")
        npy_file_p20k = np.load(input_path+primme_data+f"{i}.npy")
        # Initial container
        initial_grain_num = 20000

        # Start polar figure
        plt.close()
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca(projection='polar')

        ax.set_thetagrids(np.arange(0.0, 360.0, 45.0),fontsize=16)
        ax.set_thetamin(0.0)
        ax.set_thetamax(360.0)

        ax.set_rgrids(np.arange(0, 0.008, 0.004))
        ax.set_rlabel_position(0.0)  # 标签显示在0°
        ax.set_rlim(0.0, 0.008)  # 标签范围为[0, 5000)
        ax.set_yticklabels(['0', '4e-3'],fontsize=16)

        ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow('True')

        # spparks
        sites_list = get_all_gb_list(npy_file_s20k[0,:,:])
        slope_list = get_normal_vector_slope(npy_file_s20k, sites_list, r"spparks20k")

        # primme
        sites_list = get_all_gb_list(npy_file_p20k[0,:,:])
        slope_list = get_normal_vector_slope(npy_file_p20k, sites_list, r"primme20k")

        plt.legend(loc=(-0.10,-0.3),fontsize=16,ncol=2)
        plt.savefig(output_path + f"figures/normal_distribution_{primme_data}{i}.png", dpi=400,bbox_inches='tight')


if __name__ == '__main__':

    # 2D initial conditions
    input_folder = "/blue/michael.tonks/share/PRIMME_Inclination_npy_files/"

    input_name = ["Case2AS_T3_tstep_300_600_inclination_step",
                  "Case2BF_T8_tstep_300_1600_inclination_step",
                  "Case2BS_T1_tstep_300_1600_inclination_step",
                  "Case2CF_T4_tstep_300_400_inclination_step",
                  "Case2DF_T1_tstep_300_1600_inclination_step",
                  "Case2DS_T5_tstep_300_800_inclination_step",
                  "Case2DS_T8_tstep_300_1600_inclination_step",
                  "Case3AF_T9_tstep_300_600_inclination_step",
                  "Case3AS_T6_tstep_300_1600_inclination_step",
                  "Case3BF_T7_tstep_300_600_inclination_step",
                  "Case3BS_T10_step_300_1600_inclination_step",
                  "Case3BS_T6_tstep_300_1600_inclination_step",
                  "Case3CS_T7_tstep_300_400_inclination_step"]
    compare_data_name = "spparks_s1_tstep_300_400_600_800_1600_inclination_step"

    step_lists = [[300,600],
             [300,1600],
             [300,1600],
             [300,400],
             [300,1600],
             [300,800],
             [300,1600],
             [300,600],
             [300,1600],
             [300,600],
             [300,1600],
             [300,1600],
             [300,400],
             [300,400,600,800,1600]]

    for i in range(len(input_name)):
        compare_inclination_dsitribution(input_name[i],compare_data_name,step_lists[i],input_folder,input_folder)
        print(f"finish {input_name[i]}")





