#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 17:58:03 2022

@author: Lin
"""

import os
current_path = os.getcwd()+'/'
import sys
sys.path.append(current_path)
sys.path.append(current_path+'../../')
import numpy as np
import math
import matplotlib.pyplot as plt

def plot_test2D():
    data_num = 19
    
    file1 = 'BL_Curvature_R5_Iteration_1_20.npz'
    file2 = 'BL_Curvature_R20_Iteration_1_20.npz'
    file3 = 'BL_Curvature_R50_Iteration_1_20.npz'
    file4 = 'BL_Curvature_R80_Iteration_1_20.npz'
    function_name = 'BL_errors'
    
    r5 = np.load(file1)
    r20 = np.load(file2)
    r50 = np.load(file3)
    r80 = np.load(file4)
    
    fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    
    plt.plot(range(data_num),(r5[function_name][:data_num])/(1/5), label = "r=05")
    plt.plot(range(data_num),(r20[function_name][:data_num])/(1/20),label = "r=20")
    plt.plot(range(data_num),(r50[function_name][:data_num])/(1/50),label = "r=50")
    plt.plot(range(data_num),(r80[function_name][:data_num])/(1/80),label = "r=80")
    # plt.plot([-0.1,data_num],[0.1,0.1],'--')
    # plt.plot([-0.1,data_num],[0.5,0.5],'--')
    plt.legend()
    plt.ylim([0,6])
    plt.xlim([-0.1,20])
    
def plot_VT_test2D():
    data_num = 19
    
    file1 = 'VT_Curvature_R5_Iteration_1_20.npz'
    file2 = 'VT_Curvature_R20_Iteration_1_20.npz'
    file3 = 'VT_Curvature_R50_Iteration_1_20.npz'
    file4 = 'VT_Curvature_R80_Iteration_1_20.npz'
    function_name = 'VT_errors'
    
    r5 = np.load(file1)
    r20 = np.load(file2)
    r50 = np.load(file3)
    r80 = np.load(file4)
    
    fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    
    plt.plot(range(data_num),(r5[function_name][:data_num])/(1/5), label = "r=05")
    plt.plot(range(data_num),(r20[function_name][:data_num])/(1/20),label = "r=20")
    plt.plot(range(data_num),(r50[function_name][:data_num])/(1/50),label = "r=50")
    plt.plot(range(data_num),(r80[function_name][:data_num])/(1/80),label = "r=80")
    # plt.plot([-0.1,data_num],[0.1,0.1],'--')
    # plt.plot([-0.1,data_num],[0.5,0.5],'--')
    plt.legend()
    plt.ylim([0,6])
    plt.xlim([-0.1,20])
    
def plot_test3D():
    data_num = 19
    
    file1 = 'BL3D_Curvature_R5_Iteration_1_20.npz'
    file2 = 'BL3D_Curvature_R20_Iteration_1_20.npz'
    file3 = 'BL3D_Curvature_R50_Iteration_1_20.npz'
    # file4 = 'BL_Curvature_R80_Iteration_1_20.npz'
    function_name = 'BL_errors'
    
    r5 = np.load(file1)
    r20 = np.load(file2)
    r50 = np.load(file3)
    # r80 = np.load(file4)
    
    fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    
    plt.plot(range(data_num),(r5[function_name][:data_num])/(1/5), label = "r=05", color="royalblue")
    plt.plot(range(data_num),(r20[function_name][:data_num])/(1/20),label = "r=20", color="orange")
    plt.plot(range(data_num),(r50[function_name][:data_num])/(1/50),label = "r=50", color="green")
    # plt.plot(range(data_num),(r80[function_name][:data_num])/(1/80),label = "r=80")
    plt.plot([-0.1,data_num],[abs(r5_vv-0.2)/0.2,abs(r5_vv-0.2)/0.2],'--', color="royalblue")
    plt.plot([-0.1,data_num],[abs(r20_vv-0.05)/0.05,abs(r20_vv-0.05)/0.05],'--', color="orange")
    plt.plot([-0.1,data_num],[abs(r50_vv-0.02)/0.02,abs(r50_vv-0.02)/0.02],'--', color="green")
    plt.legend()
    plt.yscale('log')
    plt.ylim([0,2])
    plt.xlim([-0.1,20])
    
def plot_VT_test3D():
    data_num = 19
    
    file1 = 'VT3D_Curvature_R5_Iteration_1_20.npz'
    file2 = 'VT3D_Curvature_R20_Iteration_1_20.npz'
    file3 = 'VT3D_Curvature_R50_Iteration_1_20.npz'
    file4 = 'VT3D_Curvature_R80_Iteration_1_20.npz'
    file5 = 'VT3D_Curvature_R2_Iteration_1_20.npz'
    file6 = 'VT3D_Curvature_R1_Iteration_1_20.npz'
    function_name = 'VT_errors'
    
    r5 = np.load(file1)
    r20 = np.load(file2)
    r50 = np.load(file3)
    r80 = np.load(file4)
    r2 = np.load(file5)
    r1 = np.load(file6)
    
    fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    
    plt.plot(range(data_num),(r5[function_name][:data_num])/(1/5), label = "r=05", color="royalblue")
    plt.plot(range(data_num),(r20[function_name][:data_num])/(1/20),label = "r=20", color="orange")
    plt.plot(range(data_num),(r50[function_name][:data_num])/(1/50),label = "r=50", color="green")
    plt.plot(range(data_num),(r80[function_name][:data_num])/(1/80),label = "r=80", color="purple")
    plt.plot(range(data_num),(r2[function_name][:data_num])/(1/2),label = "r=2", color="red")
    plt.plot(range(data_num),(r1[function_name][:data_num])/(1),label = "r=1", color="gray")
    plt.plot([-0.1,data_num],[abs(r5_vv-0.2)/0.2,abs(r5_vv-0.2)/0.2],'--', color="royalblue")
    plt.plot([-0.1,data_num],[abs(r20_vv-0.05)/0.05,abs(r20_vv-0.05)/0.05],'--', color="orange")
    plt.plot([-0.1,data_num],[abs(r50_vv-0.02)/0.02,abs(r50_vv-0.02)/0.02],'--', color="green")
    plt.plot([-0.1,data_num],[abs(r80_vv-0.0125)/0.0125,abs(r80_vv-0.0125)/0.0125],'--', color="purple")
    plt.plot([-0.1,data_num],[abs(r2_vv-0.5)/0.5,abs(r2_vv-0.5)/0.5],'--', color="red")
    plt.plot([-0.1,data_num],[abs(r1_vv-1)/1,abs(r1_vv-1)/1],'--', color="gray")
    
    plt.legend()
    plt.yscale('log')
    plt.ylim([0,200])
    plt.xlim([-0.1,20])
    
    
r1_vv = 1.570796333
r2_vv = 0.523598778
r5_vv = 0.204886473
r20_vv = 0.049205668
r50_vv = 0.019873334
r80_vv = 0.012444896

# plot_VT_test2D()
plot_test3D()
# plot_VT_test3D()