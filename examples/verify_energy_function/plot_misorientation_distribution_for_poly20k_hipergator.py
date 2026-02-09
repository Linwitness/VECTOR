#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================
VECTOR Framework: Misorientation Distribution Analysis for Energy Function Verification
=========================================================================================

Scientific Application: Crystallographic Misorientation Analysis for Energy Function Validation
Primary Focus: Statistical Misorientation Distribution Analysis in Polycrystalline Systems

Created on Mon Jul 31 14:33:57 2023
@author: Lin

=========================================================================================
CRYSTALLOGRAPHIC MISORIENTATION ANALYSIS FRAMEWORK
=========================================================================================

This Python script implements comprehensive misorientation distribution analysis for
energy function verification in large-scale polycrystalline Monte Carlo simulations.
The analysis focuses on crystallographic texture evolution, grain boundary character
distribution, and energy function effects on crystallographic misorientation statistics.

Scientific Objectives:
- Crystallographic texture analysis through misorientation distribution characterization
- Energy function validation via grain boundary character distribution comparison
- Statistical analysis of crystallographic orientation evolution in polycrystalline systems
- Verification of energy function effects on crystallographic texture development
- Large-scale HiPerGator dataset analysis for statistical significance

Key Features:
- Quaternion-based crystallographic orientation analysis with cubic symmetry operations
- Comprehensive misorientation angle distribution calculation and statistical validation
- Multi-energy function comparison (anisotropic, isotropic, well energy formulations)
- High-resolution misorientation angle binning for detailed crystallographic analysis
- Publication-quality visualization with statistical significance assessment

Applications:
- Energy function verification through crystallographic texture analysis
- Grain boundary character distribution studies for materials science applications
- Statistical crystallographic analysis for polycrystalline materials research
- Verification of energy function effects on crystallographic texture evolution
- Large-scale simulation validation with HiPerGator computational resources
"""

# ================================================================================
# ENVIRONMENT SETUP AND PATH CONFIGURATION
# ================================================================================
import os
import sys
from examples.shared.path_setup import setup_vector_path
setup_vector_path()

import numpy as np                               # Numerical computing and array operations
from numpy import seterr                         # Numerical error handling configuration
seterr(all='raise')                             # Raise exceptions for numerical errors
import matplotlib.pyplot as plt                  # Advanced scientific visualization
import math                                      # Mathematical functions for calculations
from tqdm import tqdm                            # Progress bar for long-running operations

# ================================================================================
# VECTOR FRAMEWORK INTEGRATION: SPECIALIZED ANALYSIS MODULES
# ================================================================================
import myInput                                   # Input parameter management and file handling
import post_processing                           # Core post-processing functions for crystallographic analysis
import PACKAGE_MP_3DLinear as linear3d          # 3D linear algebra for crystallographic operations

# Crystallographic utilities from shared module
from examples.shared.crystallographic_utils import (
    euler2quaternion, symquat, quat_Multi, quaternions,
    pre_operation_misorientation, multiP_calM
)

# ================================================================================
# STATISTICAL ANALYSIS FUNCTIONS FOR MISORIENTATION CHARACTERIZATION
# ================================================================================

def simple_magnitude(freqArray):
    """
    Statistical Magnitude Calculation for Misorientation Distribution Analysis
    
    Calculates statistical measures of misorientation distribution deviation from random
    distribution for energy function comparison and validation.
    
    Scientific Purpose:
    - Quantifies crystallographic texture strength through distribution analysis
    - Provides statistical measures for energy function effect characterization
    - Enables comparative analysis of texture evolution under different energy formulations
    
    Parameters:
    -----------
    freqArray : numpy.ndarray
        Frequency distribution array for misorientation angles
        
    Returns:
    --------
    magnitude_ave : float
        Average magnitude of deviation from random distribution
    magnitude_stan : float  
        Standard deviation of magnitude distribution for statistical significance
    """
    
    # Misorientation angle analysis range (0-360 degrees for comprehensive coverage)
    xLim = [0, 360]                              # Full crystallographic rotation range
    binValue = 10.01                            # Bin width for misorientation angle analysis
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)   # Calculate number of analysis bins
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Bin center coordinates

    # Generate random distribution baseline for comparison
    freqArray_circle = np.ones(binNum)           # Uniform random distribution
    freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)  # Normalize to probability density

    # Calculate statistical magnitude measures for texture quantification
    magnitude_max = np.max(abs(freqArray - freqArray_circle))/np.average(freqArray_circle)
    magnitude_ave = np.average(abs(freqArray - freqArray_circle))/np.average(freqArray_circle)

    # Statistical standard deviation for significance assessment
    magnitude_stan = np.sqrt(np.sum((abs(freqArray - freqArray_circle)/np.average(freqArray_circle) - magnitude_ave)**2)/binNum)

    return magnitude_ave, magnitude_stan

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

        if (sites_num_list[i] < 500) or \
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

    aspect_ratio_i_list = [[] for _ in range(int(num_grains))]
    aspect_ratio_j_list = [[] for _ in range(int(num_grains))]
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            grain_id = int(table[i][j]-1)
            sites_num_list[grain_id] +=1
            aspect_ratio_i_list[grain_id].append(coord_refer_i[i][j])
            aspect_ratio_j_list[grain_id].append(coord_refer_j[i][j])

    for i in range(num_grains):
        aspect_ratio_i[i, 0] = len(list(set(aspect_ratio_i_list[i])))
        aspect_ratio_j[i, 1] = len(list(set(aspect_ratio_j_list[i])))
        if aspect_ratio_j[i, 1] == 0: aspect_ratio[i] = 0
        else: aspect_ratio[i] = aspect_ratio_i[i, 0] / aspect_ratio_j[i, 1]

    aspect_ratio = np.sum(aspect_ratio * sites_num_list) / np.sum(sites_num_list)

    return aspect_ratio

def get_normal_vector(grain_structure_figure_one, grain_num):
    nx = grain_structure_figure_one.shape[0]
    ny = grain_structure_figure_one.shape[1]
    ng = np.max(grain_structure_figure_one)
    cores = 32
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

    plt.plot(np.append(xCor,xCor[0])/180*math.pi, np.append(freqArray,freqArray[0]), linewidth=2, label=para_name)

    # fitting
    fit_coeff = np.polyfit(xCor, freqArray, 1)
    return freqArray


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


def get_line(i, j):
    """
    Efficient Row Index Calculation for Symmetric Misorientation Matrix
    
    Calculates the row index for grain pair (i,j) in symmetric misorientation energy matrix.
    This function optimizes memory usage by storing only the upper triangular portion
    of the symmetric misorientation matrix.
    
    Scientific Purpose:
    - Efficient crystallographic misorientation data storage and retrieval
    - Optimized memory management for large-scale polycrystalline simulations
    - Fast lookup for grain boundary energy calculations
    
    Parameters:
    -----------
    i, j : int
        Grain identifiers for misorientation pair calculation
        
    Returns:
    --------
    int : Row index in symmetric misorientation matrix for efficient data access
    """
    if i < j: return i+(j-1)*(j)/2
    else: return j+(i-1)*(i)/2

if __name__ == '__main__':
    # ================================================================================
    # HIPERGATOR DATASET CONFIGURATION FOR LARGE-SCALE ANALYSIS
    # ================================================================================
    """
    HiPerGator Computational Infrastructure Setup:
    - Large-scale polycrystalline simulation data from University of Florida's supercomputer
    - Multi-core parallel processing results for statistical significance
    - Comprehensive crystallographic orientation dataset analysis
    """
    
    # HiPerGator data directory structure for organized large-scale data management
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_poly_fully/results/"
    init_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_poly/IC/"
    
    # Initial configuration and simulation result files
    init_file_name = "poly_IC150_1k.init"        # Initial polycrystalline configuration
    npy_file_name_aniso = f"p_ori_fully_aveE_150_1k_multiCore64_J1_refer_1_0_0_seed56689_kt1.95.npy"

    # ================================================================================
    # TEMPORAL EVOLUTION ANALYSIS: OPTIMAL TIME STEP IDENTIFICATION
    # ================================================================================
    """
    Time Step Selection for Misorientation Analysis:
    - Identifies simulation time steps with target grain count for statistical validity
    - Ensures comparable microstructural states for crystallographic analysis
    - Optimizes statistical sampling for misorientation distribution characterization
    """
    
    # Load anisotropic energy function simulation results
    npy_file_aniso = np.load(npy_file_folder + npy_file_name_aniso)
    step_num = npy_file_aniso.shape[0]           # Total simulation time steps
    grain_num_aniso = np.zeros(step_num)         # Grain count evolution tracking
    
    # Calculate grain count evolution for optimal time step selection
    for i in tqdm(range(step_num)):
        grain_num_aniso[i] = len(set(npy_file_aniso[i,:].flatten()))
        
    expected_grain_num = 200                     # Target grain count for statistical analysis
    special_step_distribution_ave = int(np.argmin(abs(grain_num_aniso - expected_grain_num)))
    print("> Step calculation done")

    # ================================================================================
    # CRYSTALLOGRAPHIC SYMMETRY SETUP FOR MISORIENTATION CALCULATION
    # ================================================================================
    """
    Cubic Crystallographic Symmetry Operations:
    - 24-fold cubic symmetry operations for crystallographic misorientation calculation
    - Quaternion-based orientation representation for computational efficiency
    - Pre-computed symmetry matrices for optimized misorientation analysis
    """
    
    # misorientation
    grain_num = 20000                            # Total number of grains in large-scale simulation
    Osym = 24                                    # Cubic crystal symmetry operations (24-fold)
    
    # Pre-compute crystallographic symmetry operations and quaternion matrices
    symm2quat_matrix, quartAngle = pre_operation_misorientation(grain_num, init_file_folder + init_file_name, Osym)
    num_bin = 100                               # High-resolution binning for misorientation analysis
    misorientation_angle_list = np.zeros(num_bin)  # Initialize misorientation distribution storage
    print("> Pre-work done")

    # ================================================================================
    # GRAIN BOUNDARY MISORIENTATION ANALYSIS
    # ================================================================================
    """
    Comprehensive 3D Grain Boundary Misorientation Calculation:
    - Analyzes all grain boundary interfaces in 3D microstructure
    - Calculates crystallographic misorientation angles with cubic symmetry
    - Builds statistical misorientation distribution for energy function validation
    """
    
    # Get dict
    miso_dict = dict()                          # Misorientation angle cache for computational efficiency
    microstructure = npy_file_aniso[special_step_distribution_ave,:]  # Extract microstructure at target time step
    nx,ny,nz = microstructure.shape             # 3D microstructure dimensions
    
    # Comprehensive 3D grain boundary interface analysis
    for i in tqdm(range(nx)):
        for j in range(ny):
            for k in range(nz):
                # Apply periodic boundary conditions for comprehensive interface analysis
                ip,im,jp,jm,kp,km = myInput.periodic_bc3d(nx,ny,nz,i,j,k)
                
                # Identify grain boundary sites through neighbor comparison
                if ( ((microstructure[ip,j,k]-microstructure[i,j,k])!=0) or ((microstructure[im,j,k]-microstructure[i,j,k])!=0) or\
                     ((microstructure[i,jp,k]-microstructure[i,j,k])!=0) or ((microstructure[i,jm,k]-microstructure[i,j,k])!=0) or\
                     ((microstructure[i,j,kp]-microstructure[i,j,k])!=0) or ((microstructure[i,j,km]-microstructure[i,j,k])!=0) ):
                    
                    # Extract grain boundary interface information
                    central_site = microstructure[i,j,k]
                    neighboring_sites_list = [microstructure[ip,j,k], microstructure[i,jp,k], microstructure[i,j,kp], microstructure[im,j,k], microstructure[i,jm,k], microstructure[i,j,km]]
                    neighboring_sites_set = set(neighboring_sites_list).remove(central_site)
                    print(f"center: {central_site}, neighbor: {neighboring_sites_list}")
                    neighboring_sites_list_unque = list(neighboring_sites_set)
                    
                    # Calculate misorientation for all unique grain pairs at interface
                    for m in range(len(neighboring_sites_list_unque)):
                        pair_id = get_line(central_site, neighboring_sites_list_unque[m]) # get pair id
                        
                        # Calculate or extract misorientation with caching for efficiency
                        if pair_id in miso_dict:
                            misorientation_angle = miso_dict[pair_id]
                        else:
                            misorientation = multiP_calM([central_site, neighboring_sites_list_unque[m]], quartAngle, symm2quat_matrix, Osym)
                            misorientation_angle = misorientation[0] # miso angle
                            miso_dict[pair_id] = misorientation_angle

                        # Get misorientation angle distribution
                        if misorientation_angle < 0 or misorientation_angle > np.pi: print(">>> Please check the miaorientation angle calculation!") # check angle error
                        misorientation_index = int(misorientation_angle//(np.pi/num_bin)) # get angle index to check the angle distribution
                        if misorientation_index == num_bin: misorientation_angle_list[-1] += 1 # for angle of pi as 101 bins to assign to 100 bins
                        else: misorientation_angle_list[misorientation_index] += 1 # all bins
    print(f"> Misorientation dictionary (len:{len(miso_dict)}) and misorientation angle distribution done")

    # plot misorientation angle distribution
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    plt.plot(np.linspace(0,180,num_bin), misorientation_angle_list, label="misorientation angle", linewidth=2)
    # plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.xlabel(r"Misorientation Angle ($\^circ$)", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.savefig(current_path + f"/figures/misorientation_distribution_3d_{expected_grain_num}grains.png", dpi=400,bbox_inches='tight')

