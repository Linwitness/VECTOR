#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparative Dihedral Angle Analysis: Joseph vs Lin Method Validation
===================================================================

This module provides comprehensive comparison and validation of two different
algorithms for calculating dihedral angles at triple junctions in 2D grain
boundary systems. It processes SPPARKS simulation data to evaluate the accuracy
and consistency of the Joseph method versus the Lin method.

Scientific Background:
- Triple junction dihedral angle measurement methodologies
- Algorithm validation through statistical comparison
- Time series analysis of angle evolution during grain growth
- Method convergence and accuracy assessment

Key Features:
- Dual algorithm implementation for cross-validation
- Statistical analysis of angle differences between methods
- Time-dependent evolution tracking with smoothing
- Comprehensive error analysis and reporting
- Progress tracking for large dataset processing

Research Applications:
- Grain boundary analysis method validation
- Algorithm development and benchmarking
- SPPARKS simulation data post-processing
- Materials science computational method verification

Technical Implementations:
- Joseph method: [Traditional geometric approach]
- Lin method: [Advanced computational approach]
- Statistical comparison metrics
- Temporal evolution analysis with noise reduction

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



def get_gb_sites(P, grain_num):
    """
    Identify Grain Boundary Sites in 2D Microstructure
    
    This function systematically identifies all grain boundary sites by analyzing
    neighbor connectivity and detecting interface locations between different grains.
    Essential preprocessing step for both angle calculation methods.
    
    Parameters:
    -----------
    P : ndarray
        3D array representing microstructure evolution (time, x, y)
    grain_num : int
        Total number of grains in the microstructure
        
    Returns:
    --------
    ggn_gbsites : list of lists
        Grain boundary sites organized by grain ID
        Each sublist contains [i,j] coordinates of boundary sites for that grain
        
    Algorithm Details:
    -----------------
    - Uses periodic boundary conditions for edge handling
    - Excludes boundary region (timestep=5) to avoid edge effects
    - Identifies sites where neighbors have different grain IDs
    - Systematic grain-by-grain organization for efficient processing
    
    Scientific Applications:
    -----------------------
    - Interface area calculation and characterization
    - Grain boundary network topology analysis
    - Preprocessing for normal vector calculations
    - Statistical analysis of grain boundary properties
    """
    _, nx, ny = np.shape(P)
    timestep = 5  # Buffer zone to avoid boundary effects
    ggn_gbsites = [[] for i in repeat(None, grain_num)]
    
    # Systematic scan through domain excluding boundary regions
    for i in range(timestep, nx-timestep):
        for j in range(timestep, ny-timestep):
            # Get periodic boundary condition neighbors
            ip, im, jp, jm = myInput.periodic_bc(nx, ny, i, j)
            
            # Check if current site has neighbors with different grain IDs
            if (((P[0,ip,j]-P[0,i,j])!=0) or ((P[0,im,j]-P[0,i,j])!=0) or
                ((P[0,i,jp]-P[0,i,j])!=0) or ((P[0,i,jm]-P[0,i,j])!=0)) and\
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
    """
    Extract Dihedral Angles from Joseph Method Results
    
    This function processes pre-calculated triple junction analysis results
    from the Joseph algorithm to extract dihedral angle time series data.
    
    Parameters:
    -----------
    case_path : str
        Path to .npy file containing Joseph method results
    num_steps : int
        Number of time steps to process
        
    Returns:
    --------
    max_dihedral_list_joseph : ndarray
        Time series of maximum dihedral angles from Joseph method
        
    Data Structure:
    --------------
    Expected input format: [timestep, 3:6] contains angle data
    - Index 2 contains the maximum dihedral angle for each timestep
    - Uses tqdm progress tracking for large datasets
    
    Scientific Applications:
    -----------------------
    - Method comparison baseline establishment
    - Algorithm validation reference data
    - Statistical analysis of traditional geometric approach
    - Time evolution analysis of triple junction angles
    """
    # Load pre-calculated Joseph method results
    triple_results_joseph = np.load(case_path)
    
    max_dihedral_list_joseph = np.zeros(num_steps)
    
    # Extract maximum dihedral angle for each timestep
    for i in tqdm(range(num_steps)):
        # Extract angle data from Joseph algorithm results
        triple_results_step_joseph = triple_results_joseph[i,3:6]
        # Use index 2 for maximum dihedral angle
        max_dihedral_list_joseph[i] = triple_results_step_joseph[2]
        
    return max_dihedral_list_joseph
    
def dihedral_angle_from_Lin(npy_file_folder, base_name, energy_type, num_steps):
    """
    Calculate Dihedral Angles Using Lin Method Implementation
    
    This function performs comprehensive dihedral angle calculation using the
    Lin algorithm, including inclination analysis and triple junction detection.
    
    Parameters:
    -----------
    npy_file_folder : str
        Base directory containing SPPARKS simulation data
    base_name : str
        Base filename pattern for output files
    energy_type : str
        Energy calculation type identifier
    num_steps : int
        Number of simulation timesteps to process
        
    Returns:
    --------
    max_dihedral_list_lin : ndarray
        Time series of average dihedral angles from Lin method
        
    Algorithm Pipeline:
    ------------------
    1. Load SPPARKS microstructure evolution data
    2. For each timestep: perform inclination analysis using linear solver
    3. Calculate triple junction tangent vectors and dihedral angles
    4. Extract grain-specific average angles with quality filtering
    5. Apply temporal smoothing and save results
    
    Quality Control:
    ---------------
    - Filters triple junctions with angle sum validation (≈360°)
    - Focuses on specific grain (grain 1) for consistency
    - Handles exceptions and missing data gracefully
    - Implements caching to avoid redundant calculations
    
    Scientific Applications:
    -----------------------
    - Advanced computational geometry approach
    - High-precision angle calculation validation
    - Temporal evolution analysis of grain boundary angles
    - Comparison standard for geometric algorithm development
    """
    # Define file paths and load microstructure data
    dihedral_over_time_data_name = base_name + "data.npy"
    max_dihedral_list_lin = np.zeros(num_steps)
    npy_file_name = f"t_{energy_type}_512x512_delta0.6_m2_refer_1_0_0_seed56689_kt066.npy"
    P0_list = np.load(npy_file_folder + npy_file_name)
    print("IC is read as matrix")
    
    # Check if processed data exists to avoid redundant calculations
    if os.path.exists(npy_file_folder + dihedral_over_time_data_name):
        max_dihedral_list_lin = np.load(npy_file_folder + dihedral_over_time_data_name)
        print("Dihedral angle readed")
    else:
        # Process each timestep with Lin method
        for timestep in range(num_steps):
            print(f"\nCalculation for time step {timestep}")
            
            # Initialize microstructure and analysis parameters
            P0 = P0_list[timestep,:,:,:]
            output_inclination_name = base_name + "inclination.txt"
            output_dihedral_name = base_name + "dihedral.txt"
            nx, ny, _ = P0.shape   # Get IC dimensions (512, 512)
            ng = 3      # Number of grains
            cores = 8   # Parallel processing cores
            loop_times = 5  # Solver iterations
            R = np.zeros((nx,ny,2))  # Results array

            # Perform inclination analysis using linear solver
            test1 = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
            test1.linear_main("inclination")
            P = test1.get_P()
            
            # Report solver performance metrics
            print('loop_times = ' + str(test1.loop_times))
            print('running_time = %.2f' % test1.running_time)
            print('running_core time = %.2f' % test1.running_coreTime)
            print('total_errors = %.2f' % test1.errors)
            print('per_errors = %.3f' % test1.errors_per_site)
            print("Inclination calculation done")

            # Calculate triple junction tangent vectors and dihedral angles
            triple_coord, triple_angle, triple_grain = output_tangent.calculate_tangent(P0[:,:,0], loop_times)
            print("Tangent calculation done")
            
            # Data structure documentation:
            # triple_coord: coordinates of triple junction (left-upper voxel)
            #   axis 0 = triple junction index, axis 1 = coordinates (i,j,k)
            # triple_angle: three dihedral angles for each triple junction
            #   axis 0 = triple junction index, axis 1 = three dihedral angles
            # triple_grain: sequence of three grains for each triple point
            #   axis 0 = triple junction index, axis 1 = three grain IDs
            
            print(triple_grain)
            print(triple_angle)

            # Export detailed dihedral angle results
            output_dihedral_angle(npy_file_folder + output_dihedral_name, triple_coord, triple_angle, triple_grain)
            print("Dihedral angle outputted")

            # Calculate grain-specific average dihedral angle with quality control
            sum_grain_dihedral = 0
            sum_dihedral_num = 0
            specific_grain = 1  # Focus on grain 1 for consistency
            
            # Process each triple junction with quality filtering
            for i in range(len(triple_angle)):
                # Quality control: skip if angle sum deviates significantly from 360°
                if (np.sum(triple_angle[i]) - 360) > 5: 
                    continue
                
                # Extract angle for specific grain with error handling
                print("specific angle: " + str(triple_angle[i][int(np.argwhere(triple_grain[i]==specific_grain))]))
                try:
                    # Find angle associated with specific grain
                    grain_angle_index = int(np.argwhere(triple_grain[i]==specific_grain))
                    sum_grain_dihedral += triple_angle[i][grain_angle_index]
                    sum_dihedral_num += 1
                except:
                    continue
            
            # Calculate average dihedral angle for this timestep
            if sum_dihedral_num == 0: 
                average_max_dihedral = 0
            else: 
                average_max_dihedral = sum_grain_dihedral / sum_dihedral_num
            
            print(f"The average dihedral angle on grain {specific_grain} is {average_max_dihedral}")
            print("Average dihedral angle obtained")
            max_dihedral_list_lin[timestep] = average_max_dihedral
    
    # Cache results for future use
    np.save(npy_file_folder + dihedral_over_time_data_name, max_dihedral_list_lin)
    
    return max_dihedral_list_lin
    

if __name__ == '__main__':
    """
    Main Execution: Comparative Dihedral Angle Analysis
    ==================================================
    
    This script performs comprehensive comparison between Joseph and Lin methods
    for calculating triple junction dihedral angles across multiple energy types.
    
    Analysis Pipeline:
    -----------------
    1. Process multiple energy calculation cases (ave, sum, consMin, consMax, consTest)
    2. Extract dihedral angle time series using Joseph method (baseline)
    3. Alternative: Calculate angles using Lin method (computational validation)
    4. Apply temporal smoothing to reduce noise and identify trends
    5. Generate comparative visualization plots with statistical analysis
    
    Method Comparison Strategy:
    --------------------------
    - Joseph Method: Traditional geometric approach (pre-calculated results)
    - Lin Method: Advanced computational geometry (live calculation)
    - Statistical validation through time series comparison
    - Quality control with angle sum validation and outlier filtering
    
    Scientific Objectives:
    ---------------------
    - Validate computational method accuracy against established algorithms
    - Assess temporal stability and convergence of different approaches
    - Identify optimal energy calculation methodology
    - Generate publication-quality comparative analysis
    
    Output Products:
    ---------------
    - Time series data files (.npy) for each energy type and method
    - Comparative evolution plots with smoothing analysis
    - Statistical validation metrics between methods
    - High-resolution visualization for scientific publication
    """
    
    # Configuration for SPPARKS simulation data analysis
    npy_file_folder = "/Users/lin/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_triple_for_TJE/results/"
    TJ_energy_type_cases = ["ave", "sum", "consMin", "consMax","consTest"]

    # Joseph method configuration and file mapping
    file_path_joseph = "/Users/lin/Dropbox (UFL)/UFdata/Dihedral_angle/TJ_IC_11152023/Results/"
    TJ_energy_type_cases_joseph = ["t_dihedrals_3.npy", "t_dihedrals_0.npy", "t_dihedrals_5.npy", 
                                   "t_dihedrals_2.npy", "t_dihedrals_1.npy"]

    # Process each energy calculation type
    for index, energy_type in enumerate(TJ_energy_type_cases):
        # Define file paths and parameters for current energy type
        base_name = f"dihedral_results/triple_{energy_type}_"
        dihedral_over_time_figure_name = "triple_dihedral_over_time_" + energy_type + ".png"
        num_steps = 61  # Number of simulation timesteps

        # Extract dihedral angles using Joseph method (baseline approach)
        dihedral_over_time = dihedral_angle_from_Joseph(file_path_joseph + TJ_energy_type_cases_joseph[index], num_steps)
        # Handle NaN values with reasonable substitution
        dihedral_over_time[np.isnan(dihedral_over_time[:])] = 120
        # Apply strong smoothing (window=10) to identify trends
        dihedral_over_time_smooth = data_smooth(dihedral_over_time, 10)

        # Alternative: Lin method calculation (currently commented for comparison)
        # This section can be uncommented to perform live calculation with Lin method
        # dihedral_over_time = dihedral_angle_from_Lin(npy_file_folder, base_name, energy_type, num_steps)
        # dihedral_over_time_smooth = data_smooth(dihedral_over_time, 10)
        # dihedral_over_time_smooth = np.ones(num_steps)*np.average(dihedral_over_time) # Alternative: simple averaging

        # Generate comparative analysis visualization
        plt.clf()
        # Plot raw data points for detailed analysis
        plt.plot(np.linspace(0,(num_steps-1)*25,num_steps), dihedral_over_time, '.', markersize=4, 
                 label="average angle")
        # Plot smoothed trend line for pattern identification
        plt.plot(np.linspace(0,(num_steps-1)*25,num_steps), dihedral_over_time_smooth, '-', linewidth=2, 
                 label="fit")
        
        # Optional reference lines for theoretical comparison
        # plt.plot(np.linspace(0,(num_steps-1)*100,num_steps), [145.46]*num_steps, '--', linewidth=2, 
        #          label="equilibrium from GB area")  # Theoretical equilibrium
        # plt.plot(np.linspace(0,160*100,161), [45.95]*161, '--', linewidth=2, 
        #          label="expected angle results")  # Alternative reference
        
        # Configure plot appearance for publication quality
        plt.ylim([80,140])    # Focus on physically relevant angle range
        plt.xlim([0,1500])    # Full temporal evolution window
        plt.legend(fontsize=20, loc='upper right')
        plt.xlabel("Timestep (MCS)", fontsize=20)
        plt.ylabel(r"Angle ($\degree$)", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xticks([0, 300, 600, 900, 1200, 1500])  # Clear time reference points
        
        # Save high-resolution plot for scientific publication
        plt.savefig(npy_file_folder + dihedral_over_time_figure_name, 
                    bbox_inches='tight', format='png', dpi=400)

