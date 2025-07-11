#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triple Junction Energy Analysis and Dihedral Angle Calculation
==============================================================

This module provides comprehensive analysis of triple junction energies and their 
relationship with dihedral angles in 2D grain boundary systems. It processes 
SPPARKS simulation data (.npy files) to extract energy-dihedral angle correlations 
and validate theoretical predictions against computational results.

Scientific Background:
- Triple junction energy analysis in polycrystalline materials
- Dihedral angle measurement and statistical analysis
- Energy-angle relationship validation using Herring's equation
- Equilibrium angle prediction from energy minimization principles

Key Features:
- Multiple energy type analysis: ave, sum, consMin, consMax, consTest
- Site-specific energy calculation with neighbor connectivity
- Statistical analysis of dihedral angles over simulation time
- Curve fitting for energy-angle relationship prediction
- Validation against theoretical equilibrium angles (145.46°)

Applications:
- SPPARKS simulation data post-processing
- Triple junction energy model validation
- Grain boundary energy anisotropy studies
- Materials science research on grain boundary behavior

Created on Fri Mar 24 11:48:29 2023
@author: Lin
"""

import os
current_path = os.getcwd()
import numpy as np
from numpy import seterr
seterr(all='raise')  # Enable numpy error reporting for debugging
import matplotlib.pyplot as plt
import math
from itertools import repeat
from scipy.optimize import curve_fit  # Required for curve fitting analysis
import sys

# Configure system paths for VECTOR framework access
sys.path.append(current_path)
sys.path.append(current_path+'/../../')
import myInput
import PACKAGE_MP_Linear as linear2d
sys.path.append(current_path+'/../calculate_tangent/')
import output_tangent

def func(x, a, b, c):
    """
    Exponential Decay Function for Energy-Angle Relationship
    
    This function models the relationship between triple junction energy and
    dihedral angles using an exponential decay with offset.
    
    Parameters:
    -----------
    x : float or array
        Input energy values
    a, b, c : float
        Fitting parameters where:
        a = amplitude of exponential decay
        b = decay rate constant
        c = offset/baseline value
        
    Returns:
    --------
    float or array
        Fitted dihedral angle values
        
    Mathematical Model:
    ------------------
    angle = a * exp(-x * b) + c
    
    This model captures the physical relationship where:
    - Higher energies tend to produce smaller dihedral angles
    - The relationship follows exponential decay toward equilibrium
    - The offset 'c' represents the baseline angle at high energies
    
    Scientific Applications:
    -----------------------
    - Energy-angle relationship validation
    - Herring equation comparison
    - Equilibrium angle prediction
    - Triple junction energy model fitting
    """
    return a * np.exp(-x * b) + c

def get_gb_sites(P, grain_num):
    """
    Identify Grain Boundary Sites in 2D Microstructure
    
    This function systematically identifies all grain boundary sites by analyzing
    neighbor connectivity and detecting interface locations between different grains.
    
    Parameters:
    -----------
    P : ndarray
        3D array representing microstructure (layers, x, y)
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
    - Organized output by grain number for systematic analysis
    
    Scientific Applications:
    -----------------------
    - Interface area calculation
    - Grain boundary characterization
    - Normal vector calculation preparation
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
    """
    Calculate Normal Vectors for All Grain Boundary Sites
    
    This function computes the normal vectors at each grain boundary site using
    gradient calculation methods for interface orientation analysis.
    
    Parameters:
    -----------
    grain_num : int
        Total number of grains in the microstructure
    P_matrix : ndarray
        Microstructure array for gradient calculation
        
    Returns:
    --------
    norm_list : list of ndarrays
        Normal vectors organized by grain number
        Each array contains [normal_x, normal_y] for boundary sites
    boundary_site : list of lists
        Corresponding boundary site coordinates
        
    Computational Details:
    ---------------------
    - Uses myInput.get_grad() for accurate gradient calculation
    - Systematic processing grain by grain with progress tracking
    - Preserves spatial correlation between sites and normal vectors
    - Memory-efficient storage organized by grain boundaries
    
    Scientific Applications:
    -----------------------
    - Interface inclination analysis
    - Grain boundary character distribution
    - Triple junction angle calculations
    - Crystallographic orientation relationships
    """
    # Get grain boundary sites for all grains
    boundary_site = get_gb_sites(P_matrix, grain_num)
    norm_list = [np.zeros((len(boundary_site[i]), 2)) for i in range(grain_num)]
    
    # Calculate normal vectors for each grain's boundary sites
    for grain_i in range(grain_num):
        print(f"Processing grain {grain_i} boundary normals...")

        for site in range(len(boundary_site[grain_i])):
            # Calculate normal vector using gradient method
            norm = myInput.get_grad(P_matrix, boundary_site[grain_i][site][0], 
                                   boundary_site[grain_i][site][1])
            norm_list[grain_i][site,:] = list(norm)

    return norm_list, boundary_site

def get_orientation(grain_num, init_name):
    """
    Extract Euler Angles from SPPARKS Initialization File
    
    This function reads crystallographic orientations (Euler angles) from
    SPPARKS .init files for anisotropic grain boundary analysis.
    
    Parameters:
    -----------
    grain_num : int
        Total number of grains expected
    init_name : str
        Path to SPPARKS initialization file
        
    Returns:
    --------
    eulerAngle : ndarray
        Array of Euler angles [phi1, Phi, phi2] for each grain
        
    File Format Parsing:
    -------------------
    Expected format: site_id grain_id phi1 Phi phi2
    - Skips comment lines starting with '#'
    - Handles grain ID indexing (converts to 0-based)
    - Prevents duplicate angle assignment
    - Returns subset excluding last grain for compatibility
    
    Scientific Applications:
    -----------------------
    - Anisotropic grain boundary energy calculations
    - Misorientation analysis between neighboring grains
    - Texture analysis and crystallographic studies
    - Grain boundary character distribution analysis
    """
    # Initialize Euler angle array with sentinel values
    eulerAngle = np.ones((grain_num, 3)) * -10
    
    with open(init_name, 'r', encoding='utf-8') as f:
        for line in f:
            eachline = line.split()

            # Parse data lines (5 columns, not comments)
            if len(eachline) == 5 and eachline[0] != '#':
                lineN = int(eachline[1]) - 1  # Convert to 0-based indexing
                
                # Assign Euler angles if not already set
                if eulerAngle[lineN, 0] == -10:
                    eulerAngle[lineN, :] = [float(eachline[2]), float(eachline[3]), float(eachline[4])]
    
    return eulerAngle[:ng-1]  # Note: ng should be defined in calling context

def output_inclination(output_name, norm_list, site_list, orientation_list=0):
    """
    Export Grain Boundary Inclination Data to File
    
    This function generates comprehensive output files containing grain boundary
    site coordinates, normal vectors, and associated crystallographic orientations.
    
    Parameters:
    -----------
    output_name : str
        Output file path for inclination data
    norm_list : list of ndarrays
        Normal vectors organized by grain
    site_list : list of lists
        Boundary site coordinates for each grain
    orientation_list : ndarray, optional
        Euler angles for each grain (default=0 for empty)
        
    Output Format:
    --------------
    For each grain:
    - Header: "Grain N Orientation: [phi1,Phi,phi2] centroid:"
    - Data lines: "x_coord, y_coord, normal_x, normal_y"
    - Blank line separator between grains
    
    Scientific Applications:
    -----------------------
    - Post-processing for inclination distribution analysis
    - Input for crystallographic analysis software
    - Grain boundary character classification
    - Statistical analysis of interface orientations
    """
    file = open(output_name, 'w')
    
    # Write data for each grain
    for i in range(len(norm_list)):
        # Write grain header with orientation information
        if orientation_list != 0:
            file.writelines(['Grain ' + str(i+1) + ' Orientation: ' + str(orientation_list[i]) + ' centroid: ' + '\n'])
        else:
            file.writelines(['Grain ' + str(i+1) + ' Orientation: empty centroid: ' + '\n'])

        # Write site coordinates and normal vectors
        for j in range(len(norm_list[i])):
            file.writelines([str(site_list[i][j][0]) + ', ' + str(site_list[i][j][1]) + ', ' + 
                           str(norm_list[i][j][0]) + ', ' + str(norm_list[i][j][1]) + '\n'])

        file.writelines(['\n'])  # Separator between grains

    file.close()
    return

def output_dihedral_angle(output_name, triple_coord, triple_angle, triple_grain):
    """
    Export Triple Junction Dihedral Angle Analysis Results
    
    This function creates detailed output files containing triple junction
    coordinates, calculated dihedral angles, and associated grain IDs.
    
    Parameters:
    -----------
    output_name : str
        Output file path for dihedral angle data
    triple_coord : ndarray
        Coordinates of triple junction points
    triple_angle : ndarray
        Calculated dihedral angles for each triple junction
    triple_grain : ndarray
        Grain IDs associated with each triple junction
        
    Output Format:
    --------------
    Header: "triple_index triple_coordination grain_id0:dihedral0 grain_id1:dihedral1 grain_id2:dihedral2 angle_sum"
    Data: "index, x y, grain1:angle1 grain2:angle2 grain3:angle3 sum"
    
    Scientific Applications:
    -----------------------
    - Triple junction energy analysis
    - Equilibrium angle validation (should sum to 360°)
    - Statistical analysis of dihedral angle distributions
    - Comparison with theoretical predictions
    """
    file = open(output_name, 'w')
    
    # Write header for data interpretation
    file.writelines(['triple_index triple_coordination grain_id0:dihedral0 grain_id1:dihedral1 grain_id2:dihedral2 angle_sum\n'])
    
    # Write data for each triple junction
    for i in range(len(triple_coord)):
        file.writelines([str(i+1) + ', ' + str(triple_coord[i][0]) + ' ' + str(triple_coord[i][1]) + ', ' +
                         str(int(triple_grain[i][0])) + ':' + str(round(triple_angle[i][0], 2)) + ' ' +
                         str(int(triple_grain[i][1])) + ':' + str(round(triple_angle[i][1], 2)) + ' ' +
                         str(int(triple_grain[i][2])) + ':' + str(round(triple_angle[i][2], 2)) + ' ' +
                         str(round(np.sum(triple_angle[i]), 2)) + '\n'])

    file.close()
    return

def find_window(P, i, j, iteration, refer_id):
    """
    Generate Local Window Around Specified Voxel
    
    This function creates a local neighborhood window for analyzing grain
    connectivity and calculating site-specific properties with periodic boundaries.
    
    Parameters:
    -----------
    P : ndarray
        2D microstructure array
    i, j : int
        Center coordinates for window
    iteration : int
        Window half-size parameter (total size = 2*iteration+1)
    refer_id : int
        Reference grain ID for comparison
        
    Returns:
    --------
    window : ndarray
        Binary array where 1=same grain, 0=different grain
        
    Algorithm Details:
    -----------------
    - Creates square window of size (2*iteration+1) x (2*iteration+1)
    - Uses periodic boundary conditions for edge handling
    - Binary classification based on grain ID matching
    - Used for neighbor counting and energy calculations
    
    Scientific Applications:
    -----------------------
    - Local environment analysis
    - Neighbor connectivity calculation
    - Site energy normalization
    - Interface characterization
    """
    nx, ny = P.shape
    tableL = 2 * (iteration + 1) + 1
    fw_len = tableL
    fw_half = int((fw_len - 1) / 2)
    window = np.zeros((fw_len, fw_len))

    # Generate window with periodic boundary conditions
    for wi in range(fw_len):
        for wj in range(fw_len):
            global_x = (i - fw_half + wi) % nx
            global_y = (j - fw_half + wj) % ny
            
            # Binary classification: 1 if same grain, 0 if different
            if P[global_x, global_y] == refer_id:
                window[wi, wj] = 1
            else:
                window[wi, wj] = 0

    return window

def data_smooth(data_array, smooth_level=2):
    """
    Apply Moving Average Smoothing to Time Series Data
    
    This function performs temporal smoothing using a moving average filter
    to reduce noise in dihedral angle evolution data.
    
    Parameters:
    -----------
    data_array : ndarray
        Input time series data for smoothing
    smooth_level : int, optional
        Half-width of smoothing window (default=2)
        
    Returns:
    --------
    data_array_smoothed : ndarray
        Smoothed time series data
        
    Algorithm Details:
    -----------------
    - Uses symmetric moving average when possible
    - Handles boundaries with asymmetric averaging
    - Preserves data length and temporal alignment
    - Adjustable smoothing strength via smooth_level parameter
    
    Scientific Applications:
    -----------------------
    - Noise reduction in simulation data
    - Trend identification in evolution studies
    - Statistical analysis preparation
    - Visualization enhancement for publication plots
    """
    data_array_smoothed = np.zeros(len(data_array))
    
    for i in range(len(data_array)):
        # Handle left boundary
        if i < smooth_level:
            data_array_smoothed[i] = np.sum(data_array[0:i+smooth_level+1]) / (i+smooth_level+1)
        # Handle right boundary  
        elif (len(data_array) - 1 - i) < smooth_level:
            data_array_smoothed[i] = np.sum(data_array[i-smooth_level:]) / (len(data_array)-i+smooth_level)
        # Central smoothing
        else:
            data_array_smoothed[i] = np.sum(data_array[i-smooth_level:i+smooth_level+1]) / (smooth_level*2+1)

    return data_array_smoothed

if __name__ == '__main__':
    """
    Main Execution: Triple Junction Energy-Dihedral Angle Analysis
    =============================================================
    
    This script performs comprehensive analysis of the relationship between
    triple junction energies and dihedral angles across multiple energy types.
    
    Analysis Pipeline:
    -----------------
    1. Process multiple energy calculation methods (ave, sum, consMin, consMax, consTest)
    2. Calculate average triple junction energy for each method
    3. Extract corresponding average dihedral angles from previous calculations
    4. Perform curve fitting to establish energy-angle relationship
    5. Validate against theoretical predictions (Herring equation: 145.46°)
    6. Generate visualization comparing experimental and theoretical results
    
    Scientific Objectives:
    ---------------------
    - Validate energy-based triple junction models
    - Compare different energy calculation methodologies
    - Assess agreement with theoretical equilibrium angles
    - Identify optimal energy calculation approach
    - Generate publication-quality analysis plots
    
    Output Products:
    ---------------
    - Energy data files (.npy) for each calculation method
    - Fitted relationship parameters and equations
    - Expected site energy for theoretical angle match
    - Comparative visualization plot (PNG format)
    """
    
    # Configuration for SPPARKS simulation data analysis
    npy_file_folder = "/Users/lin/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_hex_for_TJE/results/"
    TJ_energy_type_cases = ["ave", "sum", "consMin", "consMax", "consTest"]  # Different energy calculation methods
    step_equalibrium_end = int(8000/100)  # Equilibrium analysis window

    # Initialize arrays for comparative analysis
    average_TJtype_energy = np.zeros(len(TJ_energy_type_cases))
    average_TJtype_dihedral_angle = np.zeros(len(TJ_energy_type_cases))
    
    # Process each energy calculation method
    for energy_type_index, energy_type in enumerate(TJ_energy_type_cases):
        print(f"\nStart {energy_type} energy type:")
        
        # Define file paths for current energy type
        npy_file_name = f"h_ori_ave_{energy_type}E_hex_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_angle.npy"
        energy_npy_file_name = f"h_ori_ave_{energy_type}E_hex_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_angle_energy.npy"

        base_name = f"dihedral_results/hex_{energy_type}_"
        energy_base_name = f"energy_results/hex_{energy_type}_"
        dihedral_over_time_data_name = energy_base_name + "energy_data.npy"

        # Check if processed data exists to avoid redundant calculations
        if os.path.exists(npy_file_folder + dihedral_over_time_data_name):
            average_TJ_energy = np.load(npy_file_folder + dihedral_over_time_data_name)
            print("Energy data readed")
        else:
            # Perform energy calculation for each timestep
            average_TJ_energy = np.zeros(step_equalibrium_end)
            
            for timestep in range(step_equalibrium_end):
                print(f"\nCalculation for time step {timestep}")
                
                # Load microstructure and energy data for current timestep
                P0_list = np.load(npy_file_folder + npy_file_name)
                P0_energy_list = np.load(npy_file_folder + energy_npy_file_name)
                print("IC is read as matrix")
                P0 = P0_list[timestep,:,:,0]      # Microstructure configuration
                P0_energy = P0_energy_list[timestep,:,:,0]  # Site energy values

                # Extract microstructure parameters
                nx, ny = P0.shape
                ng = 50      # Number of grains
                cores = 8    # Parallel processing cores
                loop_times = 5  # Analysis iterations
                R = np.zeros((nx,ny,2))  # Results storage

                # Initialize triple junction energy accumulation
                allTJ_ave_energy = 0
                num_TJ = 0
                
                # Scan through domain to identify triple junctions
                for i in range(nx-1):
                    for j in range(ny-1):
                        # Extract 2x2 neighborhood for triple junction detection
                        nei = np.zeros((2,2))
                        nei = P0[i:i+2,j:j+2]
                        energy_nei = P0_energy[i:i+2,j:j+2]
                        nei_flat = nei.flatten()
                        energy_nei_flat = energy_nei.flatten()
                        
                        # Check for triple junction: exactly 3 different grains, no voids
                        if len(set(nei_flat)) == 3 and 0 not in nei_flat:
                            # Calculate energy contribution for this triple junction
                            # Using site-specific energy normalized by neighbor connectivity
                            oneTJ_ave_energy = 0
                            
                            # Process each site in the 2x2 neighborhood
                            for k in [[i,j],[i,j+1],[i+1,j],[i+1,j+1]]:
                                # Generate local window for neighbor analysis
                                window_matrix = find_window(P0, k[0], k[1], 0, P0[k[0],k[1]])
                                window_matrix_flat = window_matrix.flatten()
                                # Count neighbor sites with different grain IDs
                                nei_sites_num = np.sum(window_matrix_flat==0)
                                # Normalize site energy by neighbor connectivity
                                oneTJ_ave_energy += P0_energy[k[0],k[1]] / nei_sites_num
                            
                            # Average over the 4 sites in triple junction neighborhood
                            oneTJ_ave_energy = oneTJ_ave_energy / 4
                            allTJ_ave_energy += oneTJ_ave_energy
                            num_TJ += 1

                # Calculate average triple junction energy for this timestep
                allTJ_ave_energy = allTJ_ave_energy / num_TJ
                average_TJ_energy[timestep] = allTJ_ave_energy

        # Save processed energy data for future use
        np.save(npy_file_folder + dihedral_over_time_data_name, average_TJ_energy)
        # Calculate time-averaged energy for this method
        average_TJtype_energy[energy_type_index] = np.average(average_TJ_energy)

        # Load corresponding dihedral angle data
        dihedral_over_time = np.load(npy_file_folder + base_name + "data.npy")
        # Calculate time-averaged dihedral angle for this method
        average_TJtype_dihedral_angle[energy_type_index] = np.average(dihedral_over_time[:step_equalibrium_end])

    # Generate comprehensive analysis plot
    dihedral_siteEnergy_cases_figure_name = "energy_results/hex_aveDihedral_over_aveEnergy_" + "old.png"
    plt.clf()
    plt.plot(average_TJtype_energy, average_TJtype_dihedral_angle, 'o', markersize=4, 
             label="average angle in energy types")

    # Perform curve fitting to establish energy-angle relationship
    # Initial parameter estimates for exponential decay model
    a = max(average_TJtype_dihedral_angle)-min(average_TJtype_dihedral_angle)  # Amplitude
    b = average_TJtype_dihedral_angle[round(len(average_TJtype_dihedral_angle)/2)]  # Decay rate
    c = min(average_TJtype_dihedral_angle)  # Baseline offset
    p0 = [a,b,c]
    
    # Fit exponential decay model to data
    popt, pcov = curve_fit(func, average_TJtype_energy, average_TJtype_dihedral_angle, p0=p0)
    print(f"The equation to fit the relationship is {round(popt[0],2)} * exp(-x * {round(popt[1],2)}) + {round(popt[2],2)}")
    
    # Generate fitted curve for visualization
    y_fit = [func(i, popt[0], popt[1], popt[2]) for i in np.linspace(0, 4, 50)]
    plt.plot(np.linspace(0, 4, 50), y_fit, '-', linewidth=2, label="fitting results")
    
    # Find energy value that produces theoretical equilibrium angle (145.46°)
    exact_list = np.linspace(0.2, 1.0, 101)
    min_level = 10
    expect_site_energy = 0
    for m in exact_list:
        if min_level > abs(func(m, popt[0], popt[1], popt[2]) - 145.46):
            min_level = abs(func(m, popt[0], popt[1], popt[2]) - 145.46)
            expect_site_energy = m
    print(f"The expected average TJ site energy is {expect_site_energy}")

    # Add theoretical reference line (Herring equation prediction)
    plt.plot(np.linspace(0,4,24), [145.46]*24, '--', linewidth=2, 
             label="Herring equation results")
    
    # Configure plot appearance for publication quality
    plt.ylim([120,160])
    plt.xlim([0,4])
    plt.legend(fontsize=14, loc='lower center')
    plt.xlabel("Coupled energy", fontsize=14)
    plt.ylabel(r"Angle ($^\circ$)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(npy_file_folder + dihedral_siteEnergy_cases_figure_name, 
                bbox_inches='tight', format='png', dpi=400)

