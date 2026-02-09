#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triple Junction Energy Analysis and Dihedral Angle Calculation
==============================================================

This module provides comprehensive analysis of triple junction energies and their
relationship with dihedral angles in 2D grain boundary systems.

Created on Fri Mar 24 11:48:29 2023
@author: Lin
"""

import os
current_path = os.getcwd()
import numpy as np
from numpy import seterr
seterr(all='raise')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

sys.path.append(current_path)
sys.path.append(current_path+'/../../')
import myInput
import PACKAGE_MP_Linear as linear2d
sys.path.append(current_path+'/../calculate_tangent/')
import output_tangent

# Import shared utilities
from utils_angles import (
    get_gb_sites_2d as get_gb_sites,
    norm_list_2d as norm_list,
    get_orientation,
    output_inclination_2d as output_inclination,
    output_dihedral_angle_2d as output_dihedral_angle,
    find_window,
    data_smooth
)


def func(x, a, b, c):
    """
    Exponential decay function for energy-angle relationship fitting.

    Parameters:
    -----------
    x : float or array
        Input energy values
    a, b, c : float
        Fitting parameters: amplitude, decay rate, offset

    Returns:
    --------
    float or array
        Fitted dihedral angle values: a * exp(-x * b) + c
    """
    return a * np.exp(-x * b) + c

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

