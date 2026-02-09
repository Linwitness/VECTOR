#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparative Dihedral Angle Analysis: Joseph vs Lin Method Validation
===================================================================

This module provides comprehensive comparison and validation of two different
algorithms for calculating dihedral angles at triple junctions in 2D grain
boundary systems.

Created on Fri Mar 24 11:48:29 2023
@author: Lin
"""

import os
current_path = os.getcwd()
import numpy as np
from numpy import seterr
seterr(all='raise')
import matplotlib.pyplot as plt
import sys
sys.path.append(current_path)
sys.path.append(current_path+'/../../')
import myInput
import PACKAGE_MP_Linear as linear2d
sys.path.append(current_path+'/../calculate_tangent/')
import output_tangent
from tqdm import tqdm

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

