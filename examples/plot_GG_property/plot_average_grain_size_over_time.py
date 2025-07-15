#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Average Grain Size Evolution Analysis and Distribution Characterization

This script provides comprehensive temporal analysis of average grain size evolution
in polycrystalline systems under different energy formulations. The analysis combines
temporal evolution tracking with statistical distribution characterization to understand
grain growth kinetics and steady-state behavior across multiple energy calculation methods.

Scientific Purpose:
- Tracks average grain size evolution over time for multiple energy methods
- Generates normalized grain size distributions at specific evolutionary states
- Compares anisotropic vs. isotropic energy formulation effects on growth kinetics
- Provides comprehensive statistical characterization of grain size distributions
- Enables identification of steady-state behavior and temporal scaling relationships

Key Features:
- Multi-energy method comparison (ave, consMin, sum, iso)
- Automated grain area calculation with caching for computational efficiency
- Equivalent circular radius calculations for grain size quantification
- Statistical binning and normalization for distribution analysis
- Temporal evolution visualization with publication-quality formatting
- Optional time-series distribution generation for steady-state analysis

Energy Methods Analyzed:
- ave: Average triple junction energy approach (baseline)
- consMin: Conservative minimum energy selection (small grain preservation)
- sum: Summation-based energy calculation (cumulative effects)
- iso: Isotropic reference case (delta=0.0, no anisotropy)

Statistical Analysis:
- Grain size calculation: R = sqrt(Area/π) (equivalent circular radius)
- Normalization: R/<R> where <R> is the average grain size
- Distribution binning: 0.16 bin width, range [-0.5, 3.5] for normalized sizes
- Frequency normalization: Area under curve = 1.0 for proper statistical comparison

Technical Specifications:
- Initial grain count: 20,000 grains
- Domain: 2D polycrystalline systems with crystallographic orientations
- Processing: 32-core parallel processing
- Temporal resolution: Full simulation timestep analysis
- Special timesteps: Optimized for ~2000-grain analysis states

Created on Mon Jul 31 14:33:57 2023
@author: Lin

Applications:
- Grain growth kinetics analysis and model validation
- Energy method benchmarking and comparison studies
- Steady-state behavior identification in polycrystalline systems
- Statistical mechanics of grain size evolution
"""

# Core scientific computing libraries
import os
current_path = os.getcwd()
import numpy as np                    # Numerical array operations and statistical analysis
from numpy import seterr
seterr(all='raise')                  # Enable numpy error checking for numerical stability
import matplotlib.pyplot as plt      # Publication-quality plotting and visualization
import math                          # Mathematical functions for size calculations
from tqdm import tqdm                # Progress bar for computationally intensive loops
import sys

# Add VECTOR framework paths for simulation analysis modules
sys.path.append(current_path)
sys.path.append(current_path+'/../../')
import myInput                       # VECTOR input parameter management
import PACKAGE_MP_Linear as linear2d # 2D linear algebra operations for grain analysis
sys.path.append(current_path+'/../calculate_tangent/')

if __name__ == '__main__':
    # =============================================================================
    # Local Data Source Configuration
    # =============================================================================
    """
    Data source: Local SPPARKS simulation results
    Simulation type: 2D polycrystalline grain growth with multiple energy methods
    Analysis focus: Temporal evolution and statistical distribution characterization
    """
    npy_file_folder = "/Users/lin/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_poly_for_GG/results/"
    
    # =============================================================================
    # Energy Method Configuration for Comparative Analysis
    # =============================================================================
    """
    Energy calculation methods for comprehensive grain growth analysis:
    - Focused on primary methods for detailed temporal characterization
    - Each method produces distinct grain growth kinetics and size distributions
    """
    TJ_energy_type_cases = ["ave"]     # Primary focus on average energy method
    TJ_energy_type_ave = "ave"         # Average energy method (baseline)
    TJ_energy_type_consMin = "consMin" # Conservative minimum energy
    TJ_energy_type_sum = "sum"         # Summation-based energy calculation
    
    # =============================================================================
    # Simulation Data File Names (32-core processing)
    # =============================================================================
    """
    File naming convention: p_ori_ave_{energy_type}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy
    - Reduced to 32-core processing for more manageable computation
    - Consistent parameters across all energy methods for direct comparison
    """
    npy_file_name_aniso_ave = f"p_ori_ave_{TJ_energy_type_ave}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMin = f"p_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_sum = f"p_ori_ave_{TJ_energy_type_sum}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_iso = "p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    
    # =============================================================================
    # Pre-computed Grain Size Data Files
    # =============================================================================
    """
    Cached grain area data for computational efficiency:
    - Avoids repeated grain area calculations for each analysis run
    - Format: 2D arrays [timestep, grain_id] containing individual grain areas
    """
    grain_size_data_name_ave = f"grain_size_p_ori_ave_{TJ_energy_type_ave}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_consMin = f"grain_size_p_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_sum = f"grain_size_p_ori_ave_{TJ_energy_type_sum}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_iso = "grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    
    # =============================================================================
    # Simulation Data Loading and Validation
    # =============================================================================
    """
    Load grain structure evolution data for all energy methods
    Each dataset contains complete temporal evolution of grain arrangements
    """
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)
    npy_file_aniso_consMin = np.load(npy_file_folder + npy_file_name_aniso_consMin)
    npy_file_aniso_sum = np.load(npy_file_folder + npy_file_name_aniso_sum)
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)
    
    # Display dataset dimensions for verification
    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    print(f"The consMin data size is: {npy_file_aniso_consMin.shape}")
    print(f"The sum data size is: {npy_file_aniso_sum.shape}")
    print(f"The iso data size is: {npy_file_iso.shape}")
    print("READING DATA DONE")
    
    # =============================================================================
    # Data Structure Initialization for Temporal Analysis
    # =============================================================================
    """
    Initialize comprehensive data arrays for temporal evolution tracking:
    - Grain numbers, areas, sizes, and average sizes for each timestep
    - Separate arrays for each energy method to enable comparative analysis
    """
    initial_grain_num = 20000                          # Initial grain count
    step_num = npy_file_aniso_ave.shape[0]            # Number of timesteps
    
    # Average energy method arrays
    grain_num_ave = np.zeros(step_num)                 # Grain count evolution
    grain_area_ave = np.zeros((step_num,initial_grain_num))     # Individual grain areas
    grain_size_ave = np.zeros((step_num,initial_grain_num))     # Individual grain sizes
    grain_ave_size_ave = np.zeros(step_num)            # Average grain size evolution
    
    # Conservative minimum energy method arrays
    grain_num_consMin = np.zeros(step_num)
    grain_area_consMin = np.zeros((step_num,initial_grain_num))
    grain_size_consMin = np.zeros((step_num,initial_grain_num))
    grain_ave_size_consMin = np.zeros(step_num)
    
    # Summation energy method arrays
    grain_num_sum = np.zeros(step_num)
    grain_area_sum = np.zeros((step_num,initial_grain_num))
    grain_size_sum = np.zeros((step_num,initial_grain_num))
    grain_ave_size_sum = np.zeros(step_num)
    
    # Isotropic reference method arrays
    grain_num_iso = np.zeros(step_num)
    grain_area_iso = np.zeros((step_num,initial_grain_num))
    grain_size_iso = np.zeros((step_num,initial_grain_num))
    grain_ave_size_iso = np.zeros(step_num)
    
    # =============================================================================
    # Statistical Distribution Analysis Configuration
    # =============================================================================
    """
    Configuration for normalized grain size distribution analysis:
    - Optimized binning parameters for grain size distribution characterization
    - Special timesteps selected for representative distribution states (~2000 grains)
    """
    bin_width = 0.16                    # Grain size distribution bin width
    x_limit = [-0.5, 3.5]             # Range for normalized grain size (R/<R>)
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)  # Number of bins
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)  # Bin centers
    
    # Distribution arrays for each energy method
    grain_size_distribution_ave = np.zeros(bin_num)
    special_step_distribution_ave = 11          # Timestep for ave distribution analysis
    grain_size_distribution_consMin = np.zeros(bin_num)
    special_step_distribution_consMin = 11      # Timestep for consMin distribution analysis
    grain_size_distribution_sum = np.zeros(bin_num)
    special_step_distribution_sum = 11          # Timestep for sum distribution analysis
    grain_size_distribution_iso = np.zeros(bin_num)
    special_step_distribution_iso = 10          # Timestep for iso distribution analysis
    
    # =============================================================================
    # Grain Area Calculation with Intelligent Caching
    # =============================================================================
    """
    Calculate grain areas from spatial grain ID data with automatic caching:
    - Checks for existing pre-computed data to avoid redundant calculations
    - Performs pixel counting for each grain at each timestep
    - Saves results for future analysis runs
    """
    
    # Average energy method grain area calculation
    if os.path.exists(npy_file_folder + grain_size_data_name_ave):
        grain_area_ave = np.load(npy_file_folder + grain_size_data_name_ave)
    else:
        print("Computing grain areas for average energy method...")
        for i in tqdm(range(step_num)):
            for j in range(npy_file_aniso_ave.shape[1]):
                for k in range(npy_file_aniso_ave.shape[2]):
                    grain_id = int(npy_file_aniso_ave[i,j,k,0])
                    grain_area_ave[i,grain_id-1] += 1  # Pixel counting for area calculation
        np.save(npy_file_folder + grain_size_data_name_ave, grain_area_ave)
    
    # Conservative minimum energy method grain area calculation
    if os.path.exists(npy_file_folder + grain_size_data_name_consMin):
        grain_area_consMin = np.load(npy_file_folder + grain_size_data_name_consMin)
    else:
        print("Computing grain areas for conservative minimum energy method...")
        for i in tqdm(range(step_num)):
            for j in range(npy_file_aniso_consMin.shape[1]):
                for k in range(npy_file_aniso_consMin.shape[2]):
                    grain_id = int(npy_file_aniso_consMin[i,j,k,0])
                    grain_area_consMin[i,grain_id-1] += 1
        np.save(npy_file_folder + grain_size_data_name_consMin, grain_area_consMin)
    
    # Summation energy method grain area calculation
    if os.path.exists(npy_file_folder + grain_size_data_name_sum):
        grain_area_sum = np.load(npy_file_folder + grain_size_data_name_sum)
    else:
        print("Computing grain areas for summation energy method...")
        for i in tqdm(range(step_num)):
            for j in range(npy_file_aniso_sum.shape[1]):
                for k in range(npy_file_aniso_sum.shape[2]):
                    grain_id = int(npy_file_aniso_sum[i,j,k,0])
                    grain_area_sum[i,grain_id-1] += 1
        np.save(npy_file_folder + grain_size_data_name_sum, grain_area_sum)
    
    # Isotropic reference method grain area calculation
    if os.path.exists(npy_file_folder + grain_size_data_name_iso):
        grain_area_iso = np.load(npy_file_folder + grain_size_data_name_iso)
    else:
        print("Computing grain areas for isotropic reference method...")
        for i in tqdm(range(step_num)):
            for j in range(npy_file_iso.shape[1]):
                for k in range(npy_file_iso.shape[2]):
                    grain_id = int(npy_file_iso[i,j,k,0])
                    grain_area_iso[i,grain_id-1] += 1
        np.save(npy_file_folder + grain_size_data_name_iso, grain_area_iso)
    
    print("GRAIN SIZE INITIALING DONE")
         
    # =============================================================================
    # Comprehensive Grain Size Analysis and Distribution Generation
    # =============================================================================
    """
    Perform temporal analysis of grain evolution with statistical characterization:
    - Calculate grain numbers and equivalent circular radii at each timestep
    - Generate normalized grain size distributions at special timesteps
    - Track average grain size evolution for comparative analysis
    """
    print("Starting comprehensive grain size analysis...")
    for i in tqdm(range(step_num)):
        # Calculate grain numbers (non-zero area grains)
        grain_num_ave[i] = np.sum(grain_area_ave[i,:] != 0)
        grain_num_consMin[i] = np.sum(grain_area_consMin[i,:] != 0)
        grain_num_sum[i] = np.sum(grain_area_sum[i,:] != 0)
        grain_num_iso[i] = np.sum(grain_area_iso[i,:] != 0)
        
        # Calculate equivalent circular radii from areas: R = sqrt(Area/π)
        grain_size_ave[i] = (grain_area_ave[i] / np.pi)**0.5
        grain_ave_size_ave[i] = np.sum(grain_size_ave[i]) / grain_num_ave[i]
        
        grain_size_consMin[i] = (grain_area_consMin[i] / np.pi)**0.5
        grain_ave_size_consMin[i] = np.sum(grain_size_consMin[i]) / grain_num_consMin[i]
        
        grain_size_sum[i] = (grain_area_sum[i] / np.pi)**0.5
        grain_ave_size_sum[i] = np.sum(grain_size_sum[i]) / grain_num_sum[i]
        
        grain_size_iso[i] = (grain_area_iso[i] / np.pi)**0.5
        grain_ave_size_iso[i] = np.sum(grain_size_iso[i]) / grain_num_iso[i]
        
        # Generate normalized grain size distributions at special timesteps
        if i == special_step_distribution_ave:
            # Average energy method distribution
            special_size_ave = grain_size_ave[i][grain_size_ave[i] != 0]  # Remove zero-size grains
            special_size_ave = special_size_ave/grain_ave_size_ave[i]     # Normalize by average size
            for j in range(len(special_size_ave)):
                grain_size_distribution_ave[int((special_size_ave[j]-x_limit[0])/bin_width)] += 1
                
        if i == special_step_distribution_consMin:
            # Conservative minimum energy method distribution
            special_size_consMin = grain_size_consMin[i][grain_size_consMin[i] != 0]
            special_size_consMin = special_size_consMin/grain_ave_size_consMin[i]
            for j in range(len(special_size_consMin)):
                grain_size_distribution_consMin[int((special_size_consMin[j]-x_limit[0])/bin_width)] += 1
                
        if i == special_step_distribution_sum:
            # Summation energy method distribution
            special_size_sum = grain_size_sum[i][grain_size_sum[i] != 0]
            special_size_sum = special_size_sum/grain_ave_size_sum[i]
            for j in range(len(special_size_sum)):
                grain_size_distribution_sum[int((special_size_sum[j]-x_limit[0])/bin_width)] += 1
                
        if i == special_step_distribution_iso:
            # Isotropic reference method distribution
            special_size_iso = grain_size_iso[i][grain_size_iso[i] != 0]
            special_size_iso = special_size_iso/grain_ave_size_iso[i]
            for j in range(len(special_size_iso)):
                grain_size_distribution_iso[int((special_size_iso[j]-x_limit[0])/bin_width)] += 1
    
    # Normalize frequency distributions (area under curve = 1.0)
    grain_size_distribution_ave = grain_size_distribution_ave/np.sum(grain_size_distribution_ave*bin_width)
    grain_size_distribution_consMin = grain_size_distribution_consMin/np.sum(grain_size_distribution_consMin*bin_width)
    grain_size_distribution_sum = grain_size_distribution_sum/np.sum(grain_size_distribution_sum*bin_width)
    grain_size_distribution_iso = grain_size_distribution_iso/np.sum(grain_size_distribution_iso*bin_width)
    print("GRAIN SIZE ANALYSIS DONE")
        
    # =============================================================================
    # Temporal Evolution Visualization
    # =============================================================================
    """
    Generate publication-quality plot showing average grain size evolution over time
    Enables comparison of growth kinetics across different energy methods
    """
    plt.clf()
    plt.plot(list(range(step_num)), grain_ave_size_ave, label="Ave case", linewidth=2)
    plt.plot(list(range(step_num)), grain_ave_size_consMin, label="ConsMin case", linewidth=2)
    plt.plot(list(range(step_num)), grain_ave_size_sum, label="Sum case", linewidth=2)
    plt.plot(list(range(step_num)), grain_ave_size_iso, label="Iso case", linewidth=2)
    
    plt.xlabel("Time step", fontsize=14)
    plt.ylabel("Grain Size", fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(current_path + "/figures/ave_grain_size_over_time.png", dpi=400,bbox_inches='tight')
    
    # =============================================================================
    # Normalized Grain Size Distribution Comparison
    # =============================================================================
    """
    Generate comparative plot of normalized grain size distributions
    Shows statistical differences in size distributions across energy methods
    """
    plt.clf()
    plt.plot(size_coordination, grain_size_distribution_ave, label="Ave case", linewidth=2)
    plt.plot(size_coordination, grain_size_distribution_consMin, label="ConsMin case", linewidth=2)
    plt.plot(size_coordination, grain_size_distribution_sum, label="Sum case", linewidth=2)
    plt.plot(size_coordination, grain_size_distribution_iso, label="Iso case", linewidth=2)
    
    plt.xlabel("R/$\langle$R$\rangle$", fontsize=14)    # Normalized grain size
    plt.ylabel("Frequency", fontsize=14)                # Probability density
    plt.legend(fontsize=14)
    plt.title(f"Grain num is around 2000", fontsize=14)  # Reference grain count
    plt.savefig(current_path + "/figures/normalized_size_distribution.png", dpi=400,bbox_inches='tight')
    
    # =============================================================================
    # Commented Time-Series Distribution Generation Code
    # =============================================================================
    """
    Optional code for generating time-series of grain size distributions
    Useful for studying steady-state behavior and temporal evolution patterns
    Currently commented out - can be activated for detailed temporal analysis
    
    Features:
    - Generates distribution plots for first 200 timesteps
    - Creates separate output folders for each energy method
    - Tracks grain count evolution alongside distribution changes
    - Useful for identifying steady-state distribution characteristics
    """







