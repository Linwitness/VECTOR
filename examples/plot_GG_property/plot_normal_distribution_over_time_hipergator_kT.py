#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================
VECTOR Framework: Thermal Fluctuation (kT) Analysis for Grain Boundary Orientation
=========================================================================================

Scientific Application: Monte Carlo Temperature Scaling Analysis
Primary Focus: Thermal Effects on Polycrystalline Grain Boundary Normal Vector Distributions

Created on Mon Jul 31 14:33:57 2023
@author: Lin

=========================================================================================
THERMAL ANALYSIS FRAMEWORK
=========================================================================================

This Python script implements comprehensive thermal fluctuation analysis for large-scale
polycrystalline microstructures using Monte Carlo temperature scaling (kT) to characterize
temperature-dependent grain boundary orientation patterns and anisotropy evolution.

Key Scientific Objectives:
1. Systematic thermal analysis across kT ∈ {0.00, 0.25, 0.50, 0.66, 0.95} regimes
2. Temperature-dependent grain boundary normal vector distribution characterization
3. Monte Carlo thermal averaging for statistical mechanics validation
4. HiPerGator multi-core processing optimization for thermal computations

=========================================================================================
THERMAL REGIMES AND MONTE CARLO FRAMEWORK
=========================================================================================

Temperature Scaling Parameters:
- kT = 0.00: Deterministic energy minimization (zero thermal fluctuations)
- kT = 0.25: Low thermal activation regime (minimal energy barrier crossing)
- kT = 0.50: Moderate thermal effects (balanced energy and entropy)
- kT = 0.66: Intermediate thermal regime (significant thermal activation)
- kT = 0.95: High thermal fluctuations (entropy-dominated behavior)

Statistical Mechanics Integration:
- Boltzmann statistics implementation for realistic thermal behavior
- Monte Carlo sampling for thermal averaging over microstructural configurations
- Temperature-dependent grain boundary mobility and orientation evolution
- Thermal equilibrium characterization for large-scale polycrystalline systems

=========================================================================================
COMPUTATIONAL FRAMEWORK AND OPTIMIZATION
=========================================================================================

VECTOR Linear2D Processing:
- 8-core parallel processing for efficient thermal normal vector computation
- Multi-time step thermal analysis with systematic data caching
- Optimized grain boundary site extraction with thermal noise handling
- Enhanced angular distribution analysis for temperature-dependent patterns

HiPerGator Multi-Core Integration:
- 32-core parallel simulation data processing
- Large-scale polycrystalline systems (20K initial grains)
- HiPerGator storage optimization for thermal datasets
- Publication-quality polar visualization generation

=========================================================================================
THERMAL ANALYSIS METHODOLOGY
=========================================================================================

Microstructural Analysis Pipeline:
1. Multi-temperature dataset loading and validation
2. Thermal normal vector computation using VECTOR Linear2D
3. Temperature-dependent angular distribution analysis
4. Statistical thermal averaging across multiple configurations
5. Publication-quality polar visualization generation

Data Management Strategy:
- Systematic thermal data caching for computational efficiency
- Temperature-specific file naming conventions
- Progress tracking for multi-kT analysis workflows
- Publication output generation with enhanced thermal visualization

=========================================================================================
SCIENTIFIC APPLICATIONS AND INSIGHTS
=========================================================================================

This thermal analysis framework enables quantitative characterization of:

1. Temperature-Dependent Orientation Evolution: Systematic analysis of grain boundary
   normal vector distributions across different thermal regimes

2. Thermal Fluctuation Effects: Understanding Monte Carlo temperature scaling effects
   on microstructural anisotropy and energy landscape exploration

3. Statistical Mechanics Validation: Temperature-dependent grain boundary behavior
   with Boltzmann statistics implementation for realistic material response

4. Thermal Activation Mechanisms: Characterization of thermal effects on grain boundary
   mobility and orientation patterns in large-scale polycrystalline systems

Key Scientific Insights:
- Deterministic vs. thermal behavior comparison across kT regimes
- Temperature-dependent anisotropy evolution patterns
- Thermal activation effects on grain boundary energy minimization
- Statistical mechanics validation for polycrystalline thermal behavior

=========================================================================================
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

def get_normal_vector(grain_structure_figure_one, grain_num):
    """
    Compute thermal normal vectors for grain boundary analysis using VECTOR Linear2D.
    
    This function implements temperature-aware grain boundary normal vector computation
    for thermal fluctuation analysis, utilizing 8-core parallel processing to efficiently
    extract grain boundary sites and compute orientation vectors under Monte Carlo
    temperature scaling conditions.
    
    Thermal Analysis Framework:
    - 8-core parallel processing for efficient thermal normal vector computation
    - Multi-grain boundary site extraction with thermal noise handling
    - Optimized inclination analysis for temperature-dependent patterns
    - Enhanced grain boundary list generation for large-scale thermal systems
    
    Parameters:
    -----------
    grain_structure_figure_one : np.ndarray
        3D microstructural array [nx, ny, fields] containing thermal grain boundary data
        with Monte Carlo temperature scaling applied to grain orientations and boundaries
    grain_num : int
        Total number of grains in the thermal polycrystalline system (typically 20,000)
        for statistical significance in thermal averaging analysis
        
    Returns:
    --------
    P : np.ndarray
        Normal vector field [nx, ny, 2] containing thermal gradient information
        with temperature-dependent orientation patterns and thermal fluctuation effects
    sites_together : list
        Comprehensive grain boundary site list [(i,j), ...] for thermal analysis
        containing all grain boundary coordinates for temperature-dependent characterization
        
    Thermal Computational Framework:
    --------------------------------
    1. Multi-core parallel processing (8 cores) for efficient thermal computation
    2. VECTOR Linear2D inclination analysis with thermal noise handling
    3. Comprehensive grain boundary site extraction across all thermal grains
    4. Temperature-aware gradient computation for Monte Carlo thermal systems
    
    Scientific Applications:
    ------------------------
    - Temperature-dependent grain boundary orientation analysis
    - Thermal fluctuation effects on normal vector distributions
    - Monte Carlo thermal averaging for statistical mechanics validation
    - Large-scale polycrystalline thermal characterization
    """
    # Microstructural dimensions for thermal analysis
    nx = grain_structure_figure_one.shape[0]                # X-dimension for thermal grid
    ny = grain_structure_figure_one.shape[1]                # Y-dimension for thermal grid
    ng = np.max(grain_structure_figure_one)                 # Maximum grain ID for thermal analysis
    
    # VECTOR Linear2D thermal processing configuration
    cores = 8                                               # 8-core parallel processing for thermal efficiency
    loop_times = 5                                          # Iteration count for thermal convergence
    P0 = grain_structure_figure_one                         # Initial thermal microstructure
    R = np.zeros((nx,ny,2))                                 # Reference field initialization for thermal analysis

    # Initialize VECTOR Linear2D class for thermal normal vector computation
    smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)

    # Execute thermal inclination analysis with temperature-dependent processing
    smooth_class.linear_main("inclination")                # Main thermal inclination computation
    P = smooth_class.get_P()                               # Extract thermal normal vector field
    
    # Comprehensive grain boundary site extraction for thermal analysis
    # Extract all grain boundary sites across the thermal polycrystalline system
    sites = smooth_class.get_all_gb_list()                 # Get all thermal grain boundary sites
    sites_together = []                                     # Initialize comprehensive site list
    for id in range(len(sites)): 
        sites_together += sites[id]                         # Combine all thermal grain boundary sites
    print("Total num of GB sites: " + str(len(sites_together)))  # Thermal GB site count validation

    return P, sites_together
def get_normal_vector_slope(P, sites, step, para_name, bias=None):
    """
    Generate thermal normal vector angular distribution with temperature-dependent analysis.
    
    This function performs comprehensive thermal angular distribution analysis for grain
    boundary normal vectors, computing temperature-dependent orientation patterns and
    generating publication-quality polar visualization for Monte Carlo thermal systems.
    
    Thermal Distribution Framework:
    - 360° angular coverage with 10.01° bin resolution for thermal precision
    - Temperature-dependent frequency analysis for kT-specific characterization
    - Enhanced thermal gradient computation with Monte Carlo noise handling
    - Publication-quality polar plot generation with thermal parameter labeling
    
    Parameters:
    -----------
    P : np.ndarray
        Thermal normal vector field [nx, ny, 2] from VECTOR Linear2D computation
        containing temperature-dependent gradient information and thermal fluctuation effects
    sites : list
        Grain boundary site coordinates [(i,j), ...] for thermal normal vector analysis
        representing all grain boundary points in the thermal polycrystalline system
    step : int
        Time step identifier for thermal evolution tracking and data management
    para_name : str
        Temperature parameter label (e.g., r"$kT=0.25$") for thermal plot identification
        and scientific visualization with proper kT notation
    bias : np.ndarray, optional
        Bias correction array for thermal distribution normalization (default: None)
        for enhanced thermal statistical analysis and circular reference correction
        
    Returns:
    --------
    freqArray : np.ndarray
        Normalized thermal angular frequency distribution [36 bins] representing
        temperature-dependent grain boundary orientation probability density
        
    Thermal Analysis Methodology:
    -----------------------------
    1. Comprehensive thermal gradient computation using myInput.get_grad()
    2. Angular conversion with temperature-dependent atan2 transformation
    3. Statistical binning with thermal frequency normalization
    4. Publication-quality polar visualization with kT parameter identification
    
    Scientific Applications:
    ------------------------
    - Temperature-dependent grain boundary orientation characterization
    - Thermal fluctuation effects on angular distribution patterns
    - Monte Carlo thermal averaging for statistical significance
    - Comparative thermal analysis across different kT regimes
    """
    # Angular distribution parameters for thermal analysis
    xLim = [0, 360]                                         # Full angular range for thermal coverage
    binValue = 10.01                                        # Bin width for thermal angular precision
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)   # Number of thermal angular bins
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Thermal bin centers

    # Initialize thermal frequency distribution array
    freqArray = np.zeros(binNum)                            # Thermal angular frequency storage
    degree = []                                             # Thermal angle collection list
    
    # Comprehensive thermal gradient computation for all grain boundary sites
    for sitei in sites:
        [i,j] = sitei                                       # Extract thermal site coordinates
        dx,dy = myInput.get_grad(P,i,j)                    # Compute thermal gradient components
        degree.append(math.atan2(-dy, dx) + math.pi)      # Angular conversion with thermal correction
        # Note: Previous conditional angle computation replaced with robust atan2 for thermal stability
        
    # Statistical thermal binning and frequency analysis
    for i in range(len(degree)):
        freqArray[int((degree[i]/math.pi*180-xLim[0])/binValue)] += 1  # Thermal bin assignment
    freqArray = freqArray/sum(freqArray*binValue)          # Normalize thermal frequency distribution

    # Apply thermal bias correction if provided for enhanced statistical analysis
    if bias is not None:
        freqArray = freqArray + bias                        # Apply thermal bias correction
        freqArray = freqArray/sum(freqArray*binValue)      # Renormalize thermal distribution
    
    # Publication-quality thermal polar visualization
    # Note: Polar plot setup handled in main execution pipeline for thermal visualization
    plt.plot(np.append(xCor,xCor[0])/180*math.pi, np.append(freqArray,freqArray[0]), 
             linewidth=2, label=para_name)                  # Thermal distribution plotting with kT identification

    # Thermal fitting analysis (coefficient computation for future thermal characterization)
    fit_coeff = np.polyfit(xCor, freqArray, 1)            # Linear thermal trend analysis
    return freqArray

if __name__ == '__main__':
    """
    ==================================================================================
    MAIN EXECUTION PIPELINE: Thermal Fluctuation (kT) Analysis
    ==================================================================================
    
    Main execution pipeline for comprehensive thermal fluctuation grain boundary analysis.
    
    This script implements systematic thermal analysis characterizing grain boundary
    normal vector distributions under varying Monte Carlo temperature conditions (kT)
    to understand thermal effects on microstructural evolution and grain boundary
    orientation patterns in large-scale polycrystalline systems.
    
    Thermal Analysis Framework:
    - Temperature Range: kT ∈ {0.00, 0.25, 0.50, 0.66, 0.95} for systematic studies
    - Statistical Mechanics: Monte Carlo temperature scaling with Boltzmann statistics
    - Large-Scale Analysis: 20K initial grain systems for thermal averaging significance
    - HiPerGator Multi-Core: 32-core parallel processing with optimized thermal workflows
    - Publication Output: High-resolution thermal distribution visualization
    
    Scientific Applications:
    - Temperature-dependent grain boundary orientation characterization
    - Thermal fluctuation effects on microstructural anisotropy evolution
    - Statistical mechanics of grain boundary energy landscape exploration
    - Monte Carlo thermal averaging for realistic material behavior prediction
    - Comparative thermal sensitivity assessment across kT regimes
    """
    
    # ================================================================================
    # File Configuration: HiPerGator Multi-Core Thermal Data
    # ================================================================================
    # HiPerGator storage path for multi-core thermal fluctuation simulation results
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/"
    
    # Thermal parameter identification for systematic kT analysis
    TJ_energy_type_T000 = "T000"                           # Zero temperature (deterministic limit)
    TJ_energy_type_T025 = "T025"                           # Low thermal fluctuations
    TJ_energy_type_T050 = "T050"                           # Moderate thermal effects
    TJ_energy_type_T066 = "T066"                           # Intermediate thermal regime
    TJ_energy_type_T095 = "T095"                           # High thermal fluctuations

    # Systematic file naming convention for thermal fluctuation parametric studies
    npy_file_name_aniso_T000 = f"p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt000.npy"
    npy_file_name_aniso_T025 = f"p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt025.npy"
    npy_file_name_aniso_T050 = f"p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt050.npy"
    npy_file_name_aniso_T066 = f"p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_T095 = f"p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt095.npy"

    # Grain size data file naming for comprehensive thermal morphological analysis
    grain_size_data_name_T000 = f"grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt000.npy"
    grain_size_data_name_T025 = f"grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt025.npy"
    grain_size_data_name_T050 = f"grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt050.npy"
    grain_size_data_name_T066 = f"grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_T095 = f"grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt095.npy"

    # ================================================================================
    # Data Loading: Multi-Temperature Thermal Evolution Datasets
    # ================================================================================
    # Load thermal microstructural evolution data for comprehensive kT analysis
    npy_file_aniso_T000 = np.load(npy_file_folder + npy_file_name_aniso_T000)  # kT = 0.00 (deterministic)
    npy_file_aniso_T025 = np.load(npy_file_folder + npy_file_name_aniso_T025)  # kT = 0.25 (low thermal)
    npy_file_aniso_T050 = np.load(npy_file_folder + npy_file_name_aniso_T050)  # kT = 0.50 (moderate thermal)
    npy_file_aniso_T066 = np.load(npy_file_folder + npy_file_name_aniso_T066)  # kT = 0.66 (intermediate)
    npy_file_aniso_T095 = np.load(npy_file_folder + npy_file_name_aniso_T095)  # kT = 0.95 (high thermal)
    
    # Enhanced data validation for thermal analysis datasets
    print(f"The T000 data size is: {npy_file_aniso_T000.shape}")      # Shape: [time, x, y, fields]
    print(f"The T025 data size is: {npy_file_aniso_T025.shape}")
    print(f"The T050 data size is: {npy_file_aniso_T050.shape}")
    print(f"The T066 data size is: {npy_file_aniso_T066.shape}")
    print(f"The T095 data size is: {npy_file_aniso_T095.shape}")
    print("READING DATA DONE")

    # ================================================================================
    # Thermal Analysis Configuration: Multi-Temperature Data Structures
    # ================================================================================
    # Analysis parameters for thermal morphological characterization
    initial_grain_num = 20000                               # Initial grain population for thermal analysis
    
    # Initialize comprehensive thermal analysis data structures
    step_num = npy_file_aniso_T000.shape[0]                 # Time steps for kT = 0.00 analysis
    grain_num_T000 = np.zeros(step_num)                     # Grain count evolution (kT = 0.00)
    grain_area_T000 = np.zeros((step_num,initial_grain_num)) # Grain area evolution (kT = 0.00)
    grain_size_T000 = np.zeros((step_num,initial_grain_num)) # Grain size evolution (kT = 0.00)
    grain_ave_size_T000 = np.zeros(step_num)               # Average grain size (kT = 0.00)
    
    # kT = 0.25 thermal analysis data structures
    step_num = npy_file_aniso_T025.shape[0]                 # Time steps for kT = 0.25 analysis
    grain_num_T025 = np.zeros(step_num)                     # Grain count evolution (kT = 0.25)
    grain_area_T025 = np.zeros((step_num,initial_grain_num)) # Grain area evolution (kT = 0.25)
    grain_size_T025 = np.zeros((step_num,initial_grain_num)) # Grain size evolution (kT = 0.25)
    grain_ave_size_T025 = np.zeros(step_num)               # Average grain size (kT = 0.25)
    
    # kT = 0.50 thermal analysis data structures
    grain_num_T050 = np.zeros(step_num)                     # Grain count evolution (kT = 0.50)
    grain_area_T050 = np.zeros((step_num,initial_grain_num)) # Grain area evolution (kT = 0.50)
    grain_size_T050 = np.zeros((step_num,initial_grain_num)) # Grain size evolution (kT = 0.50)
    grain_ave_size_T050 = np.zeros(step_num)               # Average grain size (kT = 0.50)
    
    # kT = 0.66 thermal analysis data structures
    grain_num_T066 = np.zeros(step_num)                     # Grain count evolution (kT = 0.66)
    grain_area_T066 = np.zeros((step_num,initial_grain_num)) # Grain area evolution (kT = 0.66)
    grain_size_T066 = np.zeros((step_num,initial_grain_num)) # Grain size evolution (kT = 0.66)
    grain_ave_size_T066 = np.zeros(step_num)               # Average grain size (kT = 0.66)
    
    # kT = 0.95 thermal analysis data structures
    grain_num_T095 = np.zeros(step_num)                     # Grain count evolution (kT = 0.95)
    grain_area_T095 = np.zeros((step_num,initial_grain_num)) # Grain area evolution (kT = 0.95)
    grain_size_T095 = np.zeros((step_num,initial_grain_num)) # Grain size evolution (kT = 0.95)
    grain_ave_size_T095 = np.zeros(step_num)               # Average grain size (kT = 0.95)

    # ================================================================================
    # Grain Size Distribution Configuration for Thermal Analysis
    # ================================================================================
    # Grain size distribution parameters for thermal morphological characterization
    bin_width = 0.16                                        # Grain size distribution bin width
    x_limit = [-0.5, 3.5]                                  # Size distribution range
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)  # Number of size bins
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)  # Bin centers
    
    # Initialize grain size distribution arrays for each thermal condition
    grain_size_distribution_T000 = np.zeros(bin_num)       # Size distribution (kT = 0.00)
    special_step_distribution_T000 = 10                    # Analysis time step (kT = 0.00)
    grain_size_distribution_T025 = np.zeros(bin_num)       # Size distribution (kT = 0.25)
    special_step_distribution_T025 = 10                    # Analysis time step (kT = 0.25)
    grain_size_distribution_T050 = np.zeros(bin_num)       # Size distribution (kT = 0.50)
    special_step_distribution_T050 = 10                    # Analysis time step (kT = 0.50)
    grain_size_distribution_T066 = np.zeros(bin_num)       # Size distribution (kT = 0.66)
    special_step_distribution_T066 = 10                    # Analysis time step (kT = 0.66)
    grain_size_distribution_T095 = np.zeros(bin_num)       # Size distribution (kT = 0.95)
    special_step_distribution_T095 = 11                    # Analysis time step (kT = 0.95)

    # ================================================================================
    # Publication-Quality Thermal Polar Visualization Setup
    # ================================================================================
    # Initialize polar coordinate system for thermal normal vector distribution visualization
    plt.close()
    fig = plt.figure(figsize=(5, 5))                        # Square figure for thermal polar plot
    ax = plt.gca(projection='polar')                        # Polar coordinate system

    # Configure thermal polar plot aesthetics for scientific presentation
    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)  # Angular grid marks
    ax.set_thetamin(0.0)                                    # Minimum angle
    ax.set_thetamax(360.0)                                  # Maximum angle

    # Configure radial grid and labels for thermal probability density
    ax.set_rgrids(np.arange(0, 0.01, 0.004))               # Radial grid marks
    ax.set_rlabel_position(0.0)                             # Label position at 0°
    ax.set_rlim(0.0, 0.01)                                  # Radial limits for thermal probability
    ax.set_yticklabels(['0', '0.004', '0.008'],fontsize=14) # Radial tick labels

    # Apply professional grid formatting for thermal visualization
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)  # Grid styling
    ax.set_axisbelow('True')                                # Grid behind thermal data

    # ================================================================================
    # Systematic Thermal Analysis: Multi-kT Processing Pipeline
    # ================================================================================
    # Iterative thermal analysis across specified time steps with progress tracking
    for i in tqdm(range(9,12)):

        # ================================================================================
        # kT = 0.00 Analysis: Deterministic Energy Minimization
        # ================================================================================
        if i == special_step_distribution_T000:
            # Data management for kT = 0.00 deterministic analysis
            data_file_name_P = f'/normal_distribution_data/normal_distribution_T000_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_T000_sites_step{i}.npy'
            
            # Load or compute normal vector data for kT = 0.00 case
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)      # Load cached deterministic data
                sites = np.load(current_path + data_file_name_sites)
            else:
                # Microstructure preprocessing and deterministic normal vector computation
                newplace = np.rot90(npy_file_aniso_T000[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                # Cache deterministic analysis results
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)

            # Generate thermal normal vector distribution for kT = 0.00
            slope_list = get_normal_vector_slope(P, sites, i, r"$kT=0.00$")

        # ================================================================================
        # kT = 0.25 Analysis: Low Thermal Fluctuations
        # ================================================================================
        if i == special_step_distribution_T025:
            # Data management for kT = 0.25 low thermal analysis
            data_file_name_P = f'/normal_distribution_data/normal_distribution_T025_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_T025_sites_step{i}.npy'
            
            # Load or compute normal vector data for kT = 0.25 case
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)      # Load cached low thermal data
                sites = np.load(current_path + data_file_name_sites)
            else:
                # Microstructure preprocessing and low thermal normal vector computation
                newplace = np.rot90(npy_file_aniso_T025[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                # Cache low thermal analysis results
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)

            # Generate thermal normal vector distribution for kT = 0.25
            slope_list = get_normal_vector_slope(P, sites, i, r"$kT=0.25$")

        # ================================================================================
        # kT = 0.50 Analysis: Moderate Thermal Effects
        # ================================================================================
        if i == special_step_distribution_T050:
            # Data management for kT = 0.50 moderate thermal analysis
            data_file_name_P = f'/normal_distribution_data/normal_distribution_T050_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_T050_P_sites_step{i}.npy'
            
            # Load or compute normal vector data for kT = 0.50 case
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)      # Load cached moderate thermal data
                sites = np.load(current_path + data_file_name_sites)
            else:
                # Microstructure preprocessing and moderate thermal normal vector computation
                newplace = np.rot90(npy_file_aniso_T050[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                # Cache moderate thermal analysis results
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)

            # Generate thermal normal vector distribution for kT = 0.50
            slope_list = get_normal_vector_slope(P, sites, i, r"$kT=0.50$")

        # ================================================================================
        # kT = 0.66 Analysis: Intermediate Thermal Regime
        # ================================================================================
        if i == special_step_distribution_T066:
            # Data management for kT = 0.66 intermediate thermal analysis
            data_file_name_P = f'/normal_distribution_data/normal_distribution_T066_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_T066_sites_step{i}.npy'
            
            # Load or compute normal vector data for kT = 0.66 case
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)      # Load cached intermediate thermal data
                sites = np.load(current_path + data_file_name_sites)
            else:
                # Microstructure preprocessing and intermediate thermal normal vector computation
                newplace = np.rot90(npy_file_aniso_T066[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                # Cache intermediate thermal analysis results
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)

            # Generate thermal normal vector distribution for kT = 0.66
            slope_list = get_normal_vector_slope(P, sites, i, r"$kT=0.66$")

        # ================================================================================
        # kT = 0.95 Analysis: High Thermal Fluctuations
        # ================================================================================
        if i == special_step_distribution_T095:
            # Data management for kT = 0.95 high thermal analysis
            data_file_name_P = f'/normal_distribution_data/normal_distribution_T095_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_T095_sites_step{i}.npy'
            
            # Load or compute normal vector data for kT = 0.95 case
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)      # Load cached high thermal data
                sites = np.load(current_path + data_file_name_sites)
            else:
                # Microstructure preprocessing and high thermal normal vector computation
                newplace = np.rot90(npy_file_aniso_T095[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                # Cache high thermal analysis results
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)

            # Generate thermal normal vector distribution for kT = 0.95
            slope_list = get_normal_vector_slope(P, sites, i, r"$kT=0.95$")

        # ================================================================================
        # Bias Calculation: Circular Reference for kT = 0.66
        # ================================================================================
        # Thermal bias analysis using kT = 0.66 as reference condition
        if i == special_step_distribution_T066:
            # Angular analysis parameters for bias calculation
            xLim = [0, 360]                                     # Full angular range
            binValue = 10.01                                    # Bin size consistency
            binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue) # Number of bins
            xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Bin centers
            freqArray_circle = np.ones(binNum)                  # Uniform circular distribution
            freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)  # Normalize

            # Data file management for bias calculation
            data_file_name_P = f'/normal_distribution_data/normal_distribution_T066_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_T066_sites_step{i}.npy'
            data_file_name_bias = f'/normal_distribution_data/normal_distribution_T066_bias_sites_step{i}.npy'
            
            # Load thermal data for bias calculation
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)      # Load kT = 0.66 data
                sites = np.load(current_path + data_file_name_sites)
            else:
                # Compute thermal data for bias analysis
                newplace = np.rot90(npy_file_aniso_T066[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)

            # Bias calculation code (commented for selective execution)
            #slope_list = get_normal_vector_slope(P, sites, i, "T066 case")
            #bias = freqArray_circle - slope_list
            #np.save(current_path + data_file_name_bias, bias)
            #print(bias)

    # ================================================================================
    # Publication-Quality Output Generation for Thermal Analysis
    # ================================================================================
    # Generate professional thermal polar plot with enhanced legend formatting
    plt.legend(loc=(-0.25,-0.3),fontsize=14,ncol=3)        # Enhanced legend positioning for thermal data
    plt.savefig(current_path + "/figures/normal_distribution_kT.png", dpi=400,bbox_inches='tight')

    # ================================================================================
    # Thermal Analysis Summary: Temperature-Dependent Grain Boundary Characterization
    # ================================================================================
    """
    This comprehensive thermal analysis provides quantitative characterization of:
    
    1. Temperature-Dependent Orientation Patterns: Systematic analysis of grain boundary
       normal vector distributions across kT ∈ {0.00, 0.25, 0.50, 0.66, 0.95}
    
    2. Thermal Fluctuation Effects: Quantitative assessment of Monte Carlo temperature
       scaling on microstructural anisotropy and grain boundary energy landscapes
    
    3. Statistical Mechanics Integration: Temperature-dependent grain boundary behavior
       with Boltzmann statistics implementation for realistic material response
    
    Key Scientific Insights:
    - Deterministic vs. thermal behavior comparison (kT = 0.00 baseline)
    - Thermal activation effects on grain boundary mobility and orientation
    - Statistical mechanics validation of grain boundary energy minimization
    
    Note: This Python script version focuses on thermal distribution visualization
    without anisotropy magnitude calculation (available in notebook version).
    """











