#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================
VECTOR Framework: Grain Size Distribution Analysis for Energy Function Verification
=========================================================================================

Scientific Application: Energy Function Validation through Statistical Grain Size Analysis
Primary Focus: Comparative Grain Size Distribution Analysis for Energy Function Verification

Created on Mon Jul 31 14:33:57 2023
@author: Lin

=========================================================================================
ENERGY FUNCTION VERIFICATION FRAMEWORK
=========================================================================================

This Python script implements comprehensive grain size distribution analysis for energy
function verification in Monte Carlo Potts model grain growth simulations. The analysis
focuses on comparing different energy formulations (anisotropic, isotropic, well energy)
through statistical characterization of grain size distributions and validation against
theoretical grain growth models.

Scientific Objectives:
- Energy function validation through grain size distribution comparison
- Statistical characterization of grain growth under different energy formulations
- Verification of energy function effects on grain size evolution
- Comparative analysis of anisotropic vs. isotropic energy models
- Validation of well energy functions for abnormal grain growth studies

Key Features:
- Multi-energy function comparison (aniso, aniso_abnormal, isotropic)
- Normalized grain size distribution analysis with statistical validation
- 3D grain volume calculation from pixel count data
- High-resolution visualization for publication-quality analysis
- HiPerGator data integration for large-scale simulation datasets

Applications:
- Energy function verification and validation studies
- Grain growth model comparison and benchmarking
- Statistical grain size analysis for materials science applications
- Verification of energy function effects on microstructural evolution
"""

# ================================================================================
# ENVIRONMENT SETUP AND PATH CONFIGURATION
# ================================================================================
import os
current_path = os.getcwd()                       # Current working directory for file operations
import numpy as np                               # Numerical computing and array operations
from numpy import seterr                         # Numerical error handling configuration
seterr(all='raise')                             # Raise exceptions for numerical errors
import matplotlib.pyplot as plt                  # Advanced scientific visualization
import math                                      # Mathematical functions for calculations
from tqdm import tqdm                            # Progress bar for long-running operations
import sys
sys.path.append(current_path+'/../../')          # Add VECTOR framework root directory

# ================================================================================
# VECTOR FRAMEWORK INTEGRATION: SPECIALIZED ANALYSIS MODULES
# ================================================================================
import myInput                                   # Input parameter management and file handling
import post_processing as inclination_processing # Core post-processing functions for grain analysis
import PACKAGE_MP_3DLinear as linear3d          # 3D linear algebra for grain boundary analysis


# ================================================================================
# ENERGY FUNCTION COMPARISON CONFIGURATION
# ================================================================================
"""
Energy Function Verification Setup:
- Comparative analysis of different energy formulations for grain growth simulation
- Statistical validation through grain size distribution comparison
- Energy function effect characterization on microstructural evolution
"""

if __name__ == '__main__':
    # ================================================================================
    # DATASET IDENTIFICATION AND FILE MANAGEMENT
    # ================================================================================
    """
    HiPerGator Dataset Configuration:
    - Large-scale simulation data from University of Florida's HiPerGator supercomputer
    - Multi-core parallel simulation results for statistical significance
    - Comprehensive energy function comparison dataset
    """
    
    # Primary case identification for analysis
    case_name = "264_5k"                         # Case identifier: 264x264x264 grid, 5000 initial grains
    
    # HiPerGator data directory structure for organized data management
    init_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_poly/IC/"
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_poly_fully/results/"
    
    # ================================================================================
    # ENERGY FUNCTION DATASET CONFIGURATION
    # ================================================================================
    """
    Multi-Energy Function Analysis Setup:
    - Anisotropic energy: Crystallographic orientation-dependent grain boundary energy
    - Anisotropic abnormal: Modified anisotropic energy for abnormal grain growth
    - Isotropic energy: Orientation-independent baseline energy model
    """
    
    # Anisotropic energy function simulation results
    npy_file_name_aniso = f"p_ori_fully5d_fz_aveE_f1.0_t1.0_264_5k_multiCore64_J1_refer_1_0_0_seed56689_kt1.95.npy"
    
    # Anisotropic abnormal grain growth energy function results
    npy_file_name_anisoab = f"p_ori_fully5d_fzab_aveE_f1.0_t1.0_264_5k_multiCore64_J1_refer_1_0_0_seed56689_kt1.95.npy"
    
    # Isotropic energy function baseline comparison results
    npy_file_name_iso = f"p_iso_264_5k_multiCore64_J1_refer_1_0_0_seed56689_kt1.95.npy"
    
    # Consolidated dataset array for comparative analysis
    input_npy_data = [npy_file_folder+npy_file_name_aniso, npy_file_folder+npy_file_name_anisoab, npy_file_folder+npy_file_name_iso]
    
    # Energy function labels for visualization and analysis identification
    compare_label = ["Aniso", "Aniso_Abnormal", "Iso"]

    # ================================================================================
    # GRAIN SIZE ANALYSIS PARAMETERS
    # ================================================================================
    """
    Statistical Analysis Configuration:
    - Target grain number for comparative analysis across energy functions
    - Ensures consistent statistical sampling for valid comparison
    """
    
    # Get time step with expected grain num
    expected_grain_num = 1000                    # Target grain count for statistical analysis consistency
    
    # ================================================================================
    # TEMPORAL ANALYSIS: OPTIMAL TIME STEP IDENTIFICATION
    # ================================================================================
    """
    Time Step Selection for Comparative Analysis:
    - Identifies simulation time steps where all energy functions reach target grain count
    - Ensures statistical validity by comparing equivalent grain evolution states
    - Enables fair comparison across different energy formulations
    """
    
    special_step_distribution, microstructure_list = inclination_processing.calculate_expected_step(input_npy_data, expected_grain_num)
    print(f"Steps for {compare_label} are {list(map(int, special_step_distribution))}")
    
    # ================================================================================
    # GRAIN SIZE DISTRIBUTION CALCULATION AND CACHING
    # ================================================================================
    """
    Grain Size Distribution Analysis Workflow:
    - Calculates 3D grain volumes from voxel-based microstructure data
    - Normalizes grain sizes for statistical comparison across energy functions
    - Implements data caching for computational efficiency in repeated analysis
    """
    
    # Get grain size distribution
    grain_size_list_norm_list = []              # Storage for normalized grain size distributions
    
    for i in range(len(input_npy_data)):
        # ================================================================================
        # DATA CACHING AND RETRIEVAL SYSTEM
        # ================================================================================
        """
        Efficient Data Management:
        - Checks for pre-computed grain size distribution data
        - Loads cached results when available for faster analysis
        - Computes and caches new results when data not found
        """
        
        size_data_name = f"/size_data/grain_size_data_{case_name}_{compare_label[i]}_step{special_step_distribution[i]}.npz"
        
        if os.path.exists(current_path + size_data_name):
            # Load pre-computed grain size distribution data
            grain_size_npz_file = np.load(current_path + size_data_name)
            grain_size_list_norm = grain_size_npz_file["grain_size_list_norm"]
        else:
            # ================================================================================
            # REAL-TIME GRAIN SIZE CALCULATION
            # ================================================================================
            """
            3D Grain Volume Analysis:
            - Extracts individual grain domains from 3D microstructure data
            - Calculates grain volumes through voxel counting method
            - Normalizes grain sizes for statistical distribution analysis
            """
            
            current_microstructure = microstructure_list[i]    # Current energy function microstructure
            grain_id_list = np.unique(current_microstructure)  # Extract unique grain identifiers
            grain_area_list = np.zeros(len(grain_id_list))     # Initialize grain volume storage
            
            # Calculate individual grain volumes through voxel counting
            for k in range(len(grain_id_list)):
                # Count voxels belonging to each grain (3D volume calculation)
                grain_area_list[k] = np.sum(current_microstructure==grain_id_list[k])

            # ================================================================================
            # GRAIN SIZE NORMALIZATION AND STATISTICAL PROCESSING
            # ================================================================================
            """
            Statistical Grain Size Analysis:
            - Converts 3D grain volumes to equivalent spherical radii
            - Normalizes grain sizes by average grain size for comparative analysis
            - Saves processed data for future analysis and computational efficiency
            """
            
            # Convert grain volumes to equivalent spherical radii
            grain_size_list = (grain_area_list*3/4/np.pi)**(1/3)    # Equivalent spherical radius calculation
            
            # Calculate average grain size for normalization
            grain_size_ave = np.sum(grain_size_list)/len(grain_size_list)
            
            # Normalize grain sizes by average for statistical comparison
            grain_size_list_norm = grain_size_list/grain_size_ave
            
            # Log-scale transformation for distribution analysis
            grain_size_list_norm_log = np.log10(grain_size_list_norm)
            
            # Cache computed grain size distribution for future analysis
            np.savez(current_path + size_data_name,grain_size_list_norm=grain_size_list_norm)
            
        # Accumulate normalized grain size distributions for comparative analysis
        grain_size_list_norm_list.append(grain_size_list_norm)

    # ================================================================================
    # STATISTICAL DISTRIBUTION ANALYSIS AND VISUALIZATION
    # ================================================================================
    """
    Comparative Grain Size Distribution Analysis:
    - Generates normalized grain size distribution histograms for energy function comparison
    - Implements high-resolution binning for detailed statistical characterization
    - Prepares data for publication-quality visualization and scientific analysis
    """

    # plot Normalized Grain Size Distribution figure [-2.5,1.5]
    xLim = [-0.5, 4.0]                          # Distribution range for grain size analysis
    binValue = 0.02                             # High-resolution bin width for detailed analysis
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)   # Calculate number of bins
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Bin center coordinates
    
    case_num = len(input_npy_data)              # Number of energy functions for comparison
    freqArray = np.zeros((case_num, binNum))    # Initialize frequency distribution array
    
    for i in range(case_num):
        # ================================================================================
        # HISTOGRAM GENERATION FOR STATISTICAL DISTRIBUTION ANALYSIS
        # ================================================================================
        """
        High-Resolution Distribution Binning:
        - Maps individual grain sizes to distribution bins
        - Normalizes frequency counts for probability density representation
        - Generates statistical characterization suitable for energy function comparison
        """
        
        # Map each grain size to appropriate distribution bin
        for k in range(len(grain_size_list_norm_list[i])):
            # Calculate bin index for current grain size
            freqArray[i, int((grain_size_list_norm_list[i][k]-xLim[0])/binValue)] += 1
            
        # Normalize frequency counts to probability density
        freqArray[i] = freqArray[i] / sum(freqArray[i]*binValue)
    
    # ================================================================================
    # PUBLICATION-QUALITY VISUALIZATION GENERATION
    # ================================================================================
    """
    Scientific Visualization for Energy Function Comparison:
    - Generates high-resolution publication-quality distribution plots
    - Implements professional formatting for scientific publication
    - Provides clear visual comparison of energy function effects on grain size distribution
    """
    
    plt.figure()                                # Initialize new figure for distribution comparison
    
    # Plot grain size distributions for all energy functions
    for i in range(case_num):
        plt.plot(xCor,freqArray[i], linewidth=2, label=compare_label[i])
    
    # Professional scientific formatting
    plt.xlabel(r"R/$\langle$R$\rangle$", fontsize=18)    # Normalized grain size axis label
    plt.ylabel("frequence", fontsize=18)                  # Frequency density axis label
    plt.title(f"grain num: {expected_grain_num}", fontsize=18)  # Descriptive title with grain count
    plt.xticks(fontsize=18)                              # Enhanced x-axis tick formatting
    plt.yticks(fontsize=18)                              # Enhanced y-axis tick formatting
    plt.ylim([0,2.1])                                    # Frequency axis range optimization
    plt.xlim(xLim)                                       # Grain size axis range
    
    # High-resolution figure export for publication
    plt.savefig(f'./size_figures/normalized_grain_size_distribution_{case_name}_compare.png',dpi=400,bbox_inches='tight')

