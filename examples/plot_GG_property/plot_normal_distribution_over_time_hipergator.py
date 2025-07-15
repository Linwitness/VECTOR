#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================
VECTOR Framework: Grain Boundary Energy Function Comparison Analysis
=========================================================================================

Scientific Application: Comprehensive Energy Averaging Method Characterization
Primary Focus: Comparative Analysis of Multiple Grain Boundary Energy Functions

Created on Mon Jul 31 14:33:57 2023
@author: Lin

=========================================================================================
ENERGY FUNCTION COMPARISON FRAMEWORK
=========================================================================================

This Python script implements comprehensive grain boundary energy function comparison
analysis for large-scale polycrystalline microstructures, systematically characterizing
different energy averaging methods and their effects on grain boundary orientation
patterns and anisotropy evolution under controlled thermal conditions (kT = 0.66).

Key Scientific Objectives:
1. Systematic comparison of 6 energy averaging methods with isotropic baseline
2. Grain boundary energy function validation through orientation distribution analysis
3. Bias correction methodology for enhanced comparative statistical analysis
4. HiPerGator 64-core processing optimization for energy function characterization

=========================================================================================
ENERGY AVERAGING METHODS AND COMPARATIVE FRAMEWORK
=========================================================================================

Energy Function Categories:
- ave: Average energy function (standard grain boundary energy averaging)
- consMin: Constant minimum energy approach (minimal energy selection)
- sum: Summation energy function (cumulative energy aggregation)
- min: Minimum energy function (absolute minimum energy selection)
- max: Maximum energy function (absolute maximum energy selection)
- consMax: Constant maximum energy approach (maximal energy selection)
- iso: Isotropic baseline (delta = 0.0, no anisotropic energy contribution)

Comparative Analysis Framework:
- Fixed thermal conditions: kT = 0.66 for consistent thermal background
- Delta parameter: 0.6 for anisotropic energy functions, 0.0 for isotropic baseline
- Bias correction: kT = 0.66 reference for enhanced statistical comparison
- Multi-core processing: 64-core HiPerGator optimization for energy computations

=========================================================================================
COMPUTATIONAL FRAMEWORK AND OPTIMIZATION
=========================================================================================

VECTOR Linear2D Processing:
- 8-core parallel processing for efficient normal vector computation
- Multi-time step analysis with systematic data caching
- Optimized grain boundary site extraction for energy function comparison
- Enhanced angular distribution analysis for energy-dependent patterns

HiPerGator 64-Core Integration:
- 64-core parallel simulation data processing for energy function datasets
- Large-scale polycrystalline systems (20K initial grains)
- HiPerGator storage optimization for comparative energy analysis
- Publication-quality polar visualization with bias correction

=========================================================================================
ENERGY FUNCTION ANALYSIS METHODOLOGY
=========================================================================================

Comparative Analysis Pipeline:
1. Multi-energy function dataset loading and validation
2. Systematic time step optimization for consistent grain count comparison
3. Energy-dependent normal vector computation using VECTOR Linear2D
4. Bias-corrected angular distribution analysis for fair comparison
5. Publication-quality comparative polar visualization generation

Data Management Strategy:
- Systematic energy function data caching for computational efficiency
- Energy-specific file naming conventions with parameter identification
- Progress tracking for multi-energy function analysis workflows
- Bias correction integration for enhanced comparative statistical analysis

=========================================================================================
SCIENTIFIC APPLICATIONS AND INSIGHTS
=========================================================================================

This energy function comparison framework enables quantitative characterization of:

1. Energy Function Sensitivity: Systematic analysis of grain boundary orientation
   sensitivity to different energy averaging methodologies

2. Comparative Energy Validation: Understanding energy function effects on
   microstructural anisotropy and grain boundary energy landscape characterization

3. Bias Correction Methodology: Enhanced statistical comparison through thermal
   reference bias correction for fair energy function evaluation

4. Energy Averaging Optimization: Characterization of optimal energy averaging
   approaches for realistic grain boundary energy minimization

Key Scientific Insights:
- Energy function comparative sensitivity analysis across averaging methods
- Bias correction effects on energy function orientation distribution comparison
- Energy averaging methodology validation for grain boundary energy landscapes
- Optimal energy function selection for realistic polycrystalline behavior modeling

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

def get_poly_center(micro_matrix, step):
    """
    Calculate geometric center of all non-periodic grains in microstructural matrix.
    
    This function computes the centroid coordinates and average radius for each grain
    in the microstructural system, excluding periodic boundary grains that may
    artificially affect energy function analysis due to boundary artifacts.
    
    Energy Function Analysis Framework:
    - Non-periodic grain identification to avoid boundary artifacts in energy comparison
    - Geometric centroid calculation for grain morphology characterization
    - Average radius computation for grain size distribution analysis
    - Boundary condition filtering for accurate energy function validation
    
    Parameters:
    -----------
    micro_matrix : np.ndarray
        4D microstructural array [time, nx, ny, fields] containing grain boundary data
        with energy function-dependent grain orientations and boundary configurations
    step : int
        Time step index for grain center analysis and energy function characterization
        
    Returns:
    --------
    center_list : np.ndarray
        Grain center coordinates [num_grains, 2] for all non-periodic grains
        with accurate centroid positions for energy function morphology analysis
    ave_radius_list : np.ndarray
        Average radius array [num_grains] computed from grain area assuming circular geometry
        for energy function-dependent grain size characterization
        
    Energy Function Morphological Analysis:
    ---------------------------------------
    1. Comprehensive grain identification and counting across energy functions
    2. Periodic boundary condition exclusion for accurate energy comparison
    3. Geometric centroid computation for energy-dependent morphology analysis
    4. Statistical radius calculation for energy function grain size validation
    """
    # Get the center of all non-periodic grains in matrix
    num_grains = int(np.max(micro_matrix[step,:]))          # Total grain count for energy analysis
    center_list = np.zeros((num_grains,2))                  # Initialize grain center storage
    sites_num_list = np.zeros(num_grains)                   # Initialize grain size storage
    ave_radius_list = np.zeros(num_grains)                  # Initialize radius storage
    
    # Create coordinate reference matrices for centroid calculation
    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i,j] = i                          # I-coordinate reference
            coord_refer_j[i,j] = j                          # J-coordinate reference

    # Extract grain ID table for current time step
    table = micro_matrix[step,:,:,0]                        # Grain ID matrix for energy analysis
    
    # Calculate center and radius for each grain
    for i in range(num_grains):
        sites_num_list[i] = np.sum(table == i+1)           # Count sites for grain i+1

        # Check for periodic boundary conditions or empty grains
        if (sites_num_list[i] == 0) or \
           (np.max(coord_refer_i[table == i+1]) - np.min(coord_refer_i[table == i+1]) == micro_matrix.shape[1]) or \
           (np.max(coord_refer_j[table == i+1]) - np.min(coord_refer_j[table == i+1]) == micro_matrix.shape[2]): 
          # Grains on boundary conditions are excluded from energy analysis
          center_list[i, 0] = 0                             # Set zero center for boundary grains
          center_list[i, 1] = 0
          sites_num_list[i] == 0                            # Mark as excluded
        else:
          # Calculate geometric centroid for non-periodic grains
          center_list[i, 0] = np.sum(coord_refer_i[table == i+1]) / sites_num_list[i]  # X-centroid
          center_list[i, 1] = np.sum(coord_refer_j[table == i+1]) / sites_num_list[i]  # Y-centroid
          
    # Calculate average radius assuming circular grain geometry
    ave_radius_list = np.sqrt(sites_num_list / np.pi)      # Circular equivalent radius

    return center_list, ave_radius_list

def get_poly_statistical_radius(micro_matrix, sites_list, step):
    """
    Calculate statistical radius deviation for grain morphology characterization.
    
    This function computes the maximum radius offset between actual grain boundaries
    and theoretical circular grain geometry, providing quantitative morphological
    analysis for energy function validation and grain shape characterization.
    
    Energy Function Morphological Validation:
    - Maximum radius deviation calculation for grain shape irregularity assessment
    - Statistical morphology characterization across different energy functions
    - Grain boundary curvature analysis for energy function morphological effects
    - Normalized radius offset for energy function comparative shape analysis
    
    Parameters:
    -----------
    micro_matrix : np.ndarray
        4D microstructural array [time, nx, ny, fields] containing energy function data
        with grain boundary configurations affected by different energy averaging methods
    sites_list : list
        List of grain boundary site lists for each grain in the energy function system
        containing coordinate information for morphological analysis
    step : int
        Time step index for statistical radius analysis and energy function validation
        
    Returns:
    --------
    max_radius_offset : float
        Average maximum normalized radius offset across all valid grains
        representing grain shape deviation from circular geometry for energy function analysis
        
    Energy Function Shape Analysis:
    -------------------------------
    1. Comprehensive grain center and radius calculation using get_poly_center()
    2. Individual grain boundary site radius calculation for actual morphology
    3. Maximum radius deviation identification for shape irregularity assessment
    4. Statistical averaging across all valid grains for energy function comparison
    """
    # Get the max offset of average radius and real radius
    center_list, ave_radius_list = get_poly_center(micro_matrix, step)  # Grain centers and radii
    num_grains = int(np.max(micro_matrix[step,:]))          # Total grain count

    # Initialize maximum radius offset tracking for all grains
    max_radius_offset_list = np.zeros(num_grains)           # Maximum offset storage
    
    # Calculate maximum radius offset for each grain
    for n in range(num_grains):
        center = center_list[n]                             # Grain center coordinates
        ave_radius = ave_radius_list[n]                     # Average circular radius
        sites = sites_list[n]                               # Grain boundary sites

        # Process only valid grains with non-zero radius
        if ave_radius != 0:
          # Calculate radius offset for each grain boundary site
          for sitei in sites:
              [i,j] = sitei                                 # Site coordinates
              # Calculate actual radius from center to boundary site
              current_radius = np.sqrt((i - center[0])**2 + (j - center[1])**2)
              # Calculate absolute radius deviation from average
              radius_offset = abs(current_radius - ave_radius)
              # Track maximum offset for current grain
              if radius_offset > max_radius_offset_list[n]: 
                  max_radius_offset_list[n] = radius_offset

          # Normalize maximum offset by average radius for relative comparison
          max_radius_offset_list[n] = max_radius_offset_list[n] / ave_radius

    # Calculate average normalized maximum radius offset across all valid grains
    max_radius_offset = np.average(max_radius_offset_list[max_radius_offset_list!=0])
    return max_radius_offset

def get_normal_vector(grain_structure_figure_one, grain_num):
    """
    Compute normal vectors for energy function grain boundary analysis using VECTOR Linear2D.
    
    This function implements energy function-aware grain boundary normal vector computation
    for comparative energy function analysis, utilizing 8-core parallel processing to efficiently
    extract grain boundary sites and compute orientation vectors under different energy
    averaging methodologies for comprehensive comparative characterization.
    
    Energy Function Analysis Framework:
    - 8-core parallel processing for efficient energy function normal vector computation
    - Multi-grain boundary site extraction with energy function-dependent configurations
    - Optimized inclination analysis for energy averaging method comparison
    - Enhanced grain boundary list generation for large-scale energy function systems
    
    Parameters:
    -----------
    grain_structure_figure_one : np.ndarray
        3D microstructural array [nx, ny, fields] containing energy function grain boundary data
        with specific energy averaging method applied to grain orientations and boundaries
    grain_num : int
        Total number of grains in the energy function polycrystalline system (typically 20,000)
        for statistical significance in comparative energy function analysis
        
    Returns:
    --------
    P : np.ndarray
        Normal vector field [nx, ny, 2] containing energy function-dependent gradient information
        with energy averaging method-specific orientation patterns and boundary configurations
    sites_together : list
        Comprehensive grain boundary site list [(i,j), ...] for energy function analysis
        containing all grain boundary coordinates for comparative energy function characterization
        
    Energy Function Computational Framework:
    ----------------------------------------
    1. Multi-core parallel processing (8 cores) for efficient energy function computation
    2. VECTOR Linear2D inclination analysis with energy function-dependent processing
    3. Comprehensive grain boundary site extraction across all energy function grains
    4. Energy averaging method-aware gradient computation for comparative analysis
    
    Scientific Applications:
    ------------------------
    - Energy function-dependent grain boundary orientation analysis
    - Comparative energy averaging method effects on normal vector distributions
    - Energy function validation through grain boundary site characterization
    - Large-scale polycrystalline energy function comparison
    """
    # Microstructural dimensions for energy function analysis
    nx = grain_structure_figure_one.shape[0]                # X-dimension for energy function grid
    ny = grain_structure_figure_one.shape[1]                # Y-dimension for energy function grid
    ng = np.max(grain_structure_figure_one)                 # Maximum grain ID for energy analysis
    
    # VECTOR Linear2D energy function processing configuration
    cores = 8                                               # 8-core parallel processing for efficiency
    loop_times = 5                                          # Iteration count for energy function convergence
    P0 = grain_structure_figure_one                         # Initial energy function microstructure
    R = np.zeros((nx,ny,2))                                 # Reference field initialization

    # Initialize VECTOR Linear2D class for energy function normal vector computation
    smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)

    # Execute energy function inclination analysis with method-dependent processing
    smooth_class.linear_main("inclination")                # Main energy function inclination computation
    P = smooth_class.get_P()                               # Extract energy function normal vector field
    
    # Comprehensive grain boundary site extraction for energy function analysis
    sites = smooth_class.get_all_gb_list()                 # Get all energy function grain boundary sites
    sites_together = []                                     # Initialize comprehensive site list
    for id in range(len(sites)): 
        sites_together += sites[id]                         # Combine all energy function grain boundary sites
    print("Total num of GB sites: " + str(len(sites_together)))  # Energy function GB site count validation

    return P, sites_together

def get_normal_vector_slope(P, sites, step, para_name, bias=None):
    """
    Generate energy function normal vector angular distribution with bias correction.
    
    This function performs comprehensive energy function angular distribution analysis
    for grain boundary normal vectors, computing energy averaging method-dependent
    orientation patterns and generating publication-quality polar visualization with
    bias correction for fair comparative analysis across different energy functions.
    
    Energy Function Distribution Framework:
    - 360° angular coverage with 10.01° bin resolution for energy function precision
    - Energy averaging method-dependent frequency analysis for comparative characterization
    - Enhanced energy function gradient computation with method-specific processing
    - Bias-corrected polar plot generation with energy function parameter labeling
    
    Parameters:
    -----------
    P : np.ndarray
        Energy function normal vector field [nx, ny, 2] from VECTOR Linear2D computation
        containing energy averaging method-dependent gradient information
    sites : list
        Grain boundary site coordinates [(i,j), ...] for energy function analysis
        representing all grain boundary points in the energy function system
    step : int
        Time step identifier for energy function evolution tracking and data management
    para_name : str
        Energy function parameter label (e.g., "Ave", "Min", "Max") for plot identification
        and scientific visualization with proper energy function notation
    bias : np.ndarray, optional
        Bias correction array for energy function distribution normalization
        derived from kT = 0.66 reference for enhanced comparative statistical analysis
        
    Returns:
    --------
    fit_coeff[0] : float
        Linear fitting coefficient representing energy function angular distribution slope
        for quantitative comparative analysis across different energy averaging methods
        
    Energy Function Analysis Methodology:
    -------------------------------------
    1. Comprehensive energy function gradient computation using myInput.get_grad()
    2. Angular conversion with energy function-dependent atan2 transformation
    3. Statistical binning with energy function frequency normalization
    4. Bias correction integration for fair comparative analysis
    5. Publication-quality polar visualization with energy function identification
    
    Scientific Applications:
    ------------------------
    - Energy averaging method-dependent grain boundary orientation characterization
    - Comparative energy function effects on angular distribution patterns
    - Bias-corrected energy function validation for statistical significance
    - Quantitative energy function comparison through slope coefficient analysis
    """
    # Angular distribution parameters for energy function analysis
    xLim = [0, 360]                                         # Full angular range for energy function coverage
    binValue = 10.01                                        # Bin width for energy function angular precision
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)   # Number of energy function angular bins
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Energy function bin centers

    # Initialize energy function frequency distribution array
    freqArray = np.zeros(binNum)                            # Energy function angular frequency storage
    degree = []                                             # Energy function angle collection list
    
    # Comprehensive energy function gradient computation for all grain boundary sites
    for sitei in sites:
        [i,j] = sitei                                       # Extract energy function site coordinates
        dx,dy = myInput.get_grad(P,i,j)                    # Compute energy function gradient components
        degree.append(math.atan2(-dy, dx) + math.pi)      # Angular conversion with energy function correction
        
    # Statistical energy function binning and frequency analysis
    for i in range(len(degree)):
        freqArray[int((degree[i]/math.pi*180-xLim[0])/binValue)] += 1  # Energy function bin assignment
    freqArray = freqArray/sum(freqArray*binValue)          # Normalize energy function frequency distribution

    # Apply bias correction if provided for fair energy function comparison
    if list(bias) != None:
        freqArray = freqArray + bias                        # Apply energy function bias correction
        freqArray = freqArray/sum(freqArray*binValue)      # Renormalize energy function distribution
    
    # Publication-quality energy function polar visualization
    plt.plot(np.append(xCor,xCor[0])/180*math.pi, np.append(freqArray,freqArray[0]), 
             linewidth=2, label=para_name)                  # Energy function distribution plotting

    # Energy function fitting analysis for quantitative comparison
    fit_coeff = np.polyfit(xCor, freqArray, 1)            # Linear energy function trend analysis
    return fit_coeff[0]                                     # Return slope coefficient for comparison

if __name__ == '__main__':
    """
    ==================================================================================
    MAIN EXECUTION PIPELINE: Energy Function Comparison Analysis
    ==================================================================================
    
    Main execution pipeline for comprehensive grain boundary energy function comparison.
    
    This script implements systematic energy function comparison analysis characterizing
    grain boundary normal vector distributions under different energy averaging methods
    to understand energy function effects on microstructural evolution and grain boundary
    orientation patterns in large-scale polycrystalline systems with bias correction.
    
    Energy Function Comparison Framework:
    - Energy Methods: 6 averaging approaches (ave, consMin, sum, min, max, consMax) + iso baseline
    - Fixed Thermal Conditions: kT = 0.66 for consistent comparative background
    - Delta Parameters: 0.6 for anisotropic functions, 0.0 for isotropic baseline
    - HiPerGator 64-Core: Massive parallel processing for energy function computations
    - Bias Correction: kT = 0.66 reference for enhanced statistical comparison
    
    Scientific Applications:
    - Energy averaging method sensitivity analysis for grain boundary orientation
    - Comparative energy function validation through orientation distribution analysis
    - Bias correction methodology for fair energy function statistical comparison
    - Energy function optimization for realistic grain boundary energy minimization
    """
    
    # ================================================================================
    # File Configuration: HiPerGator 64-Core Energy Function Data
    # ================================================================================
    # HiPerGator storage path for 64-core energy function simulation results
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/"
    
    # Energy function type identification for systematic comparative analysis
    TJ_energy_type_ave = "ave"                             # Average energy function
    TJ_energy_type_consMin = "consMin"                     # Constant minimum energy
    TJ_energy_type_sum = "sum"                             # Summation energy function
    TJ_energy_type_min = "min"                             # Minimum energy function
    TJ_energy_type_max = "max"                             # Maximum energy function
    TJ_energy_type_consMax = "consMax"                     # Constant maximum energy

    # Systematic file naming convention for energy function parametric studies
    # All anisotropic energy functions use delta=0.6, kT=0.66 for fair comparison
    npy_file_name_aniso_ave = f"p_ori_ave_{TJ_energy_type_ave}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMin = f"p_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_sum = f"p_ori_ave_{TJ_energy_type_sum}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_min = f"p_ori_ave_{TJ_energy_type_min}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_max = f"p_ori_ave_{TJ_energy_type_max}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMax = f"p_ori_ave_{TJ_energy_type_consMax}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    
    # Isotropic baseline with delta=0.0 for energy function comparison reference
    npy_file_name_iso = "p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    
    # Grain size data file naming for comprehensive energy function morphological analysis
    grain_size_data_name_ave = f"grain_size_p_ori_ave_{TJ_energy_type_ave}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_consMin = f"grain_size_p_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_sum = f"grain_size_p_ori_ave_{TJ_energy_type_sum}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_min = f"grain_size_p_ori_ave_{TJ_energy_type_min}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_max = f"grain_size_p_ori_ave_{TJ_energy_type_max}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_consMax = f"grain_size_p_ori_ave_{TJ_energy_type_consMax}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_iso = "grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"

    # ================================================================================
    # Data Loading: Multi-Energy Function Evolution Datasets
    # ================================================================================
    # Load energy function microstructural evolution data for comprehensive comparison
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)        # Average energy function
    npy_file_aniso_consMin = np.load(npy_file_folder + npy_file_name_aniso_consMin)  # Constant minimum
    npy_file_aniso_sum = np.load(npy_file_folder + npy_file_name_aniso_sum)        # Summation energy
    npy_file_aniso_min = np.load(npy_file_folder + npy_file_name_aniso_min)        # Minimum energy
    npy_file_aniso_max = np.load(npy_file_folder + npy_file_name_aniso_max)        # Maximum energy
    npy_file_aniso_consMax = np.load(npy_file_folder + npy_file_name_aniso_consMax)  # Constant maximum
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)                    # Isotropic baseline
    
    # Enhanced data validation for energy function analysis datasets
    print(f"The ave data size is: {npy_file_aniso_ave.shape}")                    # Average energy shape
    print(f"The consMin data size is: {npy_file_aniso_consMin.shape}")            # Constant min shape
    print(f"The sum data size is: {npy_file_aniso_sum.shape}")                    # Summation energy shape
    print(f"The min data size is: {npy_file_aniso_min.shape}")                    # Minimum energy shape
    print(f"The max data size is: {npy_file_aniso_max.shape}")                    # Maximum energy shape
    print(f"The consMax data size is: {npy_file_aniso_consMax.shape}")            # Constant max shape
    print(f"The iso data size is: {npy_file_iso.shape}")                          # Isotropic baseline shape
    print("READING DATA DONE")

    # ================================================================================
    # Energy Function Analysis Configuration: Multi-Method Data Structures
    # ================================================================================
    # Analysis parameters for energy function morphological characterization
    initial_grain_num = 20000                               # Initial grain population for energy analysis
    
    # Initialize comprehensive energy function analysis data structures
    step_num = npy_file_aniso_ave.shape[0]                  # Time steps for energy function analysis
    
    # Average energy function data structures
    grain_num_ave = np.zeros(step_num)                      # Grain count evolution (ave)
    grain_area_ave = np.zeros((step_num,initial_grain_num)) # Grain area evolution (ave)
    grain_size_ave = np.zeros((step_num,initial_grain_num)) # Grain size evolution (ave)
    grain_ave_size_ave = np.zeros(step_num)                # Average grain size (ave)
    
    # Constant minimum energy function data structures
    grain_num_consMin = np.zeros(step_num)                  # Grain count evolution (consMin)
    grain_area_consMin = np.zeros((step_num,initial_grain_num)) # Grain area evolution (consMin)
    grain_size_consMin = np.zeros((step_num,initial_grain_num)) # Grain size evolution (consMin)
    grain_ave_size_consMin = np.zeros(step_num)            # Average grain size (consMin)
    
    # Summation energy function data structures
    grain_num_sum = np.zeros(step_num)                      # Grain count evolution (sum)
    grain_area_sum = np.zeros((step_num,initial_grain_num)) # Grain area evolution (sum)
    grain_size_sum = np.zeros((step_num,initial_grain_num)) # Grain size evolution (sum)
    grain_ave_size_sum = np.zeros(step_num)                # Average grain size (sum)
    
    # Minimum energy function data structures
    grain_num_min = np.zeros(step_num)                      # Grain count evolution (min)
    grain_area_min = np.zeros((step_num,initial_grain_num)) # Grain area evolution (min)
    grain_size_min = np.zeros((step_num,initial_grain_num)) # Grain size evolution (min)
    grain_ave_size_min = np.zeros(step_num)                # Average grain size (min)
    
    # Maximum energy function data structures
    grain_num_max = np.zeros(step_num)                      # Grain count evolution (max)
    grain_area_max = np.zeros((step_num,initial_grain_num)) # Grain area evolution (max)
    grain_size_max = np.zeros((step_num,initial_grain_num)) # Grain size evolution (max)
    grain_ave_size_max = np.zeros(step_num)                # Average grain size (max)
    
    # Constant maximum energy function data structures
    grain_num_consMax = np.zeros(step_num)                  # Grain count evolution (consMax)
    grain_area_consMax = np.zeros((step_num,initial_grain_num)) # Grain area evolution (consMax)
    grain_size_consMax = np.zeros((step_num,initial_grain_num)) # Grain size evolution (consMax)
    grain_ave_size_consMax = np.zeros(step_num)            # Average grain size (consMax)
    
    # Isotropic baseline data structures
    grain_num_iso = np.zeros(step_num)                      # Grain count evolution (iso)
    grain_area_iso = np.zeros((step_num,initial_grain_num)) # Grain area evolution (iso)
    grain_size_iso = np.zeros((step_num,initial_grain_num)) # Grain size evolution (iso)
    grain_ave_size_iso = np.zeros(step_num)                # Average grain size (iso)

    # ================================================================================
    # Grain Size Distribution Configuration for Energy Function Analysis
    # ================================================================================
    # Grain size distribution parameters for energy function morphological characterization
    bin_width = 0.16                                        # Grain size distribution bin width
    x_limit = [-0.5, 3.5]                                  # Size distribution range
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)  # Number of size bins
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)  # Bin centers
    
    # Initialize grain size distribution arrays for each energy function
    grain_size_distribution_ave = np.zeros(bin_num)        # Size distribution (ave)
    special_step_distribution_ave = 11                     # Analysis time step (ave) - to get 2000 grains
    grain_size_distribution_consMin = np.zeros(bin_num)    # Size distribution (consMin)
    special_step_distribution_consMin = 11                 # Analysis time step (consMin) - to get 2000 grains
    grain_size_distribution_sum = np.zeros(bin_num)        # Size distribution (sum)
    special_step_distribution_sum = 11                     # Analysis time step (sum) - to get 2000 grains
    grain_size_distribution_min = np.zeros(bin_num)        # Size distribution (min)
    special_step_distribution_min = 30                     # Analysis time step (min) - to get 2000 grains
    grain_size_distribution_max = np.zeros(bin_num)        # Size distribution (max)
    special_step_distribution_max = 15                     # Analysis time step (max) - to get 2000 grains
    grain_size_distribution_consMax = np.zeros(bin_num)    # Size distribution (consMax)
    special_step_distribution_consMax = 11                 # Analysis time step (consMax) - to get 2000 grains
    grain_size_distribution_iso = np.zeros(bin_num)        # Size distribution (iso)
    special_step_distribution_iso = 10                     # Analysis time step (iso) - to get 2000 grains


    # ================================================================================
    # COMPARATIVE ENERGY FUNCTION NORMAL VECTOR ANALYSIS: Bias Correction Framework
    # ================================================================================
    """
    Comprehensive energy function normal vector analysis with bias correction.
    
    This analysis section implements systematic normal vector computation for all 
    energy averaging methods at specific time steps to characterize energy function
    effects on grain boundary orientation patterns with enhanced bias correction
    methodology for fair comparative statistical analysis across different approaches.
    
    Energy Function Normal Vector Framework:
    - Fixed Time Steps: Optimized for ~2000 grain population for statistical significance
    - Bias Correction: kT=0.66 reference methodology for enhanced comparative analysis
    - Normal Vector Cache: Data persistence for computational efficiency in analysis
    - Energy Method Comparison: Systematic orientation distribution characterization
    - Polar Visualization: Energy function-dependent grain boundary orientation patterns
    
    Scientific Applications:
    - Energy averaging method sensitivity analysis through normal vector distributions
    - Bias correction validation for fair energy function comparative analysis
    - Grain boundary orientation anisotropy quantification across energy methods
    - Energy function optimization through comparative orientation characterization
    """
    
    # Initialize polar visualization for energy function orientation comparison
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    # Configure polar plot parameters for energy function orientation visualization
    ax.set_thetagrids(np.arange(0.0, 360.0, 45.0),fontsize=16)    # Angular grid (45° intervals)
    ax.set_thetamin(0.0)                                          # Minimum angle (0°)
    ax.set_thetamax(360.0)                                        # Maximum angle (360°)

    # Configure radial parameters for energy function normal vector magnitude
    ax.set_rgrids(np.arange(0, 0.01, 0.004))                     # Radial grid lines
    ax.set_rlabel_position(0.0)                                   # Radial label position at 0°
    ax.set_rlim(0.0, 0.01)                                        # Radial range [0, 0.01]
    ax.set_yticklabels(['0', '4e-3', '8e-3'],fontsize=16)        # Radial tick labels

    # Configure plot aesthetics for energy function visualization
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)  # Grid styling
    ax.set_axisbelow('True')                                      # Grid below data

    # ================================================================================
    # BIAS CORRECTION DATA: kT=0.66 Reference for Energy Function Comparison
    # ================================================================================
    # Load bias correction reference data for enhanced energy function comparison
    special_step_distribution_T066_bias = 10                     # Reference time step for bias
    data_file_name_bias = f'/normal_distribution_data/normal_distribution_T066_bias_sites_step{special_step_distribution_T066_bias}.npy'
    slope_list_bias = np.load(current_path + data_file_name_bias) # Bias correction reference

    # ================================================================================
    # MINIMUM ENERGY FUNCTION NORMAL VECTOR ANALYSIS
    # ================================================================================
    # Minimum energy function normal vector computation with bias correction
    data_file_name_P = f'/normal_distribution_data/normal_distribution_min_P_step{special_step_distribution_min}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_min_sites_step{special_step_distribution_min}.npy'
    
    # Check for cached minimum energy function data to optimize computation
    if os.path.exists(current_path + data_file_name_P):
        # Load pre-computed minimum energy function normal vector data
        P = np.load(current_path + data_file_name_P)              # Normal vector field (min)
        sites = np.load(current_path + data_file_name_sites)      # Grain boundary sites (min)
    else:
        # Compute minimum energy function normal vectors with microstructural rotation
        newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
        P, sites = get_normal_vector(newplace, initial_grain_num) # Normal vector computation
        np.save(current_path + data_file_name_P, P)               # Cache normal vectors
        np.save(current_path + data_file_name_sites, sites)       # Cache boundary sites

    # Compute orientation distribution for minimum energy function with bias correction
    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_min, "Min", slope_list_bias)

    # ================================================================================
    # MAXIMUM ENERGY FUNCTION NORMAL VECTOR ANALYSIS
    # ================================================================================
    # Maximum energy function normal vector computation with bias correction
    data_file_name_P = f'/normal_distribution_data/normal_distribution_max_P_step{special_step_distribution_max}.npy'
    data_file_name_P = f'/normal_distribution_data/normal_distribution_max_P_step{special_step_distribution_max}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_max_sites_step{special_step_distribution_max}.npy'
    
    # Check for cached maximum energy function data to optimize computation
    if os.path.exists(current_path + data_file_name_P):
        # Load pre-computed maximum energy function normal vector data
        P = np.load(current_path + data_file_name_P)              # Normal vector field (max)
        sites = np.load(current_path + data_file_name_sites)      # Grain boundary sites (max)
    else:
        # Compute maximum energy function normal vectors with microstructural rotation
        newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
        P, sites = get_normal_vector(newplace, initial_grain_num) # Normal vector computation
        np.save(current_path + data_file_name_P, P)               # Cache normal vectors
        np.save(current_path + data_file_name_sites, sites)       # Cache boundary sites

    # Compute orientation distribution for maximum energy function with bias correction
    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_max, "Max", slope_list_bias)

    # ================================================================================
    # AVERAGE ENERGY FUNCTION NORMAL VECTOR ANALYSIS
    # ================================================================================
    # Average energy function normal vector computation with bias correction
    data_file_name_P = f'/normal_distribution_data/normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    
    # Check for cached average energy function data to optimize computation
    if os.path.exists(current_path + data_file_name_P):
        # Load pre-computed average energy function normal vector data
        P = np.load(current_path + data_file_name_P)              # Normal vector field (ave)
        sites = np.load(current_path + data_file_name_sites)      # Grain boundary sites (ave)
    else:
        # Compute average energy function normal vectors with microstructural rotation
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector(newplace, initial_grain_num) # Normal vector computation
        np.save(current_path + data_file_name_P, P)               # Cache normal vectors
        np.save(current_path + data_file_name_sites, sites)       # Cache boundary sites

    # Compute orientation distribution for average energy function with bias correction
    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_ave, "Ave", slope_list_bias)

    # ================================================================================
    # SUMMATION ENERGY FUNCTION NORMAL VECTOR ANALYSIS
    # ================================================================================
    # Summation energy function normal vector computation with bias correction
    data_file_name_P = f'/normal_distribution_data/normal_distribution_sum_P_step{special_step_distribution_sum}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_sum_sites_step{special_step_distribution_sum}.npy'
    
    # Check for cached summation energy function data to optimize computation
    if os.path.exists(current_path + data_file_name_P):
        # Load pre-computed summation energy function normal vector data
        P = np.load(current_path + data_file_name_P)              # Normal vector field (sum)
        sites = np.load(current_path + data_file_name_sites)      # Grain boundary sites (sum)
    else:
        # Compute summation energy function normal vectors with microstructural rotation
        newplace = np.rot90(npy_file_aniso_sum[special_step_distribution_sum,:,:,:], 1, (0,1))
        P, sites = get_normal_vector(newplace, initial_grain_num) # Normal vector computation
        np.save(current_path + data_file_name_P, P)               # Cache normal vectors
        np.save(current_path + data_file_name_sites, sites)       # Cache boundary sites

    # Compute orientation distribution for summation energy function with bias correction
    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_sum, "Sum", slope_list_bias)

    # ================================================================================
    # Constant MINIMUM ENERGY FUNCTION NORMAL VECTOR ANALYSIS
    # ================================================================================
    # Constant minimum energy function normal vector computation with bias correction
    data_file_name_P = f'/normal_distribution_data/normal_distribution_consMin_P_step{special_step_distribution_consMin}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_consMin_P_sites_step{special_step_distribution_consMin}.npy'
    
    # Check for cached Constant minimum energy function data to optimize computation
    if os.path.exists(current_path + data_file_name_P):
        # Load pre-computed Constant minimum energy function normal vector data
        P = np.load(current_path + data_file_name_P)              # Normal vector field (consMin)
        sites = np.load(current_path + data_file_name_sites)      # Grain boundary sites (consMin)
    else:
        # Compute Constant minimum energy function normal vectors with rotation
        newplace = np.rot90(npy_file_aniso_consMin[special_step_distribution_consMin,:,:,:], 1, (0,1))
        P, sites = get_normal_vector(newplace, initial_grain_num) # Normal vector computation
        np.save(current_path + data_file_name_P, P)               # Cache normal vectors
        np.save(current_path + data_file_name_sites, sites)       # Cache boundary sites

    # Compute orientation distribution for Constant minimum with bias correction
    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_consMin, "CMin", slope_list_bias)

    # ================================================================================
    # Constant MAXIMUM ENERGY FUNCTION NORMAL VECTOR ANALYSIS
    # ================================================================================
    # Constant maximum energy function normal vector computation with bias correction
    data_file_name_P = f'/normal_distribution_data/normal_distribution_consMax_P_step{special_step_distribution_consMax}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_consMax_sites_step{special_step_distribution_consMax}.npy'
    
    # Check for cached Constant maximum energy function data to optimize computation
    if os.path.exists(current_path + data_file_name_P):
        # Load pre-computed Constant maximum energy function normal vector data
        P = np.load(current_path + data_file_name_P)              # Normal vector field (consMax)
        sites = np.load(current_path + data_file_name_sites)      # Grain boundary sites (consMax)
    else:
        # Compute Constant maximum energy function normal vectors with rotation
        newplace = np.rot90(npy_file_aniso_consMax[special_step_distribution_consMax,:,:,:], 1, (0,1))
        P, sites = get_normal_vector(newplace, initial_grain_num) # Normal vector computation
        np.save(current_path + data_file_name_P, P)               # Cache normal vectors
        np.save(current_path + data_file_name_sites, sites)       # Cache boundary sites

    # Compute orientation distribution for Constant maximum with bias correction
    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_consMax, "CMax", slope_list_bias)

    # ================================================================================
    # ISOTROPIC BASELINE NORMAL VECTOR ANALYSIS
    # ================================================================================
    # Isotropic baseline normal vector computation for energy function comparison reference
    data_file_name_P = f'/normal_distribution_data/normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        # Compute isotropic baseline normal vectors with microstructural rotation
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector(newplace, initial_grain_num) # Normal vector computation
        np.save(current_path + data_file_name_P, P)               # Cache normal vectors
        np.save(current_path + data_file_name_sites, sites)       # Cache boundary sites

    # Compute orientation distribution for isotropic baseline with bias correction
    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_iso, "Iso", slope_list_bias)

    # ================================================================================
    # ENERGY FUNCTION COMPARISON VISUALIZATION: Final Plot Generation and Output
    # ================================================================================
    """
    Complete comprehensive energy function comparison visualization and analysis summary.
    
    This final section generates publication-ready polar plot visualization comparing
    all energy averaging methods with isotropic baseline to characterize energy
    function effects on grain boundary orientation patterns with bias correction for
    enhanced comparative statistical analysis across different methodological approaches.
    
    Visualization and Output Framework:
    - Polar Coordinate System: Angular grain boundary orientation distributions
    - Energy Function Legend: Comprehensive method identification and comparison
    - Bias Correction Output: Enhanced comparative framework with kT=0.66 reference
    - High-Resolution Export: 400 DPI publication-quality figure generation
    - Statistical Validation: ~2000 grain population for robust comparative analysis
    """
    
    # Add comprehensive legend for energy function comparison identification
    plt.legend(loc=(-0.12,-0.35),fontsize=16,ncol=3)
    
    # Generate high-resolution publication-ready energy function comparison plot
    plt.savefig(current_path + "/figures/poly_aniso_iso_compare_vector_distribution_after_removing_bias.png", 
               dpi=400, bbox_inches='tight')
    
    # Print comprehensive energy function analysis completion summary
    print("=" * 100)
    print("COMPREHENSIVE ENERGY FUNCTION COMPARISON ANALYSIS COMPLETED")
    print("=" * 100)
    print("Energy Function Normal Vector Analysis with Bias Correction Summary:")
    print("-" * 100)
    print(f"• Average Energy Function Analysis: Step {special_step_distribution_ave} - Standard energy averaging")
    print(f"• Constant Minimum Analysis: Step {special_step_distribution_consMin} - Constant min energy") 
    print(f"• Summation Energy Analysis: Step {special_step_distribution_sum} - Energy summation approach")
    print(f"• Minimum Energy Analysis: Step {special_step_distribution_min} - Pure minimum energy")
    print(f"• Maximum Energy Analysis: Step {special_step_distribution_max} - Pure maximum energy")
    print(f"• Constant Maximum Analysis: Step {special_step_distribution_consMax} - Constant max energy")
    print(f"• Isotropic Baseline Analysis: Step {special_step_distribution_iso} - Reference comparison")
    print("-" * 100)
    print("Bias Correction Methodology:")
    print(f"• Reference Time Step: {special_step_distribution_T066_bias} (kT=0.66)")
    print("• Bias Correction Applied: Enhanced comparative statistical framework")
    print("• Normal Vector Caching: Computational efficiency optimization implemented")
    print("-" * 100)
    print("Output Generated:")
    print("• High-Resolution Polar Plot: poly_aniso_iso_compare_vector_distribution_after_removing_bias.png")
    print("• Energy Function Comparison: All 6 anisotropic methods + isotropic baseline")
    print("• Publication Quality: 400 DPI resolution with comprehensive scientific documentation")
    print("=" * 100)











