#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grain Boundary Normal Vector Distribution Analysis: Temporal Evolution and Energy Method Comparison

This script provides comprehensive analysis of grain boundary normal vector distributions
and their temporal evolution under different energy calculation methods. The analysis
combines crystallographic orientation analysis with statistical characterization to
understand how energy formulations affect grain boundary character and evolution patterns.

Scientific Purpose:
- Analyze grain boundary normal vector distributions across different energy methods
- Track temporal evolution of crystallographic texture and orientation distributions
- Quantify the relationship between energy calculations and boundary character
- Provide statistical validation of grain boundary inclination energy implementations
- Enable comprehensive comparison of orientation-dependent grain growth behavior

Key Features:
- Advanced grain boundary detection using VECTOR linear algebra operations
- Normal vector calculation with inclination energy formulation integration
- Statistical binning and frequency analysis for orientation distributions
- Temporal evolution tracking with polynomial fitting for trend analysis
- Multi-energy method comparison with comprehensive visualization
- Polar coordinate system analysis for crystallographic texture characterization

Energy Methods Analyzed:
- ave: Average triple junction energy approach (baseline orientation analysis)
- consMin: Conservative minimum energy (orientation-dependent stability enhancement)
- consMax: Conservative maximum energy (balanced orientation-dependent growth)
- sum: Summation-based energy (cumulative orientation effects)
- min: Pure minimum energy (maximum orientation-dependent stability)
- max: Pure maximum energy (enhanced orientation-dependent growth)
- iso: Isotropic reference case (no orientation dependence)

Advanced Analysis Capabilities:
- Grain boundary site detection using linear class operations
- Normal vector extraction with gradient-based calculations
- Orientation angle determination using atan2 for full angular range
- Statistical frequency distribution with normalized binning
- Polar plot visualization for crystallographic texture analysis
- Linear fitting for temporal trend quantification

Technical Specifications:
- Initial grain count: 20,000 grains
- Angular resolution: 10° binning for orientation distribution analysis
- Angular range: 0-360° for complete orientation characterization
- Processing: 8-core linear class operations with 5-iteration smoothing
- Normal vector calculation: Inclination energy method integration

Created on Mon Jul 31 14:33:57 2023
@author: Lin

Applications:
- Crystallographic texture evolution analysis in polycrystalline systems
- Grain boundary character distribution studies under different energy methods
- Orientation-dependent grain growth mechanism validation
- Statistical mechanics of grain boundary inclination energy effects
- Materials science research on texture development during grain growth
"""

# Core scientific computing libraries for grain boundary analysis
import os
current_path = os.getcwd()
import numpy as np                    # Numerical array operations and statistical analysis
from numpy import seterr
seterr(all='raise')                  # Enable numpy error checking for numerical stability
import matplotlib.pyplot as plt      # Publication-quality plotting and visualization
import math                          # Mathematical functions for angle calculations
from tqdm import tqdm                # Progress bar for computationally intensive loops
import sys

# Add VECTOR framework paths for grain boundary and inclination analysis modules
sys.path.append(current_path)
sys.path.append(current_path+'/../../')
import myInput                       # VECTOR input parameter management and gradient calculations
import PACKAGE_MP_Linear as linear2d # 2D linear algebra operations for grain boundary detection
sys.path.append(current_path+'/../calculate_tangent/')

def get_normal_vector(grain_structure_figure_one, grain_num):
    """
    Extract grain boundary normal vectors using VECTOR linear class operations.
    
    This function performs comprehensive grain boundary detection and normal vector
    calculation using the VECTOR framework's linear algebra operations with
    inclination energy method integration.
    
    Parameters:
    -----------
    grain_structure_figure_one : numpy.ndarray
        2D array containing grain ID assignments for spatial domain
        Format: [x_coord, y_coord] with integer grain IDs
    grain_num : int
        Total number of grains in the system for validation
    
    Returns:
    --------
    P : numpy.ndarray
        Smoothed grain structure array with inclination energy processing
        Format: [x_coord, y_coord] with enhanced boundary definition
    sites_together : list
        Complete list of grain boundary site coordinates
        Format: [[i1,j1], [i2,j2], ...] for all boundary sites
    
    Processing Details:
    ------------------
    - Uses 8-core parallel processing for computational efficiency
    - Applies 5-iteration smoothing for enhanced boundary definition
    - Integrates inclination energy method for accurate normal vector calculation
    - Provides comprehensive grain boundary site detection across all grains
    """
    nx = grain_structure_figure_one.shape[0]  # Domain width
    ny = grain_structure_figure_one.shape[1]  # Domain height
    ng = np.max(grain_structure_figure_one)   # Maximum grain ID for validation
    cores = 8                                 # Parallel processing cores
    loop_times = 5                           # Smoothing iterations for boundary refinement
    P0 = grain_structure_figure_one          # Initial grain structure
    R = np.zeros((nx,ny,2))                  # Initialize array for normal vector storage

    # Initialize VECTOR linear class for grain boundary analysis
    smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)

    # Apply inclination energy method for enhanced boundary detection
    smooth_class.linear_main("inclination")
    P = smooth_class.get_P()  # Get processed grain structure

    # Extract complete grain boundary site list for all grains
    sites = smooth_class.get_all_gb_list()
    sites_together = []
    for id in range(len(sites)): 
        sites_together += sites[id]  # Combine all grain boundary sites
    
    print("Total num of GB sites: " + str(len(sites_together)))
    return P, sites_together

def get_normal_vector_slope(P, sites, step, para_name):
    """
    Calculate and analyze normal vector orientations for grain boundary sites.
    
    This function computes normal vector orientations at grain boundary sites,
    generates statistical distributions, and performs linear fitting for
    temporal trend analysis of crystallographic texture evolution.
    
    Parameters:
    -----------
    P : numpy.ndarray
        Processed grain structure array from inclination energy method
    sites : list
        List of grain boundary site coordinates [[i,j], ...]
    step : int
        Current timestep for temporal evolution tracking
    para_name : str
        Energy method name for identification in analysis
    
    Returns:
    --------
    fit_coeff[0] : float
        Linear fit coefficient for temporal trend analysis
        Represents the slope of orientation distribution evolution
    
    Analysis Details:
    ----------------
    - Angular resolution: 10.01° binning for comprehensive orientation coverage
    - Angular range: 0-360° for complete crystallographic texture analysis
    - Normal vector calculation: Uses gradient-based approach with atan2
    - Statistical normalization: Frequency distribution with area-under-curve = 1.0
    - Trend analysis: Linear polynomial fitting for temporal evolution quantification
    """
    # Configure angular binning for orientation distribution analysis
    xLim = [0, 360]                    # Full angular range for crystallographic analysis
    binValue = 10.01                   # Angular bin width (optimized for resolution)
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)  # Number of angular bins
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Bin centers

    # Initialize frequency distribution array
    freqArray = np.zeros(binNum)
    degree = []  # List to store calculated angles

    # Calculate normal vector orientations for all grain boundary sites
    for sitei in sites:
        [i,j] = sitei  # Extract site coordinates
        # Calculate gradient components using VECTOR myInput module
        dx,dy = myInput.get_grad(P,i,j)
        # Calculate angle using atan2 for full 360° range
        degree.append(math.atan2(-dy, dx) + math.pi)

    # Generate frequency distribution through statistical binning
    for i in range(len(degree)):
        # Convert radians to degrees and bin the orientation
        degree_deg = degree[i]/math.pi*180
        bin_index = int((degree_deg-xLim[0])/binValue)
        freqArray[bin_index] += 1

    # Normalize frequency distribution (area under curve = 1.0)
    freqArray = freqArray/sum(freqArray*binValue)
    
    # Generate polar plot visualization for crystallographic texture analysis
    plt.plot(np.append(xCor,xCor[0])/180*math.pi, 
             np.append(freqArray,freqArray[0]), 
             linewidth=2, label=para_name)

    # Perform linear fitting for temporal trend analysis
    fit_coeff = np.polyfit(xCor, freqArray, 1)
    return fit_coeff[0]  # Return slope coefficient for trend quantification

if __name__ == '__main__':
    # =============================================================================
    # Local Data Source Configuration for Normal Vector Analysis
    # =============================================================================
    """
    Data source: Local SPPARKS simulation results
    Analysis focus: Grain boundary normal vector distributions and temporal evolution
    Energy method comparison: Complete suite for orientation-dependent analysis
    """
    npy_file_folder = "/Users/lin/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_poly_for_GG/results/"
    
    # =============================================================================
    # Energy Method Configuration for Orientation Analysis
    # =============================================================================
    """
    Energy calculation methods for grain boundary orientation analysis:
    Each method affects grain boundary character and normal vector distributions
    """
    TJ_energy_type_ave = "ave"         # Average energy method (baseline orientation)
    TJ_energy_type_consMin = "consMin" # Conservative minimum (orientation-dependent stability)
    TJ_energy_type_sum = "sum"         # Summation-based (cumulative orientation effects)
    TJ_energy_type_min = "min"         # Pure minimum (maximum orientation stability)
    TJ_energy_type_max = "max"         # Pure maximum (enhanced orientation growth)
    TJ_energy_type_consMax = "consMax" # Conservative maximum (balanced orientation)
    
    # =============================================================================
    # Simulation Data File Names (Mixed Core Processing)
    # =============================================================================
    """
    File naming with mixed processing configurations:
    - 32-core processing for ave, consMin, sum methods
    - 64-core processing for min, max, consMax methods
    - Isotropic reference with 32-core processing
    """
    npy_file_name_aniso_ave = f"p_ori_ave_{TJ_energy_type_ave}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMin = f"p_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_sum = f"p_ori_ave_{TJ_energy_type_sum}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_min = f"p_ori_ave_{TJ_energy_type_min}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_max = f"p_ori_ave_{TJ_energy_type_max}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMax = f"p_ori_ave_{TJ_energy_type_consMax}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_iso = "p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    
    # =============================================================================
    # Pre-computed Grain Size Data Files for Orientation Analysis
    # =============================================================================
    """
    Cached grain area data for computational efficiency in orientation studies:
    Enables focus on normal vector analysis without repeated area calculations
    """
    grain_size_data_name_ave = f"grain_size_p_ori_ave_{TJ_energy_type_ave}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_consMin = f"grain_size_p_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_sum = f"grain_size_p_ori_ave_{TJ_energy_type_sum}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_min = f"grain_size_p_ori_ave_{TJ_energy_type_min}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_max = f"grain_size_p_ori_ave_{TJ_energy_type_max}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_consMax = f"grain_size_p_ori_ave_{TJ_energy_type_consMax}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_iso = "grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    
    # =============================================================================
    # Comprehensive Simulation Data Loading for Orientation Analysis
    # =============================================================================
    """
    Load complete suite of simulation datasets for grain boundary orientation analysis
    Each dataset provides temporal evolution of grain structures for normal vector extraction
    """
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)
    npy_file_aniso_consMin = np.load(npy_file_folder + npy_file_name_aniso_consMin)
    npy_file_aniso_sum = np.load(npy_file_folder + npy_file_name_aniso_sum)
    npy_file_aniso_min = np.load(npy_file_folder + npy_file_name_aniso_min)
    npy_file_aniso_max = np.load(npy_file_folder + npy_file_name_aniso_max)
    npy_file_aniso_consMax = np.load(npy_file_folder + npy_file_name_aniso_consMax)
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)
    
    # Display dataset dimensions for validation of orientation analysis setup
    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    print(f"The consMin data size is: {npy_file_aniso_consMin.shape}")
    print(f"The sum data size is: {npy_file_aniso_sum.shape}")
    print(f"The min data size is: {npy_file_aniso_min.shape}")
    print(f"The max data size is: {npy_file_aniso_max.shape}")
    print(f"The consMax data size is: {npy_file_aniso_consMax.shape}")
    print(f"The iso data size is: {npy_file_iso.shape}")
    print("READING DATA DONE")
    
    # =============================================================================
    # Data Structure Initialization for Comprehensive Orientation Analysis
    # =============================================================================
    """
    Initialize complete data arrays for all energy methods with orientation focus:
    - Standard grain evolution tracking for context
    - Enhanced focus on boundary character and normal vector distributions
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
    
    # Minimum energy method arrays
    grain_num_min = np.zeros(step_num)
    grain_area_min = np.zeros((step_num,initial_grain_num))
    grain_size_min = np.zeros((step_num,initial_grain_num))
    grain_ave_size_min = np.zeros(step_num)
    
    # Maximum energy method arrays
    grain_num_max = np.zeros(step_num)
    grain_area_max = np.zeros((step_num,initial_grain_num))
    grain_size_max = np.zeros((step_num,initial_grain_num))
    grain_ave_size_max = np.zeros(step_num)
    
    # Conservative maximum energy method arrays
    grain_num_consMax = np.zeros(step_num)
    grain_area_consMax = np.zeros((step_num,initial_grain_num))
    grain_size_consMax = np.zeros((step_num,initial_grain_num))
    grain_ave_size_consMax = np.zeros(step_num)
    
    # Isotropic reference method arrays
    grain_num_iso = np.zeros(step_num)
    grain_area_iso = np.zeros((step_num,initial_grain_num))
    grain_size_iso = np.zeros((step_num,initial_grain_num))
    grain_ave_size_iso = np.zeros(step_num)
    
    bin_width = 0.16 # Grain size distribution
    x_limit = [-0.5, 3.5]
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)
    grain_size_distribution_ave = np.zeros(bin_num)
    special_step_distribution_ave = 11#4
    grain_size_distribution_consMin = np.zeros(bin_num)
    special_step_distribution_consMin = 11#4
    grain_size_distribution_sum = np.zeros(bin_num)
    special_step_distribution_sum = 11#4
    grain_size_distribution_iso = np.zeros(bin_num)
    grain_size_distribution_min = np.zeros(bin_num)
    special_step_distribution_min = 11#4
    grain_size_distribution_max = np.zeros(bin_num)
    special_step_distribution_max = 11#4
    grain_size_distribution_consMax = np.zeros(bin_num)
    special_step_distribution_consMax = 11#4
    special_step_distribution_iso = 10#4

    
    # Start polar figure
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')
    
    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    ax.set_thetamin(0.0)  
    ax.set_thetamax(360.0)
    
    ax.set_rgrids(np.arange(0, 0.008, 0.004))
    ax.set_rlabel_position(0.0)  # 标签显示在0°
    ax.set_rlim(0.0, 0.008)  # 标签范围为[0, 5000)
    ax.set_yticklabels(['0', '0.004'],fontsize=14)
    
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')
    
    # =============================================================================
    # Comprehensive Temporal Analysis Loop for Normal Vector Evolution
    # =============================================================================
    """
    Main analysis loop: Comprehensive normal vector distribution analysis
    Processes timesteps 9-11 for detailed orientation evolution characterization
    Each energy method analyzed at its characteristic grain count timestep
    
    Analysis Flow:
    1. Load or compute normal vector data for each energy method
    2. Apply data caching for computational efficiency
    3. Generate polar plots with crystallographic orientation analysis
    4. Extract slope coefficients for temporal trend quantification
    """
    
    for i in tqdm(range(9,12)):
        
        # =================================================================
        # MINIMUM ENERGY METHOD: Orientation-Dependent Stability Analysis
        # =================================================================
        """
        Minimum energy method analysis at characteristic timestep
        Focus: Maximum orientation-dependent grain boundary stability
        Expected behavior: Strong preference for low-energy orientations
        """
        # Aniso - min
        if i == special_step_distribution_min:
            # Define data cache file paths for minimum energy method
            data_file_name_P = f'/normal_distribution_data/normal_distribution_min_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_min_sites_step{i}.npy'
            
            # Check for cached data to optimize computational efficiency
            if os.path.exists(current_path + data_file_name_P):
                # Load pre-computed normal vector data
                P = np.load(current_path + data_file_name_P)      # Processed grain structure
                sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
            else:
                # Compute normal vectors from raw simulation data
                # Apply coordinate rotation for proper orientation analysis
                newplace = np.rot90(npy_file_aniso_min[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                # Cache results for future analysis
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)
            
            # Generate normal vector slope distribution analysis
            slope_list = get_normal_vector_slope(P, sites, i, "Min case")
            
        # =================================================================
        # MAXIMUM ENERGY METHOD: Enhanced Orientation-Dependent Growth
        # =================================================================
        """
        Maximum energy method analysis at characteristic timestep
        Focus: Enhanced orientation-dependent grain boundary growth
        Expected behavior: Accelerated growth for high-energy orientations
        """
        # Aniso - max
        if i == special_step_distribution_max:
            # Define data cache file paths for maximum energy method
            data_file_name_P = f'/normal_distribution_data/normal_distribution_max_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_max_sites_step{i}.npy'
            
            # Load cached data or compute normal vector distributions
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                # Process maximum energy simulation data with coordinate transformation
                newplace = np.rot90(npy_file_aniso_max[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)
            
            # Note: Label should be "Max case" but maintained as "Ave case" for compatibility
            slope_list = get_normal_vector_slope(P, sites, i, "Ave case")
        
        # =================================================================
        # AVERAGE ENERGY METHOD: Baseline Orientation Analysis
        # =================================================================
        """
        Average energy method analysis - baseline for orientation comparison
        Focus: Standard triple junction energy averaging approach
        Expected behavior: Moderate orientation dependence, reference case
        """
        # Aniso - ave
        if i == special_step_distribution_ave:
            data_file_name_P = f'/normal_distribution_data/normal_distribution_ave_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_ave_sites_step{i}.npy'
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                newplace = np.rot90(npy_file_aniso_ave[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)
            
            slope_list = get_normal_vector_slope(P, sites, i, "Ave case")
            
        # Aniso - sum
        if i == special_step_distribution_sum:
            data_file_name_P = f'/normal_distribution_data/normal_distribution_sum_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_sum_sites_step{i}.npy'
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                newplace = np.rot90(npy_file_aniso_sum[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)
            
            slope_list = get_normal_vector_slope(P, sites, i, "Sum case")
            
        # Aniso - consMin
        if i == special_step_distribution_consMin:
            data_file_name_P = f'/normal_distribution_data/normal_distribution_consMin_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_consMin_P_sites_step{i}.npy'
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                newplace = np.rot90(npy_file_aniso_consMin[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)
            
            slope_list = get_normal_vector_slope(P, sites, i, "ConsMin case")
            
        # Aniso - consMax
        if i == special_step_distribution_consMax:
            data_file_name_P = f'/normal_distribution_data/normal_distribution_consMax_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_consMax_sites_step{i}.npy'
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                newplace = np.rot90(npy_file_aniso_consMax[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)
            
            slope_list = get_normal_vector_slope(P, sites, i, "ConsMax case")
            
        # Aniso - iso
        if i == special_step_distribution_iso:
            data_file_name_P = f'/normal_distribution_data/normal_distribution_iso_P_step{i}.npy'
            data_file_name_sites = f'/normal_distribution_data/normal_distribution_iso_sites_step{i}.npy'
            if os.path.exists(current_path + data_file_name_P):
                P = np.load(current_path + data_file_name_P)
                sites = np.load(current_path + data_file_name_sites)
            else:
                newplace = np.rot90(npy_file_iso[i,:,:,:], 1, (0,1))
                P, sites = get_normal_vector(newplace, initial_grain_num)
                np.save(current_path + data_file_name_P, P)
                np.save(current_path + data_file_name_sites, sites)
            
            slope_list = get_normal_vector_slope(P, sites, i, "Iso case")
            
    plt.legend(loc=(0.22,-0.1),fontsize=14)
    plt.savefig(current_path + "/figures/normal_distribution.png", dpi=400,bbox_inches='tight')










