#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
POLYCRYSTALLINE 20K GRAIN BOUNDARY ANALYSIS WITH HIPERGATOR INTEGRATION
================================================================================

Purpose: Large-scale polycrystalline grain boundary normal distribution analysis
         using HiPerGator supercomputing resources for virtual inclination energy
         simulations with 20,000 initial grains

Scientific Context:
- HiPerGator 3.0 supercomputing cluster optimization
- SPPARKS Monte Carlo simulations with 20,000 initial grains
- Virtual inclination energy methodology for large-scale systems
- Triple junction (TJ) energy approach comparative analysis
- Multi-core parallel processing (32-64 cores) for computational efficiency
- Anisotropic grain boundary energy parameter δ = 0.6

Analysis Framework:
1. Triple junction energy approach comparison:
   - Min: Minimum energy selection at triple junctions
   - Max: Maximum energy selection at triple junctions  
   - Ave: Average energy approach for triple junctions
   - Sum: Summation energy approach for triple junctions
   - CMin: Conservative minimum energy approach
   - CMax: Conservative maximum energy approach
2. Normal vector computation using VECTOR's 2D linear algorithms
3. Bias-corrected polar distribution visualization
4. Statistical anisotropy magnitude quantification
5. HiPerGator-optimized computational workflows

Key Features:
- 20,000 initial grain polycrystalline microstructure
- HiPerGator Blue storage integration (/blue/michael.tonks/)
- Multi-core processing: 32-64 cores for computational efficiency
- Advanced boundary condition handling for large systems
- Triple junction energy formulation comparison
- Publication-quality polar plot generation with bias correction

HiPerGator Integration:
- Blue file system storage for large datasets
- Multi-node parallel processing capabilities
- Optimized memory management for 20K grain systems
- Advanced caching mechanisms for computed normal vectors

Created on Mon Jul 31 14:33:57 2023
@author: Lin
================================================================================
"""

# ===============================================================================
# LIBRARY IMPORTS AND HIPERGATOR PATH CONFIGURATION
# ===============================================================================

# System and path management for HiPerGator environment
import os
current_path = os.getcwd()
import sys
sys.path.append(current_path)                    # Current working directory
sys.path.append(current_path+'/../../')          # VECTOR root directory
sys.path.append(current_path+'/../calculate_tangent/')  # Tangent calculation utilities

# Scientific computing libraries optimized for HiPerGator
import numpy as np
from numpy import seterr
seterr(all='raise')                              # Raise exceptions for numerical warnings
import math                                      # Mathematical functions for trigonometry

# Visualization and progress tracking for large-scale computations
import matplotlib.pyplot as plt                  # Publication-quality plotting
from tqdm import tqdm                           # Progress bar for long HiPerGator computations

# VECTOR framework modules for large-scale analysis
import myInput                                   # Custom input/output utilities
import PACKAGE_MP_Linear as linear2d            # 2D linear multi-physics package

# ===============================================================================
# STATISTICAL ANALYSIS FUNCTIONS FOR LARGE-SCALE POLYCRYSTALLINE SYSTEMS
# ===============================================================================

def simple_magnitude(freqArray):
    """
    Calculate anisotropic magnitude metrics for large-scale polycrystalline analysis.
    
    This function quantifies the deviation of grain boundary normal distributions
    from perfect circular symmetry for 20K grain systems. Essential for comparing
    different triple junction energy formulations in HiPerGator simulations.
    
    Mathematical Framework:
    - Creates uniform circular reference distribution
    - Computes absolute deviations from circular symmetry
    - Calculates statistical moments with enhanced precision for large systems
    - Normalizes by reference distribution average
    
    Parameters:
    -----------
    freqArray : numpy.ndarray
        Frequency array of grain boundary normal orientations (36 bins, 10° each)
        Derived from 20K grain boundary sites
        
    Returns:
    --------
    magnitude_ave : float
        Average relative deviation from circular symmetry
    magnitude_stan : float
        Standard deviation of relative deviations (statistical uncertainty)
        
    HiPerGator Optimization:
    - Vectorized operations for computational efficiency
    - Memory-efficient processing for large grain counts
    - Statistical robustness for massive datasets
    """
    # Define angular coordinate system for large-scale analysis
    xLim = [0, 360]                              # Full angular range (degrees)
    binValue = 10.01                             # Bin width for angular discretization
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)  # Number of angular bins
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Bin centers

    # Create ideal uniform circular distribution for comparison
    freqArray_circle = np.ones(binNum)           # Uniform distribution
    freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)  # Normalize

    # Calculate statistical measures of anisotropy for large systems
    magnitude_max = np.max(abs(freqArray - freqArray_circle))/np.average(freqArray_circle)
    magnitude_ave = np.average(abs(freqArray - freqArray_circle))/np.average(freqArray_circle)

    # Compute standard deviation for statistical uncertainty quantification
    magnitude_stan = np.sqrt(np.sum((abs(freqArray - freqArray_circle)/np.average(freqArray_circle) - magnitude_ave)**2)/binNum)

    return magnitude_ave, magnitude_stan

    # Legacy coefficient-based analysis (commented for reference)
    # coeff_high = abs(np.cos((xCor-90)/180*np.pi))
    # coeff_low = abs(np.cos((xCor)/180*np.pi))
    # return np.sum(freqArray * coeff_high)/np.sum(freqArray * coeff_low)

def find_fittingEllipse2(array):  # failure
    """
    Legacy elliptical fitting function for grain shape analysis.
    
    NOTE: This function is marked as 'failure' and retained for reference only.
    It was an experimental approach for elliptical fitting that proved insufficient
    for large-scale polycrystalline analysis.
    
    Mathematical Framework:
    - General conic equation: Ax² + Bxy + Cy² + Dx + Ey + F = 1
    - Least-squares parameter estimation using linear algebra
    - Direct matrix formulation approach
    
    Parameters:
    -----------
    array : numpy.ndarray
        Array of [x, y] coordinates for fitting
        
    Returns:
    --------
    X_mat : numpy.ndarray
        Ellipse parameters [A, B, C, D, E]
        
    Status: DEPRECATED - Use advanced methods for 20K grain systems
    """
    K_mat = []                                   # Design matrix initialization
    Y_mat = []                                   # Target vector initialization

    # Extract coordinate vectors
    X = array[:,0]                               # x-coordinates
    Y = array[:,1]                               # y-coordinates

    # Construct design matrix for conic equation fitting
    K_mat = np.hstack([X**2, X*Y, Y**2, X, Y])  # Design matrix
    Y_mat = np.ones_like(X)                      # Target vector

    # Solve least-squares problem for conic parameters
    X_mat = np.linalg.lstsq(K_mat, Y_mat)[0].squeeze()
    # Alternative formulation: X_mat = (K_mat.T*K_mat).I * K_mat.T * Y_mat

    # Debug output for ellipse equation
    print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(X_mat[0], X_mat[1], X_mat[2], X_mat[3], X_mat[4]))
    print(X_mat)

    return X_mat

# ===============================================================================
# POLYCRYSTALLINE GRAIN GEOMETRY ANALYSIS FUNCTIONS FOR 20K SYSTEMS
# ===============================================================================

def get_poly_center(micro_matrix, step):
    """
    Calculate center coordinates and average radii for all grains in polycrystalline system.
    
    This function processes complete microstructure arrays from HiPerGator simulations
    to determine grain centers and characteristic radii for 20,000 initial grain systems.
    Critical for large-scale statistical analysis of grain evolution under different
    triple junction energy formulations.
    
    Mathematical Framework:
    - Grid-based grain identification using microstructure matrix
    - Weighted centroid calculation for each grain using all interior sites
    - Boundary condition exclusion for periodic domain handling
    - Area-equivalent radius computation from grain site counts
    
    Parameters:
    -----------
    micro_matrix : numpy.ndarray
        4D microstructure array [time, x, y, data] from HiPerGator simulations
        Contains grain IDs for complete polycrystalline system evolution
    step : int
        Time step index for analysis (0-based indexing)
        
    Returns:
    --------
    center_list : numpy.ndarray
        Array of grain center coordinates [x_center, y_center] for each grain
        Shape: (num_grains, 2)
    ave_radius_list : numpy.ndarray
        Array of area-equivalent radii for each grain
        Computed as sqrt(area/π) for circular equivalence
        
    HiPerGator Optimizations:
    - Efficient grain boundary detection for massive systems
    - Memory-optimized processing for large microstructure arrays
    - Periodic boundary condition handling for simulation domains
    """
    # Get the center of all non-periodic grains in matrix
    num_grains = int(np.max(micro_matrix[step,:]))      # Total number of grains
    center_list = np.zeros((num_grains,2))              # Grain center coordinates
    sites_num_list = np.zeros(num_grains)               # Sites per grain counter
    ave_radius_list = np.zeros(num_grains)              # Area-equivalent radii
    
    # Create coordinate reference grids for spatial analysis
    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i,j] = i                      # x-coordinate reference
            coord_refer_j[i,j] = j                      # y-coordinate reference

    # Extract grain structure at specified time step
    table = micro_matrix[step,:,:,0]                    # Grain ID matrix
    
    # Calculate center and size statistics for each grain
    for i in range(num_grains):
        sites_num_list[i] = np.sum(table == i+1)        # Count sites in grain i+1

        # Exclude small grains and boundary-crossing grains for statistical accuracy
        if (sites_num_list[i] < 500) or \
           (np.max(coord_refer_i[table == i+1]) - np.min(coord_refer_i[table == i+1]) == micro_matrix.shape[1]) or \
           (np.max(coord_refer_j[table == i+1]) - np.min(coord_refer_j[table == i+1]) == micro_matrix.shape[2]): # grains on bc are ignored
          center_list[i, 0] = 0                         # Mark excluded grains
          center_list[i, 1] = 0
          sites_num_list[i] == 0
        else:
          # Calculate weighted center of mass for valid grains
          center_list[i, 0] = np.sum(coord_refer_i[table == i+1]) / sites_num_list[i]
          center_list[i, 1] = np.sum(coord_refer_j[table == i+1]) / sites_num_list[i]
          
    # Compute area-equivalent radii from grain areas
    ave_radius_list = np.sqrt(sites_num_list / np.pi)   # Circular equivalent radius

    return center_list, ave_radius_list

def get_poly_statistical_radius(micro_matrix, sites_list, step):
    """
    Calculate statistical radius deviation metrics for polycrystalline morphology analysis.
    
    This function quantifies the deviation of grain shapes from perfect circles
    by analyzing the distribution of distances from grain centers to boundary sites.
    Essential for characterizing morphological evolution in 20K grain systems under
    different triple junction energy approaches.
    
    Mathematical Framework:
    - Distance calculation from grain centers to all boundary sites
    - Radius offset computation relative to area-equivalent radius
    - Area-weighted statistical averaging for system-level metrics
    - Morphological deviation quantification
    
    Parameters:
    -----------
    micro_matrix : numpy.ndarray
        4D microstructure array from HiPerGator simulations
    sites_list : list
        List of boundary site coordinates for each grain
    step : int
        Time step index for analysis
        
    Returns:
    --------
    max_radius_offset : float
        Area-weighted average of maximum radius deviations
        Normalized by grain-equivalent radius
        Values > 0 indicate departure from circular morphology
        
    Applications:
    - Grain shape evolution tracking in large-scale systems
    - Morphological anisotropy quantification
    - Triple junction energy formulation comparison
    """
    # Get the max offset of average radius and real radius
    center_list, ave_radius_list = get_poly_center(micro_matrix, step)
    num_grains = int(np.max(micro_matrix[step,:]))       # Total grain count

    max_radius_offset_list = np.zeros(num_grains)       # Maximum offsets per grain
    
    # Calculate radius deviations for each grain
    for n in range(num_grains):
        center = center_list[n]                          # Grain center coordinates
        ave_radius = ave_radius_list[n]                  # Area-equivalent radius
        sites = sites_list[n]                            # Boundary site coordinates

        if ave_radius != 0:                              # Process valid grains only
          for sitei in sites:
              [i,j] = sitei                              # Site coordinates
              # Calculate actual distance from center
              current_radius = np.sqrt((i - center[0])**2 + (j - center[1])**2)
              # Compute relative deviation from equivalent radius
              radius_offset = abs(current_radius - ave_radius)
              if radius_offset > max_radius_offset_list[n]: 
                  max_radius_offset_list[n] = radius_offset

          # Normalize by equivalent radius for relative comparison
          max_radius_offset_list[n] = max_radius_offset_list[n] / ave_radius

    # Calculate area-weighted system average
    max_radius_offset = np.average(max_radius_offset_list[max_radius_offset_list!=0])
    area_list = np.pi*ave_radius_list*ave_radius_list    # Grain areas
    if np.sum(area_list) == 0: 
        max_radius_offset = 0                            # Handle edge case
    else: 
        # Area-weighted average for system-level metric
        max_radius_offset = np.sum(max_radius_offset_list * area_list) / np.sum(area_list)

    return max_radius_offset

def get_poly_statistical_ar(micro_matrix, step):
    """
    Calculate system-averaged aspect ratio for polycrystalline grain morphology.
    
    This function computes aspect ratios for all grains based on their spatial
    extent in x and y directions, providing a measure of grain elongation and
    morphological anisotropy in 20K grain systems.
    
    Mathematical Framework:
    - Grid-based grain boundary detection
    - Spatial extent calculation in principal directions
    - Aspect ratio computation as x-extent / y-extent
    - Site-weighted system averaging for statistical robustness
    
    Parameters:
    -----------
    micro_matrix : numpy.ndarray
        4D microstructure array from HiPerGator simulations
    step : int
        Time step index for analysis
        
    Returns:
    --------
    aspect_ratio : float
        Site-weighted average aspect ratio for entire system
        Values > 1 indicate elongation in x-direction
        Values < 1 indicate elongation in y-direction
        
    HiPerGator Applications:
    - Large-scale morphological evolution tracking
    - Anisotropy quantification under different energy formulations
    - Statistical shape characterization for massive grain datasets
    """
    # Get the average aspect ratio
    num_grains = int(np.max(micro_matrix[step,:]))       # Total number of grains
    sites_num_list = np.zeros(num_grains)               # Site count per grain
    
    # Create coordinate reference system
    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i,j] = i                      # x-coordinate grid
            coord_refer_j[i,j] = j                      # y-coordinate grid

    # Initialize aspect ratio calculation arrays
    aspect_ratio_i = np.zeros((num_grains,2))           # x-extent storage
    aspect_ratio_j = np.zeros((num_grains,2))           # y-extent storage  
    aspect_ratio = np.zeros(num_grains)                 # Individual grain ratios
    table = micro_matrix[step,:,:,0]                    # Grain ID matrix

    # Create lists to store coordinates for each grain
    aspect_ratio_i_list = [[] for _ in range(int(num_grains))]
    aspect_ratio_j_list = [[] for _ in range(int(num_grains))]
    
    # Collect all coordinates for each grain
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            grain_id = int(table[i][j]-1)               # Convert to 0-based indexing
            sites_num_list[grain_id] +=1                # Count sites
            aspect_ratio_i_list[grain_id].append(coord_refer_i[i][j])  # x-coordinates
            aspect_ratio_j_list[grain_id].append(coord_refer_j[i][j])  # y-coordinates

    # Calculate aspect ratios for each grain
    for i in range(num_grains):
        # Count unique coordinates in each direction (spatial extent)
        aspect_ratio_i[i, 0] = len(list(set(aspect_ratio_i_list[i])))  # x-extent
        aspect_ratio_j[i, 1] = len(list(set(aspect_ratio_j_list[i])))  # y-extent
        
        if aspect_ratio_j[i, 1] == 0: 
            aspect_ratio[i] = 0                         # Handle degenerate case
        else: 
            aspect_ratio[i] = aspect_ratio_i[i, 0] / aspect_ratio_j[i, 1]  # Aspect ratio

    # Calculate site-weighted system average
    # aspect_ratio = np.average(aspect_ratio[aspect_ratio!=0])  # Simple average (commented)
    aspect_ratio = np.sum(aspect_ratio * sites_num_list) / np.sum(sites_num_list)

    return aspect_ratio

# ===============================================================================
# ADVANCED GRAIN BOUNDARY NORMAL VECTOR COMPUTATION FOR HIPERGATOR SYSTEMS
# ===============================================================================

def get_normal_vector(grain_structure_figure_one, grain_num):
    """
    Compute grain boundary normal vectors using VECTOR framework linear algorithms.
    
    This function implements high-performance normal vector computation for large-scale
    polycrystalline systems using the VECTOR framework's optimized linear algorithms.
    Designed for HiPerGator 3.0 supercomputing with 32-core parallel processing for
    20,000 initial grain systems.
    
    Mathematical Framework:
    - Linear2D class implementation with multi-physics algorithms
    - Virtual inclination energy methodology for accurate normal computation
    - Parallel processing optimization for computational efficiency
    - Comprehensive grain boundary site extraction with spatial indexing
    
    Parameters:
    -----------
    grain_structure_figure_one : numpy.ndarray
        2D grain structure matrix with grain IDs
        Shape: (nx, ny) representing spatial domain
    grain_num : int
        Total number of grains in the system (up to 20,000)
        
    Returns:
    --------
    P : numpy.ndarray
        3D array containing computed grain boundary properties
        Shape: (nx, ny, 2) with [inclination, magnitude] data
    sites_together : list
        Complete list of all grain boundary site coordinates
        Combined from all grain interfaces
    sites : list
        Organized list of grain boundary sites by grain ID
        Nested structure: sites[grain_id] = [site_coordinates]
        
    HiPerGator Optimizations:
    - 32-core parallel processing for computational efficiency
    - Memory-optimized data structures for large systems
    - Linear algorithm implementation with enhanced convergence
    - Vectorized operations for maximum performance
    
    VECTOR Framework Integration:
    - linear2d.linear_class for advanced boundary computations
    - Virtual inclination energy approach for accurate normals
    - Multi-physics linear algorithms for complex grain interactions
    """
    # Extract system dimensions and parameters
    nx = grain_structure_figure_one.shape[0]         # Grid dimension x
    ny = grain_structure_figure_one.shape[1]         # Grid dimension y
    ng = np.max(grain_structure_figure_one)          # Maximum grain ID
    
    # HiPerGator computational parameters for optimal performance
    cores = 32                                       # Parallel processing cores
    loop_times = 5                                   # Convergence iterations
    
    # Initialize input arrays for VECTOR framework
    P0 = grain_structure_figure_one                  # Initial grain structure
    R = np.zeros((nx,ny,2))                         # Boundary condition array
    
    # Create VECTOR linear algorithm class instance
    smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)

    # Execute virtual inclination energy computation
    smooth_class.linear_main("inclination")         # Main algorithm execution
    P = smooth_class.get_P()                        # Extract computed properties
    
    # Extract grain boundary sites with comprehensive coverage
    # Legacy single-grain extraction (commented for reference):
    # sites = smooth_class.get_gb_list(1)
    # print(len(sites))
    # for id in range(2,grain_num+1): sites += smooth_class.get_gb_list(id)
    # print(len(sites))
    
    # Optimized bulk extraction for all grain boundaries
    sites = smooth_class.get_all_gb_list()          # All grain boundary sites by ID
    sites_together = []                              # Combined site collection
    
    # Aggregate all grain boundary sites for system-level analysis
    for id in range(len(sites)): 
        sites_together += sites[id]                  # Combine all sites
    print("Total num of GB sites: " + str(len(sites_together)))

    return P, sites_together, sites

def get_normal_vector_slope(P, sites, step, para_name, bias=None):
    """
    Calculate grain boundary normal vector distributions with bias correction.
    
    This function extracts normal vector orientations from VECTOR framework
    computation results and generates statistical distributions for anisotropy
    analysis in large-scale polycrystalline systems. Includes advanced bias
    correction for accurate triple junction energy comparisons.
    
    Mathematical Framework:
    - Normal vector angle computation from inclination fields
    - Statistical binning for angular distribution analysis
    - Bias correction implementation for systematic error reduction
    - Frequency normalization for comparative studies
    
    Parameters:
    -----------
    P : numpy.ndarray
        3D property array from VECTOR computation [nx, ny, 2]
        Contains [inclination, magnitude] at each spatial point
    sites : list
        Grain boundary site coordinates for analysis
        Format: [[x1, y1], [x2, y2], ...]
    step : int
        Time step index for multi-temporal analysis
    para_name : str
        Parameter identifier for output organization
    bias : numpy.ndarray, optional
        Bias correction array for systematic error mitigation
        Applied to inclination values for enhanced accuracy
        
    Returns:
    --------
    freqArray : numpy.ndarray
        Frequency distribution of normal vector orientations
        Binned in 10-degree intervals (36 bins total)
    xCor : numpy.ndarray
        Angular coordinates for frequency bins (bin centers)
        Range: 0° to 360° in 10° increments
        
    Applications:
    - Anisotropy magnitude calculation for triple junction energy comparison
    - Statistical morphology analysis for large-scale systems
    - Bias-corrected normal distributions for publication-quality results
    """
    # Define angular binning parameters for statistical analysis
    xLim = [0, 360]                                  # Full angular range (degrees)
    binValue = 10.01                                 # Bin width (slightly > 10° for coverage)
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)  # Number of angular bins
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Bin centers

    # Initialize frequency distribution array
    freqArray = np.zeros(binNum)                     # Angular frequency distribution
    degree = []                                      # Normal vector angles storage

    # Calculate normal vector angles for all grain boundary sites
    for sitei in sites:
        [i,j] = sitei                                # Extract site coordinates
        dx,dy = myInput.get_grad(P,i,j)             # Compute gradient components
        # Calculate angle from gradient (normal vector orientation)
        degree.append(math.atan2(-dy, dx) + math.pi) # Convert to 0-2π range
        
        # Legacy angle calculation (commented for reference):
        # if dx == 0:
        #     degree.append(math.pi/2)
        # elif dy >= 0:
        #     degree.append(abs(math.atan(-dy/dx)))
        # elif dy < 0:
        #     degree.append(abs(math.atan(dy/dx)))

    # Bin the angles into frequency distribution
    for i in range(len(degree)):
        # Convert angle to degrees and determine bin index
        bin_index = int((degree[i]/math.pi*180-xLim[0])/binValue)
        freqArray[bin_index] += 1                    # Increment frequency count

    # Normalize frequency distribution
    freqArray = freqArray/sum(freqArray*binValue)    # Normalize to probability density

    # Apply bias correction if provided (for systematic error mitigation)
    if bias is not None:
        freqArray = freqArray + bias                 # Add bias correction
        freqArray = freqArray/sum(freqArray*binValue) # Re-normalize

    # Legacy polar plotting code (commented for reference):
    # Plot
    # plt.close()
    # fig = plt.figure(figsize=(5, 5))
    # ax = plt.gca(projection='polar')

    # ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    # ax.set_thetamin(0.0)
    # ax.set_thetamax(360.0)

    # ax.set_rgrids(np.arange(0, 0.008, 0.004))
    # ax.set_rlabel_position(0.0)  # 标签显示在0°
    # ax.set_rlim(0.0, 0.008)  # 标签范围为[0, 5000)
    # ax.set_yticklabels(['0', '0.004'],fontsize=14)

    # ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    # ax.set_axisbelow('True')
    
    # Generate current plot for statistical visualization
    plt.plot(np.append(xCor,xCor[0])/180*math.pi, np.append(freqArray,freqArray[0]), 
             linewidth=2, label=para_name)           # Polar plot with parameter label

    # Advanced linear fitting for trend analysis
    fit_coeff = np.polyfit(xCor, freqArray, 1)       # Linear regression coefficients
    
    return freqArray                                 # Return frequency distribution

# ===============================================================================
# MAIN EXECUTION: LARGE-SCALE POLYCRYSTALLINE ANALYSIS WITH HIPERGATOR INTEGRATION
# ===============================================================================

if __name__ == '__main__':
    """
    MAIN EXECUTION PIPELINE: 20K POLYCRYSTALLINE ANALYSIS WITH HIPERGATOR INTEGRATION
    
    This script executes comprehensive normal distribution analysis for large-scale
    polycrystalline systems (20,000 initial grains) using HiPerGator 3.0 supercomputing
    resources. Implements comparative analysis of six triple junction energy formulations
    with advanced bias correction and publication-quality visualization.
    
    HiPerGator Integration Features:
    - Blue storage system data loading (/blue/michael.tonks/lin.yang/)
    - Multi-core parallel processing optimization (32-64 cores)
    - Memory-efficient handling of massive microstructure datasets
    - Scalable analysis pipeline for supercomputing environments
    
    Triple Junction Energy Approaches Analyzed:
    1. Ave: Average energy formulation
    2. ConsMin: Conservative minimum approach  
    3. Sum: Summation-based energy calculation
    4. Min: Minimum energy selection
    5. Max: Maximum energy selection
    6. ConsMax: Conservative maximum approach
    7. ISO: Isotropic reference case for comparison
    
    Statistical Analysis Components:
    - Grain size distribution evolution tracking
    - Normal vector anisotropy magnitude calculation
    - Bias correction implementation for systematic error mitigation
    - Comparative visualization for scientific publication
    """
    # ===============================================================================
    # HIPERGATOR DATA LOADING: BLUE STORAGE SYSTEM INTEGRATION
    # ===============================================================================
    
    # HiPerGator Blue storage file system paths for large-scale simulation data
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/"
    
    # Triple junction energy formulation identifiers for comprehensive comparison
    TJ_energy_type_ave = "ave"                       # Average energy approach
    TJ_energy_type_consMin = "consMin"               # Conservative minimum
    TJ_energy_type_sum = "sum"                       # Summation approach  
    TJ_energy_type_min = "min"                       # Minimum energy selection
    TJ_energy_type_max = "max"                       # Maximum energy selection
    TJ_energy_type_consMax = "consMax"


    npy_file_name_iso = "p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_ave = f"pT_ori_ave_{TJ_energy_type_ave}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMin = f"p_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_sum = f"p_ori_ave_{TJ_energy_type_sum}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_min = f"p_ori_ave_{TJ_energy_type_min}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_max = f"p_ori_ave_{TJ_energy_type_max}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMax = f"p_ori_ave_{TJ_energy_type_consMax}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"

    # Initial data
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)
    npy_file_aniso_consMin = np.load(npy_file_folder + npy_file_name_aniso_consMin)
    npy_file_aniso_sum = np.load(npy_file_folder + npy_file_name_aniso_sum)
    npy_file_aniso_min = np.load(npy_file_folder + npy_file_name_aniso_min)
    npy_file_aniso_max = np.load(npy_file_folder + npy_file_name_aniso_max)
    npy_file_aniso_consMax = np.load(npy_file_folder + npy_file_name_aniso_consMax)
    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    print(f"The consMin data size is: {npy_file_aniso_consMin.shape}")
    print(f"The sum data size is: {npy_file_aniso_sum.shape}")
    print(f"The min data size is: {npy_file_aniso_min.shape}")
    print(f"The max data size is: {npy_file_aniso_max.shape}")
    print(f"The consMax data size is: {npy_file_aniso_consMax.shape}")
    print(f"The iso data size is: {npy_file_iso.shape}")
    print("READING DATA DONE")

    # ===============================================================================
    # STATISTICAL ANALYSIS INITIALIZATION: 20K GRAIN SYSTEM PARAMETERS
    # ===============================================================================
    
    # Fundamental system parameters for large-scale analysis
    initial_grain_num = 20000                        # Initial grain count for statistical robustness
    step_num = npy_file_aniso_ave.shape[0]          # Total time steps in simulation

    # Grain size distribution analysis parameters
    bin_width = 0.16                                 # Logarithmic bin width for size distribution
    x_limit = [-0.5, 3.5]                          # Size range in log scale
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)  # Number of size bins
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)
    
    # Initialize grain size distribution arrays for each energy formulation
    grain_size_distribution_iso = np.zeros(bin_num)              # Isotropic reference
    special_step_distribution_iso = 10                           # Time step to get ~2000 grains
    grain_size_distribution_ave = np.zeros(bin_num)              # Average energy approach
    special_step_distribution_ave = 11                           # Time step to get ~2000 grains
    grain_size_distribution_consMin = np.zeros(bin_num)          # Conservative minimum
    special_step_distribution_consMin = 11                       # Time step to get ~2000 grains
    grain_size_distribution_sum = np.zeros(bin_num)              # Summation approach
    special_step_distribution_sum = 11                           # Time step to get ~2000 grains
    grain_size_distribution_min = np.zeros(bin_num)              # Minimum energy selection
    special_step_distribution_min = 30                           # Time step to get ~2000 grains
    grain_size_distribution_max = np.zeros(bin_num)              # Maximum energy selection
    special_step_distribution_max = 15                           # Time step to get ~2000 grains
    grain_size_distribution_consMax = np.zeros(bin_num)          # Conservative maximum
    special_step_distribution_consMax = 11                       # Time step to get ~2000 grains

    # ===============================================================================
    # PUBLICATION-QUALITY POLAR VISUALIZATION SETUP
    # ===============================================================================
    
    # Initialize polar plot for comparative normal distribution analysis
    plt.close()                                      # Clear any existing plots
    fig = plt.figure(figsize=(5, 5))                # Publication-standard figure size
    ax = plt.gca(projection='polar')                 # Polar coordinate system

    # Configure polar plot aesthetics for scientific publication
    ax.set_thetagrids(np.arange(0.0, 360.0, 45.0),fontsize=16)  # Angular grid (45° intervals)
    ax.set_thetamin(0.0)                            # Minimum angle (0°)
    ax.set_thetamax(360.0)                          # Maximum angle (360°)

    # Radial axis configuration for frequency magnitude
    ax.set_rgrids(np.arange(0, 0.01, 0.004))        # Radial grid lines
    ax.set_rlabel_position(0.0)                     # Label position at 0°
    ax.set_rlim(0.0, 0.01)  # 标签范围为[0, 5000)
    ax.set_yticklabels(['0', '4e-3', '8e-3'],fontsize=16)

    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    # Get bias from kT test
    special_step_distribution_T066_bias = 10
    data_file_name_bias = f'/normal_distribution_data/normal_distribution_T066_bias_sites_step{special_step_distribution_T066_bias}.npy'
    slope_list_bias = np.load(current_path + data_file_name_bias)

    aniso_mag = np.zeros(6)
    aniso_mag_stand = np.zeros(6)

    # Aniso - min
    data_file_name_P = f'/normal_distribution_data/normal_distribution_min_P_step{special_step_distribution_min}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_min_sites_step{special_step_distribution_min}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_min, "Min",slope_list_bias)
    aniso_mag[0], aniso_mag_stand[0] = simple_magnitude(slope_list)

    # Aniso - max
    data_file_name_P = f'/normal_distribution_data/normal_distribution_max_P_step{special_step_distribution_max}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_max_sites_step{special_step_distribution_max}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_max, "Max",slope_list_bias)
    aniso_mag[1], aniso_mag_stand[1] = simple_magnitude(slope_list)

    # Aniso - ave
    data_file_name_P = f'/normal_distribution_data/normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_ave, "Ave",slope_list_bias)
    aniso_mag[2], aniso_mag_stand[2] = simple_magnitude(slope_list)

    # Aniso - sum
    data_file_name_P = f'/normal_distribution_data/normal_distribution_sum_P_step{special_step_distribution_sum}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_sum_sites_step{special_step_distribution_sum}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_sum[special_step_distribution_sum,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_sum, "Sum",slope_list_bias)
    aniso_mag[3], aniso_mag_stand[3] = simple_magnitude(slope_list)

    # Aniso - consMin
    data_file_name_P = f'/normal_distribution_data/normal_distribution_consMin_P_step{special_step_distribution_consMin}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_consMin_P_sites_step{special_step_distribution_consMin}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_consMin[special_step_distribution_consMin,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_consMin, "CMin",slope_list_bias)
    aniso_mag[4], aniso_mag_stand[4] = simple_magnitude(slope_list)

    # Aniso - consMax
    data_file_name_P = f'/normal_distribution_data/normal_distribution_consMax_P_step{special_step_distribution_consMax}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_consMax_sites_step{special_step_distribution_consMax}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_consMax[special_step_distribution_consMax,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_consMax, "CMax",slope_list_bias)
    aniso_mag[5], aniso_mag_stand[5] = simple_magnitude(slope_list)

    # Aniso - iso
    data_file_name_P = f'/normal_distribution_data/normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_iso, "Iso",slope_list_bias)

    plt.legend(loc=(-0.12,-0.35),fontsize=16,ncol=3)
    plt.savefig(current_path + "/figures/normal_distribution_poly_20k_after_removing_bias.png", dpi=400,bbox_inches='tight')

    # PLot magnitude of anisotropy
    # data_file_name_aniso_mag = f'/normal_distribution_data/aniso_magnitude_poly_20k_energy_type.npz'
    # if os.path.exists(current_path + data_file_name_aniso_mag):
    #     data_file_aniso_mag = np.load(current_path + data_file_name_aniso_mag)
    #     aniso_mag_min=data_file_aniso_mag['aniso_mag_min']
    #     aniso_mag_max=data_file_aniso_mag['aniso_mag_max']
    #     aniso_mag_ave=data_file_aniso_mag['aniso_mag_ave']
    #     aniso_mag_sum=data_file_aniso_mag['aniso_mag_sum']
    #     aniso_mag_consMin=data_file_aniso_mag['aniso_mag_consMin']
    #     aniso_mag_consMax=data_file_aniso_mag['aniso_mag_consMax']
    # else:
    #     aniso_mag_min = np.zeros(step_num)
    #     aniso_mag_max = np.zeros(step_num)
    #     aniso_mag_ave = np.zeros(step_num)
    #     aniso_mag_sum = np.zeros(step_num)
    #     aniso_mag_consMin = np.zeros(step_num)
    #     aniso_mag_consMax = np.zeros(step_num)
    #     cores = 16
    #     loop_times = 5
    #     for i in tqdm(range(step_num)):
    #         # newplace = np.rot90(npy_file_aniso_min[i,:,:,:], 1, (0,1))
    #         # newplace = npy_file_aniso_min[i,:,:,:]
    #         # nx = newplace.shape[0]
    #         # ny = newplace.shape[1]
    #         # ng = np.max(newplace)
    #         # R = np.zeros((nx,ny,2))
    #         # P0 = newplace
    #         # smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
    #         # sites_list = smooth_class.get_all_gb_list()
    #         # P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    #         aniso_mag_min[i] = get_poly_statistical_ar(npy_file_aniso_min, i)

    #         # newplace = np.rot90(npy_file_aniso_max[i,:,:,:], 1, (0,1))
    #         # newplace = npy_file_aniso_max[i,:,:,:]
    #         # nx = newplace.shape[0]
    #         # ny = newplace.shape[1]
    #         # ng = np.max(newplace)
    #         # R = np.zeros((nx,ny,2))
    #         # P0 = newplace
    #         # smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
    #         # sites_list = smooth_class.get_all_gb_list()
    #         # P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    #         aniso_mag_max[i] = get_poly_statistical_ar(npy_file_aniso_max, i)

    #         # newplace = np.rot90(npy_file_aniso_ave[i,:,:,:], 1, (0,1))
    #         # newplace = npy_file_aniso_ave[i,:,:,:]
    #         # nx = newplace.shape[0]
    #         # ny = newplace.shape[1]
    #         # ng = np.max(newplace)
    #         # R = np.zeros((nx,ny,2))
    #         # P0 = newplace
    #         # smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
    #         # sites_list = smooth_class.get_all_gb_list()
    #         # P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    #         aniso_mag_ave[i] = get_poly_statistical_ar(npy_file_aniso_ave, i)

    #         # newplace = np.rot90(npy_file_aniso_sum[i,:,:,:], 1, (0,1))
    #         # newplace = npy_file_aniso_sum[i,:,:,:]
    #         # nx = newplace.shape[0]
    #         # ny = newplace.shape[1]
    #         # ng = np.max(newplace)
    #         # R = np.zeros((nx,ny,2))
    #         # P0 = newplace
    #         # smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
    #         # sites_list = smooth_class.get_all_gb_list()
    #         # P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    #         aniso_mag_sum[i] = get_poly_statistical_ar(npy_file_aniso_sum, i)

    #         # newplace = np.rot90(npy_file_aniso_consMin[i,:,:,:], 1, (0,1))
    #         # newplace = npy_file_aniso_consMin[i,:,:,:]
    #         # nx = newplace.shape[0]
    #         # ny = newplace.shape[1]
    #         # ng = np.max(newplace)
    #         # R = np.zeros((nx,ny,2))
    #         # P0 = newplace
    #         # smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
    #         # sites_list = smooth_class.get_all_gb_list()
    #         # P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    #         aniso_mag_consMin[i] = get_poly_statistical_ar(npy_file_aniso_consMin, i)

    #         # newplace = np.rot90(npy_file_aniso_consMax[i,:,:,:], 1, (0,1))
    #         # newplace = npy_file_aniso_consMax[i,:,:,:]
    #         # nx = newplace.shape[0]
    #         # ny = newplace.shape[1]
    #         # ng = np.max(newplace)
    #         # R = np.zeros((nx,ny,2))
    #         # P0 = newplace
    #         # smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
    #         # sites_list = smooth_class.get_all_gb_list()
    #         # P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    #         aniso_mag_consMax[i] = get_poly_statistical_ar(npy_file_aniso_consMax, i)
    #     np.savez(current_path + data_file_name_aniso_mag, aniso_mag_min=aniso_mag_min,
    #                                                       aniso_mag_max=aniso_mag_max,
    #                                                       aniso_mag_ave=aniso_mag_ave,
    #                                                       aniso_mag_sum=aniso_mag_sum,
    #                                                       aniso_mag_consMin=aniso_mag_consMin,
    #                                                       aniso_mag_consMax=aniso_mag_consMax)
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_min, label='Min case', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_max, label='Max case', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_ave, label='Ave case', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_sum, label='Sum case', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_consMin, label='ConsMin case', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_consMax, label='ConsMax case', linewidth=2)
    label_list = ["Min", "Max", "Ave", "Sum", "CMin", "CMax"]
    # plt.errorbar(np.linspace(0,len(label_list)-1,len(label_list)), aniso_mag, yerr=aniso_mag_stand, linestyle='None', marker='None',color='black',linewidth=1, capsize=2)
    plt.plot(np.linspace(0,len(label_list)-1,len(label_list)), aniso_mag, '.-', markersize=8, label='around 2000 grains', linewidth=2)
    plt.xlabel("TJ energy approach", fontsize=16)
    plt.ylabel("Anisotropic Magnitude", fontsize=16)
    plt.xticks([0,1,2,3,4,5],label_list)
    # plt.legend(fontsize=16)
    plt.ylim([-0.05,1.0])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(current_path + "/figures/anisotropic_poly_20k_magnitude_polar_ave.png", dpi=400,bbox_inches='tight')











