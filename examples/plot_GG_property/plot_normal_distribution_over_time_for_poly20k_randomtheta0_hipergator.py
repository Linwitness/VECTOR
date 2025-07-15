#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPREHENSIVE RANDOM THETA0 POLYCRYSTALLINE GRAIN BOUNDARY ANALYSIS WITH HIPERGATOR INTEGRATION

Created on Mon Jul 31 14:33:57 2023
@author: Lin

======================================================================================
SCIENTIFIC FRAMEWORK: LARGE-SCALE RANDOM THETA0 GRAIN BOUNDARY NORMAL DISTRIBUTION ANALYSIS
======================================================================================

This script implements comprehensive grain boundary normal vector distribution analysis
for large-scale polycrystalline systems (20,000 initial grains) with randomized theta0
initial orientations using HiPerGator 3.0 supercomputing resources. The analysis focuses
on selective triple junction energy formulations with advanced bias correction and
publication-quality visualization capabilities.

RANDOM THETA0 IMPLEMENTATION FEATURES:
- Randomized Initial Orientations: Enhanced statistical sampling through randomized theta0 distribution
- Selective Energy Analysis: Min, Max, Ave formulations (ConsMin, Sum, ConsMax excluded for efficiency)
- Enhanced Statistical Robustness: Improved statistical coverage via random orientation sampling
- Computational Efficiency: Reduced complexity while maintaining scientific rigor
- HiPerGator Integration: 32-core parallel processing with Blue storage system optimization

TRIPLE JUNCTION ENERGY FORMULATIONS (SELECTIVE SET):
1. Min Energy: Minimum energy selection approach for triple junction interactions
2. Max Energy: Maximum energy selection methodology for enhanced anisotropy detection
3. Ave Energy: Average energy calculation for balanced statistical representation
4. Iso Reference: Isotropic baseline (delta=0.0) for comparative analysis

ADVANCED BIAS CORRECTION METHODOLOGY:
- Reference Temperature: kT=0.66 for systematic error mitigation
- Circular Symmetry Baseline: Perfect isotropy reference for deviation quantification
- Statistical Uncertainty: Comprehensive error propagation for publication quality

HIPERGATOR 3.0 SUPERCOMPUTING INTEGRATION:
- Parallel Processing: 32-core optimization for VECTOR framework computations
- Blue Storage System: Massive dataset management (/blue/michael.tonks/lin.yang/)
- Memory Optimization: Efficient handling of 20K grain random orientation systems
- Caching System: Minimizes redundant normal vector computations for efficiency

VECTOR FRAMEWORK IMPLEMENTATION:
- Linear2D Multi-Physics: Advanced linear algorithms for grain boundary analysis
- Virtual Inclination Energy: Accurate normal vector computation methodology
- Comprehensive Site Extraction: Complete grain boundary detection and characterization
- Statistical Distribution: Angular binning (36 bins, 10-degree intervals) for analysis

SYSTEM SPECIFICATIONS:
- Initial Grain Count: 20,000 grains for statistical robustness
- Anisotropy Parameter: delta=0.6 with randomized theta0 implementation
- Temperature Parameter: kT=0.66 for enhanced statistical accuracy
- Random Seed: 56689 for reproducible random number generation
- Domain Size: Large-scale simulation domain for realistic microstructure evolution

PUBLICATION-QUALITY OUTPUT:
- High-Resolution Visualization: 400 DPI polar plots with statistical uncertainty
- Comprehensive Legends: Clear identification of energy formulations
- Statistical Summary: Anisotropy magnitude comparison with error bars
- Scientific Documentation: Complete metadata and analysis specifications
"""

# ===============================================================================
# LIBRARY IMPORTS: COMPREHENSIVE SCIENTIFIC COMPUTING AND VISUALIZATION STACK
# ===============================================================================

# Core system and path management for HiPerGator environment
import os                                           # Operating system interface for file operations
current_path = os.getcwd()                         # Working directory for organized data management

# Scientific computing foundation for large-scale random theta0 analysis
import numpy as np                                 # Numerical arrays and mathematical operations
from numpy import seterr                           # Error handling for numerical computations
seterr(all='raise')                               # Raise exceptions for numerical errors

# Advanced visualization for publication-quality polar plots
import matplotlib.pyplot as plt                    # Comprehensive plotting and visualization
import math                                        # Mathematical functions for angular calculations

# Progress tracking for long-running HiPerGator computations
from tqdm import tqdm                              # Progress bars for iterative processes

# System path configuration for VECTOR framework integration
import sys                                         # System-specific parameters and functions
sys.path.append(current_path)                     # Current directory for local modules
sys.path.append(current_path+'/../../')           # Parent directories for VECTOR framework
import myInput                                     # Custom input/output utilities for grain analysis
import PACKAGE_MP_Linear as linear2d               # VECTOR Linear2D multi-physics algorithms
sys.path.append(current_path+'/../calculate_tangent/')  # Additional computational modules

# ===============================================================================
# STATISTICAL ANISOTROPY ANALYSIS FOR RANDOM THETA0 SYSTEMS
# ===============================================================================

def simple_magnitude(freqArray):
    """
    Calculate anisotropic magnitude metrics for large-scale polycrystalline systems with random theta0.
    
    This function quantifies the deviation of grain boundary normal distributions
    from perfect circular symmetry for 20K grain systems with randomized theta0
    initial conditions. Essential for comparing different triple junction energy 
    formulations in HiPerGator simulations with bias correction applied.
    
    Mathematical Framework:
    - Creates uniform circular reference distribution for comparison
    - Computes absolute deviations from perfect circular symmetry
    - Calculates statistical moments with enhanced precision for large systems
    - Normalizes by reference distribution average for relative comparison
    
    Parameters:
    -----------
    freqArray : numpy.ndarray
        Frequency array of grain boundary normal orientations (36 bins, 10° each)
        Derived from 20K grain boundary sites with random theta0 initialization
        
    Returns:
    --------
    magnitude_ave : float
        Average relative deviation from circular symmetry
    magnitude_stan : float
        Standard deviation of relative deviations (statistical uncertainty)
        
    Random Theta0 Applications:
    - Large-scale random orientation system anisotropy quantification
    - Statistical comparison across selective triple junction energy approaches
    - Bias-corrected anisotropy measurement for publication-quality results
    """
    # Define angular coordinate system for comprehensive random theta0 analysis
    xLim = [0, 360]                                  # Full angular range (degrees)
    binValue = 10.01                                 # Bin width for angular discretization
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)  # Number of angular bins
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Bin centers

    # Create ideal uniform circular distribution for comparison baseline
    freqArray_circle = np.ones(binNum)               # Uniform distribution
    freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)  # Normalize

    # Calculate statistical measures of anisotropy for large random theta0 systems
    magnitude_max = np.max(abs(freqArray - freqArray_circle))/np.average(freqArray_circle)
    magnitude_ave = np.average(abs(freqArray - freqArray_circle))/np.average(freqArray_circle)

    # Compute standard deviation for statistical uncertainty quantification
    magnitude_stan = np.sqrt(np.sum((abs(freqArray - freqArray_circle)/np.average(freqArray_circle) - magnitude_ave)**2)/binNum)

    return magnitude_ave, magnitude_stan

    # Legacy coefficient-based analysis (commented for reference)
    # coeff_high = abs(np.cos((xCor-90)/180*np.pi))
    # coeff_low = abs(np.cos((xCor)/180*np.pi))
    # return np.sum(freqArray * coeff_high)/np.sum(freqArray * coeff_low)

def find_fittingEllipse2(array): #failure
    """
    Legacy elliptical fitting function for grain shape analysis.
    
    NOTE: This function is marked as 'failure' and retained for reference only.
    It was an experimental approach for elliptical fitting that proved insufficient
    for large-scale polycrystalline analysis with random theta0 conditions.
    
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
        
    Status: DEPRECATED - Use advanced methods for 20K random theta0 systems
    """
    K_mat = []                                       # Design matrix initialization
    Y_mat = []                                       # Target vector initialization

    # Extract coordinate vectors
    X = array[:,0]                                   # x-coordinates
    Y = array[:,1]                                   # y-coordinates

    # Construct design matrix for conic equation fitting
    K_mat = np.hstack([X**2, X*Y, Y**2, X, Y])      # Design matrix
    Y_mat = np.ones_like(X)                          # Target vector

    # Solve least-squares problem for conic parameters
    X_mat = np.linalg.lstsq(K_mat, Y_mat)[0].squeeze()
    # Alternative formulation: X_mat = (K_mat.T*K_mat).I * K_mat.T * Y_mat

    # Debug output for ellipse equation
    print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(X_mat[0], X_mat[1], X_mat[2], X_mat[3], X_mat[4]))
    print(X_mat)

    return X_mat

# ===============================================================================
# POLYCRYSTALLINE GRAIN GEOMETRY ANALYSIS FOR 20K RANDOM THETA0 SYSTEMS
# ===============================================================================

def get_poly_center(micro_matrix, step):
    """
    Calculate center coordinates and average radii for all grains in random theta0 polycrystalline system.
    
    This function processes complete microstructure arrays from HiPerGator simulations
    to determine grain centers and characteristic radii for 20,000 initial grain systems
    with randomized theta0 orientations. Critical for large-scale statistical
    analysis of grain evolution under different triple junction energy formulations.
    
    Mathematical Framework:
    - Grid-based grain identification using microstructure matrix
    - Weighted centroid calculation for each grain using all interior sites
    - Boundary condition exclusion for periodic domain handling
    - Area-equivalent radius computation from grain site counts
    - Small grain filtering (< 500 sites) for statistical robustness
    
    Parameters:
    -----------
    micro_matrix : numpy.ndarray
        4D microstructure array [time, x, y, data] from HiPerGator random theta0 simulations
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
        
    HiPerGator Random Theta0 Optimizations:
    - Efficient grain boundary detection for massive random orientation systems
    - Memory-optimized processing for large microstructure arrays
    - Periodic boundary condition handling for simulation domains
    - Statistical filtering for enhanced data quality with random initial conditions
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

        # Exclude small grains and boundary-crossing grains for random theta0 analysis
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
    Calculate statistical radius deviation metrics for random theta0 polycrystalline morphology.
    
    This function quantifies the deviation of grain shapes from perfect circles
    by analyzing the distribution of distances from grain centers to boundary sites
    in 20K grain systems with randomized theta0 orientations. Essential for characterizing
    morphological evolution under different triple junction energy approaches.
    
    Mathematical Framework:
    - Distance calculation from grain centers to all boundary sites
    - Radius offset computation relative to area-equivalent radius
    - Area-weighted statistical averaging for system-level metrics
    - Morphological deviation quantification for random theta0 systems
    
    Parameters:
    -----------
    micro_matrix : numpy.ndarray
        4D microstructure array from HiPerGator random theta0 simulations
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
        
    Random Theta0 Applications:
    - Grain shape evolution tracking in large-scale random orientation systems
    - Morphological anisotropy quantification with randomized initial conditions
    - Triple junction energy formulation comparison for random theta0 implementation
    """
    # Get the max offset of average radius and real radius
    center_list, ave_radius_list = get_poly_center(micro_matrix, step)
    num_grains = int(np.max(micro_matrix[step,:]))       # Total grain count

    max_radius_offset_list = np.zeros(num_grains)       # Maximum offsets per grain
    
    # Calculate radius deviations for each grain in random theta0 system
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

    # Calculate area-weighted system average for random theta0 analysis
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
    Calculate system-averaged aspect ratio for random theta0 polycrystalline grain morphology.
    
    This function computes aspect ratios for all grains based on their spatial
    extent in x and y directions, providing a measure of grain elongation and
    morphological anisotropy in 20K grain systems with randomized theta0 orientations.
    
    Mathematical Framework:
    - Grid-based grain boundary detection for random theta0 systems
    - Spatial extent calculation in principal directions
    - Aspect ratio computation as x-extent / y-extent
    - Site-weighted system averaging for statistical robustness
    
    Parameters:
    -----------
    micro_matrix : numpy.ndarray
        4D microstructure array from HiPerGator random theta0 simulations
    step : int
        Time step index for analysis
        
    Returns:
    --------
    aspect_ratio : float
        Site-weighted average aspect ratio for entire random theta0 system
        Values > 1 indicate elongation in x-direction
        Values < 1 indicate elongation in y-direction
        
    HiPerGator Random Theta0 Applications:
    - Large-scale morphological evolution tracking with random orientations
    - Anisotropy quantification under different triple junction energy formulations
    - Statistical shape characterization for massive random theta0 grain datasets
    """
    # Get the average aspect ratio
    num_grains = int(np.max(micro_matrix[step,:]))       # Total number of grains
    sites_num_list = np.zeros(num_grains)               # Site count per grain
    
    # Create coordinate reference system for random theta0 analysis
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
    
    # Collect all coordinates for each grain in random theta0 system
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            grain_id = int(table[i][j]-1)               # Convert to 0-based indexing
            sites_num_list[grain_id] +=1                # Count sites
            aspect_ratio_i_list[grain_id].append(coord_refer_i[i][j])  # x-coordinates
            aspect_ratio_j_list[grain_id].append(coord_refer_j[i][j])  # y-coordinates

    # Calculate aspect ratios for each grain in random theta0 system
    for i in range(num_grains):
        # Count unique coordinates in each direction (spatial extent)
        aspect_ratio_i[i, 0] = len(list(set(aspect_ratio_i_list[i])))  # x-extent
        aspect_ratio_j[i, 1] = len(list(set(aspect_ratio_j_list[i])))  # y-extent
        
        if aspect_ratio_j[i, 1] == 0: 
            aspect_ratio[i] = 0                         # Handle degenerate case
        else: 
            aspect_ratio[i] = aspect_ratio_i[i, 0] / aspect_ratio_j[i, 1]  # Aspect ratio

    # Calculate site-weighted system average for random theta0 analysis
    # aspect_ratio = np.average(aspect_ratio[aspect_ratio!=0])  # Simple average (commented)
    aspect_ratio = np.sum(aspect_ratio * sites_num_list) / np.sum(sites_num_list)

    return aspect_ratio

# ===============================================================================
# ADVANCED GRAIN BOUNDARY NORMAL VECTOR COMPUTATION FOR RANDOM THETA0 SYSTEMS
# ===============================================================================

def get_normal_vector(grain_structure_figure_one, grain_num):
    """
    Compute grain boundary normal vectors using VECTOR framework for random theta0 systems.
    
    This function implements high-performance normal vector computation for large-scale
    polycrystalline systems with randomized theta0 orientations using the VECTOR framework's
    optimized linear algorithms. Designed for HiPerGator 3.0 supercomputing with 32-core
    parallel processing for 20,000 initial grain systems with random theta0 implementation.
    
    Mathematical Framework:
    - Linear2D class implementation with multi-physics algorithms for random theta0
    - Virtual inclination energy methodology for accurate normal computation
    - Parallel processing optimization for computational efficiency (32 cores)
    - Comprehensive grain boundary site extraction with spatial indexing
    
    Parameters:
    -----------
    grain_structure_figure_one : numpy.ndarray
        2D grain structure matrix with grain IDs for random theta0 system
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
        Combined from all grain interfaces in random theta0 system
    sites : list
        Organized list of grain boundary sites by grain ID
        Nested structure: sites[grain_id] = [site_coordinates]
        
    HiPerGator Random Theta0 Optimizations:
    - 32-core parallel processing for computational efficiency
    - Memory-optimized data structures for large random orientation systems
    - Linear algorithm implementation with enhanced convergence
    - Vectorized operations for maximum performance
    
    VECTOR Framework Integration:
    - linear2d.linear_class for advanced boundary computations
    - Virtual inclination energy approach for accurate normals
    - Multi-physics linear algorithms for complex grain interactions
    - Random theta0 orientation handling with enhanced robustness
    """
    # Extract system dimensions and parameters for random theta0 analysis
    nx = grain_structure_figure_one.shape[0]         # Grid dimension x
    ny = grain_structure_figure_one.shape[1]         # Grid dimension y
    ng = np.max(grain_structure_figure_one)          # Maximum grain ID
    
    # HiPerGator computational parameters for random theta0 systems
    cores = 32                                       # Parallel processing cores
    loop_times = 5                                   # Convergence iterations
    
    # Initialize input arrays for VECTOR framework
    P0 = grain_structure_figure_one                  # Initial grain structure
    R = np.zeros((nx,ny,2))                         # Boundary condition array
    
    # Create VECTOR linear algorithm class instance for random theta0
    smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)

    # Execute virtual inclination energy computation for random theta0
    smooth_class.linear_main("inclination")         # Main algorithm execution
    P = smooth_class.get_P()                        # Extract computed properties
    
    # Extract grain boundary sites with comprehensive coverage
    # Legacy single-grain extraction (commented for reference):
    # sites = smooth_class.get_gb_list(1)
    # print(len(sites))
    # for id in range(2,grain_num+1): sites += smooth_class.get_gb_list(id)
    # print(len(sites))
    
    # Optimized bulk extraction for all grain boundaries in random theta0 system
    sites = smooth_class.get_all_gb_list()          # All grain boundary sites by ID
    sites_together = []                              # Combined site collection
    
    # Aggregate all grain boundary sites for system-level random theta0 analysis
    for id in range(len(sites)): 
        sites_together += sites[id]                  # Combine all sites
    print("Total num of GB sites: " + str(len(sites_together)))

    return P, sites_together, sites

def get_normal_vector_slope(P, sites, step, para_name, bias=None):
    """
    Calculate grain boundary normal vector distributions with bias correction for random theta0 systems.
    
    This function extracts normal vector orientations from VECTOR framework
    computation results and generates statistical distributions for anisotropy
    analysis in large-scale polycrystalline systems with randomized theta0 orientations.
    Includes advanced bias correction for accurate triple junction energy comparisons.
    
    Mathematical Framework:
    - Normal vector angle computation from inclination fields
    - Statistical binning for angular distribution analysis
    - Bias correction implementation for systematic error reduction (kT=0.66)
    - Frequency normalization for comparative studies across random theta0 formulations
    
    Parameters:
    -----------
    P : numpy.ndarray
        3D property array from VECTOR computation [nx, ny, 2]
        Contains [inclination, magnitude] at each spatial point
    sites : list
        Grain boundary site coordinates for random theta0 analysis
        Format: [[x1, y1], [x2, y2], ...]
    step : int
        Time step index for multi-temporal analysis
    para_name : str
        Parameter identifier for output organization (Min, Max, Ave, Iso)
    bias : numpy.ndarray, optional
        Bias correction array for systematic error mitigation (kT=0.66 reference)
        Applied to inclination values for enhanced accuracy
        
    Returns:
    --------
    freqArray : numpy.ndarray
        Frequency distribution of normal vector orientations
        Binned in 10-degree intervals (36 bins total)
        
    Applications:
    - Anisotropy magnitude calculation for random theta0 triple junction energy comparison
    - Statistical morphology analysis for large-scale random orientation systems
    - Bias-corrected normal distributions for publication-quality results
    """
    # Define angular binning parameters for random theta0 statistical analysis
    xLim = [0, 360]                                  # Full angular range (degrees)
    binValue = 10.01                                 # Bin width (slightly > 10° for coverage)
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)  # Number of angular bins
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Bin centers

    # Initialize frequency distribution array for random theta0 analysis
    freqArray = np.zeros(binNum)                     # Angular frequency distribution
    degree = []                                      # Normal vector angles storage

    # Calculate normal vector angles for all grain boundary sites in random theta0 system
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

    # Bin the angles into frequency distribution for random theta0 analysis
    for i in range(len(degree)):
        # Convert angle to degrees and determine bin index
        bin_index = int((degree[i]/math.pi*180-xLim[0])/binValue)
        freqArray[bin_index] += 1                    # Increment frequency count

    # Normalize frequency distribution for random theta0 comparison
    freqArray = freqArray/sum(freqArray*binValue)    # Normalize to probability density

    # Apply bias correction if provided (kT=0.66 reference for systematic error mitigation)
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
    
    # Generate current plot for random theta0 statistical visualization
    plt.plot(np.append(xCor,xCor[0])/180*math.pi, np.append(freqArray,freqArray[0]), 
             linewidth=2, label=para_name)           # Polar plot with parameter label

    # Advanced linear fitting for trend analysis in random theta0 systems
    fit_coeff = np.polyfit(xCor, freqArray, 1)       # Linear regression coefficients
    
    return freqArray                                 # Return frequency distribution

# ===============================================================================
# MAIN EXECUTION: 20K RANDOM THETA0 ANALYSIS WITH HIPERGATOR INTEGRATION
# ===============================================================================

if __name__ == '__main__':
    """
    MAIN EXECUTION PIPELINE: 20K POLYCRYSTALLINE RANDOM THETA0 ANALYSIS WITH HIPERGATOR INTEGRATION
    
    This script executes comprehensive normal distribution analysis for large-scale
    polycrystalline systems (20,000 initial grains) with randomized theta0 initial
    orientations using HiPerGator 3.0 supercomputing resources. Implements selective
    analysis of primary triple junction energy formulations with advanced bias 
    correction and publication-quality visualization.
    
    HiPerGator Integration Features:
    - Blue storage system data loading (/blue/michael.tonks/lin.yang/)
    - 32-core parallel processing for VECTOR framework computations
    - Memory-efficient handling of massive random theta0 microstructure datasets
    - Scalable analysis pipeline for supercomputing environments
    
    Random Theta0 Implementation:
    - Randomized initial grain orientation distribution for enhanced statistical coverage
    - Delta=0.6 anisotropy with randomized theta0 conditions
    - kT=0.66 temperature parameter for statistical accuracy
    - Seed=56689 for reproducible random number generation
    - Selective energy formulation analysis for computational efficiency
    
    Triple Junction Energy Approaches Analyzed (Selective Set):
    1. Min: Minimum energy selection approach
    2. Max: Maximum energy selection approach  
    3. Ave: Average energy calculation
    4. Iso: Isotropic reference case (delta=0.0) for comparison
    
    Note: ConsMin, Sum, ConsMax approaches commented out for focused analysis
    
    Statistical Analysis Components:
    - Grain boundary normal vector distribution analysis
    - Bias correction using kT=0.66 reference temperature
    - Anisotropy magnitude quantification for comparative studies
    - Publication-quality polar visualization with statistical uncertainty
    """
    # ===============================================================================
    # HIPERGATOR BLUE STORAGE DATA LOADING: RANDOM THETA0 SIMULATION DATASETS
    # ===============================================================================
    
    # HiPerGator Blue storage file system paths for large-scale random theta0 simulation data
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/"
    
    # Triple junction energy formulation identifiers for selective random theta0 comparison
    TJ_energy_type_ave = "ave"                       # Average energy approach
    TJ_energy_type_consMin = "consMin"               # Conservative minimum (commented out)
    TJ_energy_type_sum = "sum"                       # Summation approach (commented out)
    TJ_energy_type_min = "min"                       # Minimum energy selection
    TJ_energy_type_max = "max"                       # Maximum energy selection
    TJ_energy_type_consMax = "consMax"               # Conservative maximum (commented out)

    # Standardized file naming convention for HiPerGator batch processing with random theta0
    # Isotropic reference case: 32-core, delta=0.0, no random theta0 implementation
    npy_file_name_iso = "p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    
    # Random theta0 cases: delta=0.6, kT=0.66, seed=56689 for reproducibility
    npy_file_name_aniso_ave = f"p_randomtheta0_{TJ_energy_type_ave}E_20000_delta0.6_J1_refer_1_0_0_seed56689_kt0.66.npy"
    # npy_file_name_aniso_consMin = f"pm_randomtheta0_{TJ_energy_type_consMin}E_20000_delta0.6_J1_refer_1_0_0_seed56689_kt0.66.npy"  # Commented for selective analysis
    # npy_file_name_aniso_sum = f"pm_randomtheta0_{TJ_energy_type_sum}E_20000_delta0.6_J1_refer_1_0_0_seed56689_kt0.66.npy"  # Commented for selective analysis
    npy_file_name_aniso_min = f"p_randomtheta0_{TJ_energy_type_min}E_20000_delta0.6_J1_refer_1_0_0_seed56689_kt0.66.npy"
    npy_file_name_aniso_max = f"p_randomtheta0_{TJ_energy_type_max}E_20000_delta0.6_J1_refer_1_0_0_seed56689_kt0.66.npy"
    # npy_file_name_aniso_consMax = f"pm_randomtheta0_{TJ_energy_type_consMax}E_20000_delta0.6_J1_refer_1_0_0_seed56689_kt0.66.npy"  # Commented for selective analysis

    # ===============================================================================
    # SELECTIVE RANDOM THETA0 DATASET LOADING: 20K GRAIN MICROSTRUCTURE EVOLUTION DATA
    # ===============================================================================
    
    # Load datasets with comprehensive random theta0 microstructure evolution data
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)                    # Isotropic reference
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)        # Average energy random theta0
    # npy_file_aniso_consMin = np.load(npy_file_folder + npy_file_name_aniso_consMin)  # Conservative min (commented)
    # npy_file_aniso_sum = np.load(npy_file_folder + npy_file_name_aniso_sum)        # Summation (commented)
    npy_file_aniso_min = np.load(npy_file_folder + npy_file_name_aniso_min)        # Minimum energy random theta0
    npy_file_aniso_max = np.load(npy_file_folder + npy_file_name_aniso_max)        # Maximum energy random theta0
    # npy_file_aniso_consMax = np.load(npy_file_folder + npy_file_name_aniso_consMax) # Conservative max (commented)
    
    # Dataset size verification for HiPerGator computational planning
    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    # print(f"The consMin data size is: {npy_file_aniso_consMin.shape}")  # Commented for selective analysis
    # print(f"The sum data size is: {npy_file_aniso_sum.shape}")          # Commented for selective analysis
    print(f"The min data size is: {npy_file_aniso_min.shape}")
    print(f"The max data size is: {npy_file_aniso_max.shape}")
    # print(f"The consMax data size is: {npy_file_aniso_consMax.shape}")  # Commented for selective analysis
    print(f"The iso data size is: {npy_file_iso.shape}")
    print("READING DATA DONE")

    # ===============================================================================
    # STATISTICAL ANALYSIS INITIALIZATION: 20K RANDOM THETA0 SYSTEM PARAMETERS
    # ===============================================================================
    
    # Fundamental system parameters for large-scale random theta0 analysis
    initial_grain_num = 20000                        # Initial grain count for statistical robustness
    step_num = npy_file_aniso_ave.shape[0]          # Total time steps in simulation

    # Grain size distribution analysis parameters for random theta0 systems
    bin_width = 0.16                                 # Logarithmic bin width for size distribution
    x_limit = [-0.5, 3.5]                          # Size range in log scale
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)  # Number of size bins
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)
    
    # Initialize grain size distribution arrays for each random theta0 energy formulation
    grain_size_distribution_iso = np.zeros(bin_num)              # Isotropic reference
    special_step_distribution_iso = 10                           # Time step to get ~2000 grains
    grain_size_distribution_ave = np.zeros(bin_num)              # Average energy random theta0
    special_step_distribution_ave = 12                           # Time step to get ~2000 grains
    # grain_size_distribution_consMin = np.zeros(bin_num)        # Conservative minimum (commented)
    # special_step_distribution_consMin = 12                     # Time step to get ~2000 grains (commented)
    # grain_size_distribution_sum = np.zeros(bin_num)            # Summation (commented)
    # special_step_distribution_sum = 12                         # Time step to get ~2000 grains (commented)
    grain_size_distribution_iso = np.zeros(bin_num)              # Duplicate initialization (legacy)
    grain_size_distribution_min = np.zeros(bin_num)              # Minimum energy random theta0
    special_step_distribution_min = 22                           # Time step to get ~2000 grains
    grain_size_distribution_max = np.zeros(bin_num)              # Maximum energy random theta0
    special_step_distribution_max = 12                           # Time step to get ~2000 grains
    # grain_size_distribution_consMax = np.zeros(bin_num)        # Conservative maximum (commented)
    # special_step_distribution_consMax = 13                     # Time step to get ~2000 grains (commented)


    # ===============================================================================
    # BIAS CORRECTION IMPLEMENTATION: KT=0.66 REFERENCE TEMPERATURE FOR RANDOM THETA0
    # ===============================================================================
    
    # Load or compute bias correction data from kT=0.66 temperature reference study
    special_step_distribution_T066_bias = 10        # Reference time step for bias
    data_file_name_bias = f'/normal_distribution_data/normal_distribution_T066_bias_sites_step{special_step_distribution_T066_bias}.npy'
    
    if os.path.exists(current_path + data_file_name_bias):
        # Load pre-computed bias correction for efficiency
        slope_list_bias = np.load(current_path + data_file_name_bias)
    else:
        # Compute bias correction from isotropic reference case
        data_file_name_P = f'/normal_distribution_data/normal_distribution_iso_P_step{special_step_distribution_T066_bias}.npy'
        data_file_name_sites = f'/normal_distribution_data/normal_distribution_iso_sites_step{special_step_distribution_T066_bias}.npy'
        
        if os.path.exists(current_path + data_file_name_P):
            # Load cached normal vector data
            P = np.load(current_path + data_file_name_P)
            sites = np.load(current_path + data_file_name_sites)
        else:
            # Fresh computation for bias reference
            newplace = np.rot90(npy_file_iso[special_step_distribution_T066_bias,:,:,:], 1, (0,1))
            P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
            # Cache results for future bias calculations
            np.save(current_path + data_file_name_P, P)
            np.save(current_path + data_file_name_sites, sites)

        # Generate bias correction plot and compute bias array
        plt.close()
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca(projection='polar')
        slope_list = get_normal_vector_slope(P, sites, special_step_distribution_T066_bias, "Iso")
        
        # Compute bias correction array for random theta0 analysis
        xLim = [0, 360]
        binValue = 10.01
        binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
        freqArray_circle = np.ones(binNum)           # Perfect circle reference
        freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)  # Normalize
        slope_list_bias = freqArray_circle - slope_list  # Bias correction array
        # Save bias correction for future use
        np.save(current_path + data_file_name_bias, slope_list_bias)
    # ===============================================================================
    # PUBLICATION-QUALITY POLAR VISUALIZATION INITIALIZATION FOR RANDOM THETA0
    # ===============================================================================
    
    # Initialize comprehensive polar plot for random theta0 energy formulation comparison
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    # Configure polar plot aesthetics for publication-quality random theta0 analysis
    ax.set_thetagrids(np.arange(0.0, 360.0, 45.0),fontsize=16)  # Angular grid (45° intervals)
    ax.set_thetamin(0.0)                             # Minimum angle
    ax.set_thetamax(360.0)                           # Maximum angle

    ax.set_rgrids(np.arange(0, 0.01, 0.004))         # Radial grid lines
    ax.set_rlabel_position(0.0)                      # Radial label position (0°)
    ax.set_rlim(0.0, 0.01)                          # Radial range for random theta0 data
    ax.set_yticklabels(['0', '4e-3', '8e-3'],fontsize=16)  # Custom radial labels

    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)  # Grid styling
    ax.set_axisbelow('True')                         # Grid behind data

    # Initialize anisotropy magnitude storage for random theta0 comparative analysis
    aniso_mag = np.zeros(6)                          # Magnitude values (6 approaches max)
    aniso_mag_stand = np.zeros(6)                    # Standard deviations for uncertainty

    # ===============================================================================
    # MIN ENERGY FORMULATION: RANDOM THETA0 ANALYSIS
    # ===============================================================================
    
    # Load or compute normal vector data for minimum energy random theta0 formulation
    data_file_name_P = f'/normal_distribution_data/normal_distribution_rdt0_min_P_step{special_step_distribution_min}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_rdt0_min_sites_step{special_step_distribution_min}.npy'
    
    if os.path.exists(current_path + data_file_name_P):
        # Load cached data for computational efficiency
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        # Fresh computation for minimum energy random theta0 analysis
        newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        # Cache results for future analysis efficiency
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Generate frequency distribution with bias correction for minimum energy random theta0
    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_min, "Min",slope_list_bias)
    aniso_mag[0], aniso_mag_stand[0] = simple_magnitude(slope_list)

    # ===============================================================================
    # MAX ENERGY FORMULATION: RANDOM THETA0 ANALYSIS
    # ===============================================================================
    
    # Load or compute normal vector data for maximum energy random theta0 formulation
    data_file_name_P = f'/normal_distribution_data/normal_distribution_rdt0_max_P_step{special_step_distribution_max}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_rdt0_max_sites_step{special_step_distribution_max}.npy'
    
    if os.path.exists(current_path + data_file_name_P):
        # Load cached data for computational efficiency
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        # Fresh computation for maximum energy random theta0 analysis
        newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        # Cache results for future analysis efficiency
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Generate frequency distribution with bias correction for maximum energy random theta0
    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_max, "Max",slope_list_bias)
    aniso_mag[1], aniso_mag_stand[1] = simple_magnitude(slope_list)

    # ===============================================================================
    # AVERAGE ENERGY FORMULATION: RANDOM THETA0 ANALYSIS
    # ===============================================================================
    
    # Load or compute normal vector data for average energy random theta0 formulation
    data_file_name_P = f'/normal_distribution_data/normal_distribution_rdt0_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_rdt0_ave_sites_step{special_step_distribution_ave}.npy'
    
    if os.path.exists(current_path + data_file_name_P):
        # Load cached data for computational efficiency
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        # Fresh computation for average energy random theta0 analysis
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        # Cache results for future analysis efficiency
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Generate frequency distribution with bias correction for average energy random theta0
    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_ave, "Ave",slope_list_bias)
    aniso_mag[2], aniso_mag_stand[2] = simple_magnitude(slope_list)

#     # Aniso - sum
#     data_file_name_P = f'/normal_distribution_data/normal_distribution_rdt0m_sum_P_step{special_step_distribution_sum}.npy'
#     data_file_name_sites = f'/normal_distribution_data/normal_distribution_rdt0m_sum_sites_step{special_step_distribution_sum}.npy'
#     if os.path.exists(current_path + data_file_name_P):
#         P = np.load(current_path + data_file_name_P)
#         sites = np.load(current_path + data_file_name_sites)
#     else:
#         newplace = np.rot90(npy_file_aniso_sum[special_step_distribution_sum,:,:,:], 1, (0,1))
#         P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
#         np.save(current_path + data_file_name_P, P)
#         np.save(current_path + data_file_name_sites, sites)

#     slope_list = get_normal_vector_slope(P, sites, special_step_distribution_sum, "Sum",slope_list_bias)
#     aniso_mag[3], aniso_mag_stand[3] = simple_magnitude(slope_list)

#     # Aniso - consMin
#     data_file_name_P = f'/normal_distribution_data/normal_distribution_rdt0m_consMin_P_step{special_step_distribution_consMin}.npy'
#     data_file_name_sites = f'/normal_distribution_data/normal_distribution_rdt0m_consMin_P_sites_step{special_step_distribution_consMin}.npy'
#     if os.path.exists(current_path + data_file_name_P):
#         P = np.load(current_path + data_file_name_P)
#         sites = np.load(current_path + data_file_name_sites)
#     else:
#         newplace = np.rot90(npy_file_aniso_consMin[special_step_distribution_consMin,:,:,:], 1, (0,1))
#         P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
#         np.save(current_path + data_file_name_P, P)
#         np.save(current_path + data_file_name_sites, sites)

#     slope_list = get_normal_vector_slope(P, sites, special_step_distribution_consMin, "CMin",slope_list_bias)
#     aniso_mag[4], aniso_mag_stand[4] = simple_magnitude(slope_list)

#     # Aniso - consMax
#     data_file_name_P = f'/normal_distribution_data/normal_distribution_rdt0m_consMax_P_step{special_step_distribution_consMax}.npy'
#     data_file_name_sites = f'/normal_distribution_data/normal_distribution_rdt0m_consMax_sites_step{special_step_distribution_consMax}.npy'
#     if os.path.exists(current_path + data_file_name_P):
#         P = np.load(current_path + data_file_name_P)
#         sites = np.load(current_path + data_file_name_sites)
#     else:
#         newplace = np.rot90(npy_file_aniso_consMax[special_step_distribution_consMax,:,:,:], 1, (0,1))
#         P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
#         np.save(current_path + data_file_name_P, P)
#         np.save(current_path + data_file_name_sites, sites)

#     slope_list = get_normal_vector_slope(P, sites, special_step_distribution_consMax, "CMax",slope_list_bias)
#     aniso_mag[5], aniso_mag_stand[5] = simple_magnitude(slope_list)

    # Aniso - iso
    data_file_name_P = f'/normal_distribution_data/normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_iso, "Iso",slope_list_bias)

    plt.legend(loc=(-0.12,-0.35),fontsize=16,ncol=3)
    plt.savefig(current_path + "/figures/normal_distribution_poly_randomtheta0_20k_after_removing_bias.png", dpi=400,bbox_inches='tight')

    # plt.close()
    # fig = plt.figure(figsize=(5, 5))
    # label_list = ["Min", "Max", "Ave", "Sum", "CMin", "CMax"]
    # # plt.errorbar(np.linspace(0,len(label_list)-1,len(label_list)), aniso_mag, yerr=aniso_mag_stand, linestyle='None', marker='None',color='black',linewidth=1, capsize=2)
    # plt.plot(np.linspace(0,len(label_list)-1,len(label_list)), aniso_mag, '.-', markersize=8, label='around 2000 grains', linewidth=2)
    # plt.xlabel("TJ energy approach", fontsize=16)
    # plt.ylabel("Anisotropic Magnitude", fontsize=16)
    # plt.xticks([0,1,2,3,4,5],label_list)
    # # plt.legend(fontsize=16)
    # plt.ylim([-0.05,1.0])
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.savefig(current_path + "/figures/anisotropic_polym_randomtheta0_20k_magnitude_polar_ave.png", dpi=400,bbox_inches='tight')











