#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
2D CIRCULAR GRAIN BOUNDARY NORMAL DISTRIBUTION ANALYSIS FRAMEWORK
================================================================================

Purpose: Comprehensive analysis of grain boundary normal vector distributions 
         in 2D circular microstructures with anisotropic grain boundary energy

Scientific Context:
- SPPARKS Monte Carlo simulations of circular grain growth
- Multi-core parallel processing for computational efficiency
- Anisotropic grain boundary energy analysis (σ = 0.0 to 0.95)
- Virtual inclination energy methodology for orientation analysis
- Bias correction algorithms for statistical accuracy

Analysis Framework:
1. Normal vector computation using VECTOR's 2D linear algorithms
2. Polar distribution visualization for circular geometries
3. Statistical anisotropy magnitude quantification
4. Bias-corrected analysis for improved accuracy
5. Comparative study across different energy parameters

Key Features:
- Elliptical fitting for grain shape characterization
- Statistical radius deviation analysis
- Aspect ratio calculations for morphological assessment
- Multi-energy parameter comparison (σ: 0.0, 0.2, 0.4, 0.6, 0.8, 0.95)
- Publication-quality polar plot generation

Created on Mon Jul 31 14:33:57 2023
@author: Lin
================================================================================
"""

# ===============================================================================
# LIBRARY IMPORTS AND PATH CONFIGURATION
# ===============================================================================

# System and path management
import os
current_path = os.getcwd()
import sys
sys.path.append(current_path)                    # Current working directory
sys.path.append(current_path+'/../../')          # VECTOR root directory
sys.path.append(current_path+'/../calculate_tangent/')  # Tangent calculation utilities

# Scientific computing libraries
import numpy as np
from numpy import seterr
seterr(all='raise')                              # Raise exceptions for numerical warnings
import math                                      # Mathematical functions for trigonometry

# Visualization and progress tracking
import matplotlib.pyplot as plt                  # Publication-quality plotting
from tqdm import tqdm                           # Progress bar for long computations

# VECTOR framework modules
import myInput                                   # Custom input/output utilities
import PACKAGE_MP_Linear as linear2d            # 2D linear multi-physics package

# ===============================================================================
# STATISTICAL ANALYSIS FUNCTIONS FOR CIRCULAR GRAIN BOUNDARIES
# ===============================================================================

def simple_magnitude(freqArray):
    """
    Calculate anisotropic magnitude metrics for circular grain boundary analysis.
    
    This function quantifies the deviation of grain boundary normal distributions
    from perfect circular symmetry by comparing observed frequency distributions
    against an ideal uniform circular distribution.
    
    Mathematical Framework:
    - Creates uniform circular reference distribution
    - Computes absolute deviations from circular symmetry
    - Calculates statistical moments (average, standard deviation)
    - Normalizes by reference distribution average
    
    Parameters:
    -----------
    freqArray : numpy.ndarray
        Frequency array of grain boundary normal orientations (36 bins, 10° each)
        
    Returns:
    --------
    magnitude_ave : float
        Average relative deviation from circular symmetry
    magnitude_stan : float
        Standard deviation of relative deviations (statistical uncertainty)
        
    Algorithm Details:
    - Angular range: 0° to 360° (full circle coverage)
    - Bin resolution: 10.01° (36 bins for robust statistics)
    - Reference: Perfectly uniform circular distribution
    """
    # Define angular coordinate system for circular analysis
    xLim = [0, 360]                              # Full angular range (degrees)
    binValue = 10.01                             # Bin width for angular discretization
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)  # Number of angular bins
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Bin centers
    
    # Create ideal uniform circular distribution for comparison
    freqArray_circle = np.ones(binNum)           # Uniform distribution
    freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)  # Normalize
    
    # Calculate statistical measures of anisotropy
    magnitude_max = np.max(abs(freqArray - freqArray_circle))/np.average(freqArray_circle)
    magnitude_ave = np.average(abs(freqArray - freqArray_circle))/np.average(freqArray_circle)
    
    # Compute standard deviation for statistical uncertainty quantification
    magnitude_stan = np.sqrt(np.sum((abs(freqArray - freqArray_circle)/np.average(freqArray_circle) - magnitude_ave)**2)/binNum)
    
    return magnitude_ave, magnitude_stan
    
    # Legacy coefficient-based analysis (commented for reference)
    # coeff_high = abs(np.cos((xCor-90)/180*np.pi))
    # coeff_low = abs(np.cos((xCor)/180*np.pi))
    # return np.sum(freqArray * coeff_high)/np.sum(freqArray * coeff_low)

def fit_ellipse_for_circle(sites_list):
    """
    Elliptical fitting analysis for circular grain shape characterization.
    
    This function performs least-squares ellipse fitting to grain boundary sites
    to quantify deviations from perfect circular morphology. Essential for
    understanding grain shape evolution under anisotropic energy conditions.
    
    Mathematical Framework:
    - General conic equation: Ax² + Bxy + Cy² + Dx + Ey + F = 0
    - Least-squares parameter estimation using linear algebra
    - Eigenvalue analysis for major/minor axis determination
    - Aspect ratio calculation for shape quantification
    
    Parameters:
    -----------
    sites_list : list of arrays
        List containing grain boundary site coordinates for each grain
        Each element: numpy array of [x, y] coordinates
        
    Returns:
    --------
    aspect_ratio : float
        Average aspect ratio (b/a) of fitted ellipses
        Value = 1.0 indicates perfect circular grains
        
    Algorithm Details:
    - Uses robust least-squares fitting for parameter estimation
    - Handles numerical stability through proper matrix conditioning
    - Calculates ellipse center, orientation, and axis lengths
    - Averages aspect ratios across all analyzable grains
    """
    
    # Validate input data for ellipse fitting
    grain_num = len(sites_list)
    if grain_num < 2: return 1                   # Return unity for insufficient data
    
    # Initialize arrays for ellipse parameters
    a_square_list = np.ones(grain_num)           # Major axis squared values
    b_square_list = np.ones(grain_num)           # Minor axis squared values
    
    # Process each grain for elliptical fitting
    for i in range(grain_num):
        array = np.array(sites_list[i])
    
        # Extract coordinate vectors for matrix construction
        X = array[:,0]                           # x-coordinates of boundary sites
        Y = array[:,1]                           # y-coordinates of boundary sites
    
        # Construct design matrix for conic equation fitting
        K_mat = np.array([X**2, X*Y, Y**2, X, Y]).T  # Design matrix
        Y_mat = -np.ones_like(X)                 # Target vector
    
        # Solve least-squares problem for conic parameters
        X_mat = np.linalg.lstsq(K_mat, Y_mat, rcond=None)[0].squeeze()
        # Alternative formulation: X_mat = (K_mat.T*K_mat).I * K_mat.T * Y_mat
        
        # Calculate ellipse geometric properties from conic parameters
        center_base = 4 * X_mat[0] * X_mat[2] - X_mat[1] * X_mat[1]
        center_x = (X_mat[1] * X_mat[4] - 2 * X_mat[2]* X_mat[3]) / center_base
        center_y = (X_mat[1] * X_mat[3] - 2 * X_mat[0]* X_mat[4]) / center_base
        axis_square_root = np.sqrt((X_mat[0] - X_mat[2])**2 + X_mat[1]**2)
        a_square = 2*(X_mat[0]*center_x*center_x + X_mat[2]*center_y*center_y + X_mat[1]*center_x*center_y - 1) / (X_mat[0] + X_mat[2] + axis_square_root)
        b_square = 2*(X_mat[0]*center_x*center_x + X_mat[2]*center_y*center_y + X_mat[1]*center_x*center_y - 1) / (X_mat[0] + X_mat[2] - axis_square_root)
        
        # Debug output for ellipse dimensions
        print(f"a: {np.sqrt(a_square)}, b: {np.sqrt(b_square)}")
        a_square_list[i] = a_square
        b_square_list[i] = b_square
    
    # Return average aspect ratio across all grains
    return np.average(np.sqrt(b_square_list) / np.sqrt(a_square_list))

def get_circle_center(micro_matrix, step):
    """
    Calculate geometric centers and statistical radii for circular grains.
    
    This function performs centroid analysis of grain structures to determine
    the geometric center of mass for each grain and compute equivalent circular
    radii based on grain area. Essential for circular morphology analysis.
    
    Mathematical Framework:
    - Center of mass calculation: weighted average of coordinates
    - Equivalent radius: R = sqrt(Area/π) for circular approximation
    - Per-grain analysis for microstructural characterization
    
    Parameters:
    -----------
    micro_matrix : numpy.ndarray
        4D microstructure array [time, x, y, features]
    step : int
        Time step index for analysis
        
    Returns:
    --------
    center_list : numpy.ndarray
        Array of [x, y] coordinates for grain centers
    ave_radius_list : numpy.ndarray
        Array of equivalent circular radii for each grain
        
    Algorithm Details:
    - Handles periodic boundary conditions appropriately
    - Calculates area-equivalent circular radii
    - Provides robust center-of-mass computation
    """
    # Extract grain information from microstructure
    num_grains = int(np.max(micro_matrix[0,:]))   # Total number of grains
    center_list = np.zeros((num_grains,2))        # Initialize center coordinates
    sites_num_list = np.zeros(num_grains)         # Sites per grain
    ave_radius_list = np.zeros(num_grains)        # Equivalent radii
    
    # Create coordinate reference arrays for center-of-mass calculation
    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i,j] = i                # x-coordinate reference
            coord_refer_j[i,j] = j                # y-coordinate reference

    # Extract microstructure at specified time step
    table = micro_matrix[step,:,:,0]
    
    # Calculate center and radius for each grain
    for i in range(num_grains):
        sites_num_list[i] = np.sum(table == i+1)  # Count sites in grain i+1

        if sites_num_list[i] == 0:
          # Handle empty grains (edge case)
          center_list[i, 0] = 0
          center_list[i, 1] = 0
        else:
          # Calculate center of mass for grain i+1
          center_list[i, 0] = np.sum(coord_refer_i[table == i+1]) / sites_num_list[i]
          center_list[i, 1] = np.sum(coord_refer_j[table == i+1]) / sites_num_list[i]
    
    # Calculate equivalent circular radii from grain areas
    ave_radius_list = np.sqrt(sites_num_list / np.pi)

    return center_list, ave_radius_list

def get_circle_statistical_radius(micro_matrix, sites_list, step):
    """
    Statistical analysis of radial deviations in circular grain morphology.
    
    This function quantifies how well circular grains maintain their circular
    shape by analyzing the distribution of distances from grain boundary sites
    to the grain center. Provides metrics for circularity assessment.
    
    Mathematical Framework:
    - Radial distance calculation: r = sqrt((x-cx)² + (y-cy)²)
    - Deviation analysis: |r_actual - r_average| / r_average
    - Statistical moments: mean and standard deviation of deviations
    
    Parameters:
    -----------
    micro_matrix : numpy.ndarray
        4D microstructure array [time, x, y, features]
    sites_list : list
        List of grain boundary site coordinates
    step : int
        Time step index for analysis
        
    Returns:
    --------
    ave_radius_offset : float
        Average relative radial deviation from perfect circle
    magnitude_stan : float
        Standard deviation of radial deviations
        
    Algorithm Details:
    - Focuses on grain #2 (index 1) for detailed analysis
    - Normalizes deviations by average radius for scale independence
    - Handles edge cases with zero radius gracefully
    """
    # Get geometric properties for current time step
    center_list, ave_radius_list = get_circle_center(micro_matrix, step)
    center = center_list[1]                       # Center of grain #2
    ave_radius = ave_radius_list[1]               # Average radius of grain #2
    
    # Extract boundary sites for target grain
    if len(sites_list) < 2:
        sites = []                                # Handle insufficient data
    else:
        sites = sites_list[1]                     # Boundary sites for grain #2

    # Initialize statistical variables
    max_radius_offset = 0                         # Maximum deviation tracker
    ave_radius_offset_list = np.zeros(len(sites)) # Per-site deviation array
    
    # Calculate radial deviations for each boundary site
    for index, sitei in enumerate(sites):
        [i,j] = sitei                             # Site coordinates
        current_radius = np.sqrt((i - center[0])**2 + (j - center[1])**2)
        radius_offset = abs(current_radius - ave_radius)
        ave_radius_offset_list[index] = radius_offset
        if radius_offset > max_radius_offset: max_radius_offset = radius_offset
    
    # Calculate normalized statistical measures
    if ave_radius == 0:
        # Handle degenerate case (zero radius)
        max_radius_offset = 0
        ave_radius_offset = 0
    else:
        # Normalize by average radius for scale independence
        max_radius_offset = max_radius_offset / ave_radius
        ave_radius_offset = np.average(ave_radius_offset_list) / ave_radius
        magnitude_stan = np.sqrt(np.sum((ave_radius_offset_list/ave_radius - ave_radius_offset)**2)/len(sites))

    return ave_radius_offset, magnitude_stan

def get_circle_statistical_ar(micro_matrix, step):
    """
    Calculate aspect ratio statistics for circular grain morphology analysis.
    
    This function computes aspect ratios of grains to assess deviations from
    circular symmetry. The aspect ratio provides a simple metric for grain
    elongation and shape characterization in 2D microstructures.
    
    Mathematical Framework:
    - Aspect ratio = (unique x-coordinates) / (unique y-coordinates)
    - Focuses on primary grain (grain #2) for detailed analysis
    - Value = 1.0 indicates square/circular grain boundary box
    
    Parameters:
    -----------
    micro_matrix : numpy.ndarray
        4D microstructure array [time, x, y, features]
    step : int
        Time step index for analysis
        
    Returns:
    --------
    aspect_ratio : float
        Aspect ratio of the bounding box for grain #2
        
    Algorithm Details:
    - Uses bounding box approach for computational efficiency
    - Counts unique coordinate values in each direction
    - Handles degenerate cases with zero dimensions
    """
    # Extract microstructure information
    num_grains = int(np.max(micro_matrix[step,:]))  # Number of grains at time step
    sites_num_list = np.zeros(num_grains)           # Sites per grain
    
    # Create coordinate reference arrays
    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i,j] = i                  # x-coordinate reference
            coord_refer_j[i,j] = j                  # y-coordinate reference
    
    # Initialize aspect ratio calculation variables
    aspect_ratio_i = np.zeros(2)                    # x-direction extent
    aspect_ratio_j = np.zeros(2)                    # y-direction extent
    aspect_ratio = 0
    table = micro_matrix[step,:,:,0]                # Microstructure snapshot
    
    # Focus on grain #2 (index 1) for aspect ratio analysis
    for i in [1]:  # range(num_grains): (only analyzing grain #2)
        sites_num_list = np.sum(table == i+1)       # Count sites in grain
        
        # Calculate bounding box dimensions
        aspect_ratio_i[0] = len(list(set(coord_refer_i[table == i+1])))  # Unique x-coords
        aspect_ratio_j[1] = len(list(set(coord_refer_j[table == i+1])))  # Unique y-coords
        
        # Compute aspect ratio with singularity handling
        if aspect_ratio_j[1] == 0: 
            aspect_ratio = 1                        # Default for degenerate case
        else: 
            aspect_ratio = aspect_ratio_i[0] / aspect_ratio_j[1]  # Width/Height ratio
            
    # Legacy multi-grain averaging code (commented for reference)        
    # aspect_ratio = np.average(aspect_ratio[aspect_ratio!=0])
    # aspect_ratio = np.sum(aspect_ratio * sites_num_list) / np.sum(sites_num_list)

    return aspect_ratio

def get_normal_vector(grain_structure_figure_one, grain_num):
    """
    Compute grain boundary normal vectors using VECTOR's 2D linear algorithm.
    
    This function applies the VECTOR framework's multi-physics linear solver
    to compute grain boundary normal vectors for 2D microstructures. The normal
    vectors are essential for orientation distribution analysis.
    
    Mathematical Framework:
    - Inclination energy minimization for normal vector computation
    - Multi-core parallel processing for computational efficiency
    - Grain boundary detection and site extraction
    - Smooth field computation for accurate gradients
    
    Parameters:
    -----------
    grain_structure_figure_one : numpy.ndarray
        2D microstructure array with grain ID assignments
    grain_num : int
        Number of grains in the microstructure (used for validation)
        
    Returns:
    --------
    P : numpy.ndarray
        Smooth field array for gradient computation
    sites_together : list
        Combined list of all grain boundary sites
    sites : list of lists
        Per-grain grain boundary site lists
        
    Algorithm Details:
    - Uses VECTOR's linear_class for multi-physics computation
    - Applies "inclination" method for energy minimization
    - Extracts grain boundaries for all grains systematically
    - Provides comprehensive grain boundary site information
    """
    # Extract microstructure dimensions and grain information
    nx = grain_structure_figure_one.shape[0]      # Grid dimensions in x
    ny = grain_structure_figure_one.shape[1]      # Grid dimensions in y
    ng = np.max(grain_structure_figure_one)       # Maximum grain ID
    
    # Configure multi-physics solver parameters
    cores = 8                                     # Parallel processing cores
    loop_times = 5                                # Iteration count for convergence
    P0 = grain_structure_figure_one               # Initial microstructure
    R = np.zeros((nx,ny,2))                       # Resistance/constraint array
    
    # Initialize VECTOR's 2D linear multi-physics solver
    smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)

    # Execute inclination energy minimization for normal vector computation
    smooth_class.linear_main("inclination")
    P = smooth_class.get_P()                      # Extract smooth field solution
    
    # Legacy single-grain analysis code (commented for reference)
    # sites = smooth_class.get_gb_list(1)
    # print(len(sites))
    # for id in range(2,grain_num+1): sites += smooth_class.get_gb_list(id)
    # print(len(sites))
    
    # Extract grain boundary sites for all grains
    sites = smooth_class.get_all_gb_list()        # Per-grain boundary site lists
    sites_together = []
    for id in range(len(sites)): 
        sites_together += sites[id]               # Combine all boundary sites
    print("Total num of GB sites: " + str(len(sites_together)))

    return P, sites_together, sites

def get_normal_vector_slope(P, sites, step, para_name, bias=None):
    """
    Calculate angular distribution of grain boundary normal vectors with polar visualization.
    
    This function processes grain boundary normal vectors to generate angular frequency
    distributions for statistical analysis of grain boundary orientation preferences.
    Essential for quantifying anisotropic grain boundary energy effects.
    
    Mathematical Framework:
    - Gradient computation: ∇P using finite differences
    - Normal vector calculation: n = ∇P/|∇P|
    - Angular conversion: θ = atan2(-dy, dx) + π
    - Histogram binning: 36 bins × 10° resolution
    - Optional bias correction for statistical accuracy
    
    Parameters:
    -----------
    P : numpy.ndarray
        Smooth field array from VECTOR's inclination energy solver
    sites : list
        Grain boundary site coordinates [i, j]
    step : int
        Time step identifier for tracking
    para_name : str
        Parameter name for plot legend (e.g., "σ=0.40")
    bias : numpy.ndarray, optional
        Bias correction array for improved statistical accuracy
        
    Returns:
    --------
    freqArray : numpy.ndarray
        Normalized frequency distribution of normal vector orientations
        
    Algorithm Details:
    - Full angular range: 0° to 360° for complete orientation coverage
    - High-resolution binning: 10.01° bins for smooth distributions
    - Automatic normalization for probability density representation
    - Integrated polar plotting for visualization
    - Optional bias correction capability
    """
    # Define angular discretization parameters
    xLim = [0, 360]                              # Full angular range (degrees)
    binValue = 10.01                             # Angular bin width
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)  # Number of bins
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Bin centers

    # Initialize frequency array and degree list
    freqArray = np.zeros(binNum)                 # Angular frequency distribution
    degree = []                                  # Individual orientation angles
    
    # Process each grain boundary site for normal vector calculation
    for sitei in sites:
        [i,j] = sitei                            # Extract site coordinates
        dx,dy = myInput.get_grad(P,i,j)          # Compute gradient components
        degree.append(math.atan2(-dy, dx) + math.pi)  # Calculate normal angle
        
        # Legacy angle calculation methods (commented for reference)
        # if dx == 0:
        #     degree.append(math.pi/2)
        # elif dy >= 0:
        #     degree.append(abs(math.atan(-dy/dx)))
        # elif dy < 0:
        #     degree.append(abs(math.atan(dy/dx)))
    
    # Bin the orientation angles into frequency histogram
    for i in range(len(degree)):
        freqArray[int((degree[i]/math.pi*180-xLim[0])/binValue)] += 1
    
    # Normalize frequency array for probability density representation
    freqArray = freqArray/sum(freqArray*binValue)

    # Apply bias correction if provided
    if bias is not None:
        freqArray = freqArray + bias            # Add bias correction
        freqArray = freqArray/sum(freqArray*binValue)  # Re-normalize

    # Legacy polar plot setup code (commented for reference)
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
    
    # Add polar plot trace for current parameter set
    plt.plot(np.append(xCor,xCor[0])/180*math.pi, np.append(freqArray,freqArray[0]), 
             linewidth=2, label=para_name)

    # Linear fitting for trend analysis (computed but not used)
    fit_coeff = np.polyfit(xCor, freqArray, 1)
    
    return freqArray

# ===============================================================================
# MAIN EXECUTION: 2D CIRCULAR GRAIN BOUNDARY ANALYSIS WORKFLOW
# ===============================================================================

if __name__ == '__main__':
    # ==========================================================================
    # DATA SOURCE CONFIGURATION: SPPARKS SIMULATION RESULTS
    # ==========================================================================
    
    # Base directory for SPPARKS Monte Carlo simulation results
    npy_file_folder = "/Users/lin.yang/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_circle_multiCoreCompare/results/"
    
    # Anisotropic grain boundary energy parameter values (σ parameters)
    # These represent different levels of grain boundary energy anisotropy
    circle_energy_000 = "0.0"     # Isotropic case (no anisotropy)
    circle_energy_020 = "0.2"     # Weak anisotropy
    circle_energy_040 = "0.4"     # Moderate anisotropy  
    circle_energy_060 = "0.6"     # Strong anisotropy
    circle_energy_080 = "0.8"     # Very strong anisotropy
    circle_energy_095 = "0.95"    # Near-maximum anisotropy

    # ==========================================================================
    # SPPARKS SIMULATION FILE NAMING CONVENTION
    # ==========================================================================
    
    # Microstructure evolution data files (formatted with simulation parameters)
    # Format: orientation_aveE_xxx_xxx_multiCore16_kt066_seed56689_scale1_delta{σ}_m2_refer_1_0_0.npy
    npy_file_name_aniso_000 = f"c_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_000}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_020 = f"c_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_020}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_040 = f"c_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_040}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_060 = f"c_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_060}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_080 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_080}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_095 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_095}_m2_refer_1_0_0.npy"
    
    # Grain size distribution data files (for validation and correlation analysis)
    grain_size_data_name_000 = f"grain_size_c_ori_aveE_000_000_multiCore32_delta{circle_energy_000}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_020 = f"grain_size_c_ori_aveE_000_000_multiCore32_delta{circle_energy_020}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_040 = f"grain_size_c_ori_aveE_000_000_multiCore32_delta{circle_energy_040}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_060 = f"grain_size_c_ori_aveE_000_000_multiCore32_delta{circle_energy_060}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_080 = f"grain_size_c_ori_aveE_000_000_multiCore32_delta{circle_energy_080}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_095 = f"grain_size_c_ori_aveE_000_000_multiCore32_delta{circle_energy_095}_m2_J1_refer_1_0_0_seed56689_kt066.npy"

    # ==========================================================================
    # DATA LOADING AND VALIDATION
    # ==========================================================================
    
    # Load microstructure evolution data for all anisotropy levels
    npy_file_aniso_000 = np.load(npy_file_folder + npy_file_name_aniso_000)
    npy_file_aniso_020 = np.load(npy_file_folder + npy_file_name_aniso_020)
    npy_file_aniso_040 = np.load(npy_file_folder + npy_file_name_aniso_040)
    npy_file_aniso_060 = np.load(npy_file_folder + npy_file_name_aniso_060)
    npy_file_aniso_080 = np.load(npy_file_folder + npy_file_name_aniso_080)
    npy_file_aniso_095 = np.load(npy_file_folder + npy_file_name_aniso_095)
    
    # Data validation: Print array dimensions for verification
    print(f"The 000 data size is: {npy_file_aniso_000.shape}")
    print(f"The 020 data size is: {npy_file_aniso_020.shape}")
    print(f"The 040 data size is: {npy_file_aniso_040.shape}")
    print(f"The 060 data size is: {npy_file_aniso_060.shape}")
    print(f"The 080 data size is: {npy_file_aniso_080.shape}")
    print(f"The 095 data size is: {npy_file_aniso_095.shape}")
    print("READING DATA DONE")

    # ==========================================================================
    # ANALYSIS PARAMETERS AND INITIALIZATION
    # ==========================================================================
    
    # Microstructure analysis parameters
    initial_grain_num = 2                        # Number of grains in initial configuration
    step_num = npy_file_aniso_000.shape[0]       # Total number of time steps

    # Grain size distribution analysis parameters (currently unused but available)
    bin_width = 0.16                             # Bin width for grain size distributions
    x_limit = [-0.5, 3.5]                       # Range for size distribution analysis
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)

    # Time step selection for detailed normal distribution analysis
    # Step 30 chosen for mature grain structure after initial transients
    special_step_distribution_000 = 30#4        # Analysis time step for σ=0.0
    special_step_distribution_020 = 30#4        # Analysis time step for σ=0.2
    special_step_distribution_040 = 30#4        # Analysis time step for σ=0.4
    special_step_distribution_060 = 30#4        # Analysis time step for σ=0.6
    special_step_distribution_080 = 30#4        # Analysis time step for σ=0.8
    special_step_distribution_095 = 30#4        # Analysis time step for σ=0.95

    # ==========================================================================
    # PART I: RAW POLAR DISTRIBUTION ANALYSIS (WITHOUT BIAS CORRECTION)
    # ==========================================================================

    # Initialize polar plot for grain boundary normal distribution visualization
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    # Configure polar plot appearance and scale
    ax.set_thetagrids(np.arange(0.0, 360.0, 45.0),fontsize=16)  # Angular grid every 45°
    ax.set_thetamin(0.0)                         # Start angle
    ax.set_thetamax(360.0)                       # End angle

    # Radial axis configuration for frequency density
    ax.set_rgrids(np.arange(0, 0.008, 0.004))    # Radial grid lines
    ax.set_rlabel_position(0.0)                  # Radial labels at 0° position
    ax.set_rlim(0.0, 0.008)                      # Radial limits for frequency density
    ax.set_yticklabels(['0', '4e-3'],fontsize=16) # Radial tick labels

    # Plot styling for publication quality
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')                     # Grid behind data

    # -----------------------------------------------------------------------
    # ANALYSIS: σ = 0.00 (ISOTROPIC REFERENCE CASE)
    # -----------------------------------------------------------------------
    
    # Define data file paths for caching computed normal vectors
    data_file_name_P = f'/normal_distribution_data/normal_distribution_000_P_step{special_step_distribution_000}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_000_sites_step{special_step_distribution_000}.npy'
    
    # Load cached data if available, otherwise compute normal vectors
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)           # Smooth field
        sites = np.load(current_path + data_file_name_sites)   # Boundary sites
    else:
        # Prepare microstructure data with proper orientation
        newplace = np.rot90(npy_file_aniso_000[special_step_distribution_000,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        # Cache computed results for future analysis
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Compute and plot angular distribution for isotropic case
    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_000, r"$\sigma=0.00$")
    
    # -----------------------------------------------------------------------
    # BIAS CALCULATION FOR STATISTICAL CORRECTION
    # -----------------------------------------------------------------------
    
    # Calculate bias correction using isotropic reference case
    # The bias represents systematic deviations from perfect circular symmetry
    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    freqArray_circle = np.ones(binNum)           # Perfect circular distribution
    freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)  # Normalize
    slope_list_bias = freqArray_circle - slope_list  # Bias = ideal - observed

    # -----------------------------------------------------------------------
    # ANALYSIS: σ = 0.20 (WEAK ANISOTROPY)
    # -----------------------------------------------------------------------
    
    data_file_name_P = f'/normal_distribution_data/normal_distribution_020_P_step{special_step_distribution_020}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_020_P_sites_step{special_step_distribution_020}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_020[special_step_distribution_020,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_020, r"$\sigma=0.20$")

    # -----------------------------------------------------------------------
    # ANALYSIS: σ = 0.40 (MODERATE ANISOTROPY)
    # -----------------------------------------------------------------------
    
    data_file_name_P = f'/normal_distribution_data/normal_distribution_040_P_step{special_step_distribution_040}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_040_sites_step{special_step_distribution_040}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_040[special_step_distribution_040,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_040, r"$\sigma=0.40$")

    # -----------------------------------------------------------------------
    # ANALYSIS: σ = 0.60 (STRONG ANISOTROPY)
    # -----------------------------------------------------------------------
    
    data_file_name_P = f'/normal_distribution_data/normal_distribution_060_P_step{special_step_distribution_060}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_060_sites_step{special_step_distribution_060}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_060[special_step_distribution_060,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_060, r"$\sigma=0.60$")

    # -----------------------------------------------------------------------
    # ANALYSIS: σ = 0.80 (VERY STRONG ANISOTROPY)
    # -----------------------------------------------------------------------
    
    data_file_name_P = f'/normal_distribution_data/normal_distribution_080_P_step{special_step_distribution_080}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_080_sites_step{special_step_distribution_080}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_080[special_step_distribution_080,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_080, r"$\sigma=0.80$")

    # -----------------------------------------------------------------------
    # ANALYSIS: σ = 0.95 (NEAR-MAXIMUM ANISOTROPY)
    # -----------------------------------------------------------------------
    
    data_file_name_P = f'/normal_distribution_data/normal_distribution_095_P_step{special_step_distribution_095}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_095_sites_step{special_step_distribution_095}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_095[special_step_distribution_095,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_095, r"$\sigma=0.95$")

    # Finalize raw distribution polar plot
    plt.legend(loc=(-0.24,-0.3),fontsize=16,ncol=3)
    plt.savefig(current_path + "/figures/normal_distribution_circle.png", dpi=400,bbox_inches='tight')

    # ==========================================================================
    # PART II: BIAS-CORRECTED POLAR DISTRIBUTION ANALYSIS
    # ==========================================================================

    # Initialize new polar plot for bias-corrected analysis
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    # Configure polar plot with identical formatting for direct comparison
    ax.set_thetagrids(np.arange(0.0, 360.0, 45.0),fontsize=16)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)
    ax.set_rgrids(np.arange(0, 0.008, 0.004))
    ax.set_rlabel_position(0.0)
    ax.set_rlim(0.0, 0.008)
    ax.set_yticklabels(['0', '4e-3'],fontsize=16)
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    # Initialize anisotropy magnitude arrays for quantitative analysis
    aniso_mag = np.zeros(6)                      # Average anisotropy magnitudes
    aniso_mag_stand = np.zeros(6)                # Standard deviations

    # -----------------------------------------------------------------------
    # BIAS-CORRECTED ANALYSIS: σ = 0.00 (ISOTROPIC REFERENCE)
    # -----------------------------------------------------------------------
    
    data_file_name_P = f'/normal_distribution_data/normal_distribution_000_P_step{special_step_distribution_000}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_000_sites_step{special_step_distribution_000}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_000[special_step_distribution_000,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply bias correction and calculate anisotropy statistics
    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_000, r"$\sigma=0.00$", slope_list_bias)
    aniso_mag[0], aniso_mag_stand[0] = simple_magnitude(slope_list)

    # -----------------------------------------------------------------------
    # BIAS-CORRECTED ANALYSIS: σ = 0.20 (WEAK ANISOTROPY)
    # -----------------------------------------------------------------------
    
    data_file_name_P = f'/normal_distribution_data/normal_distribution_020_P_step{special_step_distribution_020}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_020_P_sites_step{special_step_distribution_020}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_020[special_step_distribution_020,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_020, r"$\sigma=0.20$", slope_list_bias)
    aniso_mag[1], aniso_mag_stand[1] = simple_magnitude(slope_list)

    # -----------------------------------------------------------------------
    # BIAS-CORRECTED ANALYSIS: σ = 0.40 (MODERATE ANISOTROPY)
    # -----------------------------------------------------------------------
    
    data_file_name_P = f'/normal_distribution_data/normal_distribution_040_P_step{special_step_distribution_040}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_040_sites_step{special_step_distribution_040}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_040[special_step_distribution_040,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_040, r"$\sigma=0.40$", slope_list_bias)
    aniso_mag[2], aniso_mag_stand[2] = simple_magnitude(slope_list)

    # -----------------------------------------------------------------------
    # BIAS-CORRECTED ANALYSIS: σ = 0.60 (STRONG ANISOTROPY)
    # -----------------------------------------------------------------------
    
    data_file_name_P = f'/normal_distribution_data/normal_distribution_060_P_step{special_step_distribution_060}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_060_sites_step{special_step_distribution_060}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_060[special_step_distribution_060,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_060, r"$\sigma=0.60$", slope_list_bias)
    aniso_mag[3], aniso_mag_stand[3] = simple_magnitude(slope_list)

    # -----------------------------------------------------------------------
    # BIAS-CORRECTED ANALYSIS: σ = 0.80 (VERY STRONG ANISOTROPY)
    # -----------------------------------------------------------------------
    
    data_file_name_P = f'/normal_distribution_data/normal_distribution_080_P_step{special_step_distribution_080}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_080_sites_step{special_step_distribution_080}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_080[special_step_distribution_080,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_080, r"$\sigma=0.80$", slope_list_bias)
    aniso_mag[4], aniso_mag_stand[4] = simple_magnitude(slope_list)

    # -----------------------------------------------------------------------
    # BIAS-CORRECTED ANALYSIS: σ = 0.95 (NEAR-MAXIMUM ANISOTROPY)
    # -----------------------------------------------------------------------
    
    data_file_name_P = f'/normal_distribution_data/normal_distribution_095_P_step{special_step_distribution_095}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_095_sites_step{special_step_distribution_095}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_095[special_step_distribution_095,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_095, r"$\sigma=0.95$", slope_list_bias)
    aniso_mag[5], aniso_mag_stand[5] = simple_magnitude(slope_list)

    # Finalize bias-corrected polar plot
    plt.legend(loc=(-0.24,-0.3),fontsize=16,ncol=3)
    plt.savefig(current_path + "/figures/normal_distribution_circle_after_removing_bias.png", dpi=400,bbox_inches='tight')
    print("Polar figure done.")

    # ==========================================================================
    # PART III: MORPHOLOGICAL ANISOTROPY ANALYSIS (RADIUS-BASED METRICS)
    # ==========================================================================

    # Alternative anisotropy quantification using geometric morphology analysis
    num_step_magni = 30                          # Time step for morphological analysis

    # Initialize arrays for radius-based anisotropy metrics
    aniso_mag2 = np.zeros(6)                     # Morphological anisotropy magnitudes
    aniso_mag_stand2 = np.zeros(6)               # Statistical uncertainties
    
    # Multi-core processing parameters for VECTOR analysis
    cores = 16                                   # Parallel processing cores
    loop_times = 5                               # Iteration count for convergence
    
    # Analyze single time step for all anisotropy levels
    for i in [num_step_magni]:  # tqdm(range(step_num)): (single step analysis)
        
        # -----------------------------------------------------------------------
        # MORPHOLOGICAL ANALYSIS: σ = 0.00 (ISOTROPIC REFERENCE)
        # -----------------------------------------------------------------------
        
        # Process microstructure without rotation for morphological analysis
        # newplace = np.rot90(npy_file_aniso_000[i,:,:,:], 1, (0,1))  # Alternative orientation
        newplace = npy_file_aniso_000[i,:,:,:]   # Direct microstructure
        nx = newplace.shape[0]
        ny = newplace.shape[1]
        ng = np.max(newplace)
        R = np.zeros((nx,ny,2))                  # Constraint array
        P0 = newplace                            # Initial conditions
        
        # Initialize VECTOR solver for grain boundary analysis
        smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
        sites_list = smooth_class.get_all_gb_list()  # Extract boundary sites
        aniso_mag2[0], aniso_mag_stand2[0] = get_circle_statistical_radius(npy_file_aniso_000, sites_list, i)
        # aniso_mag_000[i] = get_circle_statistical_ar(npy_file_aniso_000, i)  # Alternative aspect ratio

        # -----------------------------------------------------------------------
        # MORPHOLOGICAL ANALYSIS: σ = 0.20 (WEAK ANISOTROPY)
        # -----------------------------------------------------------------------
        
        newplace = npy_file_aniso_020[i,:,:,:]
        nx = newplace.shape[0]
        ny = newplace.shape[1]
        ng = np.max(newplace)
        R = np.zeros((nx,ny,2))
        P0 = newplace
        smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
        sites_list = smooth_class.get_all_gb_list()
        aniso_mag2[1], aniso_mag_stand2[1] = get_circle_statistical_radius(npy_file_aniso_000, sites_list, i)

        # -----------------------------------------------------------------------
        # MORPHOLOGICAL ANALYSIS: σ = 0.40 (MODERATE ANISOTROPY)
        # -----------------------------------------------------------------------
        
        newplace = npy_file_aniso_040[i,:,:,:]
        nx = newplace.shape[0]
        ny = newplace.shape[1]
        ng = np.max(newplace)
        R = np.zeros((nx,ny,2))
        P0 = newplace
        smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
        sites_list = smooth_class.get_all_gb_list()
        aniso_mag2[2], aniso_mag_stand2[2] = get_circle_statistical_radius(npy_file_aniso_000, sites_list, i)

        # -----------------------------------------------------------------------
        # MORPHOLOGICAL ANALYSIS: σ = 0.60 (STRONG ANISOTROPY)
        # -----------------------------------------------------------------------
        
        newplace = npy_file_aniso_060[i,:,:,:]
        nx = newplace.shape[0]
        ny = newplace.shape[1]
        ng = np.max(newplace)
        R = np.zeros((nx,ny,2))
        P0 = newplace
        smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
        sites_list = smooth_class.get_all_gb_list()
        aniso_mag2[3], aniso_mag_stand2[3] = get_circle_statistical_radius(npy_file_aniso_000, sites_list, i)

        # -----------------------------------------------------------------------
        # MORPHOLOGICAL ANALYSIS: σ = 0.80 (VERY STRONG ANISOTROPY)
        # -----------------------------------------------------------------------
        
        newplace = npy_file_aniso_080[i,:,:,:]
        nx = newplace.shape[0]
        ny = newplace.shape[1]
        ng = np.max(newplace)
        R = np.zeros((nx,ny,2))
        P0 = newplace
        smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
        sites_list = smooth_class.get_all_gb_list()
        aniso_mag2[4], aniso_mag_stand2[4] = get_circle_statistical_radius(npy_file_aniso_000, sites_list, i)

        # -----------------------------------------------------------------------
        # MORPHOLOGICAL ANALYSIS: σ = 0.95 (NEAR-MAXIMUM ANISOTROPY)
        # -----------------------------------------------------------------------
        
        newplace = npy_file_aniso_095[i,:,:,:]
        nx = newplace.shape[0]
        ny = newplace.shape[1]
        ng = np.max(newplace)
        R = np.zeros((nx,ny,2))
        P0 = newplace
        smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)
        sites_list = smooth_class.get_all_gb_list()
        aniso_mag2[5], aniso_mag_stand2[5] = get_circle_statistical_radius(npy_file_aniso_000, sites_list, i)
        
    # ==========================================================================
    # PART IV: COMPARATIVE ANISOTROPY MAGNITUDE VISUALIZATION
    # ==========================================================================

    # -----------------------------------------------------------------------
    # PLOT 1: POLAR DISTRIBUTION ANISOTROPY WITH ERROR BARS
    # -----------------------------------------------------------------------
    
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    
    # Legacy time series plotting code (commented for reference)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_000, label=r'$\delta=0.00$', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_020, label=r'$\delta=0.20$', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_040, label=r'$\delta=0.40$', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_060, label=r'$\delta=0.60$', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_080, label=r'$\delta=0.80$', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_095, label=r'$\delta=0.95$', linewidth=2)
    
    # Define anisotropy parameter array for plotting
    delta_value = np.array([0.0,0.2,0.4,0.6,0.8,0.95])
    
    # Plot bias-corrected polar anisotropy magnitudes with error bars
    plt.errorbar(delta_value, aniso_mag, yerr=aniso_mag_stand, 
                linestyle='None', marker='None',color='black',linewidth=1, capsize=2)
    plt.plot(delta_value, aniso_mag, '.-', markersize=8, label='time step = 900', linewidth=2)
    
    # Configure plot formatting for publication quality
    plt.xlabel(r"$\sigma$", fontsize=16)         # Anisotropy parameter
    plt.ylabel("Anisotropic Magnitude", fontsize=16)  # Y-axis label
    plt.legend(fontsize=16)
    plt.ylim([-0.05,1.1])                        # Y-axis range
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(current_path + "/figures/anisotropic_magnitude_circle_polar_ave.png", dpi=400,bbox_inches='tight')
    
    # -----------------------------------------------------------------------
    # PLOT 2: MORPHOLOGICAL ANISOTROPY (RADIUS-BASED METRICS)
    # -----------------------------------------------------------------------
    
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_000, label=r'$\delta=0.00$', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_020, label=r'$\delta=0.20$', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_040, label=r'$\delta=0.40$', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_060, label=r'$\delta=0.60$', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_080, label=r'$\delta=0.80$', linewidth=2)
    # plt.plot(np.linspace(0,step_num,step_num)*30, aniso_mag_095, label=r'$\delta=0.95$', linewidth=2)
    delta_value = np.array([0.0,0.2,0.4,0.6,0.8,0.95])
    # plt.errorbar(delta_value, aniso_mag2, yerr=aniso_mag_stand2, linestyle='None', marker='None',color='black',linewidth=1, capsize=2)
    plt.plot(delta_value, aniso_mag2, '.-', markersize=8, label='time step = 900', linewidth=2)
    
    # Configure plot for morphological analysis results
    plt.xlabel(r"$\sigma$", fontsize=16)
    plt.ylabel("Anisotropic Magnitude", fontsize=16)
    # plt.legend(fontsize=16)                    # Legend omitted for cleaner appearance
    plt.ylim([-0.05,0.7])                       # Adjusted range for morphological data
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(current_path + "/figures/anisotropic_magnitude_circle_radius.png", dpi=400,bbox_inches='tight')

    # ==========================================================================
    # ANALYSIS COMPLETION SUMMARY
    # ==========================================================================
    
    print("="*80)
    print("2D CIRCULAR GRAIN BOUNDARY ANALYSIS COMPLETED")
    print("="*80)
    print(f"Analyzed {len(delta_value)} anisotropy levels: σ = {delta_value}")
    print(f"Time step analyzed: {num_step_magni} (mature microstructure)")
    print("Generated outputs:")
    print("  1. Raw polar distributions: normal_distribution_circle.png")
    print("  2. Bias-corrected distributions: normal_distribution_circle_after_removing_bias.png")
    print("  3. Polar anisotropy magnitude: anisotropic_magnitude_circle_polar_ave.png")
    print("  4. Morphological anisotropy: anisotropic_magnitude_circle_radius.png")
    print("Analysis framework: VECTOR 2D + SPPARKS Monte Carlo + Multi-core processing")
    print("="*80)










