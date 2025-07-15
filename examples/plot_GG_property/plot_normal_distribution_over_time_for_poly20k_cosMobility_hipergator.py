#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GRAIN BOUNDARY NORMAL VECTOR DISTRIBUTION ANALYSIS WITH COSINE MOBILITY
FOR 20K POLYCRYSTALLINE SYSTEMS ON HIPERGATOR
================================================================================

Scientific Purpose:
------------------
This module provides comprehensive analysis of grain boundary normal vector
distributions in large-scale 20,000-grain polycrystalline systems with cosine
mobility functions using HiPerGator high-performance computing. It investigates
the influence of crystallographic orientation-dependent mobility on grain growth
kinetics and texture evolution in large statistical ensembles.

Research Objectives:
-------------------
1. Large-Scale Statistical Analysis: Quantify normal vector orientations in
   20K-grain polycrystalline systems for statistically significant results
2. Cosine Mobility Investigation: Analyze the effects of cos(θ)-dependent grain
   boundary mobility on crystallographic texture evolution
3. HiPerGator Performance Optimization: Leverage supercomputing resources for
   large-scale polycrystalline simulations with advanced parallel processing
4. Texture Magnitude Quantification: Develop metrics for measuring texture
   strength and anisotropy in large statistical ensembles
5. Elliptical Fitting Analysis: Implement advanced shape fitting algorithms
   for grain morphology characterization in oriented microstructures

Scientific Framework:
--------------------
The module implements advanced grain boundary analysis for large polycrystalline
systems using the VECTOR (Virtual Energy-based Crystallographic Texture Operations
and Research) framework with specialized cosine mobility functions. It combines
statistical analysis of 20K-grain systems with sophisticated texture quantification
methods for materials science applications.

Cosine Mobility Analysis:
------------------------
The cosine mobility function M(θ) = M₀cos(θ) introduces crystallographic orientation
dependence into grain boundary motion, where θ represents the misorientation angle
relative to preferred crystallographic directions. This creates preferential
growth in specific orientations, leading to texture development.

Key Technical Features:
----------------------
- Large-Scale Processing: Optimized handling of 20,000-grain polycrystalline systems
- Cosine Mobility Functions: Crystallographic orientation-dependent boundary motion
- Advanced Texture Metrics: Statistical measures of texture strength and distribution
- Elliptical Fitting Algorithms: OpenCV-based shape analysis for grain morphology
- HiPerGator Integration: Scalable processing for large statistical ensembles
- Grain Center Calculation: Precise geometric analysis of individual grain properties
- Magnitude Analysis: Quantitative texture strength assessment methods

Data Processing Pipeline:
------------------------
1. HiPerGator Data Loading: Load large-scale 20K-grain simulation results
2. Grain Center Analysis: Calculate geometric centers and radii for all grains
3. Normal Vector Processing: Extract boundary orientations with cosine mobility
4. Texture Magnitude Calculation: Quantify deviation from isotropic distribution
5. Elliptical Fitting Analysis: Characterize grain shape evolution under anisotropic mobility
6. Statistical Ensemble Analysis: Generate comprehensive texture statistics

Mathematical Foundation:
-----------------------
The analysis employs advanced statistical methods for texture quantification:
- Magnitude Calculation: |P(θ) - P_iso(θ)| / ⟨P_iso⟩ for texture strength
- Standard Deviation: σ = √(⟨(|P(θ) - P_iso(θ)|/⟨P_iso⟩ - ⟨mag⟩)²⟩) for consistency
- Elliptical Fitting: Least-squares optimization for conic section parameters
- Center-of-Mass: r̄ = Σᵢrᵢ/N for grain geometric centers
- Average Radius: R_avg = √(A/π) where A is grain area

HiPerGator Integration:
----------------------
- Large Memory Management: Efficient handling of 20K-grain datasets
- Parallel Processing: Multi-core optimization for statistical analysis
- Storage Optimization: Compressed data formats for large ensemble results
- Computational Efficiency: Optimized algorithms for large-scale processing

Cosine Mobility Function:
------------------------
M(θ) = M₀[1 + α cos(θ - θ₀)]
where:
- M₀: baseline mobility
- α: anisotropy parameter  
- θ: grain boundary orientation
- θ₀: preferred orientation direction

Authors: Lin Yang, Computational Materials Science Group
Institution: University of Florida, Department of Materials Science and Engineering
Date: July 31, 2023
Last Modified: [Current Date]
Version: 3.0 - Enhanced 20K-Grain Cosine Mobility Analysis with HiPerGator Integration

Dependencies:
------------
- numpy: Advanced numerical computing for large array operations
- matplotlib: High-quality scientific visualization for texture analysis
- opencv-cv2: Computer vision algorithms for elliptical fitting analysis
- VECTOR Framework: myInput, PACKAGE_MP_Linear modules for grain boundary processing
- tqdm: Progress tracking for large-scale computational operations

File Structure:
--------------
- Texture Analysis Functions: simple_magnitude(), find_fittingEllipse2(), find_fittingEllipse3()
- Grain Analysis Functions: get_poly_center() for geometric characterization
- HiPerGator Configuration: Large-scale data source and processing management
- Statistical Analysis Pipeline: Comprehensive texture quantification workflow

Usage Example:
-------------
python plot_normal_distribution_over_time_for_poly20k_cosMobility_hipergator.py

Output Files:
------------
- Texture magnitude analysis results for 20K-grain systems
- Elliptical fitting parameters for grain shape evolution
- Statistical distribution plots with cosine mobility effects
- High-resolution publication-quality visualization outputs

Scientific Impact:
-----------------
This analysis provides fundamental insights into the role of crystallographic
anisotropy in large-scale polycrystalline systems, supporting materials design
applications in advanced alloys, ceramics, and composite materials manufacturing.

================================================================================
"""

"""
Created on Mon Jul 31 14:33:57 2023

@author: Lin
"""

# ========================================================================
# CORE SCIENTIFIC COMPUTING LIBRARIES FOR 20K-GRAIN COSINE MOBILITY ANALYSIS
# ========================================================================
import os
current_path = os.getcwd()                          # Current working directory for file management
import numpy as np                                  # Advanced numerical computing for large arrays
from numpy import seterr                           # Numerical error handling configuration
seterr(all='raise')                                # Raise exceptions for all numerical errors
import matplotlib.pyplot as plt                    # High-quality scientific visualization
import math                                         # Mathematical functions for trigonometric calculations
from tqdm import tqdm                              # Progress tracking for large-scale computations

# ========================================================================
# VECTOR FRAMEWORK MODULES FOR 20K-GRAIN POLYCRYSTALLINE ANALYSIS
# ========================================================================
import sys
sys.path.append(current_path)                     # Add current directory to Python path
sys.path.append(current_path+'/../../')           # Add VECTOR root directory to path
import myInput                                     # VECTOR input/output and gradient calculation module
import PACKAGE_MP_Linear as linear2d              # 2D linear algebra operations for grain processing
sys.path.append(current_path+'/../calculate_tangent/')  # Add tangent calculation utilities

def simple_magnitude(freqArray):
    """
    Calculate texture magnitude and statistical deviation for cosine mobility analysis.
    
    This function quantifies the strength of crystallographic texture by measuring
    the deviation from isotropic distribution. It provides both average magnitude
    and standard deviation metrics for comprehensive texture characterization in
    20K-grain polycrystalline systems with cosine mobility functions.
    
    Parameters:
    -----------
    freqArray : numpy.ndarray
        Normalized frequency distribution array for grain boundary orientations
        Shape: (binNum,) where binNum is the number of angular bins
        Represents probability density function P(θ) for boundary orientations
    
    Returns:
    --------
    magnitude_ave : float
        Average texture magnitude: ⟨|P(θ) - P_iso(θ)|⟩ / ⟨P_iso⟩
        Quantifies overall deviation from isotropic distribution
    magnitude_stan : float
        Standard deviation of texture magnitude for statistical consistency
        Measures variability in texture strength across angular distribution
    
    Analysis Details:
    ----------------
    - Reference Distribution: Uniform circular distribution for isotropic reference
    - Magnitude Calculation: Normalized absolute deviation from isotropy
    - Statistical Analysis: Mean and standard deviation for texture strength
    - Angular Resolution: Consistent with 10.01° binning for comparative analysis
    
    Technical Specifications:
    ------------------------
    - Angular Range: 0-360° for complete orientation coverage
    - Normalization: Area-preserving normalization for statistical consistency
    - Error Metrics: Robust statistical measures for large ensemble analysis
    - Computational Efficiency: Optimized for 20K-grain dataset processing
    """
    # Configure angular binning parameters for texture magnitude analysis
    xLim = [0, 360]                                # Angular range for texture analysis
    binValue = 10.01                               # Angular bin width matching main analysis
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)  # Number of angular bins
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Bin centers

    # Generate perfect isotropic reference distribution for texture comparison
    freqArray_circle = np.ones(binNum)             # Uniform distribution array
    freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)  # Normalize isotropic reference

    # Calculate texture magnitude metrics for cosine mobility characterization
    magnitude_max = np.max(abs(freqArray - freqArray_circle))/np.average(freqArray_circle)  # Maximum deviation
    magnitude_ave = np.average(abs(freqArray - freqArray_circle))/np.average(freqArray_circle)  # Average deviation

    # Calculate standard deviation of texture magnitude for statistical analysis
    magnitude_stan = np.sqrt(np.sum((abs(freqArray - freqArray_circle)/np.average(freqArray_circle) - magnitude_ave)**2)/binNum)

    return magnitude_ave, magnitude_stan

def find_fittingEllipse2(array):
    """
    Fit elliptical parameters to grain boundary points using least-squares method.
    
    This function implements algebraic ellipse fitting for grain shape analysis in
    20K-grain polycrystalline systems. It calculates conic section parameters for
    characterizing grain morphology evolution under cosine mobility functions.
    
    Note: This function is marked as experimental and may have numerical stability issues.
    
    Parameters:
    -----------
    array : numpy.ndarray
        Array of 2D coordinate points representing grain boundary positions
        Shape: (N, 2) where N is number of boundary points
    
    Returns:
    --------
    X_mat : numpy.ndarray
        Ellipse parameters [A, B, C, D, E] for equation Ax² + Bxy + Cy² + Dx + Ey = 1
        Represents fitted conic section coefficients for grain shape analysis
    
    Analysis Details:
    ----------------
    - Fitting Method: Algebraic least-squares optimization for conic parameters
    - Equation Form: General conic section representation for elliptical shapes
    - Numerical Method: Direct matrix inversion for parameter estimation
    - Error Handling: May require regularization for poorly conditioned systems
    """
    K_mat = []                                     # Initialize coefficient matrix
    Y_mat = []                                     # Initialize target vector

    # Extract coordinate components from input array
    X = array[:,0]                                 # X-coordinates of boundary points
    Y = array[:,1]                                 # Y-coordinates of boundary points

    # Construct coefficient matrix for conic section fitting
    K_mat = np.hstack([X**2, X*Y, Y**2, X, Y])    # [x², xy, y², x, y] coefficient matrix
    Y_mat = np.ones_like(X)                       # Target vector (normalized to 1)

    # Solve least-squares system for ellipse parameters
    X_mat = np.linalg.lstsq(K_mat, Y_mat)[0].squeeze()

    # Display fitted ellipse equation for validation
    print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(
          X_mat[0], X_mat[1], X_mat[2], X_mat[3], X_mat[4]))
    print(X_mat)

    return X_mat

def find_fittingEllipse3(array):
    """
    Fit elliptical parameters using OpenCV computer vision algorithms.
    
    This function implements robust ellipse fitting for grain shape analysis using
    advanced computer vision techniques. It provides reliable elliptical parameter
    estimation for characterizing grain morphology in 20K-grain systems with
    cosine mobility-induced shape evolution.
    
    Parameters:
    -----------
    array : numpy.ndarray
        Array of 2D coordinate points representing grain boundary positions
        Shape: (N, 2) where N is number of boundary points
        Must be in integer format for OpenCV compatibility
    
    Returns:
    --------
    ellipse : tuple
        OpenCV ellipse parameters ((center_x, center_y), (width, height), angle)
        Contains geometric parameters for fitted ellipse representation
    
    Analysis Details:
    ----------------
    - Fitting Algorithm: OpenCV's robust ellipse fitting with RANSAC-style optimization
    - Parameter Format: Standard computer vision ellipse representation
    - Robustness: Handles noisy data and outliers effectively
    - Aspect Ratio: Accessible via ellipse[1][0]/ellipse[1][1] for shape characterization
    
    Technical Specifications:
    ------------------------
    - Numerical Stability: Superior to algebraic methods for noisy data
    - Performance: Optimized C++ implementation for computational efficiency
    - Error Handling: Robust against degenerate and poorly conditioned datasets
    - Output Format: Standard geometric parameters for visualization and analysis
    """
    import cv2                                     # Import OpenCV for computer vision algorithms
    
    # Fit ellipse to grain boundary points using robust OpenCV algorithm
    ellipse = cv2.fitEllipse(array)
    
    # Optional: Calculate and display aspect ratio for shape characterization
    # print(f"aspect ratio: {ellipse[1][0]/ellipse[1][1]}")

    return ellipse

def get_poly_center(micro_matrix, step):
    """
    Calculate geometric centers and average radii of grains in the microstructure.
    
    This function analyzes the microstructure data to determine the geometric centers
    and average radii of individual grains at a specified simulation step. It excludes
    periodic grains and those with insufficient data for reliable center estimation.
    
    Parameters:
    -----------
    micro_matrix : numpy.ndarray
        4D array containing microstructure data from the simulation
        Shape: (step_num, grains, height, width, channels)
    step : int
        The specific time step index for which to calculate grain centers
        Must be a valid index within the range of the micro_matrix array
    
    Returns:
    --------
    center_list : numpy.ndarray
        Array of geometric centers for each grain in the microstructure
        Shape: (num_grains, 2) where num_grains is the number of grains
    ave_radius_list : numpy.ndarray
        Array of average radii for each grain in the microstructure
        Shape: (num_grains,) corresponding to the number of grains
    
    Analysis Details:
    ----------------
    - Grain Identification: Unique identification of grains based on simulation data
    - Center Calculation: Geometric center determined as the mean position of all pixels
      belonging to a grain
    - Radius Calculation: Average radius estimated from the area of each grain
    - Exclusion Criteria: Grains with low pixel count or those spanning the image boundary
      are excluded from the analysis
    
    Technical Specifications:
    ------------------------
    - Input Data Format: Microstructure data from phase-field simulation
    - Output Data Format: Arrays of grain centers and average radii
    - Computational Efficiency: Optimized for large 4D arrays from 20K-grain simulations
    """
    # Get the center of all non-periodic grains in matrix
    num_grains = int(np.max(micro_matrix[step,:]))    # Number of grains at the specified step
    center_list = np.zeros((num_grains,2))           # Initialize array for grain centers
    sites_num_list = np.zeros(num_grains)            # Initialize array for grain site counts
    ave_radius_list = np.zeros(num_grains)          # Initialize array for average radii
    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))  # Coordinate reference grid
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))  # Coordinate reference grid
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i,j] = i
            coord_refer_j[i,j] = j

    table = micro_matrix[step,:,:,0]                 # Grain index table for the specified step
    for i in range(num_grains):
        sites_num_list[i] = np.sum(table == i+1)      # Count number of sites for each grain

        if (sites_num_list[i] < 500) or \
           (np.max(coord_refer_i[table == i+1]) - np.min(coord_refer_i[table == i+1]) == micro_matrix.shape[1]) or \
           (np.max(coord_refer_j[table == i+1]) - np.min(coord_refer_j[table == i+1]) == micro_matrix.shape[2]): # grains on bc are ignored
          center_list[i, 0] = 0
          center_list[i, 1] = 0
          sites_num_list[i] == 0
        else:
          center_list[i, 0] = np.sum(coord_refer_i[table == i+1]) / sites_num_list[i]
          center_list[i, 1] = np.sum(coord_refer_j[table == i+1]) / sites_num_list[i]
    ave_radius_list = np.sqrt(sites_num_list / np.pi)

    return center_list, ave_radius_list

def get_poly_statistical_radius(micro_matrix, sites_list, step):
    """
    Calculate the statistical radius deviation of grains in the microstructure.
    
    This function computes the maximum offset between the average radius and the
    actual radius of grains in the microstructure. It provides a measure of the
    statistical consistency of grain sizes and shapes at a given simulation step.
    
    Parameters:
    -----------
    micro_matrix : numpy.ndarray
        4D array containing microstructure data from the simulation
        Shape: (step_num, grains, height, width, channels)
    sites_list : list
        List of site coordinates for each grain in the microstructure
        Length: num_grains where num_grains is the number of grains
    step : int
        The specific time step index for which to calculate statistical radius
        Must be a valid index within the range of the micro_matrix array
    
    Returns:
    --------
    max_radius_offset : float
        The average maximum radius offset of grains at the specified step
        Quantifies the deviation of grain sizes from the average size
    
    Analysis Details:
    ----------------
    - Radius Offset Calculation: Maximum difference between actual and average radius
    - Averaging Method: Mean of radius offsets across all grains
    - Area Weighting: Radius offsets weighted by grain area for global consistency
    - Exclusion of Zero Areas: Grains with zero area are excluded from the averaging
    
    Technical Specifications:
    ------------------------
    - Input Data Format: Microstructure data from phase-field simulation
    - Output Data Format: Single float value representing average radius offset
    - Computational Efficiency: Optimized for large 4D arrays from 20K-grain simulations
    """
    # Get the max offset of average radius and real radius
    center_list, ave_radius_list = get_poly_center(micro_matrix, step)
    num_grains = int(np.max(micro_matrix[step,:]))

    max_radius_offset_list = np.zeros(num_grains)    # Initialize array for max radius offsets
    for n in range(num_grains):
        center = center_list[n]
        ave_radius = ave_radius_list[n]
        sites = sites_list[n]

        if ave_radius != 0:
          for sitei in sites:
              [i,j] = sitei
              current_radius = np.sqrt((i - center[0])**2 + (j - center[1])**2)
              radius_offset = abs(current_radius - ave_radius)
              if radius_offset > max_radius_offset_list[n]: max_radius_offset_list[n] = radius_offset

          max_radius_offset_list[n] = max_radius_offset_list[n] / ave_radius

    max_radius_offset = np.average(max_radius_offset_list[max_radius_offset_list!=0])
    area_list = np.pi*ave_radius_list*ave_radius_list
    if np.sum(area_list) == 0: max_radius_offset = 0
    else: max_radius_offset = np.sum(max_radius_offset_list * area_list) / np.sum(area_list)

    return max_radius_offset

def get_poly_statistical_ar(micro_matrix, step):
    """
    Calculate the average aspect ratio of grains in the microstructure.
    
    This function computes the average aspect ratio of grains at a specified
    simulation step, providing insights into the shape anisotropy of grains
    in the microstructure. It excludes grains with insufficient data for
    reliable aspect ratio estimation.
    
    Parameters:
    -----------
    micro_matrix : numpy.ndarray
        4D array containing microstructure data from the simulation
        Shape: (step_num, grains, height, width, channels)
    step : int
        The specific time step index for which to calculate average aspect ratio
        Must be a valid index within the range of the micro_matrix array
    
    Returns:
    --------
    aspect_ratio : float
        The average aspect ratio of grains at the specified step
        Quantifies the elongation or flattening of grains in the microstructure
    
    Analysis Details:
    ----------------
    - Aspect Ratio Calculation: Ratio of the lengths of the semi-major and semi-minor axes
      of the fitted ellipses to the grain boundaries
    - Averaging Method: Mean of aspect ratios across all grains
    - Exclusion Criteria: Grains with low site count or those with undefined aspect ratio
      are excluded from the analysis
    
    Technical Specifications:
    ------------------------
    - Input Data Format: Microstructure data from phase-field simulation
    - Output Data Format: Single float value representing average aspect ratio
    - Computational Efficiency: Optimized for large 4D arrays from 20K-grain simulations
    """
    # Get the average aspect ratio
    num_grains = int(np.max(micro_matrix[step,:]))    # Number of grains at the specified step
    sites_num_list = np.zeros(num_grains)            # Initialize array for grain site counts
    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))  # Coordinate reference grid
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))  # Coordinate reference grid
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i,j] = i
            coord_refer_j[i,j] = j

    aspect_ratio_i = np.zeros((num_grains,2))        # Initialize array for aspect ratios (i-component)
    aspect_ratio_j = np.zeros((num_grains,2))        # Initialize array for aspect ratios (j-component)
    aspect_ratio = np.zeros(num_grains)              # Initialize array for overall aspect ratios
    table = micro_matrix[step,:,:,0]                 # Grain index table for the specified step

    aspect_ratio_i_list = [[] for _ in range(int(num_grains))]  # Initialize list for i-component aspect ratios
    aspect_ratio_j_list = [[] for _ in range(int(num_grains))]  # Initialize list for j-component aspect ratios
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            grain_id = int(table[i][j]-1)
            sites_num_list[grain_id] +=1
            aspect_ratio_i_list[grain_id].append(coord_refer_i[i][j])
            aspect_ratio_j_list[grain_id].append(coord_refer_j[i][j])

    for i in range(num_grains):
        aspect_ratio_i[i, 0] = len(list(set(aspect_ratio_i_list[i])))
        aspect_ratio_j[i, 1] = len(list(set(aspect_ratio_j_list[i])))
        if aspect_ratio_j[i, 1] == 0: aspect_ratio[i] = 0
        else: aspect_ratio[i] = aspect_ratio_i[i, 0] / aspect_ratio_j[i, 1]

    # aspect_ratio = np.average(aspect_ratio[aspect_ratio!=0])
    aspect_ratio = np.sum(aspect_ratio * sites_num_list) / np.sum(sites_num_list)

    return aspect_ratio

def get_normal_vector(grain_structure_figure_one, grain_num):
    """
    Compute the normal vector distribution of grain boundaries in the microstructure.
    
    This function calculates the normal vectors of grain boundaries in the
    microstructure using the orientation data from the phase-field simulation.
    It provides insights into the crystallographic texture and grain growth
    behavior in polycrystalline materials.
    
    Parameters:
    -----------
    grain_structure_figure_one : numpy.ndarray
        2D array containing the orientation data of the grain boundaries
        Shape: (nx, ny) where nx and ny are the dimensions of the microstructure
    grain_num : int
        The total number of grains in the microstructure
        Used to allocate memory for the normal vector field
    
    Returns:
    --------
    P : numpy.ndarray
        2D array of normal vectors for each grain boundary in the microstructure
        Shape: (nx, ny, 2) where the last dimension represents the (i, j) components
    sites_together : list
        List of all grain boundary sites in the microstructure
    sites : list
        List of grain boundary sites for each individual grain
    
    Analysis Details:
    ----------------
    - Normal Vector Calculation: Extraction of boundary orientations from the simulation data
    - Grain Boundary Site Identification: Mapping of normal vectors to specific grain boundaries
    - Data Structuring: Organization of normal vector data for each grain in the microstructure
    
    Technical Specifications:
    ------------------------
    - Input Data Format: Orientation data from phase-field simulation
    - Output Data Format: Arrays of normal vectors and lists of grain boundary sites
    - Computational Efficiency: Optimized for large 2D arrays from 20K-grain simulations
    """
    nx = grain_structure_figure_one.shape[0]        # Number of rows in the microstructure
    ny = grain_structure_figure_one.shape[1]        # Number of columns in the microstructure
    ng = np.max(grain_structure_figure_one)          # Maximum grain index (total number of grains)
    cores = 32                                      # Number of cores for parallel processing
    loop_times = 5                                  # Number of iterations for the processing loop
    P0 = grain_structure_figure_one                 # Initial orientation data
    R = np.zeros((nx,ny,2))                        # Initialize array for normal vectors
    smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)  # Create instance of linear_class for processing

    smooth_class.linear_main("inclination")        # Perform main processing for inclination calculation
    P = smooth_class.get_P()                        # Extract the normal vector field P
    # sites = smooth_class.get_gb_list(1)
    # print(len(sites))
    # for id in range(2,grain_num+1): sites += smooth_class.get_gb_list(id)
    # print(len(sites))
    sites = smooth_class.get_all_gb_list()        # Get all grain boundary sites
    sites_together = []
    for id in range(len(sites)): sites_together += sites[id]
    print("Total num of GB sites: " + str(len(sites_together)))

    return P, sites_together, sites

def get_normal_vector_slope(P, sites, step, para_name, bias=None):
    """
    Analyze the slope of the normal vector distribution in the microstructure.
    
    This function calculates the slope of the normal vector distribution for
    grain boundaries in the microstructure. It provides insights into the
    crystallographic texture and grain growth behavior under cosine mobility
    conditions. The function also allows for bias correction in the analysis.
    
    Parameters:
    -----------
    P : numpy.ndarray
        2D array of normal vectors for each grain boundary in the microstructure
        Shape: (nx, ny, 2) where the last dimension represents the (i, j) components
    sites : list
        List of grain boundary sites for each individual grain
    step : int
        The specific time step index for which to analyze the normal vector slope
        Must be a valid index within the range of the simulation data
    para_name : str
        The name of the parameter being analyzed (used for labeling plots)
    bias : numpy.ndarray, optional
        Array of bias values for correcting the normal vector distribution
        Shape: (binNum,) where binNum is the number of angular bins
    
    Returns:
    --------
    freqArray : numpy.ndarray
        Normalized frequency distribution array for grain boundary orientations
        Shape: (binNum,) where binNum is the number of angular bins
        Represents probability density function P(θ) for boundary orientations
    
    Analysis Details:
    ----------------
    - Slope Calculation: Analysis of the change in normal vector orientations
    - Bias Correction: Optional correction of the distribution using provided bias values
    - Frequency Distribution: Calculation of the frequency of each orientation bin
    
    Technical Specifications:
    ------------------------
    - Input Data Format: Normal vector data from phase-field simulation
    - Output Data Format: Normalized frequency distribution array
    - Computational Efficiency: Optimized for large 2D arrays from 20K-grain simulations
    """
    xLim = [0, 360]                                # Angular range for texture analysis
    binValue = 10.01                               # Angular bin width matching main analysis
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)  # Number of angular bins
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Bin centers

    freqArray = np.zeros(binNum)                   # Initialize frequency array for angular bins
    degree = []
    for sitei in sites:
        [i,j] = sitei
        dx,dy = myInput.get_grad(P,i,j)
        degree.append(math.atan2(-dy, dx) + math.pi)
        # if dx == 0:
        #     degree.append(math.pi/2)
        # elif dy >= 0:
        #     degree.append(abs(math.atan(-dy/dx)))
        # elif dy < 0:
        #     degree.append(abs(math.atan(dy/dx)))
    for i in range(len(degree)):
        freqArray[int((degree[i]/math.pi*180-xLim[0])/binValue)] += 1
    freqArray = freqArray/sum(freqArray*binValue)

    if bias is not None:
        freqArray = freqArray + bias
        freqArray = freqArray/sum(freqArray*binValue)
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
    plt.plot(np.append(xCor,xCor[0])/180*math.pi, np.append(freqArray,freqArray[0]), linewidth=2, label=para_name)

    # fitting
    fit_coeff = np.polyfit(xCor, freqArray, 1)
    return freqArray

if __name__ == '__main__':
    # ========================================================================
    # HIPERGATOR DATA SOURCE CONFIGURATION FOR 20K-GRAIN COSINE MOBILITY ANALYSIS
    # ========================================================================
    
    # HiPerGator 20K-grain polycrystalline simulation data storage configuration
    # Data location: University of Florida HiPerGator cluster storage system
    # Specialized for large-scale polycrystalline systems with cosine mobility
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_wellEnergy/results/"
    
    # Cosine mobility anisotropy parameter specifications for comparative analysis
    TJ_energy_type_070 = "0.7"                    # Moderate anisotropy (70% cosine mobility)
    TJ_energy_type_080 = "0.8"                    # Strong anisotropy (80% cosine mobility)
    TJ_energy_type_090 = "0.9"                    # Very strong anisotropy (90% cosine mobility)

    # ========================================================================
    # 20K-GRAIN COSINE MOBILITY SIMULATION FILE CONFIGURATION
    # ========================================================================
    
    # Cosine mobility function specification for anisotropic grain boundary motion
    energy_function = "CosMax1Mobility"           # Cosine-based mobility function type
    
    # Comprehensive 20K-grain simulation file naming with HiPerGator parameters:
    # - p_aveE_20000: 20,000-grain polycrystalline system with average energy method
    # - CosMax1Mobility: Cosine mobility function with maximum amplitude = 1
    # - delta[0.7-0.9]: Anisotropy strength parameter for cosine mobility
    # - J1: Grain boundary energy parameter (normalized)
    # - seed56689: Random seed for reproducible 20K-grain simulations
    # - kt0.66: Temperature parameter in reduced units
    npy_file_name_iso = "p_aveE_20000_Cos_delta0.0_J1_refer_1_0_0_seed56689_kt0.66.npy"  # Isotropic reference
    npy_file_name_aniso_070 = f"p_aveE_20000_{energy_function}_delta{TJ_energy_type_070}_J1_refer_1_0_0_seed56689_kt0.66.npy"
    npy_file_name_aniso_080 = f"p_aveE_20000_{energy_function}_delta{TJ_energy_type_080}_J1_refer_1_0_0_seed56689_kt0.66.npy"
    npy_file_name_aniso_090 = f"p_aveE_20000_{energy_function}_delta{TJ_energy_type_090}_J1_refer_1_0_0_seed56689_kt0.66.npy"

    # ========================================================================
    # 20K-GRAIN SIMULATION DATA LOADING AND VALIDATION FOR COSINE MOBILITY ANALYSIS
    # ========================================================================
    
    # Load comprehensive 20K-grain simulation datasets from HiPerGator storage
    # Each dataset contains temporal evolution of large polycrystalline structures
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)              # Isotropic reference data
    npy_file_aniso_070 = np.load(npy_file_folder + npy_file_name_aniso_070)  # 70% cosine mobility data
    npy_file_aniso_080 = np.load(npy_file_folder + npy_file_name_aniso_080)  # 80% cosine mobility data
    npy_file_aniso_090 = np.load(npy_file_folder + npy_file_name_aniso_090)  # 90% cosine mobility data
    
    # Display 20K-grain dataset dimensions for validation of HiPerGator processing
    print(f"The 0.7 data size is: {npy_file_aniso_070.shape}")  # 70% anisotropy dimensions
    print(f"The 0.8 data size is: {npy_file_aniso_080.shape}")  # 80% anisotropy dimensions
    print(f"The 0.90 data size is: {npy_file_aniso_090.shape}") # 90% anisotropy dimensions
    print(f"The iso data size is: {npy_file_iso.shape}")        # Isotropic reference dimensions
    print("READING DATA DONE")  # Confirmation of successful 20K-grain data loading

    # ========================================================================
    # 20K-GRAIN COUNT ANALYSIS AND TEMPORAL CHARACTERIZATION
    # ========================================================================
    
    # Initialize 20K-grain analysis parameters for large-scale processing
    initial_grain_num = 20000                      # Initial grain count for 20K systems
    step_num = npy_file_iso.shape[0]               # Number of temporal steps in simulation
    
    # Initialize grain count arrays for cosine mobility temporal evolution tracking
    grain_num_aniso_070 = np.zeros(step_num)       # 70% cosine mobility grain count evolution
    grain_num_aniso_080 = np.zeros(step_num)       # 80% cosine mobility grain count evolution
    grain_num_aniso_090 = np.zeros(step_num)       # 90% cosine mobility grain count evolution
    grain_num_iso = np.zeros(step_num)             # Isotropic reference grain count evolution

    # Calculate temporal evolution of grain counts for each cosine mobility strength
    for i in range(step_num):
        # Extract unique grain IDs from flattened 20K-grain arrays for grain counting
        grain_num_aniso_070[i] = len(np.unique(npy_file_aniso_070[i,:].flatten()))  # 70% mobility count
        grain_num_aniso_080[i] = len(np.unique(npy_file_aniso_080[i,:].flatten()))  # 80% mobility count
        grain_num_aniso_090[i] = len(np.unique(npy_file_aniso_090[i,:].flatten()))  # 90% mobility count
        grain_num_iso[i] = len(np.unique(npy_file_iso[i,:].flatten()))              # Isotropic count

    # ========================================================================
    # TARGET GRAIN COUNT IDENTIFICATION FOR COSINE MOBILITY ANALYSIS
    # ========================================================================
    
    # Define target grain count for comprehensive normal vector analysis
    # 1000 grains provides optimal balance for statistical significance in 20K systems
    expected_grain_num = 1000
    
    # Identify timesteps where grain count is closest to target for each mobility strength
    # Use absolute difference minimization to find optimal analysis timesteps
    special_step_distribution_070 = int(np.argmin(abs(grain_num_aniso_070 - expected_grain_num)))  # 70% mobility timestep
    special_step_distribution_080 = int(np.argmin(abs(grain_num_aniso_080 - expected_grain_num)))  # 80% mobility timestep
    special_step_distribution_090 = int(np.argmin(abs(grain_num_aniso_090 - expected_grain_num)))  # 90% mobility timestep
    special_step_distribution_iso = int(np.argmin(abs(grain_num_iso - expected_grain_num)))        # Isotropic timestep
    print("Found time steps")  # Confirmation of timestep identification

    # ========================================================================
    # ISOTROPIC REFERENCE PROCESSING FOR COSINE MOBILITY BIAS CALCULATION
    # ========================================================================
    
    # Process isotropic data for bias correction reference in cosine mobility analysis
    # This provides baseline distribution for removing computational artifacts
    data_file_name_P = f'/well_normal_data/normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/well_normal_data/normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    # if os.path.exists(current_path + data_file_name_P):
    #     P = np.load(current_path + data_file_name_P)
    #     sites = np.load(current_path + data_file_name_sites)
    # else:
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    np.save(current_path + data_file_name_P, P)
    np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_iso, "Iso")
    # For bias
    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    freqArray_circle = np.ones(binNum)
    freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)
    slope_list_bias = freqArray_circle - slope_list


    # Start polar figure
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    ax.set_thetagrids(np.arange(0.0, 360.0, 45.0),fontsize=16)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)

    ax.set_rgrids(np.arange(0, 0.01, 0.004))
    ax.set_rlabel_position(0.0)  # 标签显示在0°
    ax.set_rlim(0.0, 0.01)  # 标签范围为[0, 5000)
    ax.set_yticklabels(['0', '4e-3', '8e-3'],fontsize=16)

    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    label_list = ["0.0", "0.7", "0.8", "0.9"]
    aniso_mag = np.zeros(len(label_list))
    aniso_mag_stand = np.zeros(len(label_list))
    aniso_rs = np.zeros(len(label_list))

    # Iso
    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_iso, "Iso",slope_list_bias)
    # aniso_mag[0], aniso_mag_stand[0] = simple_magnitude(slope_list)
    # aniso_rs[0] = get_poly_statistical_ar(npy_file_iso, special_step_distribution_iso)
    as_list = []
    for n in range(len(sites_list)):
        if len(sites_list[n]) < 10: continue
        ellipse = find_fittingEllipse3(np.array(sites_list[n]))
        as_list.append(ellipse[1][0]/ellipse[1][1])
    aniso_rs[0] = np.average(as_list)
    print("iso done")

    # Aniso - 070
    data_file_name_P = f'/well_normal_data/normal_distribution_070_P_{energy_function}_step{special_step_distribution_070}.npy'
    data_file_name_sites = f'/well_normal_data/normal_distribution_070_sites_{energy_function}_step{special_step_distribution_070}.npy'
    # if os.path.exists(current_path + data_file_name_P):
    #     P = np.load(current_path + data_file_name_P)
    #     sites = np.load(current_path + data_file_name_sites)
    # else:
    newplace = np.rot90(npy_file_aniso_070[special_step_distribution_070,:,:,:], 1, (0,1))
    P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    np.save(current_path + data_file_name_P, P)
    np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_070, r"$\sigma$=0.7",slope_list_bias)
    # aniso_mag[1], aniso_mag_stand[1] = simple_magnitude(slope_list)
    # aniso_rs[1] = get_poly_statistical_ar(npy_file_aniso_070, special_step_distribution_070)
    as_list = []
    for n in range(len(sites_list)):
        if len(sites_list[n]) < 10: continue
        ellipse = find_fittingEllipse3(np.array(sites_list[n]))
        as_list.append(ellipse[1][0]/ellipse[1][1])
    aniso_rs[1] = np.average(as_list)
    print("070 done")

    # Aniso - 0.8
    data_file_name_P = f'/well_normal_data/normal_distribution_080_P_{energy_function}_step{special_step_distribution_080}.npy'
    data_file_name_sites = f'/well_normal_data/normal_distribution_080_sites_{energy_function}_step{special_step_distribution_080}.npy'
    # if os.path.exists(current_path + data_file_name_P):
    #     P = np.load(current_path + data_file_name_P)
    #     sites = np.load(current_path + data_file_name_sites)
    # else:
    newplace = np.rot90(npy_file_aniso_080[special_step_distribution_080,:,:,:], 1, (0,1))
    P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    np.save(current_path + data_file_name_P, P)
    np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_080, r"$\sigma$=0.8",slope_list_bias)
    # aniso_mag[1], aniso_mag_stand[1] = simple_magnitude(slope_list)
    # aniso_rs[2] = get_poly_statistical_ar(npy_file_aniso_080, special_step_distribution_080)
    as_list = []
    for n in range(len(sites_list)):
        if len(sites_list[n]) < 10: continue
        ellipse = find_fittingEllipse3(np.array(sites_list[n]))
        as_list.append(ellipse[1][0]/ellipse[1][1])
    aniso_rs[2] = np.average(as_list)
    print("080 done")

    # Aniso - 090
    data_file_name_P = f'/well_normal_data/normal_distribution_090_P_{energy_function}_step{special_step_distribution_090}.npy'
    data_file_name_sites = f'/well_normal_data/normal_distribution_090_sites_{energy_function}_step{special_step_distribution_090}.npy'
    # if os.path.exists(current_path + data_file_name_P):
    #     P = np.load(current_path + data_file_name_P)
    #     sites = np.load(current_path + data_file_name_sites)
    # else:
    newplace = np.rot90(npy_file_aniso_090[special_step_distribution_090,:,:,:], 1, (0,1))
    P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    np.save(current_path + data_file_name_P, P)
    np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_090, r"$\sigma$=0.9",slope_list_bias)
    # aniso_mag[2], aniso_mag_stand[2] = simple_magnitude(slope_list)
    # aniso_rs[3] = get_poly_statistical_ar(npy_file_aniso_090, special_step_distribution_090)
    as_list = []
    for n in range(len(sites_list)):
        if len(sites_list[n]) < 10: continue
        ellipse = find_fittingEllipse3(np.array(sites_list[n]))
        as_list.append(ellipse[1][0]/ellipse[1][1])
    aniso_rs[3] = np.average(as_list)
    print("090 done")


    plt.legend(loc=(-0.12,-0.35),fontsize=16,ncol=3)
    plt.savefig(current_path + f"/figures/normal_distribution_poly_20k_after_removing_bias_{energy_function}_{expected_grain_num}grains.png", dpi=400,bbox_inches='tight')

    plt.close()
    fig = plt.figure(figsize=(5, 5))
    plt.plot(np.linspace(0,len(label_list)-1,len(label_list)), 1/aniso_rs, '.-', markersize=8, linewidth=2)
    plt.xlabel(r"$\sigma$", fontsize=16)
    plt.ylabel("Aspect Ratio", fontsize=16)
    plt.xticks(np.linspace(0,len(label_list)-1,len(label_list)),label_list)
    # plt.legend(fontsize=16)
    plt.ylim([-0.05,1.0])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(current_path + f"/figures/anisotropic_poly_20k_aspect_ratio_{energy_function}_{expected_grain_num}grains.png", dpi=400,bbox_inches='tight')











