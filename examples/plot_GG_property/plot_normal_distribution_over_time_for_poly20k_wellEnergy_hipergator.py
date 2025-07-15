#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPREHENSIVE WELL ENERGY GRAIN BOUNDARY ANALYSIS WITH HIPERGATOR INTEGRATION

Created on Mon Jul 31 14:33:57 2023
@author: Lin

======================================================================================
SCIENTIFIC FRAMEWORK: WELL ENERGY FUNCTION GRAIN BOUNDARY NORMAL DISTRIBUTION ANALYSIS
======================================================================================

This script implements comprehensive grain boundary normal vector distribution analysis
for large-scale polycrystalline systems (20,000 initial grains) using well energy functions
with HiPerGator 3.0 supercomputing resources. The analysis focuses on parametric studies
of anisotropy strength (σ) values ranging from 0.7 to 0.9 with advanced bias correction
and publication-quality visualization capabilities.

WELL ENERGY FUNCTION FRAMEWORK:
- Parametric Anisotropy Control: σ values (0.7, 0.8, 0.9) for systematic anisotropy variation
- Energy Function Types: Cos, Well2, CosMin1, CosMax1Mobility for comprehensive comparison
- Advanced Elliptical Fitting: OpenCV-based ellipse fitting for precise aspect ratio calculation
- Enhanced Morphological Analysis: Grain shape characterization through elliptical geometry
- Statistical Robustness: 1000-grain target for consistent comparative analysis

ANISOTROPY STRENGTH PARAMETERIZATION:
- σ = 0.0: Isotropic reference case for baseline comparison
- σ = 0.7: Moderate anisotropy for initial morphological deviation detection
- σ = 0.8: Intermediate anisotropy for balanced shape evolution analysis
- σ = 0.9: Strong anisotropy for maximum morphological response characterization

WELL ENERGY FUNCTION TYPES (COMPREHENSIVE SET):
1. Cos: Cosine-based energy function with average mobility normalization (γ_ave = 1)
2. Well2: Advanced well energy formulation for enhanced anisotropy control
3. CosMin1: Cosine function with minimum mobility normalization (γ_min = 1)
4. CosMax1Mobility: Cosine function with maximum mobility and enhanced transport

ADVANCED MORPHOLOGICAL CHARACTERIZATION:
- OpenCV Ellipse Fitting: Robust elliptical geometry analysis for grain boundaries
- Aspect Ratio Calculation: Precise morphological anisotropy quantification
- Statistical Averaging: Population-level morphological metrics for system characterization
- Comparative Analysis: Multi-energy function morphological response comparison

HIPERGATOR 3.0 SUPERCOMPUTING INTEGRATION:
- Parallel Processing: 64-core optimization for VECTOR framework computations (notebook)
- Blue Storage System: Massive dataset management (/blue/michael.tonks/lin.yang/)
- Well Energy Data Management: Specialized storage for parametric anisotropy studies
- Memory Optimization: Efficient handling of 20K grain well energy systems

VECTOR FRAMEWORK IMPLEMENTATION:
- Linear2D Multi-Physics: Advanced linear algorithms for well energy grain boundary analysis
- Virtual Inclination Energy: Accurate normal vector computation for anisotropic systems
- Comprehensive Site Extraction: Complete grain boundary detection and characterization
- Statistical Distribution: Angular binning (36 bins, 10-degree intervals) for analysis

SYSTEM SPECIFICATIONS:
- Initial Grain Count: 20,000 grains for statistical robustness
- Target Grain Count: 1,000 grains for consistent comparative analysis
- Temperature Parameter: kT=0.66 for enhanced statistical accuracy
- Random Seed: 56689 for reproducible random number generation
- Domain Size: Large-scale simulation domain for realistic well energy microstructure evolution

PUBLICATION-QUALITY OUTPUT:
- High-Resolution Visualization: 400 DPI polar plots with statistical uncertainty
- Comprehensive Legends: Clear identification of anisotropy strength parameters
- Morphological Summary: Aspect ratio evolution with anisotropy strength
- Scientific Documentation: Complete metadata and well energy analysis specifications
"""

# ===============================================================================
# LIBRARY IMPORTS: COMPREHENSIVE SCIENTIFIC COMPUTING AND VISUALIZATION STACK
# ===============================================================================

# Core system and path management for HiPerGator well energy environment
import os                                           # Operating system interface for file operations
current_path = os.getcwd()                         # Working directory for organized data management

# Scientific computing foundation for large-scale well energy analysis
import numpy as np                                 # Numerical arrays and mathematical operations
from numpy import seterr                           # Error handling for numerical computations
seterr(all='raise')                               # Raise exceptions for numerical errors

# Advanced visualization for publication-quality polar plots and aspect ratio analysis
import matplotlib.pyplot as plt                    # Comprehensive plotting and visualization
import math                                        # Mathematical functions for angular calculations

# Progress tracking for long-running HiPerGator well energy computations
from tqdm import tqdm                              # Progress bars for iterative processes

# System path configuration for VECTOR framework integration
import sys                                         # System-specific parameters and functions
sys.path.append(current_path)                     # Current directory for local modules
sys.path.append(current_path+'/../../')           # Parent directories for VECTOR framework
import myInput                                     # Custom input/output utilities for grain analysis
import PACKAGE_MP_Linear as linear2d               # VECTOR Linear2D multi-physics algorithms
sys.path.append(current_path+'/../calculate_tangent/')  # Additional computational modules

# ===============================================================================
# STATISTICAL ANISOTROPY ANALYSIS FOR WELL ENERGY SYSTEMS
# ===============================================================================

def simple_magnitude(freqArray):
    """
    Calculate anisotropic magnitude metrics for large-scale polycrystalline systems with well energy functions.
    
    This function quantifies the deviation of grain boundary normal distributions
    from perfect circular symmetry for 20K grain systems with parametric anisotropy
    control (σ values). Essential for comparing different well energy formulations
    and anisotropy strengths in HiPerGator simulations with bias correction applied.
    
    Mathematical Framework:
    - Creates uniform circular reference distribution for comparison
    - Computes absolute deviations from perfect circular symmetry
    - Calculates statistical moments with enhanced precision for large systems
    - Normalizes by reference distribution average for relative comparison
    
    Parameters:
    -----------
    freqArray : numpy.ndarray
        Frequency array of grain boundary normal orientations (36 bins, 10° each)
        Derived from 20K grain boundary sites with well energy parameterization
        
    Returns:
    --------
    magnitude_ave : float
        Average relative deviation from circular symmetry
    magnitude_stan : float
        Standard deviation of relative deviations (statistical uncertainty)
        
    Well Energy Applications:
    - Large-scale well energy system anisotropy quantification
    - Statistical comparison across parametric anisotropy strengths (σ = 0.7, 0.8, 0.9)
    - Bias-corrected anisotropy measurement for publication-quality results
    """
    # Define angular coordinate system for comprehensive well energy analysis
    xLim = [0, 360]                                  # Full angular range (degrees)
    binValue = 10.01                                 # Bin width for angular discretization
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)  # Number of angular bins
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Bin centers

    # Create ideal uniform circular distribution for comparison baseline
    freqArray_circle = np.ones(binNum)               # Uniform distribution
    freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)  # Normalize

    # Calculate statistical measures of anisotropy for large well energy systems
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
    for large-scale polycrystalline analysis with well energy conditions.
    
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
        
    Status: DEPRECATED - Use find_fittingEllipse3 with OpenCV for 20K well energy systems
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

def find_fittingEllipse3(array): 
    """
    Advanced elliptical fitting function using OpenCV for precise well energy grain morphology analysis.
    
    This function implements robust elliptical fitting using OpenCV's optimized algorithms,
    specifically designed for accurate aspect ratio calculation in large-scale polycrystalline
    systems with well energy functions. Essential for morphological characterization across
    different anisotropy strengths (σ = 0.7, 0.8, 0.9) in HiPerGator simulations.
    
    Mathematical Framework:
    - OpenCV fitEllipse implementation with robust least-squares optimization
    - Direct ellipse parameter extraction for aspect ratio calculation
    - Enhanced numerical stability for grain boundary coordinate sets
    - Comprehensive error handling for degenerate cases
    
    Parameters:
    -----------
    array : numpy.ndarray
        Array of [x, y] coordinates representing grain boundary sites
        Shape: (n_points, 2) where n_points >= 5 for ellipse fitting
        
    Returns:
    --------
    ellipse : tuple
        OpenCV ellipse representation: ((center_x, center_y), (major_axis, minor_axis), angle)
        Used for direct aspect ratio calculation: major_axis / minor_axis
        
    Well Energy Applications:
    - Precise aspect ratio quantification for parametric anisotropy studies
    - Morphological evolution tracking across well energy formulations
    - Statistical shape characterization for large-scale grain populations
    - Comparative analysis of energy function effects on grain morphology
    
    OpenCV Integration:
    - Robust numerical algorithms optimized for computer vision applications
    - Enhanced performance for large grain boundary datasets
    - Automatic handling of edge cases and numerical stability
    - Direct compatibility with aspect ratio calculation workflows
    """
    import cv2                                       # OpenCV for advanced ellipse fitting
    
    # Fit ellipse to the grain boundary points using OpenCV's robust algorithm
    ellipse = cv2.fitEllipse(array)                  # Returns ((cx, cy), (major, minor), angle)
    # print(f"aspect ratio: {ellipse[1][0]/ellipse[1][1]}")  # Debug output (commented)

    return ellipse

def get_poly_center(micro_matrix, step):
    """
    Calculate centroids and average radii for all grains in polycrystalline microstructure analysis.
    
    This function processes the complete microstructural state to extract grain centroids and
    characteristic radii for morphological analysis in well energy function studies. Essential
    for large-scale statistical characterization of grain populations across different anisotropy
    strengths, enabling comparative morphological evolution tracking in HiPerGator simulations.
    
    Mathematical Framework:
    - Centroid calculation: center_i = (1/n_i) * Σ(x_j, y_j) for grain i sites
    - Average radius estimation: R_avg = √(area/π) assuming circular approximation
    - Boundary condition filtering to exclude periodic grains at simulation edges
    - Minimum grain size threshold (500 sites) for statistical significance
    
    Parameters:
    -----------
    micro_matrix : numpy.ndarray
        4D microstructural evolution array [time, x, y, field_component]
        Contains grain ID assignments for complete polycrystalline system
    step : int
        Time step index for temporal morphological analysis
        
    Returns:
    --------
    center_list : numpy.ndarray
        Grain centroids array, shape (num_grains, 2)
        [x_center, y_center] for each grain in lattice coordinates
    ave_radius_list : numpy.ndarray
        Average radius array, shape (num_grains,)
        Characteristic radius R_avg = √(area/π) for each grain
        
    Well Energy Applications:
    - Statistical grain size distribution analysis across energy functions
    - Morphological evolution tracking in parametric anisotropy studies
    - Large-scale grain population characterization for 20K systems
    - Comparative analysis of energy function effects on grain geometry
    
    Boundary Condition Handling:
    - Automatic exclusion of grains touching simulation boundaries
    - Size-based filtering for statistical significance (minimum 500 sites)
    - Periodic boundary condition awareness for accurate centroid calculation
    """
    # Get the center of all non-periodic grains in matrix
    num_grains = int(np.max(micro_matrix[step,:]))                    # Total grain count
    center_list = np.zeros((num_grains,2))                           # Centroid storage array
    sites_num_list = np.zeros(num_grains)                            # Grain size storage
    ave_radius_list = np.zeros(num_grains)                           # Average radius storage
    
    # Create coordinate reference matrices for centroid calculation
    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i,j] = i                                   # i-coordinate reference
            coord_refer_j[i,j] = j                                   # j-coordinate reference

    table = micro_matrix[step,:,:,0]                                 # Extract grain ID field
    
    # Calculate centroid and size for each grain
    for i in range(num_grains):
        sites_num_list[i] = np.sum(table == i+1)                    # Count sites for grain i+1

        # Apply boundary condition and size filters
        if (sites_num_list[i] < 500) or \
           (np.max(coord_refer_i[table == i+1]) - np.min(coord_refer_i[table == i+1]) == micro_matrix.shape[1]) or \
           (np.max(coord_refer_j[table == i+1]) - np.min(coord_refer_j[table == i+1]) == micro_matrix.shape[2]): 
          # Grains on boundary conditions or too small are excluded
          center_list[i, 0] = 0                                      # Zero centroid for excluded grains
          center_list[i, 1] = 0
          sites_num_list[i] == 0                                     # Zero size for excluded grains
        else:
          # Calculate centroid for valid grains
          center_list[i, 0] = np.sum(coord_refer_i[table == i+1]) / sites_num_list[i]  # x-centroid
          center_list[i, 1] = np.sum(coord_refer_j[table == i+1]) / sites_num_list[i]  # y-centroid
    
    # Calculate average radius assuming circular grain approximation
    ave_radius_list = np.sqrt(sites_num_list / np.pi)               # R_avg = √(area/π)

    return center_list, ave_radius_list

def get_poly_statistical_radius(micro_matrix, sites_list, step):
    """
    Calculate statistical radius deviation for morphological shape analysis in well energy systems.
    
    This function quantifies grain shape deviation from perfect circular geometry by computing
    the maximum normalized radius offset for each grain. Essential for morphological characterization
    in well energy function studies, providing quantitative measures of grain elongation and
    shape complexity across different anisotropy strengths (σ = 0.7, 0.8, 0.9).
    
    Mathematical Framework:
    - Radius deviation: δR_max = max|R_actual - R_avg| / R_avg for each grain
    - Statistical shape parameter: averaged across all valid grains
    - Quantifies departure from circular grain geometry
    - Normalized deviation enables comparative analysis across grain sizes
    
    Parameters:
    -----------
    micro_matrix : numpy.ndarray
        4D microstructural evolution array [time, x, y, field_component]
        Contains complete polycrystalline system state
    sites_list : list
        List of coordinate arrays for each grain's boundary sites
        Used for detailed geometric analysis of individual grains
    step : int
        Time step index for temporal morphological evolution analysis
        
    Returns:
    --------
    max_radius_offset : float
        Average normalized maximum radius deviation across all grains
        Quantifies overall grain shape complexity in the microstructure
        
    Well Energy Applications:
    - Quantitative shape characterization across energy function formulations
    - Morphological evolution tracking in parametric anisotropy studies
    - Statistical analysis of grain shape complexity in large populations
    - Comparative assessment of well energy effects on grain geometry
    
    Shape Analysis Framework:
    - Individual grain radius deviation calculation
    - Population-averaged shape complexity metrics
    - Area-weighted statistical characterization
    - Boundary condition filtering for accurate statistics
    """
    # Get the max offset of average radius and real radius
    center_list, ave_radius_list = get_poly_center(micro_matrix, step)    # Extract grain centroids and radii
    num_grains = int(np.max(micro_matrix[step,:]))                        # Total grain count

    max_radius_offset_list = np.zeros(num_grains)                         # Storage for radius deviations
    
    # Calculate maximum radius deviation for each grain
    for n in range(num_grains):
        center = center_list[n]                                           # Grain centroid
        ave_radius = ave_radius_list[n]                                   # Average radius
        sites = sites_list[n]                                             # Grain boundary sites

        if ave_radius != 0:                                               # Process valid grains only
          for sitei in sites:
              [i,j] = sitei                                               # Site coordinates
              # Calculate actual radius from centroid to boundary site
              current_radius = np.sqrt((i - center[0])**2 + (j - center[1])**2)
              # Compute absolute radius deviation
              radius_offset = abs(current_radius - ave_radius)
              # Track maximum deviation for this grain
              if radius_offset > max_radius_offset_list[n]: 
                  max_radius_offset_list[n] = radius_offset

          # Normalize by average radius for dimensionless shape parameter
          max_radius_offset_list[n] = max_radius_offset_list[n] / ave_radius

    # Calculate population-averaged shape complexity metric
    max_radius_offset = np.average(max_radius_offset_list[max_radius_offset_list!=0])
    
    # Handle edge case of no valid grains
    area_list = np.pi*ave_radius_list*ave_radius_list                     # Grain area calculation
    if np.sum(area_list) == 0: 
        max_radius_offset = 0                                             # Default for degenerate case
    else: 
        # Area-weighted average for accurate population statistics
        max_radius_offset = np.sum(max_radius_offset_list * area_list) / np.sum(area_list)

    return max_radius_offset

    return max_radius_offset

def get_poly_statistical_ar(micro_matrix, step):
    """
    Calculate statistical aspect ratio for grain morphological analysis in well energy systems.
    
    This function computes grain aspect ratios using bounding box analysis to characterize
    grain elongation and morphological anisotropy. Essential for quantitative assessment
    of well energy function effects on grain geometry, enabling comparative analysis
    across different anisotropy strengths (σ = 0.7, 0.8, 0.9) in large-scale simulations.
    
    Mathematical Framework:
    - Aspect ratio calculation: AR = width / height (bounding box dimensions)
    - Bounding box analysis using coordinate span measurements
    - Area-weighted population statistics for representative morphological metrics
    - Statistical characterization across complete polycrystalline populations
    
    Parameters:
    -----------
    micro_matrix : numpy.ndarray
        4D microstructural evolution array [time, x, y, field_component]
        Contains grain ID assignments for complete polycrystalline system
    step : int
        Time step index for temporal morphological evolution analysis
        
    Returns:
    --------
    aspect_ratio : float
        Area-weighted average aspect ratio across all grains
        Quantifies overall morphological anisotropy in the microstructure
        
    Well Energy Applications:
    - Quantitative morphological characterization across energy function formulations
    - Aspect ratio evolution tracking in parametric anisotropy studies
    - Statistical shape analysis for large polycrystalline populations (20K systems)
    - Comparative assessment of well energy effects on grain elongation
    
    Computational Implementation:
    - Coordinate span analysis for bounding box determination
    - Area-weighted statistical averaging for population-representative metrics
    - Efficient processing for HiPerGator large-scale simulations
    - Comprehensive grain population characterization
    """
    # Get the average aspect ratio
    num_grains = int(np.max(micro_matrix[step,:]))                        # Total grain count
    sites_num_list = np.zeros(num_grains)                                 # Grain size storage
    
    # Create coordinate reference matrices for aspect ratio calculation
    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i,j] = i                                        # i-coordinate reference
            coord_refer_j[i,j] = j                                        # j-coordinate reference

    # Initialize aspect ratio calculation arrays
    aspect_ratio_i = np.zeros((num_grains,2))                            # i-dimension storage
    aspect_ratio_j = np.zeros((num_grains,2))                            # j-dimension storage
    aspect_ratio = np.zeros(num_grains)                                   # Final aspect ratios
    table = micro_matrix[step,:,:,0]                                      # Extract grain ID field

    # Collect coordinate lists for each grain
    aspect_ratio_i_list = [[] for _ in range(int(num_grains))]           # i-coordinates per grain
    aspect_ratio_j_list = [[] for _ in range(int(num_grains))]           # j-coordinates per grain
    
    # Populate coordinate lists for bounding box analysis
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            grain_id = int(table[i][j]-1)                                 # Extract grain ID (0-indexed)
            sites_num_list[grain_id] +=1                                  # Count grain sites
            aspect_ratio_i_list[grain_id].append(coord_refer_i[i][j])     # Store i-coordinate
            aspect_ratio_j_list[grain_id].append(coord_refer_j[i][j])     # Store j-coordinate

    # Calculate aspect ratio for each grain using bounding box dimensions
    for i in range(num_grains):
        # Bounding box width (unique i-coordinates span)
        aspect_ratio_i[i, 0] = len(list(set(aspect_ratio_i_list[i])))
        # Bounding box height (unique j-coordinates span)
        aspect_ratio_j[i, 1] = len(list(set(aspect_ratio_j_list[i])))
        
        # Calculate aspect ratio with division by zero protection
        if aspect_ratio_j[i, 1] == 0: 
            aspect_ratio[i] = 0                                           # Handle degenerate grains
        else: 
            aspect_ratio[i] = aspect_ratio_i[i, 0] / aspect_ratio_j[i, 1] # Width / Height

    # Calculate area-weighted average aspect ratio for population statistics
    aspect_ratio = np.sum(aspect_ratio * sites_num_list) / np.sum(sites_num_list)

    return aspect_ratio

def get_normal_vector(grain_structure_figure_one, grain_num):
    """
    Calculate normal vectors for grain boundary analysis using VECTOR Linear2D algorithms.
    
    This function implements advanced grain boundary normal vector computation using the
    VECTOR framework's Linear2D multi-physics solver. Essential for well energy function
    analysis, providing precise normal vector fields for energy calculation and morphological
    characterization in large-scale polycrystalline systems with parametric anisotropy control.
    
    Mathematical Framework:
    - Linear2D virtual inclination energy methodology for normal vector computation
    - Multi-physics phase field approach with grain boundary energy minimization
    - 32-core parallel processing optimization for HiPerGator 3.0 systems
    - Iterative smoothing algorithm (5 loops) for numerical stability
    
    Parameters:
    -----------
    grain_structure_figure_one : numpy.ndarray
        2D grain structure array containing grain ID assignments
        Shape: (nx, ny) representing complete polycrystalline microstructure
    grain_num : int
        Total number of grains in the microstructure for processing
        
    Returns:
    --------
    sites_together : list
        Combined list of all grain boundary sites with computed normal vectors
        Each site contains coordinates and associated normal vector information
        
    Well Energy Applications:
    - Normal vector computation for well energy function evaluation
    - Grain boundary energy analysis across parametric anisotropy studies
    - Morphological characterization using directional grain boundary properties
    - Statistical analysis of grain boundary orientation distributions
    
    VECTOR Framework Integration:
    - Linear2D class instantiation with optimized parallel processing
    - Virtual inclination energy methodology for robust normal computation
    - Comprehensive grain boundary site extraction and characterization
    - Multi-physics solver integration for advanced material modeling
    
    Computational Specifications:
    - 32-core parallel processing for HiPerGator optimization
    - 5-iteration smoothing for numerical convergence
    - Memory-efficient processing for 20K polycrystalline systems
    """
    nx = grain_structure_figure_one.shape[0]                              # x-dimension size
    ny = grain_structure_figure_one.shape[1]                              # y-dimension size
    ng = np.max(grain_structure_figure_one)                               # Maximum grain number
    cores = 32                                                            # HiPerGator parallel cores
    loop_times = 5                                                        # Smoothing iterations
    P0 = grain_structure_figure_one                                       # Initial grain structure
    R = np.zeros((nx,ny,2))                                               # Normal vector field initialization

    # Initialize VECTOR Linear2D class for advanced grain boundary analysis
    smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)

    # Execute virtual inclination energy calculation for normal vector computation
    smooth_class.linear_main("inclination")                              # Linear2D solver execution
    P = smooth_class.get_P()                                              # Extract processed grain structure
    
    # Extract comprehensive grain boundary site information
    # Alternative individual grain approach (commented for comprehensive analysis):
    # sites = smooth_class.get_gb_list(1)
    # for id in range(2,grain_num+1): sites += smooth_class.get_gb_list(id)
    
    # Extract all grain boundary sites with normal vector information
    sites = smooth_class.get_all_gb_list()                               # Complete GB site extraction
    sites_together = []                                                   # Combined site storage
    
    # Combine all grain boundary sites for comprehensive analysis
    for id in range(len(sites)): 
        sites_together += sites[id]                                       # Aggregate all GB sites
    print("Total num of GB sites: " + str(len(sites_together)))          # Debug output

    return P, sites_together, sites

def get_normal_vector_slope(P, sites, step, para_name, bias=None):
    """
    Calculate normal vector slope distribution for well energy grain boundary analysis.
    
    This function computes the angular distribution of grain boundary normal vectors
    using advanced gradient analysis. Essential for well energy function evaluation
    and morphological characterization, providing quantitative analysis of grain
    boundary orientation patterns across parametric anisotropy studies.
    
    Mathematical Framework:
    - Normal vector angle calculation: θ = atan2(-dy, dx) + π
    - Angular histogram binning with 10° resolution for statistical analysis
    - Gradient computation using VECTOR framework phase field methods
    - Statistical distribution analysis for grain boundary orientation characterization
    
    Parameters:
    -----------
    P : numpy.ndarray
        Processed grain structure from Linear2D solver with normal vector information
        Contains phase field data for gradient computation
    sites : list
        List of grain boundary site coordinates for normal vector analysis
        Each site represents a grain boundary location with computed normal vectors
    step : int
        Time step index for temporal evolution analysis (maintained for consistency)
    para_name : str
        Parameter name identifier for analysis tracking and data organization
    bias : float, optional
        Optional bias parameter for specialized normal vector calculations
        
    Returns:
    --------
    freqArray : numpy.ndarray
        Normalized frequency distribution of normal vector angles
        Binned histogram representing grain boundary orientation statistics
    xCor : numpy.ndarray
        Angular coordinate array (degrees) corresponding to frequency bins
        Range: [5°, 355°] with 10° bin spacing for comprehensive coverage
        
    Well Energy Applications:
    - Normal vector orientation analysis for energy function evaluation
    - Statistical characterization of grain boundary angular distributions
    - Morphological evolution tracking in parametric anisotropy studies
    - Comparative analysis of well energy effects on grain boundary patterns
    
    Computational Implementation:
    - High-resolution angular binning (10° resolution) for precise statistics
    - Vectorized gradient computation using VECTOR framework methods
    - Efficient processing for large grain boundary datasets from 20K systems
    """
    # Angular analysis parameters for high-resolution orientation statistics
    xLim = [0, 360]                                                       # Full angular range (degrees)
    binValue = 10.01                                                      # Bin size for angular histogram
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)                 # Number of angular bins
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Bin center coordinates

    freqArray = np.zeros(binNum)                                          # Frequency storage array
    degree = []                                                           # Angular data storage
    
    # Calculate normal vector angles for all grain boundary sites
    for sitei in sites:
        [i,j] = sitei                                                     # Extract site coordinates
        dx,dy = myInput.get_grad(P,i,j)                                   # Compute phase field gradient
        
        # Calculate normal vector angle using atan2 for full angular range
        degree.append(math.atan2(-dy, dx) + math.pi)                     # Normal vector angle calculation
        
        # Alternative angle calculation approaches (commented for reference):
        # if dx == 0:
        #     degree.append(math.pi/2)
        # elif dy >= 0:
        #     degree.append(abs(math.atan(-dy/dx)))
        # elif dy < 0:
        #     degree.append(abs(math.atan(dy/dx)))
    
    # Generate angular histogram from calculated normal vector angles
    for i in range(len(degree)):
        # Convert radians to degrees and bin the angular data
        freqArray[int((degree[i]/math.pi*180-xLim[0])/binValue)] += 1    # Histogram binning
    
    # Normalize frequency array for probability distribution
    freqArray = freqArray/sum(freqArray*binValue)                        # Probability normalization

    # Apply optional bias correction for specialized analysis
    if bias is not None:
        freqArray = freqArray + bias                                      # Add bias correction
        freqArray = freqArray/sum(freqArray*binValue)                     # Re-normalize after bias
    
    # Polar plotting code (commented for data processing focus):
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
    
    # Generate polar plot for normal vector distribution visualization
    plt.plot(np.append(xCor,xCor[0])/180*math.pi, np.append(freqArray,freqArray[0]), linewidth=2, label=para_name)

    # Polynomial fitting for trend analysis (computed but not returned)
    fit_coeff = np.polyfit(xCor, freqArray, 1)                           # Linear fit coefficients
    
    return freqArray

# ==================================================================================
# MAIN EXECUTION PIPELINE: Well Energy Function Analysis
# ==================================================================================
if __name__ == '__main__':
    """
    Main execution pipeline for comprehensive well energy function grain boundary analysis.
    
    This script performs statistical morphological characterization of large-scale polycrystalline
    systems with parametric anisotropy control using well energy functions. Implements comparative
    analysis across multiple anisotropy strengths (σ = 0.7, 0.8, 0.9) for quantitative assessment
    of energy function effects on grain morphology and boundary orientation patterns.
    
    Analysis Framework:
    - Well energy function formulations: Cos, Well2, CosMin1, CosMax1Mobility
    - Parametric anisotropy control with systematic σ variation
    - Large-scale system analysis (20K grains) for statistical significance
    - HiPerGator Blue storage integration for high-performance computing
    - Publication-quality polar visualization for scientific presentation
    
    Data Processing Pipeline:
    1. Multi-parameter well energy data loading from HiPerGator storage
    2. Temporal morphological evolution analysis across energy formulations
    3. Statistical grain boundary orientation characterization
    4. Comparative aspect ratio and shape complexity analysis
    5. Normal vector distribution visualization with polar plotting
    
    Output Products:
    - Quantitative morphological metrics across well energy parameters
    - Statistical grain boundary orientation distributions
    - Comparative energy function performance assessment
    - Publication-ready visualizations for scientific communication
    """
    
    # ================================================================================
    # File Configuration: HiPerGator Blue Storage Well Energy Data
    # ================================================================================
    # HiPerGator Blue storage path for well energy simulation results
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_wellEnergy/results/"
    
    # Parametric anisotropy strength values for systematic morphological study
    TJ_energy_type_070 = "0.7"                                           # Moderate anisotropy
    TJ_energy_type_080 = "0.8"                                           # Intermediate anisotropy  
    TJ_energy_type_090 = "0.9"                                           # Strong anisotropy
    # TJ_energy_type_095 = "0.95"                                        # Extended parameter space
    # TJ_energy_type_099 = "0.99"                                        # Near-maximum anisotropy

    # Well energy function selection for comparative morphological analysis
    energy_function = "Cos"                                              # Cosine-based energy function
    # energy_function = "Well"                                           # Alternative: Well energy
    # energy_function = "CosMin1"                                        # Alternative: CosMin1 variant
    
    # Systematic file naming convention for well energy parametric studies
    npy_file_name_iso = "p_aveE_20000_Cos_delta0.0_J1_refer_1_0_0_seed56689_kt0.66.npy"  # Isotropic reference
    npy_file_name_aniso_070 = f"p_aveE_20000_{energy_function}_delta{TJ_energy_type_070}_J1_refer_1_0_0_seed56689_kt0.66.npy"
    npy_file_name_aniso_080 = f"p_aveE_20000_{energy_function}_delta{TJ_energy_type_080}_J1_refer_1_0_0_seed56689_kt0.66.npy"
    npy_file_name_aniso_090 = f"p_aveE_20000_{energy_function}_delta{TJ_energy_type_090}_J1_refer_1_0_0_seed56689_kt0.66.npy"

    # ================================================================================
    # Data Loading: Multi-Parameter Well Energy Microstructural Evolution
    # ================================================================================
    # Load microstructural evolution data for comprehensive well energy analysis
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)          # Isotropic reference case
    npy_file_aniso_070 = np.load(npy_file_folder + npy_file_name_aniso_070)  # σ = 0.7 anisotropy
    npy_file_aniso_080 = np.load(npy_file_folder + npy_file_name_aniso_080)  # σ = 0.8 anisotropy
    npy_file_aniso_090 = np.load(npy_file_folder + npy_file_name_aniso_090)  # σ = 0.9 anisotropy
    
    # Data validation and comprehensive size verification
    print(f"The 0.7 data size is: {npy_file_aniso_070.shape}")           # Shape: [time, x, y, fields]
    print(f"The 0.8 data size is: {npy_file_aniso_080.shape}")
    print(f"The 0.9 data size is: {npy_file_aniso_090.shape}")
    print(f"The iso data size is: {npy_file_iso.shape}")
    print("READING DATA DONE")

    # ================================================================================
    # Temporal Analysis Configuration: Grain Evolution Tracking
    # ================================================================================
    # Analysis parameters for morphological evolution characterization
    initial_grain_num = 20000                                            # Initial grain population
    step_num = npy_file_aniso_070.shape[0]                               # Total evolution steps
    
    # Initialize grain count arrays for temporal evolution analysis
    grain_num_aniso_070 = np.zeros(npy_file_aniso_070.shape[0])         # σ = 0.7 grain evolution
    grain_num_aniso_080 = np.zeros(npy_file_aniso_080.shape[0])         # σ = 0.8 grain evolution
    grain_num_aniso_090 = np.zeros(npy_file_aniso_090.shape[0])         # σ = 0.9 grain evolution
    grain_num_iso = np.zeros(npy_file_iso.shape[0])                     # Isotropic reference evolution

    # ================================================================================
    # Grain Population Analysis: Temporal Evolution Quantification
    # ================================================================================
    # Calculate grain counts across temporal evolution for comparative analysis
    for i in range(npy_file_aniso_070.shape[0]):
        grain_num_aniso_070[i] = len(np.unique(npy_file_aniso_070[i,:].flatten()))  # Count unique grains
        grain_num_aniso_080[i] = len(np.unique(npy_file_aniso_080[i,:].flatten()))
        grain_num_iso[i] = len(np.unique(npy_file_iso[i,:].flatten()))
    
    # Handle different temporal dimensions for σ = 0.9 case
    for i in range(npy_file_aniso_090.shape[0]):
        grain_num_aniso_090[i] = len(np.unique(npy_file_aniso_090[i,:].flatten()))

    # ================================================================================
    # Target Grain Count Selection: 1000-Grain Analysis Point
    # ================================================================================
    # Identify time steps corresponding to target grain count for consistent comparison
    expected_grain_num = 1000                                            # Target grain count for analysis
    
    # Find optimal time steps for morphological analysis at consistent grain populations
    special_step_distribution_070 = int(np.argmin(abs(grain_num_aniso_070 - expected_grain_num)))
    special_step_distribution_080 = int(np.argmin(abs(grain_num_aniso_080 - expected_grain_num)))
    special_step_distribution_090 = int(np.argmin(abs(grain_num_aniso_090 - expected_grain_num)))
    special_step_distribution_iso = int(np.argmin(abs(grain_num_iso - expected_grain_num)))
    print("Found time steps for 1000-grain analysis")

    # ================================================================================
    # Normal Vector Analysis: Isotropic Reference Case
    # ================================================================================
    # Data management for normal vector distribution analysis
    data_file_name_P = f'/well_normal_data/normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/well_normal_data/normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    
    # Load or compute normal vector data for isotropic reference case
    # if os.path.exists(current_path + data_file_name_P):
    #     P = np.load(current_path + data_file_name_P)
    #     sites = np.load(current_path + data_file_name_sites)
    # else:
    
    # Microstructure preprocessing with coordinate system rotation for analysis
    newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
    
    # Execute comprehensive normal vector computation using VECTOR Linear2D algorithms
    P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    #     np.save(current_path + data_file_name_P, P)
    #     np.save(current_path + data_file_name_sites, sites)

    # Generate normal vector slope distribution for isotropic reference
    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_iso, "Iso")
    
    # ================================================================================
    # Bias Correction: Circular Distribution Reference
    # ================================================================================
    # Calculate bias correction for normal vector distribution analysis
    xLim = [0, 360]                                                      # Angular range for bias calculation
    binValue = 10.01                                                     # Bin size consistency
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)                # Number of bins
    freqArray_circle = np.ones(binNum)                                   # Uniform circular distribution
    freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)   # Normalize to probability
    slope_list_bias = freqArray_circle - slope_list                     # Calculate bias correction

    # ================================================================================
    # Publication-Quality Polar Visualization Setup
    # ================================================================================
    # Initialize polar coordinate system for normal vector distribution visualization
    plt.close()
    fig = plt.figure(figsize=(5, 5))                                     # Square figure for polar plot
    ax = plt.gca(projection='polar')                                     # Polar coordinate system

    # Configure polar plot aesthetics for scientific presentation
    ax.set_thetagrids(np.arange(0.0, 360.0, 45.0),fontsize=16)         # Angular grid marks
    ax.set_thetamin(0.0)                                                 # Minimum angle
    ax.set_thetamax(360.0)                                               # Maximum angle

    # Configure radial grid and labels for probability density
    ax.set_rgrids(np.arange(0, 0.015, 0.004))                          # Radial grid marks
    ax.set_rlabel_position(0.0)                                          # Label position at 0°
    ax.set_rlim(0.0, 0.015)                                             # Radial limits for probability
    ax.set_yticklabels(['0', '4e-3', '8e-3', '1.2e-3'],fontsize=16)    # Radial tick labels

    # Apply professional grid formatting
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)   # Grid styling
    ax.set_axisbelow('True')                                             # Grid behind data

    # ================================================================================
    # Parametric Anisotropy Analysis: Multi-Parameter Comparison
    # ================================================================================
    # Initialize analysis containers for parametric anisotropy characterization
    label_list = ["0.0", "0.7", "0.8", "0.9"]                          # Anisotropy parameter labels
    aniso_mag = np.zeros(len(label_list))                               # Anisotropy magnitude storage
    aniso_mag_stand = np.zeros(len(label_list))                         # Standardized magnitude storage
    aniso_rs = np.zeros(len(label_list))                                # Aspect ratio storage

    # ================================================================================
    # Isotropic Reference Analysis with Bias Correction
    # ================================================================================
    # Generate bias-corrected normal vector distribution for isotropic reference
    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_iso, "Iso",slope_list_bias)
    
    # Initialize morphological analysis containers
    as_list = []                                                         # Aspect ratio analysis storage
    
    # Process individual grains for OpenCV elliptical fitting and aspect ratio calculation
    for n in range(len(sites_list)):
        if len(sites_list[n]) < 10: continue                            # Skip grains with insufficient boundary sites
        ellipse = find_fittingEllipse3(np.array(sites_list[n]))         # OpenCV ellipse fitting
        as_list.append(ellipse[1][0]/ellipse[1][1])                     # Calculate aspect ratio
    aniso_rs[0] = np.average(as_list)                                   # Average aspect ratio for isotropic case
    print("iso done")

    # ================================================================================
    # Anisotropy Analysis: σ = 0.7 (Moderate Well Energy Anisotropy)
    # ================================================================================
    # Data management for σ = 0.7 anisotropy analysis
    data_file_name_P = f'/well_normal_data/normal_distribution_070_P_{energy_function}_step{special_step_distribution_070}.npy'
    data_file_name_sites = f'/well_normal_data/normal_distribution_070_sites_{energy_function}_step{special_step_distribution_070}.npy'
    
    # Load or compute normal vector data for σ = 0.7 case
    # if os.path.exists(current_path + data_file_name_P):
    #     P = np.load(current_path + data_file_name_P)
    #     sites = np.load(current_path + data_file_name_sites)
    # else:
    
    # Microstructure preprocessing and normal vector computation for σ = 0.7
    newplace = np.rot90(npy_file_aniso_070[special_step_distribution_070,:,:,:], 1, (0,1))
    P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    #     np.save(current_path + data_file_name_P, P)
    #     np.save(current_path + data_file_name_sites, sites)

    # Generate bias-corrected normal vector distribution for σ = 0.7 anisotropy
    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_070, r"$\sigma$=0.7",slope_list_bias)
    
    # OpenCV elliptical fitting analysis for σ = 0.7 morphological characterization
    as_list = []                                                         # Reset aspect ratio storage
    for n in range(len(sites_list)):
        if len(sites_list[n]) < 10: continue                            # Filter for adequate boundary data
        ellipse = find_fittingEllipse3(np.array(sites_list[n]))         # Advanced ellipse fitting
        as_list.append(ellipse[1][0]/ellipse[1][1])                     # Aspect ratio extraction
    aniso_rs[1] = np.average(as_list)                                   # Average aspect ratio for σ = 0.7
    print("070 done")

    # ================================================================================
    # Anisotropy Analysis: σ = 0.8 (Intermediate Well Energy Anisotropy)
    # ================================================================================
    # Data management for σ = 0.8 anisotropy analysis
    data_file_name_P = f'/well_normal_data/normal_distribution_080_P_{energy_function}_step{special_step_distribution_080}.npy'
    data_file_name_sites = f'/well_normal_data/normal_distribution_080_sites_{energy_function}_step{special_step_distribution_080}.npy'
    
    # Load or compute normal vector data for σ = 0.8 case
    # if os.path.exists(current_path + data_file_name_P):
    #     P = np.load(current_path + data_file_name_P)
    #     sites = np.load(current_path + data_file_name_sites)
    # else:
    
    # Microstructure preprocessing and normal vector computation for σ = 0.8
    newplace = np.rot90(npy_file_aniso_080[special_step_distribution_080,:,:,:], 1, (0,1))
    P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    #     np.save(current_path + data_file_name_P, P)
    #     np.save(current_path + data_file_name_sites, sites)

    # Generate bias-corrected normal vector distribution for σ = 0.8 anisotropy
    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_080, r"$\sigma$=0.8",slope_list_bias)
    
    # OpenCV elliptical fitting analysis for σ = 0.8 morphological characterization
    as_list = []                                                         # Reset aspect ratio storage
    for n in range(len(sites_list)):
        if len(sites_list[n]) < 10: continue                            # Filter for adequate boundary data
        ellipse = find_fittingEllipse3(np.array(sites_list[n]))         # Advanced ellipse fitting
        as_list.append(ellipse[1][0]/ellipse[1][1])                     # Aspect ratio extraction
    aniso_rs[2] = np.average(as_list)                                   # Average aspect ratio for σ = 0.8
    print("080 done")

    # ================================================================================
    # Anisotropy Analysis: σ = 0.9 (Strong Well Energy Anisotropy)
    # ================================================================================
    # Data management for σ = 0.9 anisotropy analysis
    data_file_name_P = f'/well_normal_data/normal_distribution_090_P_{energy_function}_step{special_step_distribution_090}.npy'
    data_file_name_sites = f'/well_normal_data/normal_distribution_090_sites_{energy_function}_step{special_step_distribution_090}.npy'
    
    # Load or compute normal vector data for σ = 0.9 case
    # if os.path.exists(current_path + data_file_name_P):
    #     P = np.load(current_path + data_file_name_P)
    #     sites = np.load(current_path + data_file_name_sites)
    # Microstructure preprocessing and normal vector computation for σ = 0.9
    newplace = np.rot90(npy_file_aniso_090[special_step_distribution_090,:,:,:], 1, (0,1))
    P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    #     np.save(current_path + data_file_name_P, P)
    #     np.save(current_path + data_file_name_sites, sites)

    # Generate bias-corrected normal vector distribution for σ = 0.9 anisotropy
    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_090, r"$\sigma$=0.9",slope_list_bias)
    
    # OpenCV elliptical fitting analysis for σ = 0.9 morphological characterization
    as_list = []                                                         # Reset aspect ratio storage
    for n in range(len(sites_list)):
        if len(sites_list[n]) < 10: continue                            # Filter for adequate boundary data
        ellipse = find_fittingEllipse3(np.array(sites_list[n]))         # Advanced ellipse fitting
        as_list.append(ellipse[1][0]/ellipse[1][1])                     # Aspect ratio extraction
    aniso_rs[3] = np.average(as_list)                                   # Average aspect ratio for σ = 0.9
    print("090 done")

    # ================================================================================
    # Publication-Quality Output Generation
    # ================================================================================
    # Generate professional polar plot with legend for scientific publication
    plt.legend(loc=(-0.12,-0.35),fontsize=16,ncol=3)                   # Legend positioning and formatting
    plt.savefig(current_path + f"/figures/normal_distribution_poly_20k_after_removing_bias_{energy_function}_{expected_grain_num}grains.png", 
                dpi=400,bbox_inches='tight')                             # High-resolution polar plot output

    # ================================================================================
    # Aspect Ratio Evolution Visualization
    # ================================================================================
    # Generate aspect ratio vs. anisotropy parameter plot for morphological analysis
    plt.close()
    fig = plt.figure(figsize=(5, 5))                                     # Square figure for consistency
    plt.plot(np.linspace(0,len(label_list)-1,len(label_list)), aniso_rs, '.-', 
             markersize=8, linewidth=2)                                  # Aspect ratio evolution plot
    
    # Configure plot aesthetics for scientific presentation
    plt.xlabel(r"$\sigma$", fontsize=16)                                # Anisotropy parameter label
    plt.ylabel("Aspect Ratio", fontsize=16)                             # Aspect ratio label
    plt.xticks(np.linspace(0,len(label_list)-1,len(label_list)),label_list)  # Parameter value labels
    plt.ylim([-0.05,1.0])                                               # Aspect ratio range
    plt.xticks(fontsize=16)                                              # Tick label formatting
    plt.yticks(fontsize=16)
    
    # Save high-resolution aspect ratio evolution plot
    plt.savefig(current_path + f"/figures/anisotropic_poly_p_20k_aspect_ratio_{energy_function}_{expected_grain_num}grains.png", 
                dpi=400,bbox_inches='tight')                             # Publication-quality output

    # ================================================================================
    # Analysis Completion: Well Energy Function Morphological Characterization
    # ================================================================================
    # This completes the comprehensive well energy function analysis pipeline,
    # providing quantitative morphological characterization across parametric
    # anisotropy studies with publication-quality visualization output.





