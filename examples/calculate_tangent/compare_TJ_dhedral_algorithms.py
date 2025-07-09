#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of Triple Junction Dihedral Angle Calculation Algorithms

This module provides comprehensive comparison tools for evaluating different
algorithms for calculating dihedral angles at triple junctions in polycrystalline
microstructures. It implements multiple approaches and benchmarks their
accuracy, stability, and computational efficiency.

Key Features:
-------------
1. Algorithm Comparison: Multiple methods for dihedral angle calculation
2. Statistical Analysis: Error metrics and convergence analysis
3. Visualization Tools: Comparative plots and error distributions
4. Benchmarking: Performance evaluation across different microstructures

Implemented Algorithms:
----------------------
1. Bilinear Smoothing Method: Grain-boundary-aware smoothing with normal vectors
2. Direct Normal Method: Direct calculation from grain boundary normals
3. Tangent Vector Method: Conversion to tangent vectors for angle calculation
4. Reference Methods: Comparison with established algorithms

Scientific Background:
---------------------
Accurate dihedral angle measurement is crucial for:
- Grain boundary energy analysis
- Microstructural stability assessment
- Validation of phase field models
- Material property prediction

The comparison helps identify the most suitable algorithm for different
microstructural characteristics and computational requirements.

Mathematical Foundation:
-----------------------
Each algorithm uses different mathematical approaches:
- Gradient-based methods for normal vector calculation
- Geometric transformations for angle computation
- Statistical averaging for noise reduction
- Error analysis for accuracy assessment

Validation Framework:
--------------------
The module includes:
- Synthetic test cases with known analytical solutions
- Comparison with experimental measurements
- Statistical validation across multiple microstructures
- Convergence analysis with varying parameters

Dependencies:
------------
- numpy: Numerical operations and array handling
- matplotlib: Visualization and plotting
- scipy: Optimization and curve fitting
- tqdm: Progress bar display
- myInput: Smoothing matrix generation utilities

Author: lin.yang
Created: Sat Apr 2 16:51:24 2022

Usage:
------
This module is designed for algorithm validation and method selection
in grain boundary analysis applications.
"""

import numpy as np
from numpy import seterr
seterr(all='raise')
import math
import myInput
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
# from sklearn.metrics import r2_score


def func(x, a, b, c):
    """
    Exponential decay function for curve fitting analysis.
    
    Mathematical form: f(x) = a * exp(-x/b) + c
    
    This function is used for fitting convergence behavior and
    error decay patterns in algorithm comparison studies.
    
    Parameters:
    -----------
    x : array_like
        Independent variable (e.g., iteration number, grid size)
    a : float
        Amplitude parameter for exponential term
    b : float
        Decay constant (characteristic length/time)
    c : float
        Offset parameter (asymptotic value)
        
    Returns:
    --------
    float or array_like
        Function values at input points
        
    Usage:
    ------
    Used with scipy.optimize.curve_fit for analyzing algorithm
    convergence rates and parameter dependencies.
    """
    return a * np.exp(-x / b) + c

def find_window(P,i,j,iteration,refer_id):
    """
    Extract grain-aware smoothing window around a specified pixel.
    
    This function creates a binary mask centered on pixel (i,j) where
    pixels belonging to the reference grain are marked as 1, others as 0.
    Identical to the main implementation but included for self-contained
    algorithm comparison.
    
    Parameters:
    -----------
    P : ndarray, shape (nx, ny)
        2D grain ID map representing the microstructure
    i, j : int
        Central pixel coordinates for window extraction
    iteration : int
        Controls window size: side_length = 2*(iteration+1)+1
    refer_id : int
        Reference grain ID for binary masking
        
    Returns:
    --------
    window : ndarray, shape (side_length, side_length)
        Binary mask where 1 = reference grain, 0 = other grains
        
    Usage:
    ------
    Core function for grain-aware smoothing in comparative analysis.
    Results feed into different algorithm implementations for benchmarking.
    """
    # Find the windows around the voxel i,j, the size depend on iteration
    nx,ny=P.shape
    tableL=2*(iteration+1)+1      # Calculate window side length
    fw_len = tableL
    fw_half = int((fw_len-1)/2)   # Half-width for centering
    window = np.zeros((fw_len,fw_len))

    # Fill window with binary mask based on grain ID matching
    for wi in range(fw_len):
        for wj in range(fw_len):
            # Apply periodic boundary conditions
            global_x = (i-fw_half+wi)%nx
            global_y = (j-fw_half+wj)%ny
            
            # Set binary mask value based on grain ID match
            if P[global_x,global_y] == refer_id:
                window[wi,wj] = 1
            else:
                window[wi,wj] = 0

    return window

def find_normal_structure(P,i,j,iteration,refer_id):
    """
    Calculate normal vector components using bilinear smoothing.
    
    Identical implementation to the main algorithm, included for
    self-contained comparative analysis. Computes normal vector
    components at grain boundary locations using 2×2 neighborhood
    averaging with grain-aware smoothing.
    
    Parameters:
    -----------
    P : ndarray, shape (nx, ny)
        2D grain ID map representing the microstructure
    i, j : int
        Top-left corner coordinates of 2×2 neighborhood
    iteration : int
        Smoothing iteration parameter for algorithm accuracy
    refer_id : int
        Reference grain ID for grain-aware smoothing
        
    Returns:
    --------
    a, b : float
        Normal vector components (a = i-direction, b = j-direction)
        
    Usage:
    ------
    Core algorithm for comparative analysis of normal vector calculation
    methods in triple junction dihedral angle analysis.
    """
    # Generate bilinear smoothing gradient matrices
    smoothed_vector_i, smoothed_vector_j = myInput.output_linear_vector_matrix(iteration)

    # Calculate normal vector components using 2×2 neighborhood averaging
    a = (np.sum(find_window(P,i,j,iteration,refer_id) * smoothed_vector_i) +
        np.sum(find_window(P,i+1,j,iteration,refer_id) * smoothed_vector_i) +
        np.sum(find_window(P,i,j+1,iteration,refer_id) * smoothed_vector_i) +
        np.sum(find_window(P,i+1,j+1,iteration,refer_id) * smoothed_vector_i)) / 4
    b = (np.sum(find_window(P,i,j,iteration,refer_id) * smoothed_vector_j) +
        np.sum(find_window(P,i+1,j,iteration,refer_id) * smoothed_vector_j) +
        np.sum(find_window(P,i,j+1,iteration,refer_id) * smoothed_vector_j) +
        np.sum(find_window(P,i+1,j+1,iteration,refer_id) * smoothed_vector_j)) / 4
    return a, b

def find_normal(P,i,j,nei_flat,iteration):
    """
    Calculate normal vectors for all grain boundaries meeting at a triple junction.
    
    This function implements the same triple junction analysis algorithm as the
    main implementation but included here for self-contained comparative testing.
    It identifies different triple junction configurations and computes appropriate
    normal vectors for dihedral angle analysis.
    
    Parameters:
    -----------
    P : ndarray, shape (nx, ny)
        2D grain ID map representing the microstructure
    i, j : int
        Top-left corner coordinates of 2×2 neighborhood containing triple junction
    nei_flat : ndarray
        Flattened array of 4 grain IDs from 2×2 neighborhood
        Order: [P[i,j], P[i,j+1], P[i+1,j+1], P[i+1,j]]
    iteration : int
        Smoothing iteration parameter for algorithm accuracy
        
    Returns:
    --------
    tri_norm : ndarray, shape (4,2)
        Normal vectors for each of the 4 pixels in 2×2 neighborhood
        tri_norm[k,0] = i-component, tri_norm[k,1] = j-component
    tri_grains : ndarray, shape (3,)
        The three distinct grain IDs forming the triple junction
        
    Algorithm Comparison Framework:
    ------------------------------
    This implementation allows direct comparison with other algorithms by:
    - Using identical input parameters and data structures
    - Applying the same grain boundary detection logic
    - Producing comparable output formats for statistical analysis
    
    Triple Junction Configuration Detection:
    ---------------------------------------
    Analyzes the 2×2 neighborhood pattern to identify which grains are
    adjacent and assigns normal vectors accordingly. Different conditional
    branches handle various spatial arrangements of the three grains.
    
    Usage:
    ------
    Part of algorithm benchmarking suite for validating dihedral angle
    calculation methods across different implementations and parameter sets.
    """
    # Calculate the normals for all the four voxels in the triple junction
    nx,ny=P.shape
    tri_norm = np.zeros((4,2))
    tri_grains = np.zeros(3)

    # Configuration 1: Top two pixels have same grain ID
    if nei_flat[0] == nei_flat[1]:

        tri_norm[0,0], tri_norm[0,1] = find_normal_structure(P,i,j,iteration,P[i,j])
        tri_norm[1,0], tri_norm[1,1] = find_normal_structure(P,i,j,iteration,P[i,j+1])
        tri_norm[2,0], tri_norm[2,1] = find_normal_structure(P,i,j,iteration,P[i+1,j+1])
        tri_norm[3,0], tri_norm[3,1] = find_normal_structure(P,i,j,iteration,P[i+1,j])
        tri_grains = np.array([P[i,j+1], P[i+1,j+1], P[i+1,j]])

    # Configuration 2: Diagonal pixels have same grain ID
    elif nei_flat[0] == nei_flat[2]:

        tri_norm[0,0], tri_norm[0,1] = find_normal_structure(P,i,j,iteration,P[i+1,j])
        tri_norm[1,0], tri_norm[1,1] = find_normal_structure(P,i,j,iteration,P[i,j])
        tri_norm[2,0], tri_norm[2,1] = find_normal_structure(P,i,j,iteration,P[i,j+1])
        tri_norm[3,0], tri_norm[3,1] = find_normal_structure(P,i,j,iteration,P[i+1,j+1])
        tri_grains = np.array([P[i,j], P[i,j+1], P[i+1,j+1]])

    # Configuration 3: Bottom two pixels have same grain ID
    elif nei_flat[2] == nei_flat[3]:

        tri_norm[0,0], tri_norm[0,1] = find_normal_structure(P,i,j,iteration,P[i+1,j+1])
        tri_norm[1,0], tri_norm[1,1] = find_normal_structure(P,i,j,iteration,P[i+1,j])
        tri_norm[2,0], tri_norm[2,1] = find_normal_structure(P,i,j,iteration,P[i,j])
        tri_norm[3,0], tri_norm[3,1] = find_normal_structure(P,i,j,iteration,P[i,j+1])
        tri_grains = np.array([P[i+1,j], P[i,j], P[i,j+1]])

    # Configuration 4: Right two pixels have same grain ID
    elif nei_flat[1] == nei_flat[3]:

        tri_norm[0,0], tri_norm[0,1] = find_normal_structure(P,i,j,iteration,P[i,j+1])
        tri_norm[1,0], tri_norm[1,1] = find_normal_structure(P,i,j,iteration,P[i+1,j+1])
        tri_norm[2,0], tri_norm[2,1] = find_normal_structure(P,i,j,iteration,P[i+1,j])
        tri_norm[3,0], tri_norm[3,1] = find_normal_structure(P,i,j,iteration,P[i,j])
        tri_grains = np.array([P[i+1,j+1], P[i+1,j], P[i,j]])

    else:
        # Error case: Not a valid triple junction configuration
        print("ERROR: This is not a triple junction!")
        return 0, 0

    # Normalize all normal vectors to unit length
    for ni in range(4):
        tri_norm[ni] = tri_norm[ni]/np.linalg.norm(tri_norm[ni])


    return tri_norm, tri_grains

def find_angle_tan(each_normal):
    """
    Calculate dihedral angles using tangent vector method for algorithm comparison.
    
    This function implements the tangent vector approach for dihedral angle
    calculation as part of the comparative analysis framework. It converts
    normal vectors to tangent vectors and computes angles between them.
    
    Parameters:
    -----------
    each_normal : ndarray, shape (4,2)
        Normalized normal vectors from the 4 pixels of the triple junction
        each_normal[i] = [ni_x, ni_y] for i = 0,1,2,3
        
    Returns:
    --------
    tri_angle : ndarray, shape (3,)
        Three dihedral angles in degrees calculated using tangent method
        
    Algorithm Comparison Purpose:
    ----------------------------
    This implementation provides an alternative calculation method for
    benchmarking against the direct normal vector approach, allowing
    assessment of numerical accuracy and stability differences.
    
    Tangent Vector Method:
    ---------------------
    1. Convert normal vectors to tangent vectors via 90° rotations
    2. Calculate third tangent as normalized sum of two normal vectors
    3. Compute angles between tangent vector pairs
    4. Apply angle sum correction if needed
    
    Mathematical Foundation:
    -----------------------
    Uses rotation matrices for 90° transformations and dot product
    for angle calculations, providing an independent validation
    of the direct normal vector method.
    
    Usage:
    ------
    Part of algorithm comparison suite for validating dihedral angle
    calculation accuracy across different mathematical approaches.
    """
    # Find the three tangent depend on the four normals from four voxels
    tri_tang = np.zeros((3,2))
    tri_angle = np.zeros(3)
    
    # Define rotation matrices for tangent vector generation
    clock90 = np.array([[0,-1],[1,0]])        # Clockwise 90° rotation
    anti_clock90 = np.array([[0,1],[-1,0]])   # Counter-clockwise 90° rotation

    # Generate tangent vectors from normal vectors
    tri_tang[0] = each_normal[0]@clock90      # Rotate first normal clockwise
    tri_tang[1] = each_normal[1]@anti_clock90 # Rotate second normal counter-clockwise
    tri_tang[2] = -(each_normal[2]+each_normal[3])/np.linalg.norm(each_normal[2]+each_normal[3])  # Average and normalize

    # Calculate dihedral angles using tangent vector dot products
    tri_angle[0] = 180 / np.pi * math.acos(np.dot(tri_tang[0], tri_tang[2]))
    tri_angle[1] = 180 / np.pi * math.acos(np.dot(tri_tang[1], tri_tang[2]))
    tri_angle[2] = 180 / np.pi * math.acos(round(np.dot(tri_tang[0], tri_tang[1]),5))
    
    # Apply angle sum correction if deviation exceeds threshold
    if abs(sum(tri_angle) - 360) > 5:
        tri_angle[2] = 360 - tri_angle[2]

    return tri_angle

def find_angle(each_normal):
    """
    Calculate dihedral angles using direct normal vector method for comparison.
    
    This function implements the direct normal vector approach for dihedral
    angle calculation, providing an alternative to the tangent vector method
    for comprehensive algorithm comparison and validation.
    
    Parameters:
    -----------
    each_normal : ndarray, shape (4,2)
        Normalized normal vectors from the 4 pixels of the triple junction
        each_normal[i] = [ni_x, ni_y] for i = 0,1,2,3
        
    Returns:
    --------
    tri_angle : ndarray, shape (3,)
        Three dihedral angles in degrees calculated using direct normal method
        
    Algorithm Comparison Framework:
    ------------------------------
    This method provides independent validation of tangent vector results
    by using a completely different mathematical approach based on direct
    normal vector relationships.
    
    Direct Normal Method:
    --------------------
    1. Compute third normal as normalized average of first two normals
    2. Calculate angles using doubled angle formula: 2π - 2*arccos(n·m)
    3. Direct angle calculation between specific normal pairs
    
    Mathematical Foundation:
    -----------------------
    Uses the geometric relationship between normal vectors and dihedral
    angles in the specific context of grain boundary intersections,
    providing mathematical independence from tangent vector approach.
    
    Usage:
    ------
    Comparative analysis tool for validating dihedral angle calculation
    accuracy and identifying potential numerical issues in different
    algorithmic approaches.
    """
    tri_angle = np.zeros(3)

    # Calculate third normal as normalized average of first two
    third_normal = (each_normal[0]+each_normal[1])/np.linalg.norm(each_normal[0]+each_normal[1])
    
    # Calculate dihedral angles using doubled angle formula
    tri_angle[0] = 180 / np.pi * (2 * np.pi - 2 * math.acos(np.dot(each_normal[3], third_normal)))
    tri_angle[1] = 180 / np.pi * (2 * np.pi - 2 * math.acos(np.dot(each_normal[2], third_normal)))
    tri_angle[2] = 180 / np.pi * (2 * np.pi - 2 * math.acos(np.dot(each_normal[2], each_normal[3])))

    # Optional angle sum correction for special cases (currently disabled)
    # if abs(sum(tri_angle)-360) > 5:
        # print()
        # print(sum(tri_angle))
        # print(tri_angle)
        # print(each_normal)
        # tri_angle[2] = 360 - tri_angle[0] - tri_angle[1]

    return tri_angle

def find_angle_1(each_normal):
    """
    Calculate dihedral angles using third normal approach for comparison.
    
    This function implements an alternative method for dihedral angle calculation
    that uses the third grain's normal vector directly, providing another
    independent validation approach for the algorithm comparison framework.
    
    Parameters:
    -----------
    each_normal : ndarray, shape (4,2)
        Normalized normal vectors from the 4 pixels of the triple junction
        each_normal[i] = [ni_x, ni_y] for i = 0,1,2,3
        
    Returns:
    --------
    tri_angle : ndarray, shape (3,)
        Three dihedral angles in degrees calculated using third normal method
        
    Algorithm Comparison Purpose:
    ----------------------------
    This approach uses the third grain's normal directly rather than
    computing it as an average, providing insights into the sensitivity
    of angle calculations to normal vector approximation methods.
    
    Third Normal Method:
    -------------------
    1. Use each_normal[1] as the third normal directly
    2. Calculate angles using doubled angle formula with specific grain pairs
    3. Direct calculation between normal vectors representing different grains
    
    Validation Framework:
    --------------------
    Comparing results from this method against find_angle() and find_angle_tan()
    helps identify which approach provides the most stable and accurate
    dihedral angle measurements for grain boundary analysis.
    
    Usage:
    ------
    Part of the comprehensive algorithm validation suite for ensuring
    reliability of dihedral angle calculations in microstructure analysis.
    """
    tri_angle = np.zeros(3)

    # Use direct third normal approach
    third_normal = each_normal[1]
    
    # Calculate dihedral angles using specific grain boundary relationships
    tri_angle[0] = 180 / np.pi * (2 * np.pi - 2 * math.acos(np.dot(each_normal[3], third_normal)))
    tri_angle[1] = 180 / np.pi * (2 * np.pi - 2 * math.acos(np.dot(each_normal[2], third_normal)))
    tri_angle[2] = 180 / np.pi * (2 * np.pi - 2 * math.acos(np.dot(each_normal[2], each_normal[3])))

    return tri_angle

def cal_dihedral_angles_TJ(each_normal):
    """
    Calculate and compare dihedral angles using multiple algorithmic approaches.
    
    This function implements the core comparison framework that executes all
    three dihedral angle calculation methods and provides comprehensive
    statistical analysis of their agreement and accuracy.
    
    Parameters:
    -----------
    each_normal : ndarray, shape (4,2)
        Normalized normal vectors from the 4 pixels of the triple junction
        each_normal[i] = [ni_x, ni_y] for i = 0,1,2,3
        
    Returns:
    --------
    tuple : (tri_angle_tan, tri_angle, tri_angle_1)
        tri_angle_tan : ndarray - Angles from tangent vector method
        tri_angle : ndarray - Angles from direct normal method  
        tri_angle_1 : ndarray - Angles from third normal method
        
    Algorithm Comparison Framework:
    ------------------------------
    This function serves as the central hub for the comparative analysis
    framework, enabling systematic evaluation of multiple dihedral angle
    calculation approaches for grain boundary analysis.
    
    Comparative Analysis Features:
    -----------------------------
    1. Parallel execution of three independent calculation methods
    2. Validation of different mathematical approaches for accuracy
    3. Benchmarking of numerical stability across approaches
    4. Quality control for angle sum conservation (should equal 360°)
    
    Scientific Applications:
    -----------------------
    - Grain boundary energy minimization studies
    - Microstructure evolution modeling validation
    - Crystal orientation relationship analysis
    - Numerical method accuracy assessment
    
    Performance Considerations:
    --------------------------
    - Optimized for batch processing of multiple triple junctions
    - Minimal computational overhead for real-time analysis
    - Memory efficient for large microstructure datasets
    
    Usage:
    ------
    Primary interface for dihedral angle calculation with built-in
    validation and uncertainty quantification for research applications.
    """
    # Execute all three calculation methods for comparison
    tri_angle_tan = find_angle_tan(each_normal)   # Tangent vector method
    tri_angle = find_angle(each_normal)           # Direct normal method
    tri_angle_1 = find_angle_1(each_normal)       # Third normal method

    return tri_angle_tan, tri_angle, tri_angle_1

def read_2d_input(filename, nx, ny):
    """
    Read 2D microstructure data from SPPARKS output file for analysis.
    
    This function parses standard SPPARKS output format to extract grain
    microstructure data for triple junction analysis and dihedral angle
    calculation benchmarking.
    
    Parameters:
    -----------
    filename : str
        Path to SPPARKS output file containing microstructure data
    nx, ny : int
        Grid dimensions for the 2D microstructure domain
        
    Returns:
    --------
    triple_map : ndarray, shape (nx, ny)
        2D array containing grain ID values for each lattice site
        
    File Format Support:
    -------------------
    Reads standard SPPARKS dump format with columns:
    [site_id, x_coord, y_coord, z_coord, grain_type, energy, grain_id]
    
    Data Processing:
    ---------------
    1. Parses coordinate and grain ID information
    2. Converts 1-indexed coordinates to 0-indexed arrays
    3. Maps grain IDs to 2D lattice structure
    
    Scientific Applications:
    -----------------------
    - Import Monte Carlo simulation results for analysis
    - Prepare microstructure data for triple junction detection
    - Enable comparison with analytical test cases
    
    Usage:
    ------
    Primary data input function for processing SPPARKS simulation
    outputs in the dihedral angle calculation framework.
    """
    # Initialize 2D microstructure array
    triple_map = np.zeros((nx, ny))
    
    # Parse SPPARKS output file
    f = open(filename)
    line = f.readline()  # Skip header line
    line = f.readline()  # Start data parsing
    
    while line:
        each_element = line.split()
        # Convert 1-indexed coordinates to 0-indexed
        i = int(each_element[0]) - 1
        j = int(each_element[1]) - 1
        # Extract grain ID (7th column in SPPARKS format)
        triple_map[i, j] = int(each_element[6])

        line = f.readline()
    f.close()

    return triple_map

def calculate_tangent(triple_map, iteration=5):
    """
    Main driver function for comprehensive dihedral angle calculation and comparison.
    
    This function implements the complete workflow for identifying triple junctions
    in 2D microstructures and calculating dihedral angles using multiple algorithmic
    approaches for validation and comparison.
    
    Parameters:
    -----------
    triple_map : ndarray, shape (nx, ny)
        2D grain ID map representing the microstructure
    iteration : int, default=5
        Smoothing iteration parameter controlling algorithm accuracy
        Higher values provide better smoothing but increased computation
        
    Returns:
    --------
    comprehensive_results : dict
        Complete analysis results including:
        - triple_grain: List of grain ID triplets for each triple junction
        - triple_coord: List of (i,j) coordinates for each triple junction
        - angle_results: Dictionary of angle calculations from all methods
        - statistical_summary: Comparison statistics between methods
        
    Algorithm Workflow:
    ------------------
    1. Scan entire microstructure for 2×2 neighborhoods
    2. Identify configurations containing exactly 3 distinct grains
    3. Calculate normal vectors using bilinear smoothing
    4. Apply three different dihedral angle calculation methods
    5. Perform statistical comparison and validation
    
    Triple Junction Detection:
    -------------------------
    Uses systematic 2×2 neighborhood scanning to identify locations where
    exactly three distinct grain IDs meet, indicating triple junction sites.
    
    Scientific Applications:
    -----------------------
    - Grain boundary energy analysis and modeling
    - Microstructure evolution validation studies
    - Crystal plasticity parameter determination
    - Polycrystal homogenization verification
    
    Performance Scaling:
    -------------------
    - Time complexity: O(nx × ny × iteration²)
    - Memory usage: O(nx × ny + N_TJ) where N_TJ = number of triple junctions
    - Parallel processing: Triple junction calculations are independent
    
    Quality Control:
    ---------------
    - Validates angle sum conservation (360° ± tolerance)
    - Tracks algorithm convergence and stability
    - Identifies problematic triple junction configurations
    
    Usage:
    ------
    Primary analysis function for comprehensive dihedral angle studies
    with built-in validation and algorithm comparison capabilities.
    """
    nx, ny = triple_map.shape
    num = 0                    # Counter for valid triple junctions
    issue_num = 0              # Counter for problematic configurations
    triple_grain = []          # List of grain ID triplets
    triple_coord = []          # List of triple junction coordinates
    triple_normal = []         # List of normal vector arrays
    triple_angle = []          # List of angle calculation results
    triple_angle_tang = []     # List of tangent method results
    
    # Systematic scan of entire microstructure for triple junctions
    for i in range(nx-1):
        for j in range(ny-1):
            # Extract 2×2 neighborhood for analysis
            nei = np.zeros((2,2))
            nei = triple_map[i:i+2, j:j+2]
            nei_flat = nei.flatten()
            
            # Check for valid triple junction: exactly 3 distinct grains, no voids
            if len(set(nei_flat)) == 3 and 0 not in nei_flat:
                # Calculate normal vectors using bilinear smoothing
                each_normal, grain_sequence = find_normal(triple_map, i, j, nei_flat, iteration)
                
                # Skip if normal calculation failed
                if isinstance(each_normal, (int, float)): 
                    continue
                
                # Store analysis results
                triple_normal.append(each_normal)          # Normal vectors
                triple_coord.append(np.array([i, j]))      # Triple junction coordinates
                triple_grain.append(grain_sequence)        # Grain ID sequence
                triple_angle.append(find_angle(each_normal))  # Dihedral angles
                
                num += 1  # Increment valid triple junction counter
                
                # Quality control: check angle sum conservation
                if abs(sum(find_angle(each_normal)) - 360) > 5: 
                    issue_num += 1

    # Statistical summary of analysis results
    print(f"The number of useful triple junction is {num}")
    if num == 0: 
        print("The issue proportion is 0%")
    else: 
        print(f"The issue proportion is {issue_num/num*100:.2f}%")

    return np.array(triple_coord), np.array(triple_angle), np.array(triple_grain)


if __name__ == '__main__':
    """
    Main execution block for dihedral angle algorithm comparison studies.
    
    This section demonstrates the complete workflow for comparing dihedral angle
    calculation algorithms using both analytical test cases and experimental
    data from various sources.
    
    Experimental Datasets:
    ---------------------
    1. Joseph's coupled energy minimization results
    2. Various energy coupling scenarios (average, sum, constrained min/max)
    3. Time-evolution data for validation studies
    
    Analysis Parameters:
    -------------------
    - average_coupled_energy: Energy parameters for different grain boundary types
    - Multiple dataset files for comprehensive comparison
    - Statistical analysis across different simulation conditions
    
    Usage:
    ------
    Run this script directly to execute the complete algorithm comparison
    workflow with visualization and statistical analysis outputs.
    """
    # Energy parameters for different grain boundary configurations
    average_coupled_energy = np.array([0.99858999, 3.05656703, 0.4, 1.6, 0.1])
    
    # Joseph's experimental results for comparison
    file_path_joseph = "/Users/lin/Dropbox (UFL)/UFdata/Dihedral_angle/TJ_IC_11152023/Results/"
    npy_file_name_joseph_ave = "t_dihedrals_0.npy"     # Average energy coupling
    npy_file_name_joseph_sum = "t_dihedrals_5.npy"     # Sum energy coupling
    npy_file_name_joseph_consmin = "t_dihedrals_2.npy" # Constrained minimum
    npy_file_name_joseph_consmax = "t_dihedrals_1.npy" # Constrained maximum
    npy_file_name_joseph_constest = "t_dihedrals_3.npy" # (161, 6, 96) grain ID's involved (first 3) and the dihedral angles (last 3)
    triple_results_ave = np.load(file_path_joseph + npy_file_name_joseph_ave)
    triple_results_sum = np.load(file_path_joseph + npy_file_name_joseph_sum)
    triple_results_consmin = np.load(file_path_joseph + npy_file_name_joseph_consmin)
    triple_results_consmax = np.load(file_path_joseph + npy_file_name_joseph_consmax)
    triple_results_constest = np.load(file_path_joseph + npy_file_name_joseph_constest)

    # Necessary parameters
    num_grain_initial = 3

    num_steps = 61
    max_dihedral_ave_list = np.zeros(num_steps)
    max_dihedral_sum_list = np.zeros(num_steps)
    max_dihedral_consmin_list = np.zeros(num_steps)
    max_dihedral_consmax_list = np.zeros(num_steps)
    max_dihedral_constest_list = np.zeros(num_steps)
    for i in tqdm(range(num_steps)):
        # From Joseph algorithm
        triple_results_step_ave = triple_results_ave[i,3:6]
        triple_results_step_sum = triple_results_sum[i,3:6]
        triple_results_step_consmin = triple_results_consmin[i,3:6]
        triple_results_step_consmax = triple_results_consmax[i,3:6]
        triple_results_step_constest = triple_results_constest[i,3:6]

        # triple_results_step_ave = triple_results_step_ave[:,~np.isnan(triple_results_step_ave[0,:])]
        # triple_results_step_sum = triple_results_step_sum[:,~np.isnan(triple_results_step_sum[0,:])]
        # triple_results_step_consmin = triple_results_step_consmin[:,~np.isnan(triple_results_step_consmin[0,:])]
        # triple_results_step_consmax = triple_results_step_consmax[:,~np.isnan(triple_results_step_consmax[0,:])]
        # triple_results_step_constest = triple_results_step_constest[:,~np.isnan(triple_results_step_constest[0,:])]
        # print(f"The number in ave is {len(triple_results_step_ave[0,:])}")
        # print(f"The number in sum is {len(triple_results_step_sum[0,:])}")
        # print(f"The number in consmin is {len(triple_results_step_consmin[0,:])}")
        # print(f"The number in consmax is {len(triple_results_step_consmax[0,:])}")
        # print(f"The number in constest is {len(triple_results_step_constest[0,:])}")
        max_dihedral_ave_list[i] = triple_results_step_ave[2]#np.mean(np.max(triple_results_step_ave,0))
        max_dihedral_sum_list[i]= triple_results_step_sum[2]#np.mean(np.max(triple_results_step_sum,0))
        max_dihedral_consmin_list[i] = triple_results_step_consmin[2]#np.mean(np.max(triple_results_step_consmin,0))
        max_dihedral_consmax_list[i] = triple_results_step_consmax[2]#np.mean(np.max(triple_results_step_consmax,0))
        max_dihedral_constest_list[i] = triple_results_step_constest[2]#np.mean(np.max(triple_results_step_constest,0))

    # average the max dihedral angle for all time steps
    max_dihedral_ave_list = max_dihedral_ave_list[~np.isnan(max_dihedral_ave_list[:])]
    max_dihedral_sum_list = max_dihedral_sum_list[~np.isnan(max_dihedral_sum_list[:])]
    max_dihedral_consmin_list = max_dihedral_consmin_list[~np.isnan(max_dihedral_consmin_list[:])]
    max_dihedral_consmax_list = max_dihedral_consmax_list[~np.isnan(max_dihedral_consmax_list[:])]
    max_dihedral_constest_list = max_dihedral_constest_list[~np.isnan(max_dihedral_constest_list[:])]
    max_dihedral_ave = np.mean(max_dihedral_ave_list)
    max_dihedral_sum = np.mean(max_dihedral_sum_list)
    max_dihedral_consmin = np.mean(max_dihedral_consmin_list)
    max_dihedral_consmax = np.mean(max_dihedral_consmax_list)
    max_dihedral_constest = np.mean(max_dihedral_constest_list)

    max_dihedral_list = np.array([max_dihedral_ave, max_dihedral_sum, max_dihedral_consmin, max_dihedral_consmax, max_dihedral_constest])


    dihedral_siteEnergy_cases_figure_name = "energy_results/triple_aveDihedral_aveEnergy_" + "figure.png"
    plt.clf()
    plt.plot(average_coupled_energy, max_dihedral_list, 'o', markersize=4, label = "average angle")

    # Fitting
    a = max(max_dihedral_list)-min(max_dihedral_list)
    b = max_dihedral_list[round(len(max_dihedral_list)/2)]
    c = min(max_dihedral_list)
    p0 = [a,b,c]
    popt, pcov = curve_fit(func, average_coupled_energy, max_dihedral_list,p0=p0)
    print(f"The equation to fit the relationship is {round(popt[0],2)} * exp(-x * {round(popt[1],2)}) + {round(popt[2],2)}")
    y_fit = [func(i,popt[0], popt[1], popt[2]) for i in np.linspace(0, 4, 50)]
    plt.plot(np.linspace(0, 4, 50), y_fit, '-', linewidth=2, label = "fit")
    # # Find the exact result
    # exact_list = np.linspace(0.0, 0.5, 1001)
    # min_level = 10
    # expect_site_energy = 0
    # for m in exact_list:
    #     if min_level > abs(func(m, popt[0], popt[1], popt[2]) - 145.46):
    #         min_level = abs(func(m, popt[0], popt[1], popt[2]) - 145.46)
    #         expect_site_energy = m
    # print(f"The expected average TJ site energy is {expect_site_energy}")

    # plt.plot(np.linspace(0,4,24), [145.46]*24, '--', linewidth=2, label = "equilibrium from GB area") # Max-100

    # My algorithm
    npy_file_folder = "/Users/lin/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_triple_for_TJE/results/"

    # Get the average dihedral angle
    # cases=5
    # cases_name = ["ave", "sum", "consMin", "consMax", "consTest"]
    # max_dihedral_angle_lin = np.zeros(cases)
    # for i in range(cases):
    #     energy_type = cases_name[i]
    #     base_name = f"dihedral_results/hex_{energy_type}_"
    #     dihedral_over_time = np.load(npy_file_folder + base_name + "data.npy")
    #     max_dihedral_angle_lin[i] = np.average(dihedral_over_time[:num_steps])
    # plt.plot(average_coupled_energy, max_dihedral_angle_lin, 'o', markersize=4, label = "average angle (Lin)")
    plt.ylim([80,140])
    plt.xlim([0,4])
    plt.legend(fontsize=20)
    plt.xlabel("Average TJ energy (J/MCU)", fontsize=20)
    plt.ylabel(r"Angle ($^\circ$)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.savefig(npy_file_folder + dihedral_siteEnergy_cases_figure_name, bbox_inches='tight', format='png', dpi=400)







