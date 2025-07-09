#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Triple Junction Dihedral Angle Analysis using Bilinear Smoothing

This module extends the 2D triple junction analysis to three-dimensional
microstructures. It provides tools for calculating dihedral angles at
triple lines (edges where three grains meet) in 3D polycrystalline materials.

Key Features:
-------------
1. 3D Bilinear Smoothing: Extension of bilinear smoothing to 3D cubic neighborhoods
2. Triple Line Detection: Identification of locations where exactly three grains meet
3. Normal Vector Calculation: 3D grain-boundary-aware normal vector computation
4. Dihedral Angle Analysis: Calculation of angles between grain boundaries meeting at triple lines

Scientific Background:
---------------------
In 3D microstructures, grain boundaries form surfaces and meet along triple lines.
The dihedral angles at these triple lines are crucial for understanding:
- Grain boundary energy relationships
- Microstructural stability
- Interfacial thermodynamics in 3D systems

Mathematical Foundation:
-----------------------
The 3D bilinear smoothing algorithm extends the 2D approach by:
- Using 2×2×2 cubic neighborhoods instead of 2×2 square neighborhoods
- Calculating gradients in three spatial directions (i, j, k)
- Averaging normal vectors across 8 corner points for stability

Algorithm Validation:
--------------------
Results can be compared with theoretical predictions and experimental
measurements of dihedral angles in 3D grain structures.

Dependencies:
------------
- numpy: Numerical operations and array handling
- myInput: 3D smoothing matrix generation and utilities
- math: Mathematical functions for angle calculations

Author: lin.yang
Created: Sat Apr 2 16:51:24 2022

Usage:
------
This module is designed for analyzing 3D microstructural data from
phase field simulations, Monte Carlo grain growth models, or experimental
3D characterization techniques.
"""

import os
current_path = os.getcwd()
import numpy as np
from numpy import seterr
seterr(all='raise')
import math
import myInput

def find_window_3d(P,i,j,k,iteration,refer_id):
    """
    Extract 3D grain-aware smoothing window around a specified voxel.
    
    This function creates a 3D binary mask centered on voxel (i,j,k) where
    pixels belonging to the reference grain are marked as 1, others as 0.
    Extends the 2D windowing concept to three-dimensional cubic neighborhoods.
    
    Parameters:
    -----------
    P : ndarray, shape (nx, ny, nz)
        3D grain ID map representing the microstructure
    i, j, k : int
        Central voxel coordinates for window extraction
    iteration : int
        Controls window size: side_length = 2*(iteration+1)+1
    refer_id : int
        Reference grain ID for binary masking
        
    Returns:
    --------
    window : ndarray, shape (side_length, side_length, side_length)
        3D binary mask where 1 = reference grain, 0 = other grains
        
    Algorithm:
    ----------
    1. Calculate window dimensions based on iteration parameter
    2. Apply periodic boundary conditions for edge voxels
    3. Create binary mask based on grain ID matching
    4. Return 3D window for smoothing operations
    
    Window Size Calculation:
    -----------------------
    side_length = 2*(iteration+1)+1
    half_width = (side_length-1)/2
    
    This ensures symmetric windows with odd dimensions for proper centering.
    
    Boundary Conditions:
    -------------------
    Uses periodic boundary conditions (modulo operation) to handle
    voxels near microstructure boundaries.
    
    Usage:
    ------
    Core function for 3D grain-aware smoothing. Results feed into
    gradient calculations for normal vector determination.
    """
    # Find the windows around the voxel i,j,k, the size depend on iteration
    nx,ny,nz=P.shape
    tableL=2*(iteration+1)+1      # Calculate window side length
    fw_len = tableL
    fw_half = int((fw_len-1)/2)   # Half-width for centering
    window = np.zeros((fw_len,fw_len,fw_len))
    
    # Fill window with binary mask based on grain ID matching
    for wi in range(fw_len):
        for wj in range(fw_len):
            for wk in range(fw_len):
                # Apply periodic boundary conditions
                global_x = (i-fw_half+wi)%nx
                global_y = (j-fw_half+wj)%ny
                global_z = (k-fw_half+wk)%nz  # Fixed: was using wj instead of wk
                
                # Set binary mask value based on grain ID match
                if P[global_x,global_y,global_z] == refer_id:
                    window[wi,wj,wk] = 1
                else:
                    window[wi,wj,wk] = 0
    
    return window

def find_normal_structure_3d(P,i,j,k,iteration,refer_id):
    """
    Calculate 3D normal vector components using bilinear smoothing.
    
    This function computes normal vector components at a grain boundary location
    by applying 3D bilinear smoothing operations to a 2×2×2 neighborhood. The
    smoothing is grain-aware, ensuring sharp grain boundary preservation in 3D.
    
    Parameters:
    -----------
    P : ndarray, shape (nx, ny, nz)
        3D grain ID map representing the microstructure
    i, j, k : int
        Top-left-front corner coordinates of 2×2×2 neighborhood to analyze
    iteration : int
        Smoothing iteration parameter for algorithm accuracy
    refer_id : int
        Reference grain ID for grain-aware smoothing
        
    Returns:
    --------
    a, b, c : float
        Normal vector components (a = i-direction, b = j-direction, c = k-direction)
        Represent local interface orientation at the specified location
        
    Algorithm:
    ----------
    1. Generate 3D bilinear smoothing gradient matrices for all three directions
    2. Extract grain-aware windows for 2×2×2 neighborhood corner points
    3. Apply smoothing kernels to each of the 8 corner points
    4. Average results across 2×2×2 neighborhood for stability
    5. Return averaged normal vector components in 3D
    
    3D Bilinear Smoothing Process:
    -----------------------------
    For each corner point (i+di, j+dj, k+dk) where di,dj,dk ∈ {0,1}:
    - Extract grain-aware smoothing window
    - Apply gradient calculation kernels (smoothed_vector_i/j/k)
    - Compute local normal vector components
    - Average across all 8 corners for final result
    
    Mathematical Foundation:
    -----------------------
    The 2×2×2 averaging reduces noise and provides more stable normal
    vector estimates compared to single-point calculations in 3D space.
    
    Usage:
    ------
    Core function for 3D normal vector calculation in triple line analysis.
    Provides grain-boundary-aware normal vectors essential for accurate
    3D dihedral angle computation.
    """
    # A basic structure to calculate normals
    # Generate 3D bilinear smoothing gradient matrices
    smoothed_vector_i, smoothed_vector_j, smoothed_vector_k = myInput.output_linear_vector_matrix3D(iteration)
    
    # Calculate normal vector i-component (gradient in i-direction)
    # Average across 2×2×2 neighborhood for stability
    a = (
        np.sum(find_window_3d(P,i,j,k,iteration,refer_id) * smoothed_vector_i) + 
        np.sum(find_window_3d(P,i+1,j,k,iteration,refer_id) * smoothed_vector_i) +  
        np.sum(find_window_3d(P,i,j+1,k,iteration,refer_id) * smoothed_vector_i) + 
        np.sum(find_window_3d(P,i+1,j+1,k,iteration,refer_id) * smoothed_vector_i) + 
        np.sum(find_window_3d(P,i,j,k+1,iteration,refer_id) * smoothed_vector_i) + 
        np.sum(find_window_3d(P,i+1,j,k+1,iteration,refer_id) * smoothed_vector_i) +  
        np.sum(find_window_3d(P,i,j+1,k+1,iteration,refer_id) * smoothed_vector_i) + 
        np.sum(find_window_3d(P,i+1,j+1,k+1,iteration,refer_id) * smoothed_vector_i)
        ) / 8
    # Calculate normal vector j-component (gradient in j-direction)
    # Average across 2×2×2 neighborhood for stability
    b = (
        np.sum(find_window_3d(P,i,j,k,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window_3d(P,i+1,j,k,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window_3d(P,i,j+1,k,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window_3d(P,i+1,j+1,k,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window_3d(P,i,j,k+1,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window_3d(P,i+1,j,k+1,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window_3d(P,i,j+1,k+1,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window_3d(P,i+1,j+1,k+1,iteration,refer_id) * smoothed_vector_j)
        ) / 8
    
    # Calculate normal vector k-component (gradient in k-direction)
    # Average across 2×2×2 neighborhood for stability
    c = (
        np.sum(find_window_3d(P,i,j,k,iteration,refer_id) * smoothed_vector_k) + 
        np.sum(find_window_3d(P,i+1,j,k,iteration,refer_id) * smoothed_vector_k) + 
        np.sum(find_window_3d(P,i,j+1,k,iteration,refer_id) * smoothed_vector_k) + 
        np.sum(find_window_3d(P,i+1,j+1,k,iteration,refer_id) * smoothed_vector_k) + 
        np.sum(find_window_3d(P,i,j,k+1,iteration,refer_id) * smoothed_vector_k) + 
        np.sum(find_window_3d(P,i+1,j,k+1,iteration,refer_id) * smoothed_vector_k) + 
        np.sum(find_window_3d(P,i,j+1,k+1,iteration,refer_id) * smoothed_vector_k) + 
        np.sum(find_window_3d(P,i+1,j+1,k+1,iteration,refer_id) * smoothed_vector_k)
        ) / 8
    
    return a, b, c

def find_normal_3d(P,i,j,k,nei_flat,iteration):
    """
    Calculate 3D normal vectors for all grain boundaries meeting at a triple line.
    
    This function identifies the configuration of a 3D triple line and computes
    the appropriate normal vectors for each grain boundary surface. It handles
    the complex 3D geometry where grain boundaries meet along edges (triple lines).
    
    Parameters:
    -----------
    P : ndarray, shape (nx, ny, nz)
        3D grain ID map representing the microstructure
    i, j, k : int
        Top-left-front corner coordinates of 2×2×2 neighborhood containing triple line
    nei_flat : ndarray
        Flattened array of 8 grain IDs from 2×2×2 neighborhood
        Order: flattened P[i:i+2, j:j+2, k:k+2]
    iteration : int
        Smoothing iteration parameter for algorithm accuracy
        
    Returns:
    --------
    tri_norm_final : ndarray, shape (3, 3)
        Final normalized normal vectors for the three grains
        tri_norm_final[grain_idx] = [nx, ny, nz] components
    tri_grains : ndarray, shape (3,)
        The three distinct grain IDs forming the triple line
        
    Algorithm:
    ----------
    1. Calculate individual normal vectors for all 8 voxels in 2×2×2 neighborhood
    2. Identify the three unique grains involved in the triple line
    3. Accumulate normal vectors for each grain based on grain ID matching
    4. Normalize final normal vectors to unit length with error handling
    5. Return organized normal vectors and grain IDs
    
    3D Triple Line Configuration:
    ----------------------------
    In 3D, triple lines occur where three grain boundary surfaces intersect
    along an edge. The 2×2×2 neighborhood captures the local geometry around
    this intersection, providing sufficient information for accurate normal
    vector calculation.
    
    Normal Vector Accumulation:
    --------------------------
    For each grain:
    - Collect normal vectors from all voxels belonging to that grain
    - Sum the contributions to get averaged normal vector
    - Normalize to unit length for geometric consistency
    
    Error Handling:
    --------------
    Includes try-catch block for normalization to handle edge cases where
    normal vectors might be zero or near-zero, setting them to zero vector
    and printing diagnostic information.
    
    Mathematical Foundation:
    -----------------------
    The accumulation and averaging process reduces noise and provides more
    stable normal vector estimates for 3D grain boundary surfaces compared
    to single-point calculations.
    
    Usage:
    ------
    Core function for 3D triple line analysis. Results feed directly into
    3D dihedral angle calculation algorithms for grain boundary characterization.
    """
    # Calculate the normals for all the eight voxels in the triple line and assign them to 3 grains
    tri_norm = np.zeros((8,3))          # Individual normal vectors for 8 voxels
    tri_norm_final = np.zeros((3,3))    # Final accumulated normal vectors for 3 grains
    tri_grains = np.zeros(3)            # Array to store the 3 unique grain IDs
    
    # Calculate normal vectors for each voxel in the 2×2×2 neighborhood
    for fi in range(len(nei_flat)):
        tri_norm[fi,0], tri_norm[fi,1], tri_norm[fi,2] = find_normal_structure_3d(P,i,j,k,iteration,nei_flat[fi])
    
    # Identify the three unique grains in the triple line
    tri_grains = np.array(list(set(nei_flat)))
    
    # Accumulate normal vectors for each grain
    for fi in range(len(nei_flat)):
        grain_sequence = np.where(tri_grains==nei_flat[fi])[0][0]  # Find grain index
        tri_norm_final[grain_sequence] += tri_norm[fi]             # Add contribution
    # Alternative explicit calculation method (commented out for reference)
    # This shows the direct approach for calculating normals at each of the 8 corner points:
    # tri_norm[0,0], tri_norm[0,1], tri_norm[0,2] = find_normal_structure_3d(P,i,j,k,iteration,P[i,j,k])
    # tri_norm[1,0], tri_norm[1,1], tri_norm[1,2] = find_normal_structure_3d(P,i,j,k,iteration,P[i,j+1,k])
    # tri_norm[2,0], tri_norm[2,1], tri_norm[2,2] = find_normal_structure_3d(P,i,j,k,iteration,P[i+1,j+1,k])
    # tri_norm[3,0], tri_norm[3,1], tri_norm[3,2] = find_normal_structure_3d(P,i,j,k,iteration,P[i+1,j,k])
    # tri_norm[4,0], tri_norm[4,1], tri_norm[4,2] = find_normal_structure_3d(P,i,j,k,iteration,P[i,j,k+1])
    # tri_norm[5,0], tri_norm[5,1], tri_norm[5,2] = find_normal_structure_3d(P,i,j,k,iteration,P[i,j+1,k+1])
    # tri_norm[6,0], tri_norm[6,1], tri_norm[6,2] = find_normal_structure_3d(P,i,j,k,iteration,P[i+1,j+1,k+1])
    # tri_norm[7,0], tri_norm[7,1], tri_norm[7,2] = find_normal_structure_3d(P,i,j,k,iteration,P[i+1,j,k+1])
    # tri_grains = np.array([P[i,j+1], P[i+1,j+1], P[i+1,j]])  # Example grain sequence
    
    # Normalize all accumulated normal vectors to unit length
    for ni in range(len(tri_norm_final)):
        try:
            # Normalize to unit vector
            tri_norm_final[ni] = tri_norm_final[ni]/np.linalg.norm(tri_norm_final[ni])
        except:
            # Handle edge cases where normalization fails (zero or near-zero vectors)
            print(tri_norm[ni])                    # Debug: print problematic normal vector
            print(f"{i},{j},{k}")                  # Debug: print location coordinates
            tri_norm_final[ni] = np.zeros(3)       # Set to zero vector as fallback
        
    
    return tri_norm_final, tri_grains

def find_angle_3d(each_normal):
    """
    Calculate 3D dihedral angles at a triple line using direct normal vector method.
    
    This function computes the three dihedral angles at a 3D triple line by
    working directly with the unit normal vectors of the three grain boundary
    surfaces. Uses the geometric relationship between normal vectors and
    the dihedral angles in 3D space.
    
    Parameters:
    -----------
    each_normal : ndarray, shape (3, 3)
        Unit normal vectors for the three grain boundary surfaces
        each_normal[i] = [nx, ny, nz] for grain boundary i
        
    Returns:
    --------
    tri_angle : ndarray, shape (3,)
        Three dihedral angles in degrees
        Angles between pairs of grain boundary surfaces at the triple line
        
    Algorithm:
    ----------
    For each pair of grain boundaries, calculate the dihedral angle using:
    θ = 2π - 2*arccos(n₁·n₂)
    
    where n₁ and n₂ are unit normal vectors to adjacent grain boundary surfaces.
    
    Mathematical Foundation:
    -----------------------
    In 3D, the dihedral angle between two surfaces is related to the angle
    between their normal vectors. The formula accounts for the geometric
    relationship in the specific context of grain boundary intersections.
    
    The three angles calculated are:
    - tri_angle[0]: Angle between surfaces of grains 1 and 2
    - tri_angle[1]: Angle between surfaces of grains 0 and 2  
    - tri_angle[2]: Angle between surfaces of grains 0 and 1
    
    Numerical Considerations:
    ------------------------
    - Uses rounding to 5 decimal places for numerical stability
    - Converts from radians to degrees for practical interpretation
    - The doubled angle formula accounts for the geometric configuration
    
    Physical Interpretation:
    -----------------------
    Dihedral angles at triple lines are fundamental geometric quantities
    that relate to grain boundary energy and microstructural stability.
    The sum of angles provides a consistency check for the calculation.
    
    Validation Note:
    ---------------
    The commented section shows optional validation where angles that
    don't sum to 360° can be corrected, but this is typically disabled
    to preserve the raw calculated values.
    
    Usage:
    ------
    Primary method for 3D dihedral angle calculation in triple line analysis.
    Results are used for grain boundary energy analysis and microstructure
    characterization in three-dimensional systems.
    """
    # Find the 3 dihedral angles based on the normals of three neighboring grains
    tri_angle = np.zeros(3)
    
    # Calculate dihedral angles using doubled angle formula
    # Formula: θ = 2π - 2*arccos(n₁·n₂) where n₁, n₂ are unit normal vectors
    tri_angle[0] = 180 / np.pi * (2 * np.pi - 2 * math.acos(round(np.dot(each_normal[1], each_normal[2]),5)))
    tri_angle[1] = 180 / np.pi * (2 * np.pi - 2 * math.acos(round(np.dot(each_normal[0], each_normal[2]),5)))
    tri_angle[2] = 180 / np.pi * (2 * np.pi - 2 * math.acos(round(np.dot(each_normal[0], each_normal[1]),5)))
    
    
    # Optional validation and correction for special cases (currently disabled)
    # This section can be enabled to handle cases where angle sum deviates significantly from 360°
    # if abs(sum(tri_angle)-360) > 5:
        # print()                           # Debug: print empty line
        # print(sum(tri_angle))             # Debug: print total angle sum
        # print(tri_angle)                  # Debug: print individual angles
        # print(each_normal)                # Debug: print normal vectors
        # tri_angle[2] = 360 - tri_angle[0] - tri_angle[1]  # Force angle sum to 360°
    
    return tri_angle

def read_3d_input(filename,nx,ny,nz):
    """
    Read 3D microstructure data from formatted input file.
    
    This function reads 3D microstructure data from a text file and creates
    a 3D grain ID map for analysis. Designed for standard output formats
    from 3D microstructure simulation tools.
    
    Parameters:
    -----------
    filename : str
        Path to input file containing 3D microstructure data
    nx, ny, nz : int
        Dimensions of the 3D microstructure grid
        
    Returns:
    --------
    triple_map : ndarray, shape (nx, ny, nz)
        3D array containing grain IDs for each voxel
        
    File Format:
    -----------
    Expected input file format:
    - First line: header (ignored)
    - Second line: header (ignored)
    - Data lines: i j k ... ... ... ... grain_id
    
    The function reads coordinates (i, j, k) and grain ID from each line,
    converting from 1-indexed file format to 0-indexed array indexing.
    Grain ID is expected in the 8th column (index 7).
    
    Coordinate Convention:
    ---------------------
    - File uses 1-indexed coordinates (starts from 1)
    - Array uses 0-indexed coordinates (starts from 0)
    - Conversion: array_index = file_coordinate - 1
    
    Usage:
    ------
    Utility function for loading 3D microstructure data from external files.
    Commonly used with:
    - 3D SPPARKS simulation output
    - Phase field simulation results
    - Experimental 3D characterization data
    - Monte Carlo grain growth model output
    """
    # Keep the 3d input macrostructure
    
    triple_map = np.zeros((nx,ny,nz))
    f = open(filename)
    line = f.readline()  # Skip first header line
    line = f.readline()  # Skip second header line
    # print(line)  # Debug: optionally print header for verification
    while line:
        each_element = line.split()
        i = int(each_element[0])-1  # Convert from 1-indexed to 0-indexed
        j = int(each_element[1])-1  # Convert from 1-indexed to 0-indexed
        k = int(each_element[2])-1  # Convert from 1-indexed to 0-indexed
        triple_map[i,j,k] = int(each_element[7])  # Grain ID is in 8th column
    
        line = f.readline()
    f.close()
    
    return triple_map

def read_3d_init(filename,nx,ny,nz,filepath=current_path+"/input/"):
    """
    Read 3D microstructure initial condition from SPPARKS-style input file.
    
    This function reads 3D microstructure data from SPPARKS initialization
    files where voxels are specified using linear indexing. Converts the
    linear site numbering to 3D array coordinates.
    
    Parameters:
    -----------
    filename : str
        Name of the input file (without path)
    nx, ny, nz : int
        Dimensions of the 3D microstructure grid
    filepath : str, optional
        Directory path containing the input file
        Default: current_path + "/input/"
        
    Returns:
    --------
    triple_map : ndarray, shape (nx, ny, nz)
        3D array containing grain IDs for each voxel
        
    File Format:
    -----------
    SPPARKS initialization file format:
    - First 4 lines: headers (ignored)
    - Data lines: site_number grain_id
    
    Site numbering convention:
    - Linear indexing: site = 1 + i + j*nx + k*nx*ny
    - Converts to 3D coordinates: (i, j, k)
    
    Coordinate Conversion Algorithm:
    ------------------------------
    Given linear site number (1-indexed):
    1. Convert to 0-indexed: site_0 = site - 1
    2. Extract k: k = site_0 // (nx * ny)
    3. Extract j: j = (site_0 - k*nx*ny) // nx
    4. Extract i: i = site_0 - k*nx*ny - j*nx
    
    Mathematical Foundation:
    -----------------------
    The conversion assumes row-major ordering where:
    - i varies fastest (innermost loop)
    - j varies next
    - k varies slowest (outermost loop)
    
    Usage:
    ------
    Specialized function for reading SPPARKS initial condition files.
    Commonly used for:
    - Loading initial grain structures for phase field simulations
    - Processing Monte Carlo simulation initial conditions
    - Converting between linear and 3D coordinate systems
    """
    triple_map = np.zeros((nx,ny,nz))
    f = open(filepath + filename)
    
    # Skip the first 4 header lines in SPPARKS format
    line = f.readline()  # Header line 1
    line = f.readline()  # Header line 2
    line = f.readline()  # Header line 3
    line = f.readline()  # Header line 4
    
    while line:
        each_element = line.split()
        
        # Convert linear site number to 3D coordinates
        # Linear index is 1-indexed, convert to 0-indexed first
        linear_index = int(each_element[0]) - 1
        
        # Extract 3D coordinates from linear index
        k = linear_index // (nx * ny)                           # z-coordinate
        j = (linear_index - (nx * ny) * k) // nx                # y-coordinate  
        i = linear_index - (nx * ny) * k - nx * j               # x-coordinate
        
        # Assign grain ID to the calculated 3D position
        triple_map[i,j,k] = int(each_element[1])
        
        line = f.readline()
    f.close()
    
    return triple_map

def get_3d_ic1(nx,ny,nz):
    """
    Generate 3D verification initial condition with 120°-120°-120° dihedral angles.
    
    This function creates a synthetic 3D microstructure with three grains
    meeting at triple lines with theoretically ideal 120° dihedral angles.
    Used for algorithm validation and accuracy testing.
    
    Parameters:
    -----------
    nx, ny, nz : int
        Dimensions of the 3D microstructure grid
        
    Returns:
    --------
    triple_map : ndarray, shape (nx, ny, nz)
        3D array with grain IDs (1, 2, 3) forming ideal triple line configuration
        
    Geometric Configuration:
    -----------------------
    Creates three grains with boundaries meeting at 120° angles:
    - Grain 1: Left-bottom region
    - Grain 2: Left-top region  
    - Grain 3: Right region with angled boundary
    
    Mathematical Foundation:
    -----------------------
    The grain boundaries are positioned to create theoretical 120° dihedral
    angles, which correspond to the equilibrium configuration for equal
    grain boundary energies in 2D cross-sections.
    
    Boundary Equations:
    ------------------
    - Vertical boundary at x = nx/2 separating left and right regions
    - Horizontal boundary at y = ny/2 separating top and bottom in left region
    - Angled boundaries in right region: j = nx/2 ± (i - nx/2) * √3
    
    Physical Interpretation:
    -----------------------
    This configuration represents the theoretical equilibrium state for
    three grains with equal surface tensions, providing a benchmark for
    testing dihedral angle calculation algorithms.
    
    Usage:
    ------
    Used for algorithm validation by comparing calculated dihedral angles
    with the theoretical 120° values. Any significant deviation indicates
    algorithm issues or numerical errors.
    """
    # Get verification IC (120, 120, 120)
    triple_map = np.zeros((nx,ny,nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Left-bottom region: Grain 1
                if i < nx/2 and j < ny/2: 
                    triple_map[i,j,k] = 1
                # Left-top region: Grain 2
                elif i < nx/2 and j >= ny/2: 
                    triple_map[i,j,k] = 2
                # Right region, lower angled boundary: Grain 1
                elif i >= nx/2 and j < nx/2 - (i - nx/2) * math.sqrt(3): 
                    triple_map[i,j,k] = 1
                # Right region, upper angled boundary: Grain 2
                elif i >= nx/2 and j >= nx/2 + (i - nx/2) * math.sqrt(3) - 1: 
                    triple_map[i,j,k] = 2
                # Right region, middle area: Grain 3
                else: 
                    triple_map[i,j,k] = 3
    
    return triple_map

def get_3d_ic2(nx,ny,nz):
    """
    Generate 3D verification initial condition with 90°-90°-180° dihedral angles.
    
    This function creates a synthetic 3D microstructure with three grains
    meeting at triple lines with 90°-90°-180° dihedral angles. Used for
    algorithm validation with orthogonal grain boundary configuration.
    
    Parameters:
    -----------
    nx, ny, nz : int
        Dimensions of the 3D microstructure grid
        
    Returns:
    --------
    triple_map : ndarray, shape (nx, ny, nz)
        3D array with grain IDs (1, 2, 3) forming orthogonal triple line configuration
        
    Geometric Configuration:
    -----------------------
    Creates three grains with orthogonal boundaries:
    - Grain 1: Left-bottom quadrant
    - Grain 2: Left-top quadrant
    - Grain 3: Entire right half
    
    Boundary Configuration:
    ----------------------
    - Vertical boundary at x = nx/2 (180° angle)
    - Horizontal boundary at y = ny/2 in left half (90° angles)
    
    Physical Interpretation:
    -----------------------
    This configuration represents a special case where grain boundary
    energies are highly asymmetric, leading to one straight boundary
    and two perpendicular boundaries meeting at 90° angles.
    
    Usage:
    ------
    Validation test case for algorithms handling non-equilibrium dihedral
    angle configurations. Tests algorithm robustness for extreme cases.
    """
    # Get verification IC (90, 90, 180)
    triple_map = np.zeros((nx,ny,nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Left-bottom quadrant: Grain 1
                if i < nx/2 and j < ny/2: 
                    triple_map[i,j,k] = 1
                # Left-top quadrant: Grain 2
                elif i < nx/2 and j >= ny/2: 
                    triple_map[i,j,k] = 2
                # Entire right half: Grain 3
                else: 
                    triple_map[i,j,k] = 3
    
    return triple_map
         
def get_3d_ic3(nx,ny,nz):
    """
    Generate 3D verification initial condition with 105°-120°-135° dihedral angles.
    
    This function creates a synthetic 3D microstructure with three grains
    meeting at triple lines with slightly asymmetric dihedral angles.
    Used for testing algorithm accuracy with near-equilibrium configurations.
    
    Parameters:
    -----------
    nx, ny, nz : int
        Dimensions of the 3D microstructure grid
        
    Returns:
    --------
    triple_map : ndarray, shape (nx, ny, nz)
        3D array with grain IDs (1, 2, 3) forming asymmetric triple line configuration
        
    Geometric Configuration:
    -----------------------
    Similar to IC1 but with modified boundary angles:
    - Grain 1: Left-bottom and modified right region
    - Grain 2: Left-top and modified right region
    - Grain 3: Central right region
    
    Boundary Modifications:
    ----------------------
    The right region boundaries are slightly modified from the 120° case:
    - Lower boundary: j < nx/2 - (i - nx/2) * √3 (same as IC1)
    - Upper boundary: j >= nx/2 + (i - nx/2) - 1 (linear instead of √3 slope)
    
    Physical Interpretation:
    -----------------------
    Represents a slightly perturbed equilibrium state, useful for testing
    algorithm sensitivity to small deviations from perfect symmetry.
    
    Usage:
    ------
    Validation test for algorithm stability and accuracy with realistic
    microstructural configurations that deviate slightly from ideal geometries.
    """
    # Get verification IC (105, 120, 135)
    triple_map = np.zeros((nx,ny,nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Left-bottom region: Grain 1
                if i < nx/2 and j < ny/2: 
                    triple_map[i,j,k] = 1
                # Left-top region: Grain 2
                elif i < nx/2 and j >= ny/2: 
                    triple_map[i,j,k] = 2
                # Right region, lower angled boundary: Grain 1
                elif i >= nx/2 and j < nx/2 - (i - nx/2) * math.sqrt(3): 
                    triple_map[i,j,k] = 1
                # Right region, upper linear boundary: Grain 2 (modified slope)
                elif i >= nx/2 and j >= nx/2 + (i - nx/2)-1: 
                    triple_map[i,j,k] = 2
                # Right region, middle area: Grain 3
                else: 
                    triple_map[i,j,k] = 3
    
    return triple_map 

def get_3d_ic4(nx,ny,nz):
    """
    Generate 3D verification initial condition with 45°-45°-270° dihedral angles.
    
    This function creates a synthetic 3D microstructure with three grains
    meeting at triple lines with extreme dihedral angle configuration.
    Used for testing algorithm robustness with highly asymmetric cases.
    
    Parameters:
    -----------
    nx, ny, nz : int
        Dimensions of the 3D microstructure grid
        
    Returns:
    --------
    triple_map : ndarray, shape (nx, ny, nz)
        3D array with grain IDs (1, 2, 3) forming extreme angle triple line configuration
        
    Geometric Configuration:
    -----------------------
    Creates three grains with highly asymmetric boundaries:
    - Grain 1: Small triangular region in left-bottom
    - Grain 2: Small triangular region in left-top
    - Grain 3: Large majority region
    
    Boundary Equations:
    ------------------
    - Diagonal boundary: j = i (45° slope)
    - Diagonal boundary: j = nx - i (135° slope, or -45°)
    - These create two small 45° wedges with one large 270° region
    
    Physical Interpretation:
    -----------------------
    This represents an extreme non-equilibrium configuration where
    one grain dominates with a very large dihedral angle (270°),
    while two small grains have acute angles (45° each).
    
    Mathematical Note:
    -----------------
    The 270° angle tests algorithm handling of reflex angles and
    ensures proper angle calculation for extreme configurations.
    
    Usage:
    ------
    Stress test for dihedral angle calculation algorithms. Validates
    robustness for highly asymmetric grain boundary configurations
    that might occur during grain growth or recrystallization processes.
    """
    # Get verification IC (45, 45, 270)
    triple_map = np.zeros((nx,ny,nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Small triangular region 1: upper-left triangle above main diagonal
                if i < nx/2 and j <= ny/2 and j > i: 
                    triple_map[i,j,k] = 1
                # Small triangular region 2: lower-right triangle below anti-diagonal
                elif i < nx/2 and j > ny/2 and j <= nx - i: 
                    triple_map[i,j,k] = 2
                # Large majority region: Grain 3 (270° angle)
                else: 
                    triple_map[i,j,k] = 3
    
    return triple_map       

def calculate_tangent(triple_map,iteration=5):
    """
    Calculate 3D dihedral angles for all triple lines in a 3D microstructure.
    
    This is the main function that systematically scans a 3D microstructure
    to identify triple lines and calculate their dihedral angles using
    3D bilinear smoothing and normal vector analysis.
    
    Parameters:
    -----------
    triple_map : ndarray, shape (nx, ny, nz)
        3D grain ID map representing the microstructure
    iteration : int, default=5
        Smoothing iteration parameter for algorithm accuracy
        Higher values provide more accurate but computationally expensive results
        
    Returns:
    --------
    triple_coord : ndarray, shape (n_triples, 3)
        Coordinates of triple line locations (top-left-front corner of 2×2×2 neighborhood)
    triple_angle : ndarray, shape (n_triples, 3)
        Dihedral angles for each triple line (in degrees)
    triple_grain : ndarray, shape (n_triples, 3)
        Grain IDs involved in each triple line
        
    Algorithm:
    ----------
    1. Scan entire 3D microstructure with 2×2×2 sliding window
    2. Identify locations where exactly 3 distinct grains meet
    3. Calculate normal vectors using 3D bilinear smoothing
    4. Compute dihedral angles from normal vectors
    5. Validate results and report statistics
    
    Triple Line Detection:
    ---------------------
    A valid triple line is identified when:
    - A 2×2×2 neighborhood contains exactly 3 distinct grain IDs
    - No background (ID=0) voxels are present
    - Proper geometric configuration exists
    
    Boundary Exclusion:
    ------------------
    Excludes regions near domain boundaries (5 voxel buffer) to avoid
    edge effects and ensure reliable smoothing window extraction.
    
    Quality Control:
    ---------------
    The function tracks:
    - Total number of triple lines found
    - Percentage with angle sum issues (deviation from 360°)
    - Error reporting for invalid configurations
    
    Mathematical Foundation:
    -----------------------
    Uses 3D bilinear smoothing for grain-boundary-aware normal vector
    calculation, ensuring accurate geometric representation of grain
    boundary surfaces and reliable dihedral angle measurements.
    
    Performance Considerations:
    --------------------------
    For large 3D datasets, computation time scales with volume.
    The boundary exclusion reduces computational load while maintaining
    accuracy for interior triple lines.
    
    Usage:
    ------
    Primary analysis function for 3D triple line characterization in
    polycrystalline microstructures. Results are used for:
    - 3D grain boundary energy analysis
    - Microstructure validation
    - Comparison with theoretical predictions
    - Quality assessment of 3D simulations
    """
    nx,ny,nz = triple_map.shape
    num = 0                    # Counter for valid triple lines
    issue_num = 0             # Counter for problematic angle calculations
    triple_grain = []         # List to store grain ID sequences
    triple_coord = []         # List to store triple line coordinates
    triple_normal = []        # List to store normal vector arrays
    triple_angle = []         # List to store dihedral angle arrays
    
    # Scan 3D microstructure with 2×2×2 sliding window
    # Exclude boundary regions (5 voxel buffer) to avoid edge effects
    for i in range(5,nx-6):
        for j in range(5,ny-6):
            for k in range(5,nz-6):
                # Extract 2×2×2 neighborhood
                nei = np.zeros((2,2,2))
                nei = triple_map[i:i+2,j:j+2,k:k+2]
                nei_flat = nei.flatten()
                # print(nei_flat)  # Debug output
                
                # Check if this is a valid triple line
                if len(set(nei_flat)) == 3 and 0 not in nei_flat:
                    
                    # Calculate normal vectors and grain sequence for this triple line
                    each_normal, tri_grains = find_normal_3d(triple_map,i,j,k,nei_flat,iteration) # Get basic normals and grain id sequence
                    
                    # Store results for this triple line
                    triple_normal.append(each_normal)            # Save the normal vectors
                    triple_coord.append(np.array([i,j,k]))       # Save the coordinate of the triple point
                    triple_grain.append(tri_grains)              # Save the grain id sequence
                    triple_angle.append(find_angle_3d(each_normal)) # Save the 3 dihedral angles
                    
                    num += 1 # Increment count of valid triple lines
                    
                    # Check for angle sum issues (should be 360°)
                    if abs(sum(find_angle_3d(each_normal))-360) > 5: 
                        issue_num += 1
            
    # Report analysis statistics
    print(f"The number of useful triple junction is {num}")
    print(f"The issue propotion is {issue_num/num*100}%")
    
    return np.array(triple_coord), np.array(triple_angle), np.array(triple_grain)



if __name__ == '__main__':
    """
    Main execution block for 3D triple line dihedral angle analysis.
    
    This section demonstrates the complete workflow for analyzing dihedral angles
    at triple lines in three-dimensional polycrystalline microstructures using
    the VECTOR framework with 3D bilinear smoothing algorithms.
    
    Workflow Options:
    ----------------
    1. Real microstructure analysis from SPPARKS simulation files
    2. Synthetic verification cases with known theoretical angles
    3. Algorithm validation and accuracy testing
    4. Performance benchmarking for large 3D datasets
    
    Available Test Cases:
    --------------------
    - IC1: 120°-120°-120° (equilibrium configuration)
    - IC2: 90°-90°-180° (orthogonal configuration)
    - IC3: 105°-120°-135° (near-equilibrium configuration)
    - IC4: 45°-45°-270° (extreme asymmetric configuration)
    
    Data Structure:
    --------------
    - Input: 3D array (nx, ny, nz) with grain IDs
    - Processing: 2×2×2 neighborhood analysis
    - Output: Arrays of coordinates, angles, and grain IDs
    
    Validation Approach:
    -------------------
    By using synthetic test cases with known theoretical dihedral angles,
    the accuracy and robustness of the 3D bilinear smoothing algorithm
    can be systematically validated.
    """
    # Configuration parameters
    filename = "500x500x50_50kgrains.init"  # SPPARKS input file
    iteration = 3                           # Smoothing iteration parameter
    
    # Option 1: Read real 3D microstructure from SPPARKS initialization file
    # nx, ny, nz = 500, 500, 50
    # triple_map = read_3d_init(filename,nx,ny,nz)
    
    # Option 2: Generate verification IC with 120°-120°-120° dihedral angles (equilibrium)
    # nx, ny, nz = 50, 50, 50
    # triple_map = get_3d_ic1(nx,ny,nz)
    
    # Option 3: Generate verification IC with 90°-90°-180° dihedral angles (orthogonal)
    # nx, ny, nz = 50, 50, 50
    # triple_map = get_3d_ic2(nx,ny,nz)
    
    # Option 4: Generate verification IC with 105°-120°-135° dihedral angles (near-equilibrium)
    nx, ny, nz = 50, 50, 50
    triple_map = get_3d_ic3(nx,ny,nz)
    
    # Option 5: Generate verification IC with 45°-45°-270° dihedral angles (extreme case)
    # nx, ny, nz = 50, 50, 50
    # triple_map = get_3d_ic4(nx,ny,nz)
    
    # Perform 3D triple line dihedral angle analysis
    triple_coord, triple_angle, triple_grain = calculate_tangent(triple_map, iteration)
    
    # Output Data Structure Documentation:
    # -----------------------------------
    # triple_coord: Coordinates of triple line locations
    #   - Shape: (n_triples, 3)
    #   - Content: [i, j, k] coordinates of top-left-front corner of 2×2×2 neighborhood
    #   - Usage: Spatial mapping of triple lines in 3D microstructure
    #
    # triple_angle: Dihedral angles for each triple line
    #   - Shape: (n_triples, 3) 
    #   - Content: Three dihedral angles in degrees for each triple line
    #   - Usage: Quantitative analysis of grain boundary geometry
    #   - Validation: Sum should equal 360° for each triple line
    #
    # triple_grain: Grain ID sequences for each triple line
    #   - Shape: (n_triples, 3)
    #   - Content: Three grain IDs involved in each triple line
    #   - Usage: Grain identity tracking and topological analysis
    
    # Optional: Save results for further analysis
    # np.save('triple_data_105_120_135',triple_map)
    # 
    # Additional analysis options:
    # - Statistical analysis of angle distributions
    # - Comparison with theoretical predictions
    # - Visualization of triple line networks
    # - Error analysis and convergence studies






