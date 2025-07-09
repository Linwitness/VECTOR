#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triple Junction Dihedral Angle Analysis using Bilinear Smoothing
================================================================

This module implements advanced algorithms for calculating dihedral angles at triple 
junctions in polycrystalline microstructures. It employs bilinear smoothing techniques 
to obtain accurate grain boundary normal vectors and compute the geometric angles 
where three grains meet.

Scientific Context:
- Triple junctions are critical features in polycrystalline materials
- Dihedral angles affect grain boundary mobility and microstructure evolution
- Accurate angle measurement requires sophisticated normal vector calculation
- Comparison with analytical solutions (120° for isotropic systems) validates algorithms

Key Features:
- Bilinear smoothing for robust normal vector calculation
- Automatic triple junction detection in 2D microstructures
- Dihedral angle calculation using tangent vector geometry
- Statistical analysis and error quantification
- Comparison framework for algorithm validation

Algorithm Overview:
1. Detect triple junction locations in grain maps
2. Calculate local normal vectors using bilinear smoothing
3. Compute tangent vectors perpendicular to grain boundaries
4. Calculate dihedral angles from tangent vector geometry
5. Validate results against analytical or reference solutions

Created on Sat Apr  2 16:51:24 2022
@author: lin.yang

Dependencies:
- numpy: Numerical computations and array operations
- matplotlib: Visualization of results and validation plots
- tqdm: Progress tracking for batch processing
- myInput: Custom bilinear smoothing matrix generation
"""

import numpy as np
from numpy import seterr
seterr(all='raise')  # Enable floating point error detection for debugging
import math
import myInput
from tqdm import tqdm
import matplotlib.pyplot as plt

def find_window(P,i,j,iteration,refer_id):
    """
    Extract local grain-aware smoothing window around a specific grid point.
    
    This function creates a binary mask window centered on grid point (i,j) that 
    identifies neighboring points belonging to the same grain as the reference grain ID.
    The window is used for grain-aware bilinear smoothing operations.
    
    Parameters:
    -----------
    P : ndarray
        2D grain ID map with shape (nx, ny)
    i, j : int
        Center coordinates for window extraction
    iteration : int
        Smoothing iteration parameter determining window size
    refer_id : int
        Reference grain ID to match against
        
    Returns:
    --------
    window : ndarray
        Binary mask array with shape (tableL, tableL) where:
        1 = neighboring point belongs to reference grain
        0 = neighboring point belongs to different grain
        
    Algorithm:
    ----------
    1. Calculate window size: tableL = 2*(iteration+1)+1
    2. Center window on input coordinates (i,j)
    3. Apply periodic boundary conditions for domain edges
    4. Compare grain ID at each window position with reference grain ID
    5. Generate binary mask preserving grain boundaries
    
    Window Size:
    -----------
    - iteration=1: 5×5 window
    - iteration=2: 7×7 window  
    - iteration=3: 9×9 window
    
    Boundary Conditions:
    -------------------
    Uses modulo arithmetic for periodic wrapping, ensuring consistent
    behavior at domain boundaries.
    
    Usage:
    ------
    Essential for grain-aware smoothing that maintains sharp grain
    boundary definitions while reducing numerical noise within grains.
    """
    # Extract domain dimensions
    nx,ny=P.shape
    
    # Calculate window size based on iteration parameter
    tableL=2*(iteration+1)+1
    fw_len = tableL
    fw_half = int((fw_len-1)/2)
    
    # Initialize binary window mask
    window = np.zeros((fw_len,fw_len))
    
    # Populate window with grain membership information
    for wi in range(fw_len):
        for wj in range(fw_len):
            # Apply periodic boundary conditions
            global_x = (i-fw_half+wi)%nx
            global_y = (j-fw_half+wj)%ny
            
            # Binary classification: same grain vs different grain
            if P[global_x,global_y] == refer_id:
                window[wi,wj] = 1  # Same grain as reference
            else:
                window[wi,wj] = 0  # Different grain
    
    return window

def find_normal_structure(P,i,j,iteration,refer_id):
    """
    Calculate normal vector components using bilinear smoothing for a specific grain.
    
    This function computes normal vector components at a grain boundary location
    by applying bilinear smoothing operations to a 2×2 neighborhood. The smoothing
    is grain-aware, ensuring sharp grain boundary preservation.
    
    Parameters:
    -----------
    P : ndarray
        2D grain ID map with shape (nx, ny)
    i, j : int
        Top-left corner coordinates of 2×2 neighborhood to analyze
    iteration : int
        Smoothing iteration parameter for algorithm accuracy
    refer_id : int
        Reference grain ID for grain-aware smoothing
        
    Returns:
    --------
    a, b : float
        Normal vector components (a = i-direction, b = j-direction)
        Represent local interface orientation at the specified location
        
    Algorithm:
    ----------
    1. Generate bilinear smoothing gradient matrices
    2. Extract grain-aware windows for 2×2 neighborhood points
    3. Apply smoothing kernels to each of the 4 corner points
    4. Average results across 2×2 neighborhood for stability
    5. Return averaged normal vector components
    
    Bilinear Smoothing Process:
    --------------------------
    For each corner point (i+di, j+dj) where di,dj ∈ {0,1}:
    - Extract grain-aware smoothing window
    - Apply gradient calculation kernels (smoothed_vector_i/j)
    - Compute local normal vector components
    - Average across all 4 corners for final result
    
    Mathematical Foundation:
    -----------------------
    The 2×2 averaging reduces noise and provides more stable normal
    vector estimates compared to single-point calculations.
    
    Usage:
    ------
    Core function for normal vector calculation in triple junction analysis.
    Provides grain-boundary-aware normal vectors essential for accurate
    dihedral angle computation.
    """
    # Generate bilinear smoothing gradient matrices
    smoothed_vector_i, smoothed_vector_j = myInput.output_linear_vector_matrix(iteration)
    
    # Calculate normal vector i-component (vertical gradient)
    # Average across 2×2 neighborhood for stability
    a = (np.sum(find_window(P,i,j,iteration,refer_id) * smoothed_vector_i) + 
        np.sum(find_window(P,i+1,j,iteration,refer_id) * smoothed_vector_i) +  
        np.sum(find_window(P,i,j+1,iteration,refer_id) * smoothed_vector_i) + 
        np.sum(find_window(P,i+1,j+1,iteration,refer_id) * smoothed_vector_i)) / 4
    
    # Calculate normal vector j-component (horizontal gradient)  
    # Average across 2×2 neighborhood for stability
    b = (np.sum(find_window(P,i,j,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window(P,i+1,j,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window(P,i,j+1,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window(P,i+1,j+1,iteration,refer_id) * smoothed_vector_j)) / 4
    
    return a, b

def find_normal(P,i,j,nei_flat,iteration):
    """
    Calculate normal vectors for all grain boundaries meeting at a triple junction.
    
    This function identifies the configuration of a triple junction and computes
    the appropriate normal vectors for dihedral angle analysis. It handles different
    triple junction orientations and ensures proper normal vector assignments.
    
    Parameters:
    -----------
    P : ndarray
        2D grain ID map with shape (nx, ny)
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
        
    Algorithm:
    ----------
    1. Identify triple junction configuration by analyzing grain ID pattern
    2. Calculate normal vectors for each pixel using grain-aware smoothing
    3. Assign appropriate reference grain IDs for each normal calculation
    4. Return organized normal vectors and grain IDs
    
    Triple Junction Configurations:
    ------------------------------
    The function handles multiple possible arrangements of three grains
    meeting at a 2×2 pixel neighborhood. Different conditional branches
    identify the specific configuration and assign normal vectors accordingly.
    
    Mathematical Foundation:
    -----------------------
    Each normal vector is calculated using bilinear smoothing with respect
    to the appropriate reference grain, ensuring sharp grain boundary
    preservation and accurate geometric representation.
    
    Usage:
    ------
    Core function for triple junction analysis. Results feed directly into
    dihedral angle calculation algorithms.
    """
    # Calculate the nomals for all the four voxels in the triple junction
    nx,ny=P.shape
    tri_norm = np.zeros((4,2))
    tri_grains = np.zeros(3)
    
    if nei_flat[0] == nei_flat[1]:

        tri_norm[0,0], tri_norm[0,1] = find_normal_structure(P,i,j,iteration,P[i,j])
        tri_norm[1,0], tri_norm[1,1] = find_normal_structure(P,i,j,iteration,P[i,j+1])
        tri_norm[2,0], tri_norm[2,1] = find_normal_structure(P,i,j,iteration,P[i+1,j+1])
        tri_norm[3,0], tri_norm[3,1] = find_normal_structure(P,i,j,iteration,P[i+1,j])
        tri_grains = np.array([P[i,j+1], P[i+1,j+1], P[i+1,j]])
        
        
    elif nei_flat[0] == nei_flat[2]:
        
        tri_norm[0,0], tri_norm[0,1] = find_normal_structure(P,i,j,iteration,P[i+1,j])
        tri_norm[1,0], tri_norm[1,1] = find_normal_structure(P,i,j,iteration,P[i,j])
        tri_norm[2,0], tri_norm[2,1] = find_normal_structure(P,i,j,iteration,P[i,j+1])
        tri_norm[3,0], tri_norm[3,1] = find_normal_structure(P,i,j,iteration,P[i+1,j+1])
        tri_grains = np.array([P[i,j], P[i,j+1], P[i+1,j+1]])
        
    elif nei_flat[2] == nei_flat[3]:
        
        tri_norm[0,0], tri_norm[0,1] = find_normal_structure(P,i,j,iteration,P[i+1,j+1])
        tri_norm[1,0], tri_norm[1,1] = find_normal_structure(P,i,j,iteration,P[i+1,j])
        tri_norm[2,0], tri_norm[2,1] = find_normal_structure(P,i,j,iteration,P[i,j])
        tri_norm[3,0], tri_norm[3,1] = find_normal_structure(P,i,j,iteration,P[i,j+1])
        tri_grains = np.array([P[i+1,j], P[i,j], P[i,j+1]])
        
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
    Calculate dihedral angles at a triple junction using tangent vector method.
    
    This function computes the three dihedral angles at a triple junction by:
    1. Converting normal vectors to tangent vectors via 90° rotations
    2. Computing angles between adjacent tangent vectors
    3. Ensuring angle sum equals 360° (correcting for orientation if needed)
    
    Parameters:
    -----------
    each_normal : ndarray, shape (4,2)
        Normalized normal vectors from the 4 pixels of the triple junction
        each_normal[i] = [ni_x, ni_y] for i = 0,1,2,3
        
    Returns:
    --------
    tri_angle : ndarray, shape (3,)
        Three dihedral angles in degrees
        Sum should equal 360° for valid triple junction
        
    Algorithm:
    ----------
    1. Convert normal vectors to tangent vectors:
       - tang[0] = normal[0] rotated 90° clockwise
       - tang[1] = normal[1] rotated 90° counter-clockwise  
       - tang[2] = -(normal[2] + normal[3]) normalized
    
    2. Calculate angles between tangent pairs:
       - angle[0] = angle(tang[0], tang[2])
       - angle[1] = angle(tang[1], tang[2])
       - angle[2] = angle(tang[0], tang[1])
    
    3. Validate and correct angle sum to 360°
    
    Mathematical Foundation:
    -----------------------
    Rotation matrices:
    - Clockwise 90°: [[0,-1],[1,0]]
    - Counter-clockwise 90°: [[0,1],[-1,0]]
    
    Angle calculation: θ = arccos(tang_i · tang_j)
    
    Physical Interpretation:
    -----------------------
    Dihedral angles represent the angles between grain boundaries meeting
    at the triple junction. The sum constraint (360°) ensures geometric
    consistency.
    
    Usage:
    ------
    Primary method for dihedral angle calculation in triple junction analysis.
    Results are used for grain boundary energy analysis and microstructure
    characterization.
    """
    # Find the three tangent depend on the four normals from four voxels
    tri_tang = np.zeros((3,2))
    tri_angle = np.zeros(3)
    
    # Define rotation matrices for 90° rotations
    clock90 = np.array([[0,-1],[1,0]])           # Clockwise 90°
    anti_clock90 = np.array([[0,1],[-1,0]])      # Counter-clockwise 90°
    
    # Convert normal vectors to tangent vectors
    tri_tang[0] = each_normal[0]@clock90         # Rotate normal[0] clockwise 90°
    tri_tang[1] = each_normal[1]@anti_clock90    # Rotate normal[1] counter-clockwise 90°
    tri_tang[2] = -(each_normal[2]+each_normal[3])/np.linalg.norm(each_normal[2]+each_normal[3])  # Average and normalize
    
    # Calculate dihedral angles using dot product
    tri_angle[0] = 180 / np.pi * math.acos(np.dot(tri_tang[0], tri_tang[2]))
    tri_angle[1] = 180 / np.pi * math.acos(np.dot(tri_tang[1], tri_tang[2]))
    tri_angle[2] = 180 / np.pi * math.acos(round(np.dot(tri_tang[0], tri_tang[1]),5))
    
    # Ensure angle sum equals 360° (correct for orientation if needed)
    if abs(sum(tri_angle) - 360) > 5:
        tri_angle[2] = 360 - tri_angle[2]
    
    return tri_angle

def find_angle(each_normal):
    """
    Calculate dihedral angles using direct normal vector method.
    
    This function provides an alternative approach to dihedral angle calculation
    by working directly with normal vectors rather than converting to tangent vectors.
    Uses geometric relationships and angle doubling formulas.
    
    Parameters:
    -----------
    each_normal : ndarray, shape (4,2)
        Normalized normal vectors from the 4 pixels of the triple junction
        each_normal[i] = [ni_x, ni_y] for i = 0,1,2,3
        
    Returns:
    --------
    tri_angle : ndarray, shape (3,)
        Three dihedral angles in degrees
        
    Algorithm:
    ----------
    1. Compute third normal as average of first two normals
    2. Calculate angles using doubled angle formula: 2π - 2*arccos(n·m)
    3. Direct angle calculation between normals 2 and 3
    
    Mathematical Foundation:
    -----------------------
    The algorithm uses the relationship:
    dihedral_angle = 2π - 2*arccos(n1·n2)
    
    where n1, n2 are unit normal vectors to adjacent grain boundaries.
    
    Note:
    -----
    This method provides an alternative to the tangent vector approach.
    Some special cases may require additional handling (see commented code).
    
    Usage:
    ------
    Alternative method for dihedral angle calculation. Can be used for
    validation or in cases where tangent vector method encounters issues.
    """
    tri_angle = np.zeros(3)
    
    # Calculate third normal as normalized average of first two
    third_normal = (each_normal[0]+each_normal[1])/np.linalg.norm(each_normal[0]+each_normal[1])
    
    # Calculate dihedral angles using doubled angle formula
    tri_angle[1] = 180 / np.pi * (2 * np.pi - 2 * math.acos(np.dot(each_normal[3], third_normal)))
    tri_angle[2] = 180 / np.pi * (2 * np.pi - 2 * math.acos(np.dot(each_normal[2], third_normal)))
    tri_angle[0] = 180 / np.pi * (2 * np.pi - 2 * math.acos(np.dot(each_normal[2], each_normal[3])))
    
    
    # A way to ignore the issue of some special triple angle
    # if abs(sum(tri_angle)-360) > 5:
        # print()
        # print(sum(tri_angle))
        # print(tri_angle)
        # print(each_normal)
        # tri_angle[2] = 360 - tri_angle[0] - tri_angle[1]
    
    return tri_angle

def read_2d_input(filename,nx,ny):
    """
    Read 2D microstructure data from input file.
    
    This function reads microstructure data from a formatted input file
    and creates a 2D grain ID map for analysis.
    
    Parameters:
    -----------
    filename : str
        Path to input file containing microstructure data
    nx, ny : int
        Dimensions of the microstructure grid
        
    Returns:
    --------
    triple_map : ndarray, shape (nx, ny)
        2D array containing grain IDs for each pixel
        
    File Format:
    -----------
    Expected input file format:
    - First line: header (ignored)
    - Second line: header (ignored)  
    - Data lines: i j ... ... ... ... grain_id
    
    The function reads coordinates (i, j) and grain ID from each line,
    converting from 1-indexed file format to 0-indexed array indexing.
    
    Usage:
    ------
    Utility function for loading microstructure data from external files.
    Commonly used with SPPARKS output or other simulation data formats.
    """
    # Keep the 2d input macrostructure
    
    triple_map = np.zeros((nx,ny))
    f = open(filename)
    line = f.readline()  # Skip first header line
    line = f.readline()  # Skip second header line
    # print(line)
    while line:
        each_element = line.split()
        i = int(each_element[0])-1  # Convert from 1-indexed to 0-indexed
        j = int(each_element[1])-1  # Convert from 1-indexed to 0-indexed
        triple_map[i,j] = int(each_element[6])  # Grain ID is in 7th column
    
        line = f.readline()
    f.close()
    
    return triple_map

def calculate_tangent(triple_map,iteration=5):
    """
    Calculate dihedral angles for all triple junctions in a 2D microstructure.
    
    This is the main function that systematically scans a 2D microstructure
    to identify triple junctions and calculate their dihedral angles using
    bilinear smoothing and normal vector analysis.
    
    Parameters:
    -----------
    triple_map : ndarray, shape (nx, ny)
        2D grain ID map representing the microstructure
    iteration : int, default=5
        Smoothing iteration parameter for bilinear algorithm accuracy
        Higher values provide more accurate but computationally expensive results
        
    Returns:
    --------
    triple_coord : ndarray, shape (n_triples, 2)
        Coordinates of triple junction locations (top-left corner of 2×2 neighborhood)
    triple_angle : ndarray, shape (n_triples, 3)
        Dihedral angles for each triple junction (in degrees)
    triple_grain : ndarray, shape (n_triples, 3)
        Grain IDs involved in each triple junction
        
    Algorithm:
    ----------
    1. Scan entire microstructure with 2×2 sliding window
    2. Identify locations where exactly 3 distinct grains meet
    3. Calculate normal vectors using bilinear smoothing
    4. Compute dihedral angles from normal vectors
    5. Validate results and report statistics
    
    Triple Junction Detection:
    -------------------------
    A valid triple junction is identified when:
    - A 2×2 neighborhood contains exactly 3 distinct grain IDs
    - No background (ID=0) pixels are present
    - Proper geometric configuration exists
    
    Quality Control:
    ---------------
    The function tracks:
    - Total number of triple junctions found
    - Percentage with angle sum issues (deviation from 360°)
    - Error reporting for invalid configurations
    
    Mathematical Foundation:
    -----------------------
    Uses bilinear smoothing for grain-boundary-aware normal vector calculation,
    ensuring accurate geometric representation of grain boundaries and
    reliable dihedral angle measurements.
    
    Usage:
    ------
    Primary analysis function for triple junction characterization in
    polycrystalline microstructures. Results are used for grain boundary
    energy analysis and microstructure validation.
    """
    nx, ny = triple_map.shape
    num = 0                    # Counter for valid triple junctions
    issue_num = 0             # Counter for problematic angle calculations
    triple_grain = []         # List to store grain ID sequences
    triple_coord = []         # List to store triple junction coordinates
    triple_normal = []        # List to store normal vector arrays
    triple_angle = []         # List to store dihedral angle arrays
    triple_angle_tang = []    # List for tangent-based angle calculations
    
    # Scan microstructure with 2×2 sliding window
    for i in range(nx-1):
        for j in range(ny-1):
            # Extract 2×2 neighborhood
            nei = np.zeros((2,2))
            nei = triple_map[i:i+2,j:j+2]
            nei_flat = nei.flatten()
            
            # Check if this is a valid triple junction
            if len(set(nei_flat)) == 3 and 0 not in nei_flat:
                
                # print(str(i)+" "+str(j))  # Debug output
                
                # Calculate normal vectors and grain sequence for this triple junction
                each_normal, grain_sequence = find_normal(triple_map,i,j,nei_flat,iteration) # Get basic normals and grain id sequence
                if isinstance(each_normal,(int, float)): continue  # Skip if calculation failed
                
                # Store results for this triple junction
                triple_normal.append(each_normal)           # Save the normal vectors
                triple_coord.append(np.array([i,j]))        # Save the coordinate of the triple point
                triple_grain.append(grain_sequence)         # Save the grain id sequence
                triple_angle.append(find_angle(each_normal)) # Save the 3 dihedral angles

                num += 1 # Increment count of valid triple junctions
                
                # Check for angle sum issues (should be 360°)
                if abs(sum(find_angle(each_normal))-360) > 5: 
                    issue_num += 1
            
    # Report analysis statistics
    print(f"The number of useful triple junction is {num}")
    if num==0: 
        print("The issue propotion is 0%")
    else: 
        print(f"The issue propotion is {issue_num/num*100}%")
    
    return np.array(triple_coord), np.array(triple_angle), np.array(triple_grain)



if __name__ == '__main__':
    """
    Main execution block for triple junction dihedral angle analysis.
    
    This section demonstrates the complete workflow for analyzing dihedral angles
    in polycrystalline microstructures using the VECTOR framework with bilinear
    smoothing algorithms.
    
    Workflow:
    ---------
    1. Load microstructure data from .npy files
    2. Set analysis parameters (iteration count, grain numbers)
    3. Calculate dihedral angles for all time steps
    4. Compare with reference results (Joseph's data)
    5. Generate error statistics and visualizations
    
    Data Structure:
    --------------
    - Input: 4D array (time, nx, ny, components)
    - Processing: 2D slices for each time step
    - Output: Arrays of coordinates, angles, and grain IDs
    
    Validation:
    ----------
    Results are compared against reference calculations to ensure
    accuracy and consistency of the bilinear smoothing approach.
    """
    
    # File paths for input data and reference results
    file_path = "/Users/lin/Dropbox (UFL)/UFdata/Dihedral_angle/IC/"
    npy_file_name = "h_ori_ave_aveE_hex_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066_angle.npy"

    # Joseph's reference results for validation
    file_path_joseph = "/Users/lin/Dropbox (UFL)/UFdata/Dihedral_angle/output/"    
    npy_file_name_joseph = "hex_dihedrals.npy" # (161, 6, 96) grain ID's involved (first 3) and the dihedral angles (last 3)
    triple_results = np.load(file_path_joseph + npy_file_name_joseph)
    
    # Analysis parameters
    iteration = 5              # Bilinear smoothing iteration parameter
    num_grain_initial = 48    # Initial number of grains in microstructure
    
    # Load microstructure data
    triple_map = np.load(file_path + npy_file_name)
    triple_map = triple_map[:,:,:,0]  # Extract grain ID component
    
    # Main analysis function call (commented for demonstration)
    # triple_coord, triple_angle, triple_grain = calculate_tangent(triple_map, iteration)
    # 
    # Output data structure:
    # triple_coord: Coordinates of triple junctions (left-upper voxel)
    #   - axis 0: index of triple junction
    #   - axis 1: coordinate (i,j)
    # 
    # triple_angle: Three dihedral angles for each triple junction
    #   - axis 0: index of triple junction
    #   - axis 1: three dihedral angles
    # 
    # triple_grain: Sequence of three grains for each triple junction
    #   - axis 0: index of triple junction
    #   - axis 1: three grain IDs
    
    # Time-series analysis setup
    num_steps = triple_map.shape[0]
    error_list = np.zeros(num_steps)         # Error tracking for our method
    error_list_joseph = np.zeros(num_steps)  # Error tracking for reference method
    angle_list = np.zeros(num_steps)         # Angle statistics
    angle_list_joseph = np.zeros(num_steps)
    for i in tqdm(range(num_steps)):
        triple_map_step = triple_map[i,:,:]
        num_grains = len(list(set(list(triple_map_step.reshape(-1)))))
        if num_grains < num_grain_initial: break
    
        triple_coord, triple_angle, triple_grain = calculate_tangent(triple_map_step, iteration)
        error_list[i] = np.mean(abs((triple_angle) - 120))
        angle_list[i] = np.mean(triple_angle[:,0])
        
        # From Joseph
        triple_results_step = triple_results[i,3:6,:]
        triple_results_step = triple_results_step[:,~np.isnan(triple_results_step[0,:])]
        print(f"The number in Joseph is {len(triple_results_step[0,:])}")
        error_list_joseph[i] = np.mean(abs((triple_results_step) - 120))
        angle_list_joseph[i] = np.mean(triple_results_step[0,:])
        
        
    plt.close()
    plt.plot(np.linspace(0,num_steps-1,num_steps), error_list, linewidth = 2, label='Linear algorithm error')
    plt.plot(np.linspace(0,num_steps-1,num_steps), error_list_joseph, linewidth = 2, label='Joseph algorithm error')
    plt.xlabel("Time step", fontsize=20)
    plt.ylabel(r"angle error ($^\circ$)", fontsize=20)
    plt.legend(fontsize=20)
    plt.show()
    
    print(r"Analytical result is 120$^\circ$")
    print(f"Linear result is {np.mean(angle_list[:80])}, Joseph result is {np.mean(angle_list_joseph[:80])}.")
    
    






