#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Grain Boundary Normal Vector Distribution Analysis for HiPerGator
==================================================================

This script analyzes the distribution of grain boundary normal vectors in 3D 
microstructures using data from SPPARKS Monte Carlo simulations run on the 
University of Florida HiPerGator supercomputing cluster.

The analysis focuses on comparing different energy formulations:
- Isotropic (iso): Reference case with delta=0.0
- Anisotropic ave: Average energy with delta=0.6
- Anisotropic min: Minimum energy with delta=0.6  
- Anisotropic max: Maximum energy with delta=0.6

The script generates polar plots showing grain boundary normal distributions
in three 2D projections (XY, XZ, YZ) both with and without bias correction.

Created on Mon Jul 31 14:33:57 2023
@author: Lin
Last modified: For comprehensive 3D grain boundary orientation analysis
"""

# ============================================================================
# IMPORT SECTION: Core Libraries and Path Configuration
# ============================================================================

import os
current_path = os.getcwd()  # Get current working directory for relative path operations

# Scientific computing and numerical analysis libraries
import numpy as np  # Fundamental package for scientific computing with Python
from numpy import seterr  # NumPy error handling configuration
seterr(all='raise')  # Configure NumPy to raise exceptions on all floating-point errors

# Visualization and plotting libraries
import matplotlib.pyplot as plt  # Comprehensive plotting library for 2D graphics

# Mathematical operations and progress tracking
import math  # Built-in mathematical functions (trigonometry, constants, etc.)
from tqdm import tqdm  # Fast, extensible progress bar library

# Python system and path management
import sys  # System-specific parameters and functions
sys.path.append(current_path)  # Add current directory to Python path
sys.path.append(current_path+'/../../')  # Add parent directories for VECTOR framework access

# VECTOR framework imports for grain boundary analysis
import myInput  # Custom input/output utilities for grain data processing
import PACKAGE_MP_Linear as linear2d  # 2D linear grain boundary analysis package
import PACKAGE_MP_3DLinear as linear3d  # 3D linear grain boundary analysis package

# Additional path for tangent calculation utilities
sys.path.append(current_path+'/../calculate_tangent/')  # Access to specialized calculation modules

# ============================================================================
# FUNCTION DEFINITIONS: 2D Grain Boundary Analysis
# ============================================================================

def get_normal_vector(grain_structure_figure_one):
    """
    Calculate normal vectors for grain boundaries in 2D microstructures.
    
    This function processes a 2D grain structure to identify grain boundaries
    and compute their normal vectors using the VECTOR framework's linear 
    smoothing algorithm.
    
    Parameters:
    -----------
    grain_structure_figure_one : numpy.ndarray
        2D array representing the grain structure where each element contains
        a grain ID number. Shape: (nx, ny)
    
    Returns:
    --------
    tuple: (P, sites_together)
        P : numpy.ndarray
            Processed grain structure with computed inclination data
        sites_together : list
            Flattened list of all grain boundary site coordinates
    
    Algorithm Overview:
    -------------------
    1. Extract microstructure dimensions and grain count
    2. Initialize linear smoothing class with multicore processing
    3. Execute inclination calculation using linear algorithm
    4. Collect all grain boundary sites from individual grain boundaries
    5. Return processed data and boundary site coordinates
    """
    # Extract microstructure dimensions
    nx = grain_structure_figure_one.shape[0]  # Grid size in x-direction
    ny = grain_structure_figure_one.shape[1]  # Grid size in y-direction
    ng = np.max(grain_structure_figure_one)   # Maximum grain ID (total grain count)
    
    # Configure processing parameters for optimal performance
    cores = 8        # Number of CPU cores for parallel processing
    loop_times = 5   # Number of smoothing iterations for convergence
    
    # Prepare input data structures
    P0 = grain_structure_figure_one  # Original grain structure data
    R = np.zeros((nx,ny,2))          # Initialize result array for 2D vectors
    
    # Initialize the VECTOR linear smoothing class for 2D analysis
    smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)

    # Execute the main inclination calculation algorithm
    smooth_class.linear_main("inclination")
    P = smooth_class.get_P()  # Retrieve processed grain structure with inclinations
    
    # Collect grain boundary sites from all grains
    # Note: Commented legacy code shows alternative grain-by-grain approach
    # sites = smooth_class.get_gb_list(1)
    # print(len(sites))
    # for id in range(2,grain_num+1): sites += smooth_class.get_gb_list(id)
    # print(len(sites))
    
    # Use optimized method to get all grain boundary sites at once
    sites = smooth_class.get_all_gb_list()  # Returns list of lists (one per grain)
    sites_together = []  # Initialize flattened site list
    
    # Flatten the nested list structure to create single list of boundary sites
    for id in range(len(sites)): 
        sites_together += sites[id]
    
    # Output total number of grain boundary sites for verification
    print("Total num of GB sites: " + str(len(sites_together)))

    return P, sites_together

# ============================================================================
# FUNCTION DEFINITIONS: 3D Grain Boundary Analysis
# ============================================================================

def get_normal_vector_3d(grain_structure_figure_one):
    """
    Calculate normal vectors for grain boundaries in 3D microstructures.
    
    This function extends the 2D analysis to three dimensions, processing 
    volumetric grain structures to identify grain boundaries and compute 
    their normal vectors using the VECTOR framework's 3D linear smoothing algorithm.
    
    Parameters:
    -----------
    grain_structure_figure_one : numpy.ndarray
        3D array representing the grain structure where each element contains
        a grain ID number. Shape: (nx, ny, nz)
    
    Returns:
    --------
    tuple: (P, sites_together)
        P : numpy.ndarray
            Processed 3D grain structure with computed inclination data
        sites_together : list
            Flattened list of all 3D grain boundary site coordinates
    
    Algorithm Overview:
    -------------------
    1. Extract 3D microstructure dimensions and grain count
    2. Initialize 3D linear smoothing class with multicore processing
    3. Execute 3D inclination calculation using linear algorithm
    4. Collect all grain boundary sites from 3D volume
    5. Return processed data and 3D boundary site coordinates
    
    Key Differences from 2D:
    -------------------------
    - Handles additional z-dimension for volumetric analysis
    - Uses 3D gradient calculations for normal vector computation
    - Processes significantly larger datasets (3D volumes vs 2D slices)
    - Requires expanded memory allocation for 3D vector fields
    """
    # Extract 3D microstructure dimensions
    nx = grain_structure_figure_one.shape[0]  # Grid size in x-direction
    ny = grain_structure_figure_one.shape[1]  # Grid size in y-direction  
    nz = grain_structure_figure_one.shape[2]  # Grid size in z-direction
    ng = np.max(grain_structure_figure_one)   # Maximum grain ID (total grain count)
    
    # Configure processing parameters for 3D analysis
    cores = 8        # Number of CPU cores for parallel processing (critical for 3D)
    loop_times = 5   # Number of smoothing iterations for convergence
    
    # Prepare 3D input data structures
    P0 = grain_structure_figure_one[:,:,:,np.newaxis]  # Add extra dimension for processing
    R = np.zeros((nx,ny,nz,3))  # Initialize result array for 3D vectors (x,y,z components)
    
    # Initialize the VECTOR 3D linear smoothing class
    # 'np' parameter specifies NumPy backend for computations
    smooth_class = linear3d.linear3d_class(nx,ny,nz,ng,cores,loop_times,P0,R,'np')

    # Execute the main 3D inclination calculation algorithm
    smooth_class.linear3d_main("inclination")
    P = smooth_class.get_P()  # Retrieve processed 3D grain structure with inclinations
    
    # Collect all 3D grain boundary sites
    sites = smooth_class.get_all_gb_list()  # Returns list of lists (one per grain)
    sites_together = []  # Initialize flattened site list
    
    # Flatten the nested list structure for 3D boundary sites
    for id in range(len(sites)): 
        sites_together += sites[id]
    
    # Output total number of 3D grain boundary sites for verification
    print(f"Total num of GB sites: {len(sites_together)}")

    return P, sites_together

# ============================================================================
# FUNCTION DEFINITIONS: 2D Normal Vector Distribution Analysis
# ============================================================================

def get_normal_vector_slope(P, sites, step, para_name):
    """
    Calculate and plot the angular distribution of grain boundary normal vectors in 2D.
    
    This function computes the distribution of grain boundary orientations by analyzing
    the angles of normal vectors at grain boundary sites. The results are binned into
    angular intervals and plotted as a polar histogram.
    
    Parameters:
    -----------
    P : numpy.ndarray
        Processed grain structure with computed inclination data from get_normal_vector()
    sites : list
        List of grain boundary site coordinates [(i1,j1), (i2,j2), ...]
    step : int
        Current simulation time step (for labeling purposes)
    para_name : str
        Parameter name for plot legend (e.g., "Iso", "Ave", "Min", "Max")
    
    Returns:
    --------
    int : 0 (success indicator)
    
    Algorithm Overview:
    -------------------
    1. Define angular range (0-360°) and bin size for histogram
    2. Calculate gradient (dx, dy) at each grain boundary site
    3. Convert gradients to angles using atan2 function
    4. Bin angles into histogram with normalization
    5. Plot results on current polar axes
    
    Mathematical Details:
    ---------------------
    - Uses atan2(-dy, dx) + π to map angles to [0, 2π] range
    - Bins are 10.01° wide to avoid edge effects
    - Frequency normalization: freqArray = freqArray/sum(freqArray*binValue)
    - Converts degrees to radians for polar plotting
    """
    # Define angular range and binning parameters
    xLim = [0, 360]    # Angular range in degrees (full circle)
    binValue = 10.01   # Bin width in degrees (slightly larger than 10° to avoid edge effects)
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)  # Number of bins
    
    # Calculate bin center positions for plotting
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)

    # Initialize arrays for angle distribution
    freqArray = np.zeros(binNum)  # Frequency array for histogram
    degree = []                   # List to store calculated angles
    
    # Process each grain boundary site
    for sitei in sites:
        [i,j] = sitei  # Extract site coordinates
        
        # Calculate gradient components at the site using myInput utility
        dx,dy = myInput.get_grad(P,i,j)
        
        # Convert gradient to angle (normal vector orientation)
        # atan2(-dy, dx) gives angle in [-π, π], adding π shifts to [0, 2π]
        degree.append(math.atan2(-dy, dx) + math.pi)
    
    # Bin the angles into histogram
    for i in range(len(degree)):
        # Convert angle to bin index and increment frequency
        freqArray[int((degree[i]/math.pi*180-xLim[0])/binValue)] += 1
    
    # Normalize frequency array to create probability density
    freqArray = freqArray/sum(freqArray*binValue)

    # Plot the distribution on polar axes
    plt.plot(xCor/180*math.pi, freqArray, linewidth=2, label=para_name)

    return 0

# ============================================================================
# FUNCTION DEFINITIONS: 3D Normal Vector Distribution Analysis
# ============================================================================

def get_normal_vector_slope_3d(P, sites, step, para_name, angle_index=0, bias=None):
    """
    Calculate and plot the angular distribution of grain boundary normal vectors in 3D.
    
    This function extends the 2D analysis to three dimensions by projecting 3D normal
    vectors onto 2D planes (XY, XZ, or YZ) and computing angular distributions.
    Optional bias correction can be applied to account for geometric effects.
    
    Parameters:
    -----------
    P : numpy.ndarray
        Processed 3D grain structure with computed inclination data
    sites : list
        List of 3D grain boundary site coordinates [(i1,j1,k1), (i2,j2,k2), ...]
    step : int
        Current simulation time step (for labeling purposes)
    para_name : str
        Parameter name for plot legend (e.g., "Iso", "Ave", "Min", "Max")
    angle_index : int, optional (default=0)
        Projection plane selector:
        - 0: XY plane (dx, dy components)
        - 1: XZ plane (dx, dz components)
        - 2: YZ plane (dy, dz components)
    bias : numpy.ndarray, optional (default=None)
        Bias correction array to subtract from frequency distribution
    
    Returns:
    --------
    numpy.ndarray
        Normalized frequency array representing the angular distribution
    
    Algorithm Overview:
    -------------------
    1. Define angular range (0-360°) and bin size for histogram
    2. Calculate 3D gradient (dx, dy, dz) at each grain boundary site
    3. Project 3D gradient onto selected 2D plane
    4. Normalize projected gradient components
    5. Convert to angles and bin into histogram
    6. Apply bias correction if provided
    7. Plot results with periodic boundary conditions
    
    Mathematical Details:
    ---------------------
    - 3D gradients projected to 2D: (dx_fake, dy_fake)
    - Normalization: magnitude = sqrt(dx_fake² + dy_fake²)
    - Skip sites with near-zero magnitude (< 1e-5) to avoid singularities
    - Angle calculation: atan2(-dy_fake_norm, dx_fake_norm) + π
    - Periodic plotting: append first value to close polar curve
    
    Bias Correction:
    ----------------
    - Geometric bias arises from discrete sampling on regular grids
    - Bias array represents expected deviation from uniform distribution
    - Correction: freqArray_corrected = freqArray_raw + bias
    - Re-normalization ensures probability conservation
    """
    # Define angular range and binning parameters
    xLim = [0, 360]    # Angular range in degrees (full circle)
    binValue = 10.01   # Bin width in degrees (slightly larger than 10° to avoid edge effects)
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)  # Number of bins
    
    # Calculate bin center positions for plotting
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)

    # Initialize arrays for angle distribution
    freqArray = np.zeros(binNum)  # Frequency array for histogram
    degree = []                   # List to store calculated angles
    
    # Process each 3D grain boundary site
    for sitei in sites:
        [i,j,k] = sitei  # Extract 3D site coordinates
        
        # Calculate 3D gradient components at the site using myInput utility
        dx,dy,dz = myInput.get_grad3d(P,i,j,k)
        
        # Project 3D gradient onto selected 2D plane
        if angle_index == 0:      # XY plane projection
            dx_fake = dx
            dy_fake = dy
        elif angle_index == 1:    # XZ plane projection
            dx_fake = dx
            dy_fake = dz
        elif angle_index == 2:    # YZ plane projection
            dx_fake = dy
            dy_fake = dz

        # Normalize projected gradient to unit vector
        magnitude = math.sqrt(dy_fake**2+dx_fake**2)
        if magnitude < 1e-5: 
            continue  # Skip sites with near-zero gradient (numerical singularities)
            
        # Calculate normalized components
        dy_fake_norm = dy_fake / magnitude
        dx_fake_norm = dx_fake / magnitude

        # Convert normalized gradient to angle (normal vector orientation)
        # atan2(-dy_fake_norm, dx_fake_norm) gives angle in [-π, π], adding π shifts to [0, 2π]
        degree.append(math.atan2(-dy_fake_norm, dx_fake_norm) + math.pi)
    
    # Bin the angles into histogram
    for n in range(len(degree)):
        # Convert angle to bin index and increment frequency
        freqArray[int((degree[n]/math.pi*180-xLim[0])/binValue)] += 1
    
    # Normalize frequency array to create probability density
    freqArray = freqArray/sum(freqArray*binValue)

    # Apply bias correction if provided
    if bias is not None:
        freqArray = freqArray + bias      # Add bias correction
        freqArray = freqArray/sum(freqArray*binValue)  # Re-normalize after correction

    # Plot the distribution on polar axes with periodic boundary conditions
    # Append first value to close the polar curve for continuous appearance
    plt.plot(np.append(xCor,xCor[0])/180*math.pi, np.append(freqArray,freqArray[0]),linewidth=2,label=para_name)

    return freqArray

# ============================================================================
# MAIN EXECUTION: HiPerGator 3D Grain Boundary Analysis Pipeline
# ============================================================================

if __name__ == '__main__':
    # ========================================================================
    # DATA LOADING: HiPerGator Simulation Results
    # ========================================================================
    
    # Define file paths and naming conventions for HiPerGator cluster data
    # HiPerGator path: University of Florida supercomputing cluster storage
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_poly/results/"
    
    # Energy formulation types for anisotropic grain boundary analysis
    TJ_energy_type_ave = "ave"  # Average energy method
    TJ_energy_type_min = "min"  # Minimum energy method
    TJ_energy_type_max = "max"  # Maximum energy method

    # Construct file names following HiPerGator naming convention
    # Format: p[2]_ori_ave_{energy_type}E_{grid_size}_{initial_grains}_multiCore{cores}_delta{delta}_m{mobility}_J{coupling}_refer_{orientation}_seed{seed}_kt{temperature}.npy
    npy_file_name_aniso_ave = f"p2_ori_ave_{TJ_energy_type_ave}E_264_5k_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt1.95.npy"
    npy_file_name_aniso_min = f"p_ori_ave_{TJ_energy_type_min}E_264_5k_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt1.95.npy"
    npy_file_name_aniso_max = f"p_ori_ave_{TJ_energy_type_max}E_264_5k_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt1.95.npy"
    
    # Isotropic reference case (delta=0.0 for no anisotropy)
    npy_file_name_iso = "p_ori_ave_aveE_264_5k_multiCore64_delta0.0_m2_J1_refer_1_0_0_seed56689_kt1.95.npy"

    # Load simulation data from HiPerGator storage
    # Data format: 4D arrays (time_steps, nx, ny, nz) containing grain IDs
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)  # Anisotropic average energy
    npy_file_aniso_min = np.load(npy_file_folder + npy_file_name_aniso_min)  # Anisotropic minimum energy
    npy_file_aniso_max = np.load(npy_file_folder + npy_file_name_aniso_max)  # Anisotropic maximum energy
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)              # Isotropic reference
    
    # Display loaded data dimensions for verification
    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    print(f"The min data size is: {npy_file_aniso_min.shape}")
    print(f"The max data size is: {npy_file_aniso_max.shape}")
    print(f"The iso data size is: {npy_file_iso.shape}")
    print("READING DATA DONE")

    # ========================================================================
    # GRAIN EVOLUTION ANALYSIS: Temporal Grain Count Tracking
    # ========================================================================
    
    # Initialize parameters for grain evolution analysis
    initial_grain_num = 5000                    # Initial number of grains in simulation
    step_num = npy_file_aniso_min.shape[0]      # Total number of time steps in simulation
    
    # Initialize arrays to track grain count evolution for each energy method
    grain_num_aniso_ave = np.zeros(step_num)    # Grain count vs time for average energy
    grain_num_aniso_min = np.zeros(step_num)    # Grain count vs time for minimum energy
    grain_num_aniso_max = np.zeros(step_num)    # Grain count vs time for maximum energy
    grain_num_iso = np.zeros(step_num)          # Grain count vs time for isotropic reference

    # Calculate grain count at each time step for all energy methods
    # Method: Count unique grain IDs in flattened 3D structure
    for i in range(step_num):
        grain_num_aniso_ave[i] = len(set(npy_file_aniso_ave[i,:].flatten()))  # Unique grain IDs
        grain_num_aniso_min[i] = len(set(npy_file_aniso_min[i,:].flatten()))  # Unique grain IDs
        grain_num_aniso_max[i] = len(set(npy_file_aniso_max[i,:].flatten()))  # Unique grain IDs
        grain_num_iso[i] = len(set(npy_file_iso[i,:].flatten()))              # Unique grain IDs

    # ========================================================================
    # TARGET GRAIN SELECTION: Find Optimal Analysis Time Step
    # ========================================================================
    
    # Define target grain count for comparative analysis
    expected_grain_num = 200  # Target grain count for detailed normal distribution analysis
    
    # Find time steps where grain count is closest to target for each energy method
    # This ensures fair comparison between different energy formulations
    special_step_distribution_ave = int(np.argmin(abs(grain_num_aniso_ave - expected_grain_num)))
    special_step_distribution_min = int(np.argmin(abs(grain_num_aniso_min - expected_grain_num)))
    special_step_distribution_max = int(np.argmin(abs(grain_num_aniso_max - expected_grain_num)))
    special_step_distribution_iso = int(np.argmin(abs(grain_num_iso - expected_grain_num)))

    # ========================================================================
    # VISUALIZATION SETUP: XY Plane Polar Distribution Analysis
    # ========================================================================
    
    # Initialize polar plot for XY plane normal distribution analysis
    plt.close()  # Close any existing plots to ensure clean visualization
    fig = plt.figure(figsize=(5, 5))      # Create square figure for polar plot
    ax = plt.gca(projection='polar')      # Set up polar coordinate system

    # Configure angular grid and labels
    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)  # Angular grid every 20°
    ax.set_thetamin(0.0)    # Minimum angle (0°)
    ax.set_thetamax(360.0)  # Maximum angle (360°)

    # Configure radial grid and limits
    ax.set_rgrids(np.arange(0, 0.008, 0.004))  # Radial grid lines at 0, 0.004, 0.008
    ax.set_rlabel_position(0.0)                # Position radial labels at 0°
    ax.set_rlim(0.0, 0.008)                    # Radial limits from 0 to 0.008
    ax.set_yticklabels(['0', '0.004'],fontsize=14)  # Custom radial tick labels

    # Configure plot appearance
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)  # Grid styling
    ax.set_axisbelow('True')  # Place grid below plot data

    # ========================================================================
    # XY PLANE ANALYSIS: Isotropic Reference Case
    # ========================================================================
    
    # Define file paths for cached 3D normal vector data (XY plane)
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    
    # Load or compute isotropic reference data
    if os.path.exists(current_path + data_file_name_P):
        # Load pre-computed data to save computational time
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Compute 3D normal vectors for isotropic case
        # Apply 90° rotation to align data with analysis coordinate system
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        
        # Cache results for future use
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Compute and plot XY plane normal distribution for isotropic case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso")
    
    # ========================================================================
    # BIAS CALCULATION: XY Plane Geometric Bias Correction
    # ========================================================================
    
    # Calculate bias correction for XY plane analysis
    # Bias arises from discrete grid sampling and geometric constraints
    xLim = [0, 360]    # Angular range for bias calculation
    binValue = 10.01   # Bin size for consistency with distribution calculation
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)  # Number of bins
    
    # Create uniform distribution (expected for isotropic case)
    freqArray_circle = np.ones(binNum)  # Uniform frequency array
    freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)  # Normalize
    
    # Calculate bias as difference between uniform and observed distributions
    slope_list_bias = freqArray_circle - slope_list

    # ========================================================================
    # XY PLANE ANALYSIS: Anisotropic Average Energy Case
    # ========================================================================
    
    # Define file paths for cached average energy data (XY plane)
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    
    # Load or compute anisotropic average energy data
    if os.path.exists(current_path + data_file_name_P):
        # Load pre-computed data to save computational time
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Compute 3D normal vectors for anisotropic average energy case
        # Apply 90° rotation to align data with analysis coordinate system
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        
        # Cache results for future use
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Compute and plot XY plane normal distribution for average energy case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave")

    # ========================================================================
    # XY PLANE ANALYSIS: Anisotropic Minimum Energy Case
    # ========================================================================
    
    # Define file paths for cached minimum energy data (XY plane)
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_min_P_step{special_step_distribution_min}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_min_P_sites_step{special_step_distribution_min}.npy'
    
    # Load or compute anisotropic minimum energy data
    if os.path.exists(current_path + data_file_name_P):
        # Load pre-computed data to save computational time
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Compute 3D normal vectors for anisotropic minimum energy case
        # Apply 90° rotation to align data with analysis coordinate system
        newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        
        # Cache results for future use
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Compute and plot XY plane normal distribution for minimum energy case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_min, "Min")

    # ========================================================================
    # XY PLANE ANALYSIS: Anisotropic Maximum Energy Case
    # ========================================================================
    
    # Define file paths for cached maximum energy data (XY plane)
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_max_P_step{special_step_distribution_max}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_max_sites_step{special_step_distribution_max}.npy'
    
    # Load or compute anisotropic maximum energy data
    if os.path.exists(current_path + data_file_name_P):
        # Load pre-computed data to save computational time
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Compute 3D normal vectors for anisotropic maximum energy case
        # Apply 90° rotation to align data with analysis coordinate system
        newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        
        # Cache results for future use
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Compute and plot XY plane normal distribution for maximum energy case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_max, "Max")

    # ========================================================================
    # XY PLANE VISUALIZATION: Finalize and Save Plot
    # ========================================================================
    
    # Add legend and axis labels for XY plane plot
    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=3)  # Position legend below plot
    plt.text(0.0, 0.0095, "x", fontsize=14)           # Label x-axis direction
    plt.text(np.pi/2, 0.0095, "y", fontsize=14)       # Label y-axis direction
    
    # Save XY plane plot with high resolution
    plt.savefig(current_path + f"/figures/normal_distribution_3d_xy_{expected_grain_num}grains.png", 
                dpi=400, bbox_inches='tight')

    # ========================================================================
    # VISUALIZATION SETUP: XZ Plane Polar Distribution Analysis
    # ========================================================================
    
    # Initialize new polar plot for XZ plane normal distribution analysis
    plt.close()  # Close previous plot
    fig = plt.figure(figsize=(5, 5))      # Create square figure for polar plot
    ax = plt.gca(projection='polar')      # Set up polar coordinate system

    # Configure angular grid and labels (identical to XY plane setup)
    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)  # Angular grid every 20°
    ax.set_thetamin(0.0)    # Minimum angle (0°)
    ax.set_thetamax(360.0)  # Maximum angle (360°)

    # Configure radial grid and limits (identical to XY plane setup)
    ax.set_rgrids(np.arange(0, 0.008, 0.004))  # Radial grid lines at 0, 0.004, 0.008
    ax.set_rlabel_position(0.0)                # Position radial labels at 0°
    ax.set_rlim(0.0, 0.008)                    # Radial limits from 0 to 0.008
    ax.set_yticklabels(['0', '0.004'],fontsize=14)  # Custom radial tick labels

    # Configure plot appearance (identical to XY plane setup)
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)  # Grid styling
    ax.set_axisbelow('True')  # Place grid below plot data

    # ========================================================================
    # XZ PLANE ANALYSIS: Isotropic Reference Case
    # ========================================================================
    
    # Reuse cached data from XY plane analysis (same grain structure, different projection)
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    
    # Load or compute isotropic reference data
    if os.path.exists(current_path + data_file_name_P):
        # Load pre-computed data (should exist from XY plane analysis)
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Compute 3D normal vectors if not already available
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        
        # Cache results for future use
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Compute XZ plane normal distribution (angle_index=1 for XZ projection)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 1)
    
    # ========================================================================
    # BIAS CALCULATION: XZ Plane Geometric Bias Correction
    # ========================================================================
    
    # Calculate bias correction specific to XZ plane analysis
    xLim = [0, 360]    # Angular range for bias calculation
    binValue = 10.01   # Bin size for consistency with distribution calculation
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)  # Number of bins
    
    # Create uniform distribution (expected for isotropic case in XZ plane)
    freqArray_circle = np.ones(binNum)  # Uniform frequency array
    freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)  # Normalize
    
    # Calculate XZ plane bias as difference between uniform and observed distributions
    slope_list_bias_1 = freqArray_circle - slope_list

    # ========================================================================
    # XZ PLANE ANALYSIS: Anisotropic Energy Cases
    # ========================================================================
    
    # Anisotropic Average Energy - XZ Plane Analysis
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    
    # Load cached average energy data (should exist from XY plane analysis)
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Compute if not available (fallback)
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Compute XZ plane normal distribution for average energy (angle_index=1)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 1)

    # Anisotropic Minimum Energy - XZ Plane Analysis
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_min_P_step{special_step_distribution_min}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_min_P_sites_step{special_step_distribution_min}.npy'
    
    # Load cached minimum energy data (should exist from XY plane analysis)
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Compute if not available (fallback)
        newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Compute XZ plane normal distribution for minimum energy (angle_index=1)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_min, "Min", 1)

    # Anisotropic Maximum Energy - XZ Plane Analysis
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_max_P_step{special_step_distribution_max}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_max_sites_step{special_step_distribution_max}.npy'
    
    # Load cached maximum energy data (should exist from XY plane analysis)
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Compute if not available (fallback)
        newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Compute XZ plane normal distribution for maximum energy (angle_index=1)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_max, "Max", 1)

    # ========================================================================
    # XZ PLANE VISUALIZATION: Finalize and Save Plot
    # ========================================================================
    
    # Add legend and axis labels for XZ plane plot
    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=3)  # Position legend below plot
    plt.text(0.0, 0.0095, "x", fontsize=14)           # Label x-axis direction
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)       # Label z-axis direction
    
    # Save XZ plane plot with high resolution
    plt.savefig(current_path + f"/figures/normal_distribution_3d_xz_{expected_grain_num}grains.png", 
                dpi=400, bbox_inches='tight')


    # ========================================================================
    # VISUALIZATION SETUP: YZ Plane Polar Distribution Analysis
    # ========================================================================
    
    # Initialize new polar plot for YZ plane normal distribution analysis
    plt.close()  # Close previous plot
    fig = plt.figure(figsize=(5, 5))      # Create square figure for polar plot
    ax = plt.gca(projection='polar')      # Set up polar coordinate system

    # Configure angular grid and labels (identical to previous plane setups)
    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)  # Angular grid every 20°
    ax.set_thetamin(0.0)    # Minimum angle (0°)
    ax.set_thetamax(360.0)  # Maximum angle (360°)

    # Configure radial grid and limits (identical to previous plane setups)
    ax.set_rgrids(np.arange(0, 0.008, 0.004))  # Radial grid lines at 0, 0.004, 0.008
    ax.set_rlabel_position(0.0)                # Position radial labels at 0°
    ax.set_rlim(0.0, 0.008)                    # Radial limits from 0 to 0.008
    ax.set_yticklabels(['0', '0.004'],fontsize=14)  # Custom radial tick labels

    # Configure plot appearance (identical to previous plane setups)
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)  # Grid styling
    ax.set_axisbelow('True')  # Place grid below plot data

    # ========================================================================
    # YZ PLANE ANALYSIS: Isotropic Reference Case
    # ========================================================================
    
    # Reuse cached data from previous plane analyses (same grain structure, different projection)
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    
    # Load or compute isotropic reference data
    if os.path.exists(current_path + data_file_name_P):
        # Load pre-computed data (should exist from previous plane analyses)
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Compute 3D normal vectors if not already available
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        
        # Cache results for future use
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Compute YZ plane normal distribution (angle_index=2 for YZ projection)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 2)
    
    # ========================================================================
    # BIAS CALCULATION: YZ Plane Geometric Bias Correction
    # ========================================================================
    
    # Calculate bias correction specific to YZ plane analysis
    xLim = [0, 360]    # Angular range for bias calculation
    binValue = 10.01   # Bin size for consistency with distribution calculation
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)  # Number of bins
    
    # Create uniform distribution (expected for isotropic case in YZ plane)
    freqArray_circle = np.ones(binNum)  # Uniform frequency array
    freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)  # Normalize
    
    # Calculate YZ plane bias as difference between uniform and observed distributions
    slope_list_bias_2 = freqArray_circle - slope_list

    # ========================================================================
    # YZ PLANE ANALYSIS: Anisotropic Energy Cases
    # ========================================================================
    
    # Anisotropic Average Energy - YZ Plane Analysis
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    
    # Load cached average energy data (should exist from previous plane analyses)
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Compute if not available (fallback)
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Compute YZ plane normal distribution for average energy (angle_index=2)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 2)

     # Anisotropic Minimum Energy - YZ Plane Analysis
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_min_P_step{special_step_distribution_min}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_min_P_sites_step{special_step_distribution_min}.npy'
    
    # Load cached minimum energy data (should exist from previous plane analyses)
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Compute if not available (fallback)
        newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Compute YZ plane normal distribution for minimum energy (angle_index=2)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_min, "Min", 2)

    # Anisotropic Maximum Energy - YZ Plane Analysis
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_max_P_step{special_step_distribution_max}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_max_sites_step{special_step_distribution_max}.npy'
    
    # Load cached maximum energy data (should exist from previous plane analyses)
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Compute if not available (fallback)
        newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Compute YZ plane normal distribution for maximum energy (angle_index=2)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_max, "Max", 2)

    # ========================================================================
    # YZ PLANE VISUALIZATION: Finalize and Save Plot
    # ========================================================================
    
    # Add legend and axis labels for YZ plane plot
    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=3)  # Position legend below plot
    plt.text(0.0, 0.0095, "y", fontsize=14)           # Label y-axis direction
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)       # Label z-axis direction
    
    # Save YZ plane plot with high resolution
    plt.savefig(current_path + f"/figures/normal_distribution_3d_yz_{expected_grain_num}grains.png", 
                dpi=400, bbox_inches='tight')



    # ========================================================================
    # BIAS-CORRECTED ANALYSIS SECTION
    # ========================================================================
    # This section repeats the three-plane analysis (XY, XZ, YZ) with bias 
    # correction applied to remove geometric artifacts from discrete sampling.
    # The bias correction accounts for systematic deviations from uniform 
    # distribution that arise from grid-based discretization effects.
    # ========================================================================

    # ========================================================================
    # BIAS-CORRECTED VISUALIZATION: XY Plane Analysis
    # ========================================================================
    
    # Initialize new polar plot for bias-corrected XY plane analysis
    plt.close()  # Close previous plot
    fig = plt.figure(figsize=(5, 5))      # Create square figure for polar plot
    ax = plt.gca(projection='polar')      # Set up polar coordinate system

    # Configure angular grid and labels (identical to uncorrected setup)
    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)  # Angular grid every 20°
    ax.set_thetamin(0.0)    # Minimum angle (0°)
    ax.set_thetamax(360.0)  # Maximum angle (360°)

    # Configure radial grid and limits (identical to uncorrected setup)
    ax.set_rgrids(np.arange(0, 0.008, 0.004))  # Radial grid lines at 0, 0.004, 0.008
    ax.set_rlabel_position(0.0)                # Position radial labels at 0°
    ax.set_rlim(0.0, 0.008)                    # Radial limits from 0 to 0.008
    ax.set_yticklabels(['0', '0.004'],fontsize=14)  # Custom radial tick labels

    # Configure plot appearance (identical to uncorrected setup)
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)  # Grid styling
    ax.set_axisbelow('True')  # Place grid below plot data

    # ========================================================================
    # BIAS-CORRECTED XY PLANE: Isotropic Reference Case
    # ========================================================================
    
    # Reuse cached data from previous XY plane analysis
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    
    # Load cached isotropic data (should definitely exist by now)
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Fallback computation (should not be needed)
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply XY plane bias correction to isotropic case
    # Using previously calculated slope_list_bias from uncorrected XY analysis
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 0, slope_list_bias)

    # ========================================================================
    # BIAS-CORRECTED XY PLANE: Anisotropic Energy Cases
    # ========================================================================
    
    # Anisotropic Average Energy - Bias-Corrected XY Plane
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    
    # Load cached average energy data
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Fallback computation (should not be needed)
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply XY plane bias correction to average energy case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 0, slope_list_bias)

    # Anisotropic Minimum Energy - Bias-Corrected XY Plane
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_min_P_step{special_step_distribution_min}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_min_P_sites_step{special_step_distribution_min}.npy'
    
    # Load cached minimum energy data
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Fallback computation (should not be needed)
        newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply XY plane bias correction to minimum energy case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_min, "Min", 0, slope_list_bias)

    # Anisotropic Maximum Energy - Bias-Corrected XY Plane
    # Note: Comment in original code says "Aniso - sum" but this is actually maximum energy
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_max_P_step{special_step_distribution_max}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_max_sites_step{special_step_distribution_max}.npy'
    
    # Load cached maximum energy data
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Fallback computation (should not be needed)
        newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply XY plane bias correction to maximum energy case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_max, "Max", 0, slope_list_bias)

    # ========================================================================
    # BIAS-CORRECTED XY PLANE: Finalize and Save Plot
    # ========================================================================
    
    # Add legend and axis labels for bias-corrected XY plane plot
    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=3)  # Position legend below plot
    plt.text(0.0, 0.0095, "x", fontsize=14)           # Label x-axis direction
    plt.text(np.pi/2, 0.0095, "y", fontsize=14)       # Label y-axis direction
    
    # Save bias-corrected XY plane plot with descriptive filename
    plt.savefig(current_path + f"/figures/normal_distribution_3d_xy_{expected_grain_num}grains_after_removing_bias.png", 
                dpi=400, bbox_inches='tight')

    # ========================================================================
    # BIAS-CORRECTED VISUALIZATION: XZ Plane Analysis
    # ========================================================================
    
    # Initialize new polar plot for bias-corrected XZ plane analysis
    plt.close()  # Close previous plot
    fig = plt.figure(figsize=(5, 5))      # Create square figure for polar plot
    ax = plt.gca(projection='polar')      # Set up polar coordinate system

    # Configure angular grid and labels (identical to previous setups)
    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)  # Angular grid every 20°
    ax.set_thetamin(0.0)    # Minimum angle (0°)
    ax.set_thetamax(360.0)  # Maximum angle (360°)

    # Configure radial grid and limits (identical to previous setups)
    ax.set_rgrids(np.arange(0, 0.008, 0.004))  # Radial grid lines at 0, 0.004, 0.008
    ax.set_rlabel_position(0.0)                # Position radial labels at 0°
    ax.set_rlim(0.0, 0.008)                    # Radial limits from 0 to 0.008
    ax.set_yticklabels(['0', '0.004'],fontsize=14)  # Custom radial tick labels

    # Configure plot appearance (identical to previous setups)
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)  # Grid styling
    ax.set_axisbelow('True')  # Place grid below plot data

    # ========================================================================
    # BIAS-CORRECTED XZ PLANE: All Energy Cases with XZ-Specific Bias
    # ========================================================================
    
    # Apply XZ plane bias correction (slope_list_bias_1) to all energy cases
    # Note: XZ plane has different geometric bias than XY plane due to different projection
    
    # Isotropic Reference - Bias-Corrected XZ Plane
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    
    # Load cached isotropic data
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Fallback computation (should not be needed at this point)
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply XZ-specific bias correction to isotropic case (angle_index=1 for XZ, bias=slope_list_bias_1)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 1, slope_list_bias_1)

    # Anisotropic Average Energy - Bias-Corrected XZ Plane
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    
    # Load cached average energy data
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Fallback computation (should not be needed)
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply XZ-specific bias correction to average energy case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 1, slope_list_bias_1)

    # Anisotropic Minimum Energy - Bias-Corrected XZ Plane
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_min_P_step{special_step_distribution_min}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_min_P_sites_step{special_step_distribution_min}.npy'
    
    # Load cached minimum energy data
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Fallback computation (should not be needed)
        newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply XZ-specific bias correction to minimum energy case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_min, "Min", 1, slope_list_bias_1)

    # Anisotropic Maximum Energy - Bias-Corrected XZ Plane
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_max_P_step{special_step_distribution_max}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_max_sites_step{special_step_distribution_max}.npy'
    
    # Load cached maximum energy data
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Fallback computation (should not be needed)
        newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply XZ-specific bias correction to maximum energy case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_max, "Max", 1, slope_list_bias_1)

    # ========================================================================
    # BIAS-CORRECTED XZ PLANE: Finalize and Save Plot
    # ========================================================================
    
    # Add legend and axis labels for bias-corrected XZ plane plot
    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=3)  # Position legend below plot
    plt.text(0.0, 0.0095, "x", fontsize=14)           # Label x-axis direction
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)       # Label z-axis direction
    
    # Save bias-corrected XZ plane plot with descriptive filename
    plt.savefig(current_path + f"/figures/normal_distribution_3d_xz_{expected_grain_num}grains_after_removing_bias.png", 
                dpi=400, bbox_inches='tight')

    # ========================================================================
    # BIAS-CORRECTED VISUALIZATION: YZ Plane Analysis
    # ========================================================================
    
    # Initialize new polar plot for bias-corrected YZ plane analysis
    plt.close()  # Close previous plot
    fig = plt.figure(figsize=(5, 5))      # Create square figure for polar plot
    ax = plt.gca(projection='polar')      # Set up polar coordinate system

    # Configure angular grid and labels (identical to previous setups)
    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)  # Angular grid every 20°
    ax.set_thetamin(0.0)    # Minimum angle (0°)
    ax.set_thetamax(360.0)  # Maximum angle (360°)

    # Configure radial grid and limits (identical to previous setups)
    ax.set_rgrids(np.arange(0, 0.008, 0.004))  # Radial grid lines at 0, 0.004, 0.008
    ax.set_rlabel_position(0.0)                # Position radial labels at 0°
    ax.set_rlim(0.0, 0.008)                    # Radial limits from 0 to 0.008
    ax.set_yticklabels(['0', '0.004'],fontsize=14)  # Custom radial tick labels

    # Configure plot appearance (identical to previous setups)
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)  # Grid styling
    ax.set_axisbelow('True')  # Place grid below plot data

    # ========================================================================
    # BIAS-CORRECTED YZ PLANE: All Energy Cases with YZ-Specific Bias
    # ========================================================================
    
    # Apply YZ plane bias correction (slope_list_bias_2) to all energy cases
    # Note: YZ plane has unique geometric bias different from both XY and XZ planes
    
    # Isotropic Reference - Bias-Corrected YZ Plane
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    
    # Load cached isotropic data
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Fallback computation (should not be needed at this point)
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply YZ-specific bias correction to isotropic case (angle_index=2 for YZ, bias=slope_list_bias_2)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 2, slope_list_bias_2)

    # Anisotropic Average Energy - Bias-Corrected YZ Plane
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    
    # Load cached average energy data
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Fallback computation (should not be needed)
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply YZ-specific bias correction to average energy case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 2, slope_list_bias_2)

    # Anisotropic Minimum Energy - Bias-Corrected YZ Plane
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_min_P_step{special_step_distribution_min}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_min_P_sites_step{special_step_distribution_min}.npy'
    
    # Load cached minimum energy data
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Fallback computation (should not be needed)
        newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply YZ-specific bias correction to minimum energy case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_min, "Min", 2, slope_list_bias_2)

    # Anisotropic Maximum Energy - Bias-Corrected YZ Plane
    data_file_name_P = f'/3D_normal_distribution_data/3D_normal_distribution_max_P_step{special_step_distribution_max}.npy'
    data_file_name_sites = f'/3D_normal_distribution_data/3D_normal_distribution_max_sites_step{special_step_distribution_max}.npy'
    
    # Load cached maximum energy data
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)        # Processed grain structure
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Fallback computation (should not be needed)
        newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply YZ-specific bias correction to maximum energy case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_max, "Max", 2, slope_list_bias_2)

    # ========================================================================
    # BIAS-CORRECTED YZ PLANE: Finalize and Save Plot
    # ========================================================================
    
    # Add legend and axis labels for bias-corrected YZ plane plot
    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=3)  # Position legend below plot
    plt.text(0.0, 0.0095, "y", fontsize=14)           # Label y-axis direction
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)       # Label z-axis direction
    
    # Save bias-corrected YZ plane plot with descriptive filename
    plt.savefig(current_path + f"/figures/normal_distribution_3d_yz_{expected_grain_num}grains_after_removing_bias.png", 
                dpi=400, bbox_inches='tight')

# ============================================================================
# ANALYSIS COMPLETION AND SUMMARY
# ============================================================================
# 
# This script has completed a comprehensive 3D grain boundary normal vector 
# distribution analysis for HiPerGator supercomputing cluster data, including:
#
# 1. Data Loading: SPPARKS Monte Carlo simulation results for four energy methods
# 2. Grain Evolution: Temporal grain count tracking and optimal time step selection
# 3. 3D Normal Vector Computation: VECTOR framework implementation for all cases
# 4. Multi-Planar Analysis: XY, XZ, and YZ plane projections for complete 3D characterization
# 5. Bias Correction: Systematic removal of geometric discretization artifacts
# 6. High-Quality Visualization: Publication-ready polar plots with proper legends
#
# Output Files Generated:
# - 6 polar distribution plots (3 planes × 2 bias states)
# - Cached normal vector data for computational efficiency
# - High-resolution figures suitable for scientific publication
#
# The analysis enables comparison of grain boundary orientation distributions
# between isotropic and anisotropic energy formulations, revealing the impact
# of grain boundary anisotropy on microstructural evolution in 3D systems.
# ============================================================================







