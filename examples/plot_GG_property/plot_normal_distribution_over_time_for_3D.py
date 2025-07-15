#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Grain Boundary Normal Vector Distribution Analysis Script

This script analyzes the orientation distribution of grain boundary normal vectors
in 3D polycrystalline microstructures by calculating and visualizing the angular
distribution patterns across different projection planes (XY, XZ, YZ).

The analysis compares multiple energy formulations for grain boundary evolution:
- Anisotropic cases: ave, min, sum energy methods with delta=0.6
- Isotropic reference case: ave energy method with delta=0.0

Key Features:
- 3D normal vector calculation using VECTOR framework's linear3d module
- Multi-planar projection analysis (XY, XZ, YZ planes)
- Polar coordinate visualization for orientation distributions
- Computational caching for efficiency with large 3D datasets
- Comparative analysis across different energy formulations

Scientific Context:
This script supports research in materials science focusing on grain boundary
character distribution (GBCD) analysis in 3D polycrystalline systems.
The orientation analysis helps understand texture evolution and anisotropic
effects in microstructural development.

Created on Mon Jul 31 14:33:57 2023
@author: Lin
"""

# ===============================================================================
# IMPORT LIBRARIES AND DEPENDENCIES
# ===============================================================================

# Standard library imports
import os                           # Operating system interface for file path operations
current_path = os.getcwd()         # Get current working directory for relative path construction
import math                        # Mathematical functions for trigonometric calculations
import sys                         # System-specific parameters and functions for path manipulation

# Scientific computing libraries
import numpy as np                 # Numerical computing: arrays, linear algebra, mathematical functions
from numpy import seterr           # Configure numpy error handling behavior
seterr(all='raise')               # Set numpy to raise exceptions on all numerical errors

# Visualization library
import matplotlib.pyplot as plt    # 2D plotting library for creating polar plots and figures

# Progress tracking
from tqdm import tqdm             # Progress bar utility for long-running loops (imported but not used)

# Custom VECTOR framework modules
sys.path.append(current_path)                    # Add current directory to Python path
sys.path.append(current_path+'/../../')          # Add VECTOR root directory to Python path
import myInput                                   # Custom input/gradient calculation functions
import PACKAGE_MP_Linear as linear2d            # 2D linear smoothing and normal vector computation
import PACKAGE_MP_3DLinear as linear3d          # 3D linear smoothing and normal vector computation
sys.path.append(current_path+'/../calculate_tangent/')  # Add tangent calculation utilities path

# ===============================================================================
# FUNCTION DEFINITIONS
# ===============================================================================

def get_normal_vector(grain_structure_figure_one):
    """
    Calculate normal vectors for grain boundaries in 2D microstructures.
    
    This function implements the 2D linear smoothing algorithm from the VECTOR
    framework to compute grain boundary normal vectors for orientation analysis.
    
    Algorithm Overview:
    1. Initialize 2D linear smoothing class with microstructure data
    2. Execute iterative linear smoothing to refine grain boundary locations
    3. Extract all grain boundary sites from the smoothed field
    4. Return smoothed phase field P and consolidated grain boundary site list
    
    Parameters:
    -----------
    grain_structure_figure_one : ndarray
        2D array representing the grain structure where each element contains
        the grain ID. Shape: (nx, ny) where nx, ny are spatial dimensions.
    
    Returns:
    --------
    P : ndarray
        Smoothed phase field array with refined grain boundary representation.
        Shape: (nx, ny, 2) containing spatial gradient information.
    sites_together : list
        Consolidated list of all grain boundary sites as [i, j] coordinates.
        Each site represents a location where grain boundary normal vectors
        can be calculated for orientation analysis.
    
    Technical Details:
    ------------------
    - Uses 8-core parallel processing for computational efficiency
    - Performs 5 iterations of linear smoothing for convergence
    - Employs "inclination" mode for gradient-based boundary detection
    """
    # Extract spatial dimensions from input grain structure
    nx = grain_structure_figure_one.shape[0]  # Number of grid points in x-direction
    ny = grain_structure_figure_one.shape[1]  # Number of grid points in y-direction
    ng = np.max(grain_structure_figure_one)   # Maximum grain ID (total number of grains)
    
    # Set computational parameters for linear smoothing algorithm
    cores = 8          # Number of CPU cores for parallel processing
    loop_times = 5     # Number of smoothing iterations for convergence
    
    # Initialize input arrays for the linear smoothing class
    P0 = grain_structure_figure_one          # Initial grain structure (nx, ny)
    R = np.zeros((nx,ny,2))                 # Initialize gradient array (nx, ny, 2)
    
    # Create 2D linear smoothing class instance
    smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)

    # Execute the main linear smoothing algorithm in "inclination" mode
    # This mode optimizes for grain boundary normal vector calculation
    smooth_class.linear_main("inclination")
    
    # Extract the smoothed phase field containing refined boundary information
    P = smooth_class.get_P()
    
    # Get grain boundary site lists for all grains
    # sites = smooth_class.get_gb_list(1)
    # print(len(sites))
    # for id in range(2,grain_num+1): sites += smooth_class.get_gb_list(id)
    # print(len(sites))
    
    # Extract all grain boundary sites from the smoothed structure
    sites = smooth_class.get_all_gb_list()  # Returns list of lists, one per grain
    
    # Consolidate all grain boundary sites into a single list
    sites_together = []
    for id in range(len(sites)): 
        sites_together += sites[id]  # Flatten the nested list structure
    
    # Output total number of grain boundary sites for verification
    print("Total num of GB sites: " + str(len(sites_together)))

    return P, sites_together

def get_normal_vector_3d(grain_structure_figure_one):
    """
    Calculate normal vectors for grain boundaries in 3D microstructures.
    
    This function extends the 2D algorithm to 3D using the VECTOR framework's
    3D linear smoothing capabilities for comprehensive volumetric grain boundary
    analysis in three-dimensional polycrystalline materials.
    
    Algorithm Overview:
    1. Initialize 3D linear smoothing class with volumetric microstructure data
    2. Execute iterative 3D linear smoothing with "inclination" mode
    3. Extract all 3D grain boundary sites from the smoothed volume
    4. Return smoothed 3D phase field and consolidated grain boundary coordinates
    
    Parameters:
    -----------
    grain_structure_figure_one : ndarray
        3D array representing the grain structure where each element contains
        the grain ID. Shape: (nx, ny, nz) for volumetric microstructure data.
    
    Returns:
    --------
    P : ndarray
        Smoothed 3D phase field array with refined grain boundary representation.
        Shape: (nx, ny, nz, 3) containing 3D spatial gradient information.
    sites_together : list
        Consolidated list of all 3D grain boundary sites as [i, j, k] coordinates.
        Each site represents a volumetric location for 3D normal vector analysis.
    
    Technical Details:
    ------------------
    - Extends 2D algorithm to full 3D volumetric processing
    - Uses 'np' mode for numpy-optimized 3D computations
    - Maintains same iteration count (5) and core usage (8) as 2D version
    - Handles significantly larger datasets due to 3D nature (nx×ny×nz elements)
    """
    # Extract 3D spatial dimensions from input grain structure
    nx = grain_structure_figure_one.shape[0]  # Number of voxels in x-direction
    ny = grain_structure_figure_one.shape[1]  # Number of voxels in y-direction  
    nz = grain_structure_figure_one.shape[2]  # Number of voxels in z-direction
    ng = np.max(grain_structure_figure_one)   # Maximum grain ID (total number of grains)
    
    # Set computational parameters for 3D linear smoothing algorithm
    cores = 8          # Number of CPU cores for parallel 3D processing
    loop_times = 5     # Number of smoothing iterations for 3D convergence
    
    # Initialize input arrays for the 3D linear smoothing class
    P0 = grain_structure_figure_one[:,:,:,np.newaxis]  # Add channel dimension: (nx,ny,nz,1)
    R = np.zeros((nx,ny,nz,3))                         # Initialize 3D gradient array: (nx,ny,nz,3)
    
    # Create 3D linear smoothing class instance with numpy optimization mode
    smooth_class = linear3d.linear3d_class(nx,ny,nz,ng,cores,loop_times,P0,R,'np')

    # Execute the main 3D linear smoothing algorithm in "inclination" mode
    # This mode is optimized for 3D grain boundary normal vector calculation
    smooth_class.linear3d_main("inclination")
    
    # Extract the smoothed 3D phase field containing refined boundary information
    P = smooth_class.get_P()
    
    # Extract all 3D grain boundary sites from the smoothed volume
    sites = smooth_class.get_all_gb_list()  # Returns list of lists, one per grain
    
    # Consolidate all 3D grain boundary sites into a single list
    sites_together = []
    for id in range(len(sites)): 
        sites_together += sites[id]  # Flatten the nested list structure
    
    # Output total number of 3D grain boundary sites for verification
    print(f"Total num of GB sites: {len(sites_together)}")

    return P, sites_together

def get_normal_vector_slope(P, sites, step, para_name):
    """
    Calculate and plot angular distribution of 2D grain boundary normal vectors.
    
    This function computes the orientation distribution of grain boundary normal
    vectors in 2D by calculating angles from gradient information and creating
    a histogram for statistical analysis and visualization.
    
    Algorithm Overview:
    1. Set up angular binning parameters (0-360 degrees, 10-degree bins)
    2. Calculate gradients at each grain boundary site using myInput.get_grad()
    3. Convert gradients to angles using atan2 for proper quadrant handling
    4. Bin the angles and normalize to create probability density
    5. Plot the angular distribution on the current polar axis
    
    Parameters:
    -----------
    P : ndarray
        Smoothed phase field from get_normal_vector() containing gradient information.
        Shape: (nx, ny, 2) with spatial and gradient components.
    sites : list
        List of grain boundary sites as [i, j] coordinates from get_normal_vector().
    step : int
        Time step identifier for the analysis (used for data organization).
    para_name : str
        Parameter name for plot legend (e.g., "Ave case", "Min case").
    
    Returns:
    --------
    int
        Returns 0 upon successful completion of angle calculation and plotting.
    
    Mathematical Details:
    ---------------------
    - Angle calculation: θ = atan2(-dy, dx) + π 
    - Ensures angles are in range [0, 2π] for consistent orientation mapping
    - Binning: 10.01-degree bins to avoid exact boundary conditions
    - Normalization: freqArray = freqArray / sum(freqArray * binValue)
    """
    # Set up angular coordinate system and binning parameters
    xLim = [0, 360]                                           # Angular range in degrees
    binValue = 10.01                                          # Bin width (slightly > 10 to avoid boundary issues)
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)     # Number of bins for histogram
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Bin center coordinates

    # Initialize frequency array for angular histogram
    freqArray = np.zeros(binNum)  # Will store count of orientations in each angular bin
    degree = []                   # List to store calculated angles for each grain boundary site
    
    # Calculate normal vector angles for each grain boundary site
    for sitei in sites:
        [i,j] = sitei  # Extract i,j coordinates of the grain boundary site
        
        # Calculate spatial gradients using custom gradient function
        dx,dy = myInput.get_grad(P,i,j)  # Returns gradient components at site (i,j)
        
        # Convert gradients to angle using atan2 for proper quadrant determination
        # Note: -dy is used to match coordinate system convention
        # Adding π ensures angles are in range [0, 2π] instead of [-π, π]
        degree.append(math.atan2(-dy, dx) + math.pi)
        
        # Alternative angle calculation methods (commented out for reference):
        # if dx == 0:
        #     degree.append(math.pi/2)
        # elif dy >= 0:
        #     degree.append(abs(math.atan(-dy/dx)))
        # elif dy < 0:
        #     degree.append(abs(math.atan(dy/dx)))
    
    # Populate frequency histogram by binning the calculated angles
    for i in range(len(degree)):
        # Convert angle from radians to degrees and determine appropriate bin
        bin_index = int((degree[i]/math.pi*180-xLim[0])/binValue)
        freqArray[bin_index] += 1
    
    # Normalize frequency array to create probability density function
    # Division by sum(freqArray*binValue) ensures total probability = 1
    freqArray = freqArray/sum(freqArray*binValue)
    
    # Plot the angular distribution on the current polar coordinate system
    plt.plot(xCor/180*math.pi, freqArray, linewidth=2, label=para_name)

    return 0

def get_normal_vector_slope_3d(P, sites, step, para_name, angle_index=0):
    """
    Calculate and plot angular distribution of 3D grain boundary normal vectors.
    
    This function extends the 2D orientation analysis to 3D by projecting 3D normal
    vectors onto specific 2D planes (XY, XZ, YZ) and analyzing their angular
    distributions. This enables comprehensive characterization of 3D grain boundary
    orientation texture across multiple viewing planes.
    
    Algorithm Overview:
    1. Set up angular binning parameters (0-360 degrees, 10-degree bins)
    2. Calculate 3D gradients at each grain boundary site using myInput.get_grad3d()
    3. Project 3D gradients onto specified 2D plane based on angle_index
    4. Normalize projected gradients and convert to angles
    5. Bin angles and normalize to create probability density
    6. Plot distribution with periodic boundary for complete circular representation
    
    Parameters:
    -----------
    P : ndarray
        Smoothed 3D phase field from get_normal_vector_3d() with gradient information.
        Shape: (nx, ny, nz, 3) containing 3D spatial and gradient components.
    sites : list
        List of 3D grain boundary sites as [i, j, k] coordinates.
    step : int
        Time step identifier for the analysis (used for data organization).
    para_name : str
        Parameter name for plot legend (e.g., "Ave case", "Min case", "Iso case").
    angle_index : int, optional (default=0)
        Selects the projection plane for 2D analysis:
        - 0: XY plane projection (dx_fake=dx, dy_fake=dy)
        - 1: XZ plane projection (dx_fake=dx, dy_fake=dz)  
        - 2: YZ plane projection (dx_fake=dy, dy_fake=dz)
    
    Returns:
    --------
    int
        Returns 0 upon successful completion of 3D angle calculation and plotting.
    
    Mathematical Details:
    ---------------------
    - 3D gradient projection preserves directional information in selected plane
    - Gradient normalization: ensures unit vector representation before angle calculation
    - Skip condition: |∇| < 1e-5 filters out numerical noise at grain interiors
    - Periodic plotting: appends first point to end for complete circular visualization
    
    Technical Implementation:
    ------------------------
    - Handles large 3D datasets efficiently through vectorized operations
    - Provides flexibility for multi-planar texture analysis
    - Compatible with polar coordinate visualization requirements
    """
    # Set up angular coordinate system and binning parameters
    xLim = [0, 360]                                           # Angular range in degrees
    binValue = 10.01                                          # Bin width (slightly > 10 to avoid boundary issues)
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)     # Number of bins for histogram
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Bin center coordinates

    # Initialize arrays for 3D angular analysis
    freqArray = np.zeros(binNum)  # Frequency histogram for angular distribution
    degree = []                   # List to store calculated angles for each 3D grain boundary site
    # degree_shadow = []          # Optional debugging array for gradient components
    
    # Calculate 3D normal vector angles for each grain boundary site
    for sitei in sites:
        [i,j,k] = sitei  # Extract i,j,k coordinates of the 3D grain boundary site
        
        # Calculate 3D spatial gradients using custom 3D gradient function
        dx,dy,dz = myInput.get_grad3d(P,i,j,k)  # Returns 3D gradient components at site (i,j,k)
        
        # Project 3D gradient onto specified 2D plane based on angle_index
        # dy_fake = math.sqrt(dy**2 + dz**2)  # Alternative: magnitude in YZ plane
        
        if angle_index == 0:      # XY plane projection
            dx_fake = dx          # X-component remains the same
            dy_fake = dy          # Y-component remains the same
        elif angle_index == 1:    # XZ plane projection  
            dx_fake = dx          # X-component remains the same
            dy_fake = dz          # Z-component becomes "Y" for 2D analysis
        elif angle_index == 2:    # YZ plane projection
            dx_fake = dy          # Y-component becomes "X" for 2D analysis
            dy_fake = dz          # Z-component becomes "Y" for 2D analysis
        
        # Normalize projected gradient to unit vector (essential for accurate angle calculation)
        gradient_magnitude = math.sqrt(dy_fake**2+dx_fake**2)
        if gradient_magnitude < 1e-5: 
            continue  # Skip sites with negligible gradients (likely grain interiors)
            
        # Calculate normalized gradient components
        dy_fake_norm = dy_fake / gradient_magnitude
        dx_fake_norm = dx_fake / gradient_magnitude

        # Convert normalized gradients to angle with proper quadrant handling
        # Note: -dy_fake_norm maintains consistent orientation convention
        degree.append(math.atan2(-dy_fake_norm, dx_fake_norm) + math.pi)
        
        # Optional debugging: store site location and gradient information
        # degree_shadow.append([i,j,k,dz])
    
    # Populate frequency histogram by binning the calculated 3D projection angles
    for n in range(len(degree)):
        # Convert angle from radians to degrees and determine appropriate bin
        bin_index = int((degree[n]/math.pi*180-xLim[0])/binValue)
        freqArray[bin_index] += 1
        
        # Optional debugging: print angles that fall in first bin
        # if int((degree[n]/math.pi*180-xLim[0])/binValue) == 0:
        #     print(f"loc: {degree_shadow[n][0]},{degree_shadow[n][1]},{degree_shadow[n][2]} : {degree[n]/np.pi*180} and {degree_shadow[n][3]}")
    
    # Normalize frequency array to create probability density function
    freqArray = freqArray/sum(freqArray*binValue)

    # Plot the 3D angular distribution with periodic boundary for complete circle
    # Append first point to end to close the circular plot properly
    plt.plot(np.append(xCor,xCor[0])/180*math.pi, np.append(freqArray,freqArray[0]),linewidth=2,label=para_name)

    return 0

# ===============================================================================
# MAIN EXECUTION: 3D GRAIN BOUNDARY NORMAL VECTOR DISTRIBUTION ANALYSIS
# ===============================================================================

if __name__ == '__main__':
    # -----------------------------------------------------------------------
    # DATA SOURCE CONFIGURATION
    # -----------------------------------------------------------------------
    
    # Base directory containing SPPARKS simulation results
    # These are 3D polycrystalline evolution simulations with different energy formulations
    npy_file_folder = "/Users/lin/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/3d_poly_for_GG/results/"
    
    # Energy formulation type definitions for comparative analysis
    # TJ_energy_type_cases = ["ave"] #["ave", "sum", "consMin", "consMax", "consTest"]  # Full option list
    TJ_energy_type_ave = "ave"  # Average energy formulation
    TJ_energy_type_min = "min"  # Minimum energy formulation  
    TJ_energy_type_sum = "sum"  # Sum energy formulation
    
    # -----------------------------------------------------------------------
    # SIMULATION FILE SPECIFICATION
    # -----------------------------------------------------------------------
    
    # Anisotropic simulation files (delta=0.6 introduces grain boundary energy anisotropy)
    # File naming convention: p_ori_ave_{energy_type}E_100_20k_multiCore64_delta{delta}_m2_J1_refer_1_0_0_seed56689_kt066.npy
    # Parameters breakdown:
    # - 100: grid size parameter
    # - 20k: initial grain count (20,000 grains)
    # - multiCore64: 64-core parallel processing
    # - delta0.6: anisotropy strength (0.6 for anisotropic, 0.0 for isotropic)
    # - m2: mobility parameter
    # - J1: interaction strength
    # - refer_1_0_0: reference orientation
    # - seed56689: random seed for reproducibility
    # - kt066: temperature parameter
    npy_file_name_aniso_ave = f"p_ori_ave_{TJ_energy_type_ave}E_100_20k_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_min = f"p_ori_ave_{TJ_energy_type_min}E_100_20k_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_sum = f"p_ori_ave_{TJ_energy_type_sum}E_100_20k_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    
    # Isotropic reference simulation file (delta=0.0 for comparison baseline)
    npy_file_name_iso = "p_ori_ave_aveE_100_20k_multiCore64_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    
    # -----------------------------------------------------------------------
    # DATA LOADING AND VALIDATION
    # -----------------------------------------------------------------------
    
    # Load 3D microstructural evolution data from numpy arrays
    # Each file contains time series of 3D grain structures: shape (time_steps, nx, ny, nz)
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)  # Anisotropic average energy case
    npy_file_aniso_min = np.load(npy_file_folder + npy_file_name_aniso_min)  # Anisotropic minimum energy case
    npy_file_aniso_sum = np.load(npy_file_folder + npy_file_name_aniso_sum)  # Anisotropic sum energy case
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)              # Isotropic reference case
    
    # Display dataset information for verification and debugging
    print(f"The ave data size is: {npy_file_aniso_ave.shape}")  # Expected: (time_steps, nx, ny, nz)
    print(f"The min data size is: {npy_file_aniso_min.shape}")  # Should match ave dimensions
    print(f"The sum data size is: {npy_file_aniso_sum.shape}")  # Should match ave dimensions  
    print(f"The iso data size is: {npy_file_iso.shape}")        # Should match ave dimensions
    print("READING DATA DONE")  # Confirmation of successful data loading
    
    # -----------------------------------------------------------------------
    # GRAIN EVOLUTION ANALYSIS SETUP
    # -----------------------------------------------------------------------
    
    # Initialize grain count tracking arrays for temporal evolution analysis
    initial_grain_num = 20000                        # Starting grain count from simulation parameters
    step_num = npy_file_aniso_ave.shape[0]          # Number of time steps in the simulation
    
    # Create arrays to track grain count evolution for each energy formulation
    grain_num_aniso_ave = np.zeros(step_num)        # Average energy anisotropic case
    grain_num_aniso_min = np.zeros(step_num)        # Minimum energy anisotropic case  
    grain_num_aniso_sum = np.zeros(step_num)        # Sum energy anisotropic case
    grain_num_iso = np.zeros(step_num)              # Isotropic reference case
    
    # -----------------------------------------------------------------------
    # GRAIN COUNT CALCULATION ACROSS TIME STEPS
    # -----------------------------------------------------------------------
    
    # Calculate the number of unique grains at each time step
    # This provides insight into grain growth kinetics under different energy formulations
    for i in range(step_num):
        # Count unique grain IDs by flattening 3D array and finding unique values
        grain_num_aniso_ave[i] = len(set(npy_file_aniso_ave[i,:].flatten()))  # Anisotropic average
        grain_num_aniso_min[i] = len(set(npy_file_aniso_min[i,:].flatten()))  # Anisotropic minimum
        grain_num_aniso_sum[i] = len(set(npy_file_aniso_sum[i,:].flatten()))  # Anisotropic sum
        grain_num_iso[i] = len(set(npy_file_iso[i,:].flatten()))              # Isotropic reference
    
    # -----------------------------------------------------------------------
    # GRAIN SIZE DISTRIBUTION ANALYSIS PARAMETERS
    # -----------------------------------------------------------------------
    
    # Set up binning parameters for grain size distribution analysis
    # Note: These variables are defined but not actively used in the current normal vector analysis
    bin_width = 0.16                                # Width of each size bin for grain size distribution
    x_limit = [-0.5, 3.5]                          # Range for grain size analysis (logarithmic scale)
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)  # Number of bins for size distribution
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)  # Bin centers
    
    # Initialize grain size distribution arrays (for potential future analysis)
    grain_size_distribution_ave = np.zeros(bin_num)  # Average case size distribution
    special_step_distribution_ave = 2                # Time step for detailed ave analysis
    grain_size_distribution_min = np.zeros(bin_num)  # Minimum case size distribution
    special_step_distribution_min = 2                # Time step for detailed min analysis  
    grain_size_distribution_sum = np.zeros(bin_num)  # Sum case size distribution
    special_step_distribution_sum = 2                # Time step for detailed sum analysis
    grain_size_distribution_iso = np.zeros(bin_num)  # Isotropic case size distribution
    special_step_distribution_iso = 2                # Time step for detailed iso analysis

    # ===============================================================================
    # XY PLANE POLAR VISUALIZATION: GRAIN BOUNDARY NORMAL VECTOR DISTRIBUTIONS
    # ===============================================================================
    
    # -----------------------------------------------------------------------
    # XY PLANE POLAR PLOT INITIALIZATION
    # -----------------------------------------------------------------------
    
    # Clear any existing plots and create new polar coordinate figure
    plt.close()                                    # Close previous figures to prevent overlay
    fig = plt.figure(figsize=(5, 5))              # Create square figure for circular polar plot
    ax = plt.gca(projection='polar')              # Set up polar coordinate system
    
    # Configure angular grid and labels for the polar plot
    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)  # Angular grid every 20 degrees
    ax.set_thetamin(0.0)                          # Set minimum angle to 0 degrees
    ax.set_thetamax(360.0)                        # Set maximum angle to 360 degrees
    
    # Configure radial grid and labels
    ax.set_rgrids(np.arange(0, 0.008, 0.004))     # Radial grid lines at 0, 0.004, 0.008
    ax.set_rlabel_position(0.0)                   # Position radial labels at 0 degrees
    ax.set_rlim(0.0, 0.008)                       # Set radial limits from 0 to 0.008 (probability density)
    ax.set_yticklabels(['0', '0.004'],fontsize=14) # Custom radial tick labels
    
    # Configure plot appearance and grid
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)  # Black grid with transparency
    ax.set_axisbelow('True')                      # Place grid behind data plots
    
    # -----------------------------------------------------------------------
    # XY PLANE ANALYSIS: ANISOTROPIC AVERAGE ENERGY CASE
    # -----------------------------------------------------------------------
    
    # Set up file paths for computational caching of expensive 3D normal vector calculations
    data_file_name_P = f'/normal_distribution_data/3D_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/3D_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    
    # Check if pre-computed data exists to avoid redundant expensive calculations
    if os.path.exists(current_path + data_file_name_P):
        # Load cached results for efficiency
        P = np.load(current_path + data_file_name_P)          # Smoothed phase field
        sites = np.load(current_path + data_file_name_sites)  # Grain boundary sites
    else:
        # Perform fresh 3D normal vector calculation if cache doesn't exist
        # Apply 90-degree rotation to align with coordinate system convention
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)  # Compute 3D normal vectors and sites
        
        # Cache results for future use to avoid recomputation
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)
    
    # Generate XY plane angular distribution (angle_index=0 for XY projection)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave case")
        
    # -----------------------------------------------------------------------
    # XY PLANE ANALYSIS: ANISOTROPIC MINIMUM ENERGY CASE
    # -----------------------------------------------------------------------
    
    # Set up file paths for minimum energy case caching
    data_file_name_P = f'/normal_distribution_data/3D_normal_distribution_min_P_step{special_step_distribution_min}.npy'
    data_file_name_sites = f'/normal_distribution_data/3D_normal_distribution_min_P_sites_step{special_step_distribution_min}.npy'
    
    # Load cached data or compute fresh results for minimum energy formulation
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        # Process minimum energy case with coordinate system rotation
        newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        # Cache results for efficiency in future runs
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)
    
    # Generate XY plane angular distribution for minimum energy case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_min, "min case")
        
    # -----------------------------------------------------------------------
    # XY PLANE ANALYSIS: ANISOTROPIC SUM ENERGY CASE
    # -----------------------------------------------------------------------
    
    # Set up file paths for sum energy case caching
    data_file_name_P = f'/normal_distribution_data/3D_normal_distribution_sum_P_step{special_step_distribution_sum}.npy'
    data_file_name_sites = f'/normal_distribution_data/3D_normal_distribution_sum_sites_step{special_step_distribution_sum}.npy'
    
    # Load cached data or compute fresh results for sum energy formulation
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        # Process sum energy case with coordinate system rotation
        newplace = np.rot90(npy_file_aniso_sum[special_step_distribution_sum,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        # Cache results for efficiency in future runs
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)
    
    # Generate XY plane angular distribution for sum energy case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_sum, "Sum case")
        
    # -----------------------------------------------------------------------
    # XY PLANE ANALYSIS: ISOTROPIC REFERENCE CASE
    # -----------------------------------------------------------------------
    
    # Set up file paths for isotropic reference case caching
    data_file_name_P = f'/normal_distribution_data/3D_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/3D_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    
    # Load cached data or compute fresh results for isotropic reference
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        # Process isotropic case (delta=0.0) with coordinate system rotation
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        # Cache results for efficiency in future runs
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)
    
    # Generate XY plane angular distribution for isotropic reference case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso case")
    
    # -----------------------------------------------------------------------
    # XY PLANE PLOT FINALIZATION AND EXPORT
    # -----------------------------------------------------------------------
    
    # Add legend and axis labels for the XY plane polar plot
    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)  # Position legend below plot with 2 columns
    plt.text(0.0, 0.0095, "x", fontsize=14)           # Label x-axis direction
    plt.text(np.pi/2, 0.0095, "y", fontsize=14)       # Label y-axis direction
    
    # Save XY plane normal distribution plot with high resolution
    plt.savefig(current_path + "/figures/normal_distribution_3d_xy.png", dpi=400,bbox_inches='tight')
    
    # ===============================================================================
    # XZ PLANE POLAR VISUALIZATION: GRAIN BOUNDARY NORMAL VECTOR DISTRIBUTIONS  
    # ===============================================================================
    
    # -----------------------------------------------------------------------
    # XZ PLANE POLAR PLOT INITIALIZATION
    # -----------------------------------------------------------------------
    
    # Clear previous plot and create new polar figure for XZ plane analysis
    plt.close()                                    # Close XY plane figure
    fig = plt.figure(figsize=(5, 5))              # Create new square figure
    ax = plt.gca(projection='polar')              # Initialize polar coordinate system
    
    # Configure angular grid and labels (identical to XY plane for consistency)
    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)  # 20-degree angular increments
    ax.set_thetamin(0.0)                          # Start angle at 0 degrees
    ax.set_thetamax(360.0)                        # End angle at 360 degrees
    
    # Configure radial grid and labels (matching XY plane scale)
    ax.set_rgrids(np.arange(0, 0.008, 0.004))     # Radial grid at 0, 0.004, 0.008
    ax.set_rlabel_position(0.0)                   # Position radial labels at 0 degrees
    ax.set_rlim(0.0, 0.008)                       # Radial limits for probability density
    ax.set_yticklabels(['0', '0.004'],fontsize=14) # Custom radial tick labels
    
    # Configure plot appearance
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)  # Grid styling
    ax.set_axisbelow('True')                      # Grid behind data
    
    # -----------------------------------------------------------------------
    # XZ PLANE ANALYSIS: ANISOTROPIC AVERAGE ENERGY CASE
    # -----------------------------------------------------------------------
    
    # Reuse cached data from XY plane analysis (same P and sites arrays)
    data_file_name_P = f'/normal_distribution_data/3D_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/3D_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    
    # Load cached 3D normal vector data (no need to recompute)
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        # Fallback computation if cache is missing
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)
    
    # Generate XZ plane angular distribution (angle_index=1 for XZ projection)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave case", 1)
        
    # -----------------------------------------------------------------------
    # XZ PLANE ANALYSIS: ANISOTROPIC MINIMUM ENERGY CASE
    # -----------------------------------------------------------------------
    
    # Load cached minimum energy case data
    data_file_name_P = f'/normal_distribution_data/3D_normal_distribution_min_P_step{special_step_distribution_min}.npy'
    data_file_name_sites = f'/normal_distribution_data/3D_normal_distribution_min_P_sites_step{special_step_distribution_min}.npy'
    
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        # Fallback computation for minimum energy case
        newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)
    
    # Generate XZ plane angular distribution for minimum energy case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_min, "min case", 1)
        
    # -----------------------------------------------------------------------
    # XZ PLANE ANALYSIS: ANISOTROPIC SUM ENERGY CASE
    # -----------------------------------------------------------------------
    
    # Load cached sum energy case data
    data_file_name_P = f'/normal_distribution_data/3D_normal_distribution_sum_P_step{special_step_distribution_sum}.npy'
    data_file_name_sites = f'/normal_distribution_data/3D_normal_distribution_sum_sites_step{special_step_distribution_sum}.npy'
    
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        # Fallback computation for sum energy case
        newplace = np.rot90(npy_file_aniso_sum[special_step_distribution_sum,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)
    
    # Generate XZ plane angular distribution for sum energy case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_sum, "Sum case", 1)
        
    # -----------------------------------------------------------------------
    # XZ PLANE ANALYSIS: ISOTROPIC REFERENCE CASE
    # -----------------------------------------------------------------------
    
    # Load cached isotropic reference case data
    data_file_name_P = f'/normal_distribution_data/3D_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/3D_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        # Fallback computation for isotropic reference case
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)
    
    # Generate XZ plane angular distribution for isotropic reference case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso case", 1)
    
    # -----------------------------------------------------------------------
    # XZ PLANE PLOT FINALIZATION AND EXPORT
    # -----------------------------------------------------------------------
    
    # Add legend and axis labels for the XZ plane polar plot
    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)  # Legend positioning
    plt.text(0.0, 0.0095, "x", fontsize=14)           # Label x-axis direction
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)       # Label z-axis direction
    
    # Save XZ plane normal distribution plot with high resolution
    plt.savefig(current_path + "/figures/normal_distribution_3d_xz.png", dpi=400,bbox_inches='tight')
    
    # ===============================================================================
    # YZ PLANE POLAR VISUALIZATION: GRAIN BOUNDARY NORMAL VECTOR DISTRIBUTIONS
    # ===============================================================================
    
    # -----------------------------------------------------------------------
    # YZ PLANE POLAR PLOT INITIALIZATION
    # -----------------------------------------------------------------------
    
    # Clear previous plot and create new polar figure for YZ plane analysis
    plt.close()                                    # Close XZ plane figure
    fig = plt.figure(figsize=(5, 5))              # Create new square figure
    ax = plt.gca(projection='polar')              # Initialize polar coordinate system
    
    # Configure angular grid and labels (consistent with XY and XZ planes)
    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)  # 20-degree angular increments
    ax.set_thetamin(0.0)                          # Start angle at 0 degrees
    ax.set_thetamax(360.0)                        # End angle at 360 degrees
    
    # Configure radial grid and labels (matching previous planes)
    ax.set_rgrids(np.arange(0, 0.008, 0.004))     # Radial grid at 0, 0.004, 0.008
    ax.set_rlabel_position(0.0)                   # Position radial labels at 0 degrees
    ax.set_rlim(0.0, 0.008)                       # Radial limits for probability density
    ax.set_yticklabels(['0', '0.004'],fontsize=14) # Custom radial tick labels
    
    # Configure plot appearance (consistent styling)
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)  # Grid styling
    ax.set_axisbelow('True')                      # Grid behind data
    
    # -----------------------------------------------------------------------
    # YZ PLANE ANALYSIS: ANISOTROPIC AVERAGE ENERGY CASE
    # -----------------------------------------------------------------------
    
    # Reuse cached data from previous analyses (same 3D normal vector computation)
    data_file_name_P = f'/normal_distribution_data/3D_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/3D_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    
    # Load cached 3D normal vector data (efficient reuse)
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        # Fallback computation if cache is missing
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)
    
    # Generate YZ plane angular distribution (angle_index=2 for YZ projection)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave case", 2)
        
    # -----------------------------------------------------------------------
    # YZ PLANE ANALYSIS: ANISOTROPIC MINIMUM ENERGY CASE
    # -----------------------------------------------------------------------
    
    # Load cached minimum energy case data
    data_file_name_P = f'/normal_distribution_data/3D_normal_distribution_min_P_step{special_step_distribution_min}.npy'
    data_file_name_sites = f'/normal_distribution_data/3D_normal_distribution_min_P_sites_step{special_step_distribution_min}.npy'
    
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        # Fallback computation for minimum energy case
        newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)
    
    # Generate YZ plane angular distribution for minimum energy case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_min, "min case", 2)
        
    # -----------------------------------------------------------------------
    # YZ PLANE ANALYSIS: ANISOTROPIC SUM ENERGY CASE
    # -----------------------------------------------------------------------
    
    # Load cached sum energy case data
    data_file_name_P = f'/normal_distribution_data/3D_normal_distribution_sum_P_step{special_step_distribution_sum}.npy'
    data_file_name_sites = f'/normal_distribution_data/3D_normal_distribution_sum_sites_step{special_step_distribution_sum}.npy'
    
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        # Fallback computation for sum energy case
        newplace = np.rot90(npy_file_aniso_sum[special_step_distribution_sum,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)
    
    # Generate YZ plane angular distribution for sum energy case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_sum, "Sum case", 2)
        
    # -----------------------------------------------------------------------
    # YZ PLANE ANALYSIS: ISOTROPIC REFERENCE CASE
    # -----------------------------------------------------------------------
    
    # Load cached isotropic reference case data
    data_file_name_P = f'/normal_distribution_data/3D_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/3D_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        # Fallback computation for isotropic reference case
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)
    
    # Generate YZ plane angular distribution for isotropic reference case
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso case", 2)
    
    # -----------------------------------------------------------------------
    # YZ PLANE PLOT FINALIZATION AND EXPORT
    # -----------------------------------------------------------------------
    
    # Add legend and axis labels for the YZ plane polar plot
    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)  # Legend positioning
    plt.text(0.0, 0.0095, "y", fontsize=14)           # Label y-axis direction  
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)       # Label z-axis direction
    
    # Save YZ plane normal distribution plot with high resolution
    plt.savefig(current_path + "/figures/normal_distribution_3d_yz.png", dpi=400,bbox_inches='tight')

# ===============================================================================
# ANALYSIS COMPLETION SUMMARY
# ===============================================================================

# This script has successfully completed the following operations:
# 
# 1. **3D Data Processing**: Loaded and processed four different energy formulation 
#    datasets (ave, min, sum anisotropic cases + isotropic reference)
#
# 2. **Grain Evolution Analysis**: Tracked grain count evolution across time steps
#    for all energy formulations to understand growth kinetics
#
# 3. **3D Normal Vector Computation**: Applied VECTOR framework's 3D linear smoothing
#    algorithm to extract grain boundary normal vectors with caching for efficiency
#
# 4. **Multi-Planar Projection Analysis**: Generated orientation distributions for
#    three orthogonal projection planes (XY, XZ, YZ) enabling comprehensive 3D texture analysis
#
# 5. **Comparative Visualization**: Created polar plots comparing anisotropic vs isotropic
#    behavior across different energy formulations and projection planes
#
# **Output Files Generated**:
# - /figures/normal_distribution_3d_xy.png: XY plane orientation distributions
# - /figures/normal_distribution_3d_xz.png: XZ plane orientation distributions  
# - /figures/normal_distribution_3d_yz.png: YZ plane orientation distributions
#
# **Cached Data Files** (for computational efficiency):
# - /normal_distribution_data/3D_normal_distribution_*_P_step*.npy: Smoothed phase fields
# - /normal_distribution_data/3D_normal_distribution_*_sites_step*.npy: Grain boundary sites
#
# This comprehensive analysis enables researchers to understand grain boundary character
# distribution (GBCD) effects in 3D polycrystalline materials under different energy
# formulations and assess the impact of anisotropy on microstructural texture evolution.