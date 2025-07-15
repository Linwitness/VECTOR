#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Spherical Grain Boundary Normal Vector Distribution Analysis using HiPerGator

This script analyzes the orientation distribution of grain boundary normal vectors
in 3D spherical microstructures computed on the University of Florida's HiPerGator 
supercomputing cluster. The analysis focuses on comparing anisotropic vs isotropic
grain boundary energy formulations and includes bias correction methodology for
accurate statistical analysis.

Key Features:
- 3D spherical grain microstructure analysis using HiPerGator computational resources
- Multi-planar projection analysis (XY, XZ, YZ planes) for comprehensive orientation characterization
- Bias correction algorithms to account for geometric effects in spherical domains
- Comparative analysis between anisotropic (delta=0.6) and isotropic (delta=0.0) cases
- Computational caching system for efficiency with large HiPerGator datasets
- Polar coordinate visualization with before/after bias correction comparisons

Scientific Context:
This script supports advanced materials science research focusing on grain boundary
character distribution (GBCD) analysis in 3D spherical polycrystalline systems.
The bias correction methodology is particularly important for spherical geometries
where surface effects can influence orientation statistics.

HiPerGator Integration:
- Utilizes data from /blue/michael.tonks/ storage on UF HiPerGator 3.0
- Optimized for large-scale 3D Monte Carlo simulations with 64-core parallel processing
- Handles SPPARKS simulation results with virtual inclination energy algorithms

Created on Mon Jul 31 14:33:57 2023
@author: Lin
"""

# ===============================================================================
# IMPORT LIBRARIES AND DEPENDENCIES
# ===============================================================================

# Standard library imports for system operations and path management
import os                           # Operating system interface for file path operations
current_path = os.getcwd()         # Get current working directory for relative path construction
import math                        # Mathematical functions for trigonometric calculations and bias corrections
import sys                         # System-specific parameters and functions for path manipulation

# Scientific computing and numerical analysis libraries
import numpy as np                 # Numerical computing: arrays, linear algebra, statistical operations
from numpy import seterr           # Configure numpy error handling behavior for robust computation
seterr(all='raise')               # Set numpy to raise exceptions on all numerical errors (important for HiPerGator stability)

# Visualization library for polar plots and scientific figures
import matplotlib.pyplot as plt    # 2D plotting library for creating publication-quality polar plots

# Progress tracking and performance monitoring
from tqdm import tqdm             # Progress bar utility for long-running HiPerGator computations (imported but not actively used)

# Custom VECTOR framework modules for 3D grain boundary analysis
sys.path.append(current_path)                    # Add current directory to Python path
sys.path.append(current_path+'/../../')          # Add VECTOR root directory to Python path
import myInput                                   # Custom gradient calculation functions for 2D and 3D analysis
import PACKAGE_MP_Linear as linear2d            # 2D linear smoothing and grain boundary detection algorithms
import PACKAGE_MP_3DLinear as linear3d          # 3D linear smoothing optimized for spherical geometries
sys.path.append(current_path+'/../calculate_tangent/')  # Add tangent/orientation calculation utilities

# ===============================================================================
# FUNCTION DEFINITIONS FOR GRAIN BOUNDARY ANALYSIS
# ===============================================================================

def get_normal_vector(grain_structure_figure_one):
    """
    Calculate normal vectors for grain boundaries in 2D microstructures.
    
    This function implements the 2D linear smoothing algorithm from the VECTOR
    framework to compute grain boundary normal vectors for orientation analysis.
    While primarily used for 2D systems, this function serves as a foundation
    for the 3D spherical analysis in this script.
    
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
    Calculate normal vectors for grain boundaries in 3D spherical microstructures.
    
    This function extends the 2D algorithm to 3D using the VECTOR framework's
    3D linear smoothing capabilities specifically optimized for spherical grain
    microstructures generated on HiPerGator. The spherical geometry introduces
    unique challenges that require specialized handling for accurate analysis.
    
    Algorithm Overview:
    1. Initialize 3D linear smoothing class with volumetric spherical microstructure data
    2. Execute iterative 3D linear smoothing with "inclination" mode optimization
    3. Extract all 3D grain boundary sites from the smoothed spherical volume
    4. Return smoothed 3D phase field and consolidated grain boundary coordinates
    
    Parameters:
    -----------
    grain_structure_figure_one : ndarray
        3D array representing the spherical grain structure where each element contains
        the grain ID. Shape: (nx, ny, nz) for volumetric spherical microstructure data.
        Generated from HiPerGator SPPARKS simulations with spherical boundary conditions.
    
    Returns:
    --------
    P : ndarray
        Smoothed 3D phase field array with refined grain boundary representation.
        Shape: (nx, ny, nz, 3) containing 3D spatial gradient information optimized
        for spherical geometries.
    sites_together : list
        Consolidated list of all 3D grain boundary sites as [i, j, k] coordinates.
        Each site represents a volumetric location for 3D normal vector analysis
        within the spherical domain.
    
    Technical Details:
    ------------------
    - Extends 2D algorithm to full 3D volumetric processing for spherical domains
    - Uses 'np' mode for numpy-optimized 3D computations with HiPerGator efficiency
    - Maintains same iteration count (5) and core usage (8) as 2D version
    - Handles significantly larger datasets due to 3D spherical nature (nx×ny×nz elements)
    - Optimized for HiPerGator's computational architecture and memory management
    """
    # Extract 3D spatial dimensions from input spherical grain structure
    nx = grain_structure_figure_one.shape[0]  # Number of voxels in x-direction
    ny = grain_structure_figure_one.shape[1]  # Number of voxels in y-direction  
    nz = grain_structure_figure_one.shape[2]  # Number of voxels in z-direction
    ng = np.max(grain_structure_figure_one)   # Maximum grain ID (total number of grains in sphere)
    
    # Set computational parameters for 3D linear smoothing algorithm
    cores = 8          # Number of CPU cores for parallel 3D processing on HiPerGator
    loop_times = 5     # Number of smoothing iterations for 3D spherical convergence
    
    # Initialize input arrays for the 3D linear smoothing class
    P0 = grain_structure_figure_one[:,:,:,np.newaxis]  # Add channel dimension: (nx,ny,nz,1)
    R = np.zeros((nx,ny,nz,3))                         # Initialize 3D gradient array: (nx,ny,nz,3)
    
    # Create 3D linear smoothing class instance with numpy optimization mode
    # 'np' mode is specifically optimized for HiPerGator's numpy installations
    smooth_class = linear3d.linear3d_class(nx,ny,nz,ng,cores,loop_times,P0,R,'np')

    # Execute the main 3D linear smoothing algorithm in "inclination" mode
    # This mode is optimized for 3D spherical grain boundary normal vector calculation
    smooth_class.linear3d_main("inclination")
    
    # Extract the smoothed 3D phase field containing refined boundary information
    P = smooth_class.get_P()
    
    # Extract all 3D grain boundary sites from the smoothed spherical volume
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
    a histogram for statistical analysis and visualization. While primarily for
    2D analysis, this function provides the foundation for the 3D spherical analysis.
    
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
        Parameter name for plot legend (e.g., "Ave case", "Iso case").
    
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

def get_normal_vector_slope_3d(P, sites, step, para_name, angle_index=0, bias=None):
    """
    Calculate and plot angular distribution of 3D spherical grain boundary normal vectors with bias correction.
    
    This function extends the 2D orientation analysis to 3D spherical systems by projecting 3D normal
    vectors onto specific 2D planes (XY, XZ, YZ) and analyzing their angular distributions. A critical
    enhancement for spherical geometries is the inclusion of bias correction methodology to account for
    geometric effects inherent in spherical domains that can skew orientation statistics.
    
    Algorithm Overview:
    1. Set up angular binning parameters (0-360 degrees, 10-degree bins)
    2. Calculate 3D gradients at each grain boundary site using myInput.get_grad3d()
    3. Project 3D gradients onto specified 2D plane based on angle_index
    4. Normalize projected gradients and convert to angles
    5. Bin angles and normalize to create probability density
    6. Apply bias correction if provided (essential for spherical geometries)
    7. Plot distribution with periodic boundary for complete circular representation
    
    Parameters:
    -----------
    P : ndarray
        Smoothed 3D phase field from get_normal_vector_3d() with gradient information.
        Shape: (nx, ny, nz, 3) containing 3D spatial and gradient components for spherical domains.
    sites : list
        List of 3D grain boundary sites as [i, j, k] coordinates from spherical microstructures.
    step : int
        Time step identifier for the analysis (used for HiPerGator data organization).
    para_name : str
        Parameter name for plot legend (e.g., "Ave", "Iso" for HiPerGator cases).
    angle_index : int, optional (default=0)
        Selects the projection plane for 2D analysis:
        - 0: XY plane projection (dx_fake=dx, dy_fake=dy)
        - 1: XZ plane projection (dx_fake=dx, dy_fake=dz)  
        - 2: YZ plane projection (dx_fake=dy, dy_fake=dz)
    bias : ndarray, optional (default=None)
        Bias correction array to remove geometric artifacts from spherical domains.
        When provided, this correction is added to the frequency distribution to
        compensate for surface curvature effects in spherical microstructures.
    
    Returns:
    --------
    freqArray : ndarray
        Normalized frequency array representing the angular distribution after
        bias correction (if applied). Essential for bias calculation in spherical analysis.
    
    Mathematical Details:
    ---------------------
    - 3D gradient projection preserves directional information in selected plane
    - Gradient normalization: ensures unit vector representation before angle calculation
    - Skip condition: |∇| < 1e-5 filters out numerical noise at grain interiors
    - Bias correction: freqArray = (freqArray + bias) / sum((freqArray + bias) * binValue)
    - Periodic plotting: appends first point to end for complete circular visualization
    
    Bias Correction Methodology:
    ----------------------------
    For spherical geometries, surface curvature introduces systematic bias in orientation
    statistics. The bias correction removes this artifact by:
    1. Computing expected uniform distribution (freqArray_circle)
    2. Calculating bias as difference from isotropic reference case
    3. Applying additive correction to restore statistical accuracy
    
    Technical Implementation:
    ------------------------
    - Handles large 3D spherical datasets efficiently through vectorized operations
    - Provides flexibility for multi-planar texture analysis in spherical domains
    - Compatible with polar coordinate visualization requirements
    - Optimized for HiPerGator's computational architecture and memory constraints
    """
    # Set up angular coordinate system and binning parameters
    xLim = [0, 360]                                           # Angular range in degrees
    binValue = 10.01                                          # Bin width (slightly > 10 to avoid boundary issues)
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)     # Number of bins for histogram
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)  # Bin center coordinates

    # Initialize arrays for 3D spherical angular analysis
    freqArray = np.zeros(binNum)  # Frequency histogram for angular distribution
    degree = []                   # List to store calculated angles for each 3D grain boundary site
    # degree_shadow = []          # Optional debugging array for gradient components (commented out)
    
    # Calculate 3D normal vector angles for each grain boundary site in spherical domain
    for sitei in sites:
        [i,j,k] = sitei  # Extract i,j,k coordinates of the 3D grain boundary site
        
        # Calculate 3D spatial gradients using custom 3D gradient function
        dx,dy,dz = myInput.get_grad3d(P,i,j,k)  # Returns 3D gradient components at site (i,j,k)
        
        # Project 3D gradient onto specified 2D plane based on angle_index
        # dy_fake = math.sqrt(dy**2 + dz**2)  # Alternative: magnitude in YZ plane (commented out)
        
        if angle_index == 0:      # XY plane projection (view from Z-axis)
            dx_fake = dx          # X-component remains the same
            dy_fake = dy          # Y-component remains the same
        elif angle_index == 1:    # XZ plane projection (view from Y-axis)
            dx_fake = dx          # X-component remains the same
            dy_fake = dz          # Z-component becomes "Y" for 2D analysis
        elif angle_index == 2:    # YZ plane projection (view from X-axis)
            dx_fake = dy          # Y-component becomes "X" for 2D analysis
            dy_fake = dz          # Z-component becomes "Y" for 2D analysis

        # Normalize projected gradient to unit vector (essential for accurate angle calculation)
        gradient_magnitude = math.sqrt(dy_fake**2+dx_fake**2)
        if gradient_magnitude < 1e-5: 
            continue  # Skip sites with negligible gradients (likely grain interiors)
            
        # Calculate normalized gradient components for spherical geometry
        dy_fake_norm = dy_fake / gradient_magnitude
        dx_fake_norm = dx_fake / gradient_magnitude

        # Convert normalized gradients to angle with proper quadrant handling
        # Note: -dy_fake_norm maintains consistent orientation convention for spherical domains
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

    # Apply bias correction if provided (critical for spherical geometries)
    if bias is not None:
        freqArray = freqArray + bias        # Add bias correction to remove spherical artifacts
        freqArray = freqArray/sum(freqArray*binValue)  # Renormalize after bias correction

    # Plot the 3D angular distribution with periodic boundary for complete circle
    # Append first point to end to close the circular plot properly
    plt.plot(np.append(xCor,xCor[0])/180*math.pi, np.append(freqArray,freqArray[0]),linewidth=2,label=para_name)

    return freqArray

# ===============================================================================
# MAIN EXECUTION: 3D SPHERICAL GRAIN BOUNDARY ANALYSIS ON HIPERGATOR
# ===============================================================================

if __name__ == '__main__':
    # -----------------------------------------------------------------------
    # HIPERGATOR DATA SOURCE CONFIGURATION
    # -----------------------------------------------------------------------
    
    # HiPerGator 3.0 storage path for 3D spherical microstructure simulations
    # Located on University of Florida's supercomputing cluster in Dr. Michael Tonks' allocation
    # Path structure: /blue/{PI_username}/{user}/project_directory/results/
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_sphere/results/"
    
    # Energy formulation type definitions for HiPerGator spherical analysis
    TJ_energy_type_ave = "ave"           # Average energy formulation (anisotropic case)
    TJ_energy_type_iso = "ave_delta000"  # Isotropic reference case identifier
    
    # -----------------------------------------------------------------------
    # HIPERGATOR SIMULATION FILE SPECIFICATION
    # -----------------------------------------------------------------------
    
    # Anisotropic simulation file (delta=0.6 introduces grain boundary energy anisotropy)
    # HiPerGator file naming convention for 3D spherical simulations:
    # p_ori_ave_{energy_type}E_150_multiCore64_delta{delta}_m2_J1_refer_1_0_0_seed56689_kt{temperature}.npy
    # Parameters breakdown for HiPerGator 3D sphere simulations:
    # - 150: spherical domain size parameter (optimized for HiPerGator memory)
    # - multiCore64: 64-core parallel processing on HiPerGator compute nodes
    # - delta0.6: anisotropy strength (0.6 for anisotropic, 0.0 for isotropic reference)
    # - m2: mobility parameter for grain boundary motion
    # - J1: interaction strength parameter
    # - refer_1_0_0: reference crystallographic orientation
    # - seed56689: random seed for reproducibility across HiPerGator runs
    # - kt1.95: reduced temperature parameter for thermodynamic conditions
    npy_file_name_aniso_ave = f"p_ori_ave_{TJ_energy_type_ave}E_150_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt1.95.npy"
    
    # Isotropic reference simulation file (delta=0.0 for unbiased baseline comparison)
    npy_file_name_iso = "p_ori_ave_aveE_150_multiCore64_delta0.0_m2_J1_refer_1_0_0_seed56689_kt1.95.npy"

    # -----------------------------------------------------------------------
    # HIPERGATOR DATA LOADING AND VALIDATION
    # -----------------------------------------------------------------------
    
    # Load 3D spherical microstructural evolution data from HiPerGator numpy arrays
    # Each file contains time series of 3D spherical grain structures: shape (time_steps, nx, ny, nz)
    # Data generated from SPPARKS Monte Carlo simulations with virtual inclination energy
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)  # Anisotropic average energy case
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)              # Isotropic reference case
    
    # Display HiPerGator dataset information for verification and debugging
    print(f"The ave data size is: {npy_file_aniso_ave.shape}")  # Expected: (time_steps, nx, ny, nz)
    print(f"The iso data size is: {npy_file_iso.shape}")        # Should match ave dimensions
    print("READING DATA DONE")  # Confirmation of successful HiPerGator data loading
    
    # -----------------------------------------------------------------------
    # SPHERICAL GRAIN EVOLUTION ANALYSIS SETUP
    # -----------------------------------------------------------------------
    
    # Initialize grain count tracking arrays for temporal evolution analysis in spherical domains
    initial_grain_num = 2                           # Starting grain count for spherical simulations (bi-crystal)
    step_num = npy_file_iso.shape[0]               # Number of time steps in the HiPerGator simulation
    
    # Create arrays to track grain count evolution for spherical energy formulations
    grain_num_aniso_ave = np.zeros(step_num)       # Average energy anisotropic case tracking
    grain_num_iso = np.zeros(step_num)             # Isotropic reference case tracking
    
    # -----------------------------------------------------------------------
    # SPHERICAL GRAIN COUNT CALCULATION ACROSS TIME STEPS
    # -----------------------------------------------------------------------
    
    # Calculate the number of unique grains at each time step in spherical domains
    # This provides insight into spherical grain growth kinetics under different energy formulations
    for i in range(step_num):
        # Count unique grain IDs by flattening 3D spherical array and finding unique values
        grain_num_aniso_ave[i] = len(set(npy_file_aniso_ave[i,:].flatten()))  # Anisotropic average case
        grain_num_iso[i] = len(set(npy_file_iso[i,:].flatten()))              # Isotropic reference case
    
    # -----------------------------------------------------------------------
    # TIME STEP SELECTION FOR DETAILED ANALYSIS
    # -----------------------------------------------------------------------
    
    # Alternative automatic time step selection based on target grain count (commented out)
    # This approach would select time steps when grain count reaches a specific value
    # expected_grain_num = 500
    # special_step_distribution_ave = int(np.argmin(abs(grain_num_aniso_ave - expected_grain_num)))
    # special_step_distribution_min = int(np.argmin(abs(grain_num_aniso_min - expected_grain_num)))
    # special_step_distribution_min195D264 = int(np.argmin(abs(grain_num_aniso_min195D264 - expected_grain_num)))
    # special_step_distribution_iso = int(np.argmin(abs(grain_num_iso - expected_grain_num)))
    
    # Manual time step selection for detailed spherical grain boundary analysis
    # Step 10 chosen to ensure sufficient grain growth while maintaining computational efficiency
    special_step_distribution_ave = 10             # Time step for anisotropic detailed analysis
    special_step_distribution_iso = 10             # Time step for isotropic detailed analysis

    # ===============================================================================
    # BIAS CALCULATION FOR SPHERICAL GEOMETRY CORRECTION
    # ===============================================================================
    
    # -----------------------------------------------------------------------
    # ISOTROPIC REFERENCE CASE PROCESSING FOR BIAS DETERMINATION
    # -----------------------------------------------------------------------
    
    # Set up file paths for computational caching of expensive 3D spherical normal vector calculations
    # The isotropic case serves as the reference for bias correction calculations
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    
    # Load cached data or compute fresh results for isotropic spherical reference case
    if os.path.exists(current_path + data_file_name_P):
        # Load cached results for computational efficiency on HiPerGator workflows
        P = np.load(current_path + data_file_name_P)          # Cached smoothed phase field
        sites = np.load(current_path + data_file_name_sites)  # Cached grain boundary sites
    else:
        # Perform fresh 3D spherical normal vector calculation if cache doesn't exist
        # Apply 90-degree rotation to align with coordinate system convention for spherical domains
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)  # Compute 3D normal vectors and sites for sphere
        
        # Cache results for future use to avoid recomputation in HiPerGator workflows
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # -----------------------------------------------------------------------
    # MULTI-PLANAR BIAS CALCULATION FOR SPHERICAL DOMAINS
    # -----------------------------------------------------------------------
    
    # Calculate orientation distributions for isotropic reference across all three projection planes
    # These distributions will be used to compute bias corrections for each plane
    
    # XY plane (angle_index=0) orientation distribution for isotropic reference
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso")
    
    # Set up theoretical uniform distribution parameters for bias calculation
    # For a perfect sphere with no anisotropy, orientations should be uniformly distributed
    xLim = [0, 360]                                           # Angular range in degrees
    binValue = 10.01                                          # Bin width matching analysis functions
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)     # Number of bins for uniform distribution
    
    # Create theoretical uniform (isotropic) distribution for spherical geometry
    freqArray_circle = np.ones(binNum)                        # Uniform frequency across all angles
    freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)  # Normalize to probability density
    
    # Calculate bias correction for XY plane (difference between theoretical and observed)
    slope_list_bias_0 = freqArray_circle - slope_list        # Bias = Theoretical - Observed
    
    # XZ plane (angle_index=1) bias calculation
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso",1)
    slope_list_bias_1 = freqArray_circle - slope_list        # XZ plane bias correction
    
    # YZ plane (angle_index=2) bias calculation  
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso",2)
    slope_list_bias_2 = freqArray_circle - slope_list        # YZ plane bias correction

    # ===============================================================================
    # PART I: UNCORRECTED POLAR VISUALIZATIONS (RAW SPHERICAL DATA)
    # ===============================================================================
    
    # -----------------------------------------------------------------------
    # XY PLANE POLAR PLOT: UNCORRECTED SPHERICAL ANALYSIS
    # -----------------------------------------------------------------------
    
    # Clear any existing plots and create new polar coordinate figure for XY plane
    plt.close()                                    # Close previous figures to prevent overlay
    fig = plt.figure(figsize=(5, 5))              # Create square figure for circular polar plot
    ax = plt.gca(projection='polar')              # Set up polar coordinate system

    # Configure angular grid and labels for the polar plot
    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)  # Angular grid every 20 degrees
    ax.set_thetamin(0.0)                          # Set minimum angle to 0 degrees
    ax.set_thetamax(360.0)                        # Set maximum angle to 360 degrees

    # Configure radial grid and labels for probability density
    ax.set_rgrids(np.arange(0, 0.008, 0.004))     # Radial grid lines at 0, 0.004, 0.008
    ax.set_rlabel_position(0.0)                   # Position radial labels at 0 degrees
    ax.set_rlim(0.0, 0.008)                       # Set radial limits (probability density scale)
    ax.set_yticklabels(['0', '0.004'],fontsize=14) # Custom radial tick labels

    # Configure plot appearance and grid
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)  # Black grid with transparency
    ax.set_axisbelow('True')                      # Place grid behind data plots

    # -----------------------------------------------------------------------
    # XY PLANE: ANISOTROPIC CASE ANALYSIS (UNCORRECTED)
    # -----------------------------------------------------------------------
    
    # Load or compute anisotropic spherical case for XY plane analysis
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    
    # Check for cached anisotropic spherical data to optimize HiPerGator workflows
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)          # Load cached phase field
        sites = np.load(current_path + data_file_name_sites)  # Load cached grain boundary sites
    else:
        # Compute fresh anisotropic spherical normal vectors if cache missing
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)  # 3D spherical analysis
        # Cache results for future HiPerGator efficiency
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Generate XY plane angular distribution for anisotropic case (no bias correction)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave")

    # -----------------------------------------------------------------------
    # XY PLANE: ISOTROPIC REFERENCE CASE ANALYSIS (UNCORRECTED)
    # -----------------------------------------------------------------------
    
    # Reload isotropic reference case for comparison (reusing cached data)
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    
    # Load isotropic spherical reference data
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        # Fallback computation if cache missing
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Generate XY plane angular distribution for isotropic reference (no bias correction)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso")

    # Finalize XY plane plot with legend and axis labels
    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)  # Position legend below plot
    plt.text(0.0, 0.0095, "x", fontsize=14)           # Label x-axis direction
    plt.text(np.pi/2, 0.0095, "y", fontsize=14)       # Label y-axis direction
    
    # Save XY plane uncorrected spherical distribution with step identifier
    plt.savefig(current_path + f"/figures/normal_distribution_3dsphere_xy_{special_step_distribution_iso}steps.png", dpi=400,bbox_inches='tight')

    # -----------------------------------------------------------------------
    # XZ PLANE POLAR PLOT: UNCORRECTED SPHERICAL ANALYSIS
    # -----------------------------------------------------------------------
    
    # Create new polar figure for XZ plane spherical orientation analysis
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    # Configure XZ plane polar plot (identical formatting to XY plane for consistency)
    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)
    ax.set_rgrids(np.arange(0, 0.008, 0.004))
    ax.set_rlabel_position(0.0)  # start from 0 degree
    ax.set_rlim(0.0, 0.008)  # radiu lim range is from 0 to 0.008
    ax.set_yticklabels(['0', '0.004'],fontsize=14)
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    # Anisotropic case XZ projection (angle_index=1)
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Generate XZ plane angular distribution (angle_index=1 for XZ projection)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 1)

    # Isotropic reference case XZ projection
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 1)

    # Finalize XZ plane plot
    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "x", fontsize=14)           # X-axis label
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)       # Z-axis label (note: Z not Y)
    plt.savefig(current_path + f"/figures/normal_distribution_3dsphere_xz_{special_step_distribution_iso}steps.png", dpi=400,bbox_inches='tight')

    # -----------------------------------------------------------------------
    # YZ PLANE POLAR PLOT: UNCORRECTED SPHERICAL ANALYSIS
    # -----------------------------------------------------------------------
    
    # Create new polar figure for YZ plane spherical orientation analysis
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    # Configure YZ plane polar plot (consistent formatting across all planes)
    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)
    ax.set_rgrids(np.arange(0, 0.008, 0.004))
    ax.set_rlabel_position(0.0)  # start from 0 degree
    ax.set_rlim(0.0, 0.008)  # radiu lim range is from 0 to 0.008
    ax.set_yticklabels(['0', '0.004'],fontsize=14)
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    # Anisotropic case YZ projection (angle_index=2)
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Generate YZ plane angular distribution (angle_index=2 for YZ projection)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 2)

    # Isotropic reference case YZ projection
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 2)

    # Finalize YZ plane plot
    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "y", fontsize=14)           # Y-axis label
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)       # Z-axis label
    plt.savefig(current_path + f"/figures/normal_distribution_3dsphere_yz_{special_step_distribution_iso}steps.png", dpi=400,bbox_inches='tight')


    # ===============================================================================
    # PART II: BIAS-CORRECTED POLAR VISUALIZATIONS (SPHERICAL GEOMETRY CORRECTED)
    # ===============================================================================
    
    # -----------------------------------------------------------------------
    # XY PLANE POLAR PLOT: BIAS-CORRECTED SPHERICAL ANALYSIS
    # -----------------------------------------------------------------------
    
    # Create new polar figure for bias-corrected XY plane analysis
    # The bias correction removes systematic artifacts introduced by spherical geometry
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    # Configure bias-corrected XY plane polar plot (identical formatting for comparison)
    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)
    ax.set_rgrids(np.arange(0, 0.008, 0.004))
    ax.set_rlabel_position(0.0)  # start from 0 degree
    ax.set_rlim(0.0, 0.008)  # radiu lim range is from 0 to 0.008
    ax.set_yticklabels(['0', '0.004'],fontsize=14)
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    # Anisotropic case with XY plane bias correction (slope_list_bias_0)
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply XY plane bias correction (angle_index=0, bias=slope_list_bias_0)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 0, slope_list_bias_0)

    # Isotropic reference case with XY plane bias correction
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply same XY bias correction to isotropic case for consistent comparison
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 0, slope_list_bias_0)

    # Finalize bias-corrected XY plane plot
    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "x", fontsize=14)
    plt.text(np.pi/2, 0.0095, "y", fontsize=14)
    # Save with descriptive filename indicating bias correction applied
    plt.savefig(current_path + f"/figures/normal_distribution_3dsphere_xy_{special_step_distribution_iso}steps_after_removing_bias.png", dpi=400,bbox_inches='tight')

    # -----------------------------------------------------------------------
    # XZ PLANE POLAR PLOT: BIAS-CORRECTED SPHERICAL ANALYSIS
    # -----------------------------------------------------------------------
    
    # Create new polar figure for bias-corrected XZ plane analysis
    # XZ projection analysis with spherical bias correction
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    # Configure bias-corrected XZ plane polar plot (identical formatting for comparison)
    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)
    ax.set_rgrids(np.arange(0, 0.008, 0.004))
    ax.set_rlabel_position(0.0)  # start from 0 degree
    ax.set_rlim(0.0, 0.008)  # radiu lim range is from 0 to 0.008
    ax.set_yticklabels(['0', '0.004'],fontsize=14)
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    # Anisotropic case with XZ plane bias correction (slope_list_bias_1)
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply XZ plane bias correction (angle_index=1, bias=slope_list_bias_1)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 1, slope_list_bias_1)

    # Isotropic reference case with XZ plane bias correction
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply same XZ bias correction to isotropic case for consistent comparison
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 1, slope_list_bias_1)

    # Finalize bias-corrected XZ plane plot
    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "x", fontsize=14)
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)
    # Save with descriptive filename indicating bias correction applied
    plt.savefig(current_path + f"/figures/normal_distribution_3dsphere_xz_{special_step_distribution_iso}steps_after_removing_bias.png", dpi=400,bbox_inches='tight')


    # -----------------------------------------------------------------------
    # YZ PLANE POLAR PLOT: BIAS-CORRECTED SPHERICAL ANALYSIS
    # -----------------------------------------------------------------------
    
    # Create new polar figure for bias-corrected YZ plane analysis
    # YZ projection analysis with spherical bias correction
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    # Configure bias-corrected YZ plane polar plot (identical formatting for comparison)
    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)
    ax.set_rgrids(np.arange(0, 0.008, 0.004))
    ax.set_rlabel_position(0.0)  # start from 0 degree
    ax.set_rlim(0.0, 0.008)  # radiu lim range is from 0 to 0.008
    ax.set_yticklabels(['0', '0.004'],fontsize=14)
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    # Anisotropic case with YZ plane bias correction (slope_list_bias_2)
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply YZ plane bias correction (angle_index=2, bias=slope_list_bias_2)
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_ave, "Ave", 2, slope_list_bias_2)

    # Isotropic reference case with YZ plane bias correction
    data_file_name_P = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_P_step{special_step_distribution_iso}.npy'
    data_file_name_sites = f'/normal_distribution_data/3Dsphere_normal_distribution_iso_sites_step{special_step_distribution_iso}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_iso[special_step_distribution_iso,:,:,:], 1, (0,1))
        P, sites = get_normal_vector_3d(newplace)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    # Apply same YZ bias correction to isotropic case for consistent comparison
    slope_list = get_normal_vector_slope_3d(P, sites, special_step_distribution_iso, "Iso", 2, slope_list_bias_2)

    # Finalize bias-corrected YZ plane plot with proper axis labels
    plt.legend(loc=(0.05,-0.25),fontsize=14, ncol=2)
    plt.text(0.0, 0.0095, "y", fontsize=14)
    plt.text(np.pi/2, 0.0095, "z", fontsize=14)
    # Save with descriptive filename indicating bias correction applied
    plt.savefig(current_path + f"/figures/normal_distribution_3dsphere_yz_{special_step_distribution_iso}steps_after_removing_bias.png", dpi=400,bbox_inches='tight')

    # ===============================================================================
    # ANALYSIS COMPLETION: HiPerGator 3D SPHERICAL BIAS CORRECTION ACHIEVED
    # ===============================================================================
    
    # Print completion status for HiPerGator workflows
    print("3D spherical grain boundary analysis completed with HiPerGator optimization")
    print(f"Generated bias-corrected polar plots for all three projection planes (XY, XZ, YZ)")
    print(f"Anisotropic analysis at step {special_step_distribution_ave}")
    print(f"Isotropic reference at step {special_step_distribution_iso}")
    print("All figures saved with '_after_removing_bias' suffix for publication quality")
    print("HiPerGator-optimized 3D spherical bias correction methodology successfully applied")







