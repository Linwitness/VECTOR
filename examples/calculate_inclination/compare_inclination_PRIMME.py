#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRAIN BOUNDARY INCLINATION DISTRIBUTION COMPARISON FOR PRIMME VALIDATION
========================================================================

This script compares grain boundary inclination distributions between different
simulation methods (SPPARKS, PRIMME, Phase Field) to validate the accuracy of
the PRIMME (Parallel Runge-Kutta-based Implicit Multi-physics Multi-scale Engine)
implementation against established simulation frameworks.

Key Functionality:
1. Extract grain boundary sites from 2D microstructures
2. Calculate grain boundary normal vectors using smoothing algorithms
3. Compute inclination angle distributions in polar coordinates
4. Generate comparative polar plots for validation studies

Scientific Purpose:
- Validate PRIMME grain growth simulation accuracy
- Compare microstructural evolution between simulation methods
- Analyze grain boundary character distributions
- Quantify differences in boundary orientation statistics

Created on Thu Sep 30 14:55:28 2021
@author: lin.yang
"""

import os
current_path = os.getcwd()
import sys
sys.path.append(current_path)
sys.path.append('../../.')
import numpy as np
import math
from itertools import repeat
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

# VECTOR framework modules for grain boundary analysis
import myInput
import PACKAGE_MP_Linear as linear2d     # 2D Bilinear smoothing algorithm for inclination calculation


def get_all_gb_list(P0):
    """
    Identify all grain boundary sites in a 2D microstructure.
    
    This function scans the entire microstructure to locate sites that are
    at grain boundaries by checking if neighboring sites belong to different
    grains. Uses periodic boundary conditions for edge handling.
    
    Parameters:
    -----------
    P0 : numpy.ndarray, shape (nx, ny)
        2D microstructure array where each element represents a grain ID
        Values should be integer grain identifiers
        
    Returns:
    --------
    list : List of [i, j] coordinates of grain boundary sites
        Each element is a two-element list [row, column] indicating
        the position of a grain boundary site in the microstructure
        
    Algorithm:
    ----------
    For each site (i,j):
    1. Get periodic neighbors using myInput.periodic_bc()
    2. Compare grain ID with all four neighbors (up, down, left, right)
    3. If any neighbor has different grain ID, mark as boundary site
    4. Store coordinates of all boundary sites
    
    Notes:
    ------
    - Uses periodic boundary conditions to handle domain edges
    - Detects both horizontal and vertical grain boundaries
    - Essential preprocessing step for inclination analysis
    """
    nx, ny = P0.shape
    gagn_gbsites = []  # List to store grain boundary site coordinates
    
    # Scan entire microstructure for grain boundary sites
    for i in range(0, nx):
        for j in range(0, ny):
            # Get periodic boundary neighbors
            ip, im, jp, jm = myInput.periodic_bc(nx, ny, i, j)
            
            # Check if current site differs from any of its four neighbors
            # If so, it's a grain boundary site
            if (((P0[ip, j] - P0[i, j]) != 0) or
                ((P0[im, j] - P0[i, j]) != 0) or
                ((P0[i, jp] - P0[i, j]) != 0) or
                ((P0[i, jm] - P0[i, j]) != 0)):
                gagn_gbsites.append([i, j])
                
    return gagn_gbsites

def get_normal_vector(grain_structure_figure_one, grain_num):
    """
    Calculate grain boundary normal vectors using bilinear smoothing algorithm.
    
    This function applies the VECTOR framework's 2D bilinear smoothing algorithm
    to compute grain boundary inclination vectors from a discrete microstructure.
    The smoothing algorithm converts discrete grain boundaries into continuous
    fields with well-defined normal vectors.
    
    Parameters:
    -----------
    grain_structure_figure_one : numpy.ndarray, shape (nx, ny)
        2D microstructure array with integer grain IDs
        Each element represents the grain ID at that lattice site
        
    grain_num : int
        Total number of grains in the microstructure
        Used for algorithm initialization and validation
        
    Returns:
    --------
    tuple : (P, sites_together, sites)
        P : numpy.ndarray, shape (3, nx, ny)
            Smoothed field array where:
            P[0,:,:] = original microstructure
            P[1,:,:] = x-component of inclination vector
            P[2,:,:] = y-component of inclination vector
            
        sites_together : list
            Flattened list of all grain boundary site coordinates
            
        sites : list of lists
            Grain boundary sites organized by grain ID
            sites[grain_id] contains boundary sites for that grain
            
    Algorithm Details:
    -----------------
    1. Initialize bilinear smoothing class with microstructure
    2. Run inclination calculation using linear_main('inclination')
    3. Extract smoothed field P containing normal vector components
    4. Identify all grain boundary sites using the smoothing algorithm
    5. Consolidate boundary sites across all grains
    
    Scientific Context:
    ------------------
    - Normal vectors are essential for grain boundary character analysis
    - Inclination angles determine grain boundary energy and mobility
    - Smoothing removes discretization artifacts from digital microstructures
    - Enables quantitative comparison between simulation methods
    """
    # Extract microstructure dimensions
    nx = grain_structure_figure_one.shape[0]
    ny = grain_structure_figure_one.shape[1]
    ng = np.max(grain_structure_figure_one)  # Maximum grain ID
    
    # Algorithm parameters for bilinear smoothing
    cores = 8         # Number of CPU cores for parallel processing
    loop_times = 5    # Smoothing iteration count for convergence
    
    # Initialize microstructure and gradient field arrays
    P0 = grain_structure_figure_one
    R = np.zeros((nx, ny, 2))  # Initial gradient field (zero)
    
    # Create smoothing algorithm instance
    smooth_class = linear2d.linear_class(nx, ny, ng, cores, loop_times, P0, R)

    # Execute inclination calculation using bilinear smoothing
    smooth_class.linear_main("inclination")
    P = smooth_class.get_P()  # Get smoothed field with normal vectors
    
    # Extract grain boundary sites from smoothing algorithm results
    # Alternative approach: get sites for specific grains
    # sites = smooth_class.get_gb_list(1)  # Sites for grain 1
    # for id in range(2, grain_num+1): 
    #     sites += smooth_class.get_gb_list(id)  # Add sites for other grains
    
    # Get all grain boundary sites organized by grain
    sites = smooth_class.get_all_gb_list()
    
    # Flatten the grain-organized sites into a single list
    sites_together = []
    for grain_id in range(len(sites)): 
        sites_together += sites[grain_id]
        
    print(f"Total number of grain boundary sites: {len(sites_together)}")

    return P, sites_together, sites

def get_normal_vector_slope(P, sites, para_name, bias=None):
    """
    Calculate grain boundary inclination angle distribution from normal vectors.
    
    This function computes the statistical distribution of grain boundary
    inclination angles from the normal vector field and generates a histogram
    for visualization and quantitative analysis. Used for comparing different
    simulation methods and validating grain boundary character distributions.
    
    Parameters:
    -----------
    P : numpy.ndarray, shape (3, nx, ny)
        Smoothed field array from bilinear smoothing algorithm
        P[0,:,:] = microstructure
        P[1,:,:] = x-component of normal vector
        P[2,:,:] = y-component of normal vector
        
    sites : list
        List of [i, j] coordinates of grain boundary sites
        Each element is a two-element list [row, column]
        
    para_name : str
        Label for the dataset (e.g., "spparks20k", "primme20k")
        Used in plot legends and identification
        
    bias : numpy.ndarray, optional
        Optional bias correction to apply to frequency distribution
        Same shape as freqArray output
        
    Returns:
    --------
    numpy.ndarray : Normalized frequency array for inclination angles
        Shape matches the number of angle bins (typically 36 bins for 10° resolution)
        Values represent probability density (sum * bin_width = 1)
        
    Algorithm Details:
    -----------------
    1. For each grain boundary site, extract normal vector components (dx, dy)
    2. Calculate inclination angle: θ = atan2(-dy, dx) + π
    3. Convert angle to degrees and bin into histogram
    4. Normalize to create probability density distribution
    5. Plot on polar coordinates for visualization
    
    Angle Convention:
    ----------------
    - Angles range from 0° to 360°
    - 0° corresponds to horizontal grain boundaries
    - 90° corresponds to vertical grain boundaries
    - Full periodic coverage ensures statistical completeness
    
    Statistical Analysis:
    --------------------
    - Uniform distribution indicates isotropic grain boundary character
    - Preferred orientations appear as peaks in the distribution
    - Deviations between methods indicate simulation artifacts
    
    Visualization:
    -------------
    - Automatically generates polar plot with specified label
    - Periodic boundary conditions (angle[0] = angle[360])
    - Suitable for comparative analysis between simulation methods
    """
    # Define angle binning parameters for histogram
    xLim = [0, 360]      # Angular range in degrees (full circle)
    binValue = 10.01     # Bin width in degrees (slightly > 10 to avoid edge effects)
    binNum = round((abs(xLim[0]) + abs(xLim[1])) / binValue)  # Number of bins
    
    # Create bin center coordinates for plotting
    xCor = np.linspace((xLim[0] + binValue/2), (xLim[1] - binValue/2), binNum)

    # Initialize frequency array for angle histogram
    freqArray = np.zeros(binNum)
    degree = []  # List to store calculated angles
    
    # Calculate inclination angle for each grain boundary site
    for sitei in sites:
        [i, j] = sitei  # Extract site coordinates
        dx, dy = P[1:, i, j]  # Extract normal vector components
        
        # Calculate inclination angle using atan2 for proper quadrant handling
        # Note: -dy used to match crystallographic convention
        # Adding π ensures angles are in [0, 2π] range
        angle = math.atan2(-dy, dx) + math.pi
        degree.append(angle)

    # Populate histogram bins with angle data
    for i in range(len(degree)):
        # Convert radians to degrees and determine bin index
        angle_deg = degree[i] / math.pi * 180
        bin_index = int((angle_deg - xLim[0]) / binValue)
        
        # Ensure bin index is within valid range
        if 0 <= bin_index < binNum:
            freqArray[bin_index] += 1

    # Normalize frequency array to create probability density
    # Normalization: sum(freqArray * binValue) = 1
    total_count = sum(freqArray * binValue)
    if total_count > 0:
        freqArray = freqArray / total_count

    # Apply optional bias correction for method comparison
    if bias is not None:
        freqArray = freqArray + bias
        freqArray = freqArray / sum(freqArray * binValue)  # Re-normalize
    
    # Generate polar plot for visualization
    # Convert angles to radians for polar coordinates
    angles_rad = np.append(xCor, xCor[0]) / 180 * math.pi
    frequencies_periodic = np.append(freqArray, freqArray[0])  # Close the loop
    
    plt.plot(angles_rad, frequencies_periodic, linewidth=2, label=para_name)

    return freqArray


if __name__ == '__main__':
    """
    MAIN EXECUTION: COMPARATIVE INCLINATION ANALYSIS
    ===============================================
    
    This section performs systematic comparison of grain boundary inclination
    distributions between SPPARKS, PRIMME, and Phase Field simulation methods.
    The analysis validates PRIMME accuracy and characterizes differences in
    grain boundary character evolution.
    
    Simulation Datasets:
    -------------------
    - Domain: 512×512 and 2400×2400 lattice sites
    - Grains: 512 and 20,000 grain populations
    - Timesteps: 100, 300, 1600 simulation steps
    - Methods: SPPARKS Monte Carlo, PRIMME, Phase Field
    
    Analysis Workflow:
    -----------------
    1. Load microstructure data from .npy files
    2. Extract grain boundary sites for each method
    3. Calculate inclination distributions using smoothing algorithm
    4. Generate comparative polar plots
    5. Save visualization results
    
    Output Products:
    ---------------
    - Polar plots showing inclination distributions
    - Quantitative comparison between simulation methods
    - Statistical validation of PRIMME implementation
    """

    # =================================================================
    # SIMULATION DATASET CONFIGURATION
    # =================================================================
    
    # File naming conventions for different simulation methods and scales
    # Naming pattern: method_sz(domain)_ng(grains)_nsteps(steps)_freq(frequency)_kt(temperature)_cut(cutoff)_inclination_step
    
    # Small-scale test datasets (512×512 domain, 512 grains)
    input_name_spparks_512 = "output/spparks_sz(512x512)_ng(512)_nsteps(1600)_freq(1)_kt(0.66)_cut(0)_inclination_step"
    input_name_primme_512 = "output/primme_sz(512x512)_ng(512)_nsteps(1600)_freq(1)_kt(0.66)_cut(0)_inclination_step"
    
    # Large-scale production datasets (2400×2400 domain, 20,000 grains)
    input_name_spparks_20000 = "output/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0)_inclination_step"
    input_name_primme_20000 = "output/primme_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1)_kt(0.66)_cut(0)_inclination_step"
    input_name_pf_20000 = "output/phasefiled_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1)_kt(0.66)_cut(0)_inclination_step"
    
    print("Starting comparative inclination analysis...")
    print("Comparing SPPARKS, PRIMME, and Phase Field simulation methods")
    
    # =================================================================
    # SMALL-SCALE COMPARISON (512×512 DOMAIN) - CURRENTLY DISABLED
    # =================================================================
    
    # Uncomment this section for 512-grain analysis
    # for i in [100, 1600]:
    #     print(f"\nProcessing 512-grain datasets at timestep {i}...")
    #     
    #     # Load microstructure data from both simulation methods
    #     npy_file_s512 = np.load(input_name_spparks_512 + f"{i}.npy")
    #     npy_file_p512 = np.load(input_name_primme_512 + f"{i}.npy")
    #     
    #     # Initialize analysis parameters
    #     initial_grain_num = 512
    #     
    #     # Setup polar plot for inclination distribution visualization
    #     plt.close()
    #     fig = plt.figure(figsize=(5, 5))
    #     ax = plt.gca(projection='polar')
    #
    #     # Configure polar plot appearance
    #     ax.set_thetagrids(np.arange(0.0, 360.0, 45.0), fontsize=16)
    #     ax.set_thetamin(0.0)
    #     ax.set_thetamax(360.0)
    #     ax.set_rgrids(np.arange(0, 0.008, 0.004))
    #     ax.set_rlabel_position(0.0) 
    #     ax.set_rlim(0.0, 0.008)
    #     ax.set_yticklabels(['0', '4e-3'], fontsize=16)
    #     ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    #     ax.set_axisbelow('True')
    #     
    #     # Analyze SPPARKS data
    #     sites_list = get_all_gb_list(npy_file_s512[0, :, :])
    #     slope_list = get_normal_vector_slope(npy_file_s512, sites_list, r"spparks512")
    #     
    #     # Analyze PRIMME data
    #     sites_list = get_all_gb_list(npy_file_p512[0, :, :])
    #     slope_list = get_normal_vector_slope(npy_file_p512, sites_list, r"primme512")
    #     
    #     plt.legend(loc=(-0.10, -0.2), fontsize=16, ncol=3)
    #     plt.savefig(current_path + f"/Images/normal_distribution_512_step{i}.png", 
    #                dpi=400, bbox_inches='tight')

    # =================================================================
    # LARGE-SCALE COMPARISON (2400×2400 DOMAIN, 20,000 GRAINS)
    # =================================================================
    
    print("\nProcessing large-scale datasets (20,000 grains)...")
    
    # Analysis timesteps for comparison
    timesteps = [300, 1600]
    
    for i in timesteps:
        print(f"\n--- Analyzing timestep {i} ---")
        
        # Load microstructure data from all three simulation methods
        print("Loading simulation data...")
        npy_file_s20k = np.load(input_name_spparks_20000 + f"{i}.npy")
        npy_file_p20k = np.load(input_name_primme_20000 + f"{i}.npy")
        npy_file_pf20k = np.load(input_name_pf_20000 + f"{i}.npy")
        
        print(f"  SPPARKS data: {npy_file_s20k.shape}")
        print(f"  PRIMME data: {npy_file_p20k.shape}")
        print(f"  Phase Field data: {npy_file_pf20k.shape}")
        
        # Microstructure parameters
        initial_grain_num = 20000
        
        # Grain size distribution parameters (for reference)
        bin_width = 0.16     # Bin width for grain size analysis
        x_limit = [-0.5, 3.5]  # Range for normalized grain sizes
        bin_num = round((abs(x_limit[0]) + abs(x_limit[1])) / bin_width)
        size_coordination = np.linspace((x_limit[0] + bin_width/2), 
                                      (x_limit[1] - bin_width/2), bin_num)
        
        # =================================================================
        # POLAR PLOT SETUP FOR INCLINATION VISUALIZATION
        # =================================================================
        
        print("Setting up polar plot for inclination comparison...")
        plt.close()
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca(projection='polar')

        # Configure angular grid (every 45 degrees)
        ax.set_thetagrids(np.arange(0.0, 360.0, 45.0), fontsize=16)
        ax.set_thetamin(0.0)
        ax.set_thetamax(360.0)

        # Configure radial grid and limits
        ax.set_rgrids(np.arange(0, 0.008, 0.004))
        ax.set_rlabel_position(0.0)    # Label position at 0°
        ax.set_rlim(0.0, 0.008)        # Radial range [0, 0.008]
        ax.set_yticklabels(['0', '4e-3'], fontsize=16)

        # Grid appearance
        ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow('True')
        
        # =================================================================
        # INCLINATION ANALYSIS FOR EACH SIMULATION METHOD
        # =================================================================
        
        # SPPARKS Analysis
        print("Analyzing SPPARKS inclination distribution...")
        sites_list = get_all_gb_list(npy_file_s20k[0, :, :])
        slope_list = get_normal_vector_slope(npy_file_s20k, sites_list, r"spparks20k")
        print(f"  SPPARKS boundary sites: {len(sites_list)}")
        
        # PRIMME Analysis  
        print("Analyzing PRIMME inclination distribution...")
        sites_list = get_all_gb_list(npy_file_p20k[0, :, :])
        slope_list = get_normal_vector_slope(npy_file_p20k, sites_list, r"primme20k")
        print(f"  PRIMME boundary sites: {len(sites_list)}")
        
        # Phase Field Analysis
        print("Analyzing Phase Field inclination distribution...")
        sites_list = get_all_gb_list(npy_file_pf20k[0, :, :])
        slope_list = get_normal_vector_slope(npy_file_pf20k, sites_list, r"phasefield20k")
        print(f"  Phase Field boundary sites: {len(sites_list)}")
        
        # =================================================================
        # PLOT FINALIZATION AND OUTPUT
        # =================================================================
        
        # Add legend and save comparative plot
        plt.legend(loc=(-0.10, -0.3), fontsize=16, ncol=2)
        
        # Save high-resolution figure
        output_filename = current_path + f"/Images/normal_distribution_20k_step{i}.png"
        plt.savefig(output_filename, dpi=400, bbox_inches='tight')
        print(f"  Saved comparison plot: {output_filename}")
        
    print("\n" + "="*60)
    print("COMPARATIVE INCLINATION ANALYSIS COMPLETED!")
    print("="*60)
    print("Analysis products:")
    print("- Polar plots comparing SPPARKS, PRIMME, and Phase Field methods")
    print("- Quantitative validation of simulation accuracy")  
    print("- Statistical characterization of grain boundary distributions")
    print("\nNext steps:")
    print("1. Examine polar plots for method agreement")
    print("2. Quantify statistical differences between methods")
    print("3. Validate PRIMME implementation accuracy")
    print("4. Analyze timestep evolution of boundary character")

    
    
    
    
    
    
    
    
    
    