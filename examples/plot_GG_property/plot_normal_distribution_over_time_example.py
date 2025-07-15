#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normal Vector Distribution Analysis Example for Grain Boundary Orientations

This script provides comprehensive analysis of normal vector distributions in polycrystalline
grain boundaries, focusing on crystallographic orientation characterization and statistical
analysis of grain boundary plane orientations. The analysis utilizes advanced computational
methods to extract and visualize normal vector distributions from large-scale simulation data.

Scientific Objectives:
- Normal Vector Characterization: Statistical analysis of grain boundary normal vector distributions
- Crystallographic Analysis: Orientation-dependent grain boundary plane characterization
- Polar Visualization: Advanced polar coordinate plotting for orientation data
- Bias Removal: Statistical correction for computational and sampling biases

Key Features:
- High-Performance Data Processing: Efficient handling of large-scale simulation datasets
- Advanced Visualization: Polar coordinate plots with enhanced formatting
- Statistical Analysis: Comprehensive normal vector distribution characterization
- HiPerGator Integration: Optimized for University of Florida supercomputing cluster

Technical Specifications:
- System Scale: 20,000 grain polycrystalline systems for statistical significance
- Data Format: NumPy array processing with compressed data storage
- Visualization: Matplotlib polar plots with publication-quality formatting
- Processing: Multi-core optimized algorithms for large dataset analysis

Applications:
- Crystallographic Texture Analysis: Understanding preferred grain boundary orientations
- Materials Characterization: Statistical analysis of polycrystalline microstructures
- Simulation Validation: Verification of grain boundary orientation distributions
- Research Applications: Advanced materials science and computational metallurgy

Created on Mon Jul 31 14:33:57 2023
@author: Lin
"""

# Essential Scientific Computing and Visualization Library Imports
# Core libraries for numerical computing, visualization, and system operations
import os
current_path = os.getcwd()  # Current working directory for file path management
import numpy as np          # High-performance numerical computing for array operations
from numpy import seterr    # NumPy error handling configuration
seterr(all='raise')        # Configure NumPy to raise exceptions for numerical errors
import matplotlib.pyplot as plt  # Advanced scientific visualization library
import math                 # Mathematical functions for calculations
from tqdm import tqdm      # Progress bar functionality for long computations
import sys                 # System-specific parameters and functions

# VECTOR Framework Module Imports
# Add necessary paths to Python system path for module accessibility
sys.path.append(current_path)              # Current directory modules
sys.path.append(current_path+'/../../')    # Parent directory VECTOR modules
import myInput                             # Input parameter configuration module
import PACKAGE_MP_Linear as linear2d       # 2D linear grain boundary processing package
import post_processing                     # Post-processing utilities for grain analysis
sys.path.append(current_path+'/../calculate_tangent/')  # Tangent calculation utilities

if __name__ == '__main__':
    # ========================================================================
    # DATA LOADING AND CONFIGURATION SECTION
    # ========================================================================
    
    # HiPerGator Simulation Data File Configuration
    # Define file paths and parameters for large-scale polycrystalline simulation data
    # processed on University of Florida's HiPerGator supercomputing cluster
    
    # HiPerGator data storage path - contains simulation results from multi-core processing
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/" # .npy folder
    
    # Energy calculation method specification - "ave" method for averaged energy calculations
    TJ_energy_type_ave = "ave"  # Triple junction energy calculation method
    
    # Comprehensive simulation file naming convention with key parameters:
    # - pT_ori_ave: Polycrystalline texture with orientation and averaged energy
    # - 20000: Initial grain count for statistical significance
    # - multiCore32: 32-core parallel processing configuration
    # - delta0.6: Mobility parameter for grain boundary motion
    # - m2_J1: Energy parameters for grain boundary interactions
    # - seed56689: Random seed for reproducible simulations
    # - kt066: Temperature parameter in reduced units
    npy_file_name_aniso_ave = f"pT_ori_ave_{TJ_energy_type_ave}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy" # .npy name

    # ========================================================================
    # SIMULATION DATA LOADING AND VALIDATION
    # ========================================================================
    
    # Load large-scale polycrystalline simulation data from HiPerGator processing
    # Data contains temporal evolution of grain orientations and boundary configurations
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)
    print(f"The ave data size is: {npy_file_aniso_ave.shape}")  # Display data dimensions for validation
    print("READING DATA DONE")  # Confirmation of successful data loading

    # ========================================================================
    # GRAIN COUNT ANALYSIS AND TEMPORAL STEP SELECTION
    # ========================================================================
    
    # Statistical Analysis Configuration
    # Set target grain count for normal vector distribution analysis
    # 200 grains provides optimal balance between statistical significance and computational efficiency
    expected_grains = 200  # Target grain count for distribution analysis
    
    # Calculate simulation timestep corresponding to target grain count
    # Uses post-processing algorithms to identify when grain coarsening reaches 200 grains
    # Returns: special_step_distribution_ave - timestep index for 200-grain configuration
    # The underscore variable contains additional metadata (unused in this analysis)
    special_step_distribution_ave, _ = post_processing.calculate_expected_step([npy_file_name_aniso_ave], expected_grains) # get steps for 200 grains

    # ========================================================================
    # ADVANCED POLAR COORDINATE VISUALIZATION SETUP
    # ========================================================================
    
    # Initialize high-quality polar coordinate figure for normal vector distribution
    # Polar plots are optimal for crystallographic orientation data visualization
    plt.close()  # Clear any existing plots to prevent interference
    fig = plt.figure(figsize=(5, 5))  # Square figure for balanced polar representation
    ax = plt.gca(projection='polar')  # Create polar coordinate axis system

    # Angular Coordinate Configuration (Theta Axis)
    # Configure angular grid lines and labels for crystallographic orientation analysis
    ax.set_thetagrids(np.arange(0.0, 360.0, 45.0),fontsize=16)  # 45-degree angular grid intervals
    ax.set_thetamin(0.0)    # Minimum angular range (0 degrees)
    ax.set_thetamax(360.0)  # Maximum angular range (360 degrees - full circle)

    # Radial Coordinate Configuration (R Axis)
    # Configure radial grid lines for probability density visualization
    ax.set_rgrids(np.arange(0, 0.01, 0.004))  # Radial grid lines every 0.004 units
    ax.set_rlabel_position(0.0)  # Position radial labels at 0-degree angle
    ax.set_rlim(0.0, 0.01)       # Radial axis limits for probability density range
    ax.set_yticklabels(['0', '4e-3', '8e-3'],fontsize=16)  # Custom radial axis labels

    # Enhanced Grid and Formatting Configuration
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)  # Professional grid styling
    ax.set_axisbelow('True')  # Ensure grid lines appear behind data plots

    # ========================================================================
    # NORMAL VECTOR EXTRACTION AND PROCESSING
    # ========================================================================
    
    # Anisotropic Normal Vector Analysis for Averaged Energy Method
    # Process simulation data to extract grain boundary normal vector distributions
    
    # Optimized Data Caching System
    # Check for pre-computed normal vector data to avoid redundant calculations
    # Caching significantly improves performance for repeated analysis
    data_file_name = f'/normal_distribution_data/normal_distribution_ave_step{special_step_distribution_ave}.npz' # tmp data file name
    
    if os.path.exists(current_path + data_file_name):
        # Load pre-computed normal vector data from cache
        # NPZ format provides efficient compressed storage for numerical arrays
        inclination_npz_data = np.load(current_path + data_file_name)
        P = inclination_npz_data["P"]        # Normal vector probability distributions
        sites = inclination_npz_data["sites"] # Grain boundary site information
    else:
        # Compute normal vectors from raw simulation data
        # Apply 90-degree rotation to align coordinate system for analysis
        # Rotation parameters: (axis1=0, axis2=1) for x-y plane rotation
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        
        # Extract normal vector distributions using advanced computational geometry
        # Returns: P - probability distributions, sites - boundary sites, sites_list - detailed site data
        P, sites, sites_list = post_processing.get_normal_vector(newplace)
        
        # Cache computed data for future analysis efficiency
        # NPZ format ensures data integrity and fast I/O operations
        np.savez(current_path + data_file_name, P=P, sites=sites)
    
    # ========================================================================
    # STATISTICAL ANALYSIS AND VISUALIZATION
    # ========================================================================
    
    # Generate Normal Vector Slope Distribution Analysis
    # Compute and visualize inclination angle distributions with bias correction
    # Parameters: P - probability data, sites - boundary sites, timestep, method identifier
    slope_list = post_processing.get_normal_vector_slope(P, sites, special_step_distribution_ave, "Ave") # plot inclinaitn distribution
    
    # Enhanced Figure Formatting and Export
    # Configure legend positioning and styling for publication-quality output
    plt.legend(loc=(-0.12,-0.35),fontsize=16,ncol=3)  # Position legend below plot with 3 columns
    
    # Export high-resolution figure with optimized formatting
    # DPI=400 ensures publication-quality resolution
    # bbox_inches='tight' eliminates excess whitespace for professional appearance
    plt.savefig(current_path + "/figures/normal_distribution_poly_20k_after_removing_bias.png", dpi=400,bbox_inches='tight')












