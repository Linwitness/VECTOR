#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Polycrystalline Microstructure Visualization for Oriented Grain Systems
=========================================================================

This module provides specialized visualization capabilities for analyzing large-scale
polycrystalline microstructures with oriented grains obtained from SPPARKS Monte Carlo
simulations. It focuses on comparative analysis of different energy calculation methods
in systems with realistic crystallographic orientation distributions.

Scientific Background:
- Large-scale 2D grain growth with crystallographic orientations
- Comparative analysis of energy calculation methodologies
- Multi-core parallel processing optimization for HiPerGator cluster
- High-quality microstructure imaging for scientific publication

Key Features:
- Multi-method energy comparison with oriented grain systems
- HiPerGator cluster optimization (64-core parallel processing)
- Post-processing integration for advanced analysis capabilities
- Consistent visualization parameters across energy methods
- Publication-quality figure generation

Research Applications:
- Anisotropic grain growth simulation analysis
- Energy method validation in oriented systems
- Large-scale parallel simulation documentation
- Crystallographic texture effect studies

Technical Specifications:
- Initial grain count: 20,000 grains with crystallographic orientations
- Parallel processing: 64-core optimization
- Energy methods: 6 different calculation approaches
- Domain parameters: delta=0.6, kt=0.66
- Output format: High-resolution PNG images

Created on Mon Jul 31 14:33:57 2023
@author: Lin
"""

import os
current_path = os.getcwd()
import numpy as np
from numpy import seterr
seterr(all='raise')  # Enable numpy error reporting for debugging
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm  # Progress tracking for batch processing
import sys

# Configure system paths for VECTOR framework access
sys.path.append(current_path)
sys.path.append(current_path+'/../../')
import myInput
import PACKAGE_MP_Linear as linear2d
import post_processing  # Advanced post-processing capabilities
sys.path.append(current_path+'/../calculate_tangent/')

def plot_structure_figure(step, structure_figure, figure_path):
    """
    Generate High-Quality Oriented Microstructure Visualization
    
    This function creates publication-quality microstructure plots for oriented
    grain systems with consistent color mapping and clean formatting suitable
    for scientific publications and comparative analysis.
    
    Parameters:
    -----------
    step : int
        Simulation timestep to visualize
    structure_figure : ndarray
        4D array containing oriented microstructure evolution data (time, x, y, features)
    figure_path : str
        Base path for output figure (energy method suffix will be appended)
        
    Algorithm Details:
    -----------------
    - Uses initial timestep for consistent color range normalization
    - Applies 90-degree rotation for proper crystallographic orientation
    - Removes all axes and ticks for clean scientific presentation
    - Saves at 400 DPI for high-quality publication figures
    - Uses rainbow colormap optimized for oriented grain distinction
    
    Visualization Features:
    ----------------------
    - Consistent color range across all energy methods for comparison
    - No interpolation for pixel-perfect grain boundary representation
    - Tight bounding box to eliminate white space
    - Timestep encoding in filename for temporal tracking
    - Optimized for oriented grain system visualization
    
    Scientific Applications:
    -----------------------
    - Comparative anisotropic microstructure evolution analysis
    - Energy method effect visualization in oriented systems
    - Publication-quality figure generation for crystallographic studies
    - Large-scale oriented grain growth documentation
    """
    # Close any existing plots to prevent memory issues
    plt.close()
    fig, ax = plt.subplots()

    # Extract oriented microstructure data for visualization
    cv_initial = np.squeeze(structure_figure[0])    # Initial state for color normalization
    cv0 = np.squeeze(structure_figure[step])        # Current timestep data
    cv0 = np.rot90(cv0, 1)                         # Rotate for proper crystallographic orientation

    # Create microstructure plot with consistent color mapping for oriented grains
    im = ax.imshow(cv0, vmin=np.min(cv_initial), vmax=np.max(cv_initial), 
                   cmap='rainbow', interpolation='none')
    
    # Optional colorbar configuration (currently disabled for clean appearance)
    # cb = fig.colorbar(im)
    # cb.set_ticks([10000,20000])
    # cb.set_ticklabels(['1e4', '2e4'])
    # cb.ax.tick_params(labelsize=20)
    
    # Remove all axes and ticks for clean scientific presentation
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.tick_params(which='both', size=0, labelsize=0)

    # Save high-resolution figure with timestep information
    plt.savefig(figure_path + f"_ts{step*30}.png", dpi=400, bbox_inches='tight')


if __name__ == '__main__':
    """
    Main Execution: Comprehensive Oriented Microstructure Analysis
    
    This section orchestrates large-scale visualization of oriented polycrystalline
    microstructures using advanced energy calculation methods. It systematically
    processes HiPerGator cluster simulation results for comparative analysis
    of different triple junction energy formulations in oriented grain systems.
    
    System Configuration:
    --------------------
    - Initial grains: 20,000 oriented polycrystalline grains
    - HiPerGator optimization: 64-core parallel processing
    - Simulation parameters: delta=0.6, kt=0.66, m=2, J=1
    - Reference orientation: [1,0,0] crystallographic direction
    - Random seed: 56689 for reproducible oriented distributions
    
    Energy Method Analysis:
    ----------------------
    - ave (Average): Arithmetic mean of triple junction energies
    - consMin (Conservative Minimum): Weighted minimum energy approach
    - sum (Summation): Total energy accumulation method
    - min (Minimum): Pure minimum energy selection
    - max (Maximum): Maximum energy criterion
    - consMax (Conservative Maximum): Weighted maximum energy approach
    
    Timestep Selection Strategy:
    ---------------------------
    - Target grain count: 2000 grains (10% of initial population)
    - Method-specific timesteps optimized for each energy formulation
    - Growth kinetics vary significantly between energy methods
    - Final states represent quasi-equilibrium microstructures
    
    Output Specifications:
    ---------------------
    - High-resolution PNG figures for publication quality
    - Consistent color mapping across all energy methods
    - Microstructure snapshots at growth-converged states
    - Organized file structure for comparative analysis
    
    Scientific Applications:
    -----------------------
    - Energy method validation in oriented polycrystalline systems
    - Anisotropic grain growth mechanism analysis
    - Large-scale oriented microstructure evolution studies
    - Computational method benchmarking for crystallographic systems
    """
    
    # Data file location on HiPerGator cluster storage
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/"
    
    # Triple junction energy calculation method definitions
    # Each method provides different physical interpretation of grain boundary energetics
    TJ_energy_type_ave = "ave"         # Arithmetic average of adjacent grain energies
    TJ_energy_type_consMin = "consMin" # Conservative minimum energy approach
    TJ_energy_type_sum = "sum"         # Summation of all contributing energies
    TJ_energy_type_min = "min"         # Pure minimum energy selection
    TJ_energy_type_max = "max"         # Maximum energy criterion
    TJ_energy_type_consMax = "consMax" # Conservative maximum energy approach

    # Construct standardized file names for oriented grain simulation data
    # Common parameters: 20k grains, 64-core, delta=0.6, kt=0.66, seed=56689
    npy_file_name_aniso_ave = f"p_ori_ave_{TJ_energy_type_ave}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMin = f"pm_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_sum = f"pm_ori_ave_{TJ_energy_type_sum}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_min = f"pm_ori_ave_{TJ_energy_type_min}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_max = f"pm_ori_ave_{TJ_energy_type_max}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMax = f"pm_ori_ave_{TJ_energy_type_consMax}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"

    # Load oriented microstructure evolution data for all energy methods
    # Each dataset contains 4D arrays: [time, x, y, grain_features]
    print("Loading oriented microstructure evolution data...")
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)
    npy_file_aniso_consMin = np.load(npy_file_folder + npy_file_name_aniso_consMin)
    npy_file_aniso_sum = np.load(npy_file_folder + npy_file_name_aniso_sum)
    npy_file_aniso_min = np.load(npy_file_folder + npy_file_name_aniso_min)
    npy_file_aniso_max = np.load(npy_file_folder + npy_file_name_aniso_max)
    npy_file_aniso_consMax = np.load(npy_file_folder + npy_file_name_aniso_consMax)
    
    # Display dataset dimensions for verification
    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    print(f"The consMin data size is: {npy_file_aniso_consMin.shape}")
    print(f"The sum data size is: {npy_file_aniso_sum.shape}")
    print(f"The min data size is: {npy_file_aniso_min.shape}")
    print(f"The max data size is: {npy_file_aniso_max.shape}")
    print(f"The consMax data size is: {npy_file_aniso_consMax.shape}")
    print("READING DATA DONE")

    # System initialization parameters
    initial_grain_num = 20000           # Starting number of oriented grains
    step_num = npy_file_aniso_ave.shape[0]  # Total simulation timesteps

    # Method-specific timestep selection for 2000-grain target states
    # Each energy method exhibits different growth kinetics requiring optimization
    special_step_distribution_ave = 11     # Average method: uniform growth rate
    special_step_distribution_consMin = 11 # Conservative min: similar to average
    special_step_distribution_sum = 11     # Summation: rapid early growth
    special_step_distribution_min = 30     # Minimum: slower convergence due to energy barriers
    special_step_distribution_max = 15     # Maximum: intermediate growth rate
    special_step_distribution_consMax = 11 # Conservative max: similar to average

    # Alternative timestep selection for clear figure generation (currently disabled)
    # These provide later-stage microstructures with enhanced grain size contrast
    #special_step_distribution_ave = 334     # Extended growth for clearer boundaries
    #special_step_distribution_consMin = 334 # Extended growth for clearer boundaries
    #special_step_distribution_sum = 334     # Extended growth for clearer boundaries
    #special_step_distribution_min = 334     # Extended growth for clearer boundaries
    #special_step_distribution_max = 334     # Extended growth for clearer boundaries
    #special_step_distribution_consMax = 334 # Extended growth for clearer boundaries

    # Generate comparative microstructure visualizations
    print("Generating oriented microstructure visualizations...")
    figure_path = current_path + "/figures/microstructure_poly20k"
    
    # Create publication-quality figures for each energy method
    # Extract grain ID data (first feature dimension) for visualization
    plot_structure_figure(special_step_distribution_min, npy_file_aniso_min[:,:,:,0], figure_path + "_min")
    plot_structure_figure(special_step_distribution_max, npy_file_aniso_max[:,:,:,0], figure_path + "_max")
    plot_structure_figure(special_step_distribution_ave, npy_file_aniso_ave[:,:,:,0], figure_path + "_ave")
    plot_structure_figure(special_step_distribution_sum, npy_file_aniso_sum[:,:,:,0], figure_path + "_sum")
    plot_structure_figure(special_step_distribution_consMin, npy_file_aniso_consMin[:,:,:,0], figure_path + "_consmin")
    plot_structure_figure(special_step_distribution_consMax, npy_file_aniso_consMax[:,:,:,0], figure_path + "_consmax")
    
    print("Oriented microstructure analysis complete!")










