#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Polycrystalline Microstructure Analysis for Delta Parameter Studies
=====================================================================

This module provides specialized visualization capabilities for analyzing
medium-scale polycrystalline microstructures with systematic delta parameter
variation. It focuses on understanding the effects of anisotropy parameters
on grain growth evolution in 512-grain oriented systems using rainbow
colormap visualization for enhanced grain distinction.

Scientific Background:
- Medium-scale 2D polycrystalline grain growth (512 initial grains)
- Systematic delta parameter sensitivity analysis (0.0 to 0.95)
- Oriented grain systems with crystallographic reference directions
- Target grain count studies (convergence to ~10 grain final states)
- Multi-core processing optimization for computational efficiency

Key Features:
- Delta parameter sweep for anisotropy sensitivity analysis
- Rainbow colormap visualization for enhanced grain identification
- Target-based timestep selection (10-grain convergence criterion)
- Grain size distribution analysis capabilities
- Multi-core processing variations for computational studies

Research Applications:
- Anisotropy parameter effect quantification in polycrystalline systems
- Grain growth kinetics analysis under varying delta conditions
- Computational efficiency studies with multi-core processing
- Grain size distribution evolution documentation
- Algorithm validation for intermediate-scale microstructures

Technical Specifications:
- Initial grain count: 512 oriented grains
- Target final count: ~10 grains for convergence analysis
- Delta range: [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
- Processing: 8-16 core parallel optimization
- Visualization: Rainbow colormap for grain distinction

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
sys.path.append(current_path+'/../calculate_tangent/')

def plot_structure_figure(step, structure_figure, figure_path):
    """
    Generate Colorful Polycrystalline Microstructure Visualization
    
    This function creates publication-quality microstructure plots for
    medium-scale polycrystalline systems using rainbow colormap visualization
    that enhances grain identification and provides clear distinction between
    individual grains in complex microstructures.
    
    Parameters:
    -----------
    step : int
        Simulation timestep to visualize
    structure_figure : ndarray
        4D array containing polycrystalline microstructure evolution data
    figure_path : str
        Base path for output figure (delta suffix will be appended)
        
    Algorithm Details:
    -----------------
    - Uses initial timestep for consistent color range normalization
    - Applies 90-degree rotation for proper grain orientation
    - Rainbow colormap for maximum grain distinction capability
    - Removes all axes and ticks for clean scientific presentation
    - Saves at 400 DPI for high-quality publication figures
    
    Visualization Features:
    ----------------------
    - Rainbow colormap optimized for grain identification
    - Consistent color range across all delta parameter studies
    - No interpolation for pixel-perfect grain boundary representation
    - Timestep encoding in filename for temporal tracking
    - Enhanced contrast for complex polycrystalline structures
    
    Scientific Applications:
    -----------------------
    - Delta parameter effect visualization in polycrystalline systems
    - Grain growth evolution analysis with enhanced identification
    - Multi-grain system documentation for intermediate scales
    - Publication-quality figure generation for parameter studies
    """
    # Close any existing plots to prevent memory issues
    plt.close()
    fig, ax = plt.subplots()

    # Extract polycrystalline microstructure data for visualization
    cv_initial = np.squeeze(structure_figure[0])    # Initial state for color normalization
    cv0 = np.squeeze(structure_figure[step])        # Current timestep data
    cv0 = np.rot90(cv0, 1)                         # Rotate for proper grain orientation

    # Create colorful polycrystalline microstructure plot
    im = ax.imshow(cv0, vmin=np.min(cv_initial), vmax=np.max(cv_initial), 
                   cmap='rainbow', interpolation='none')  # Rainbow for grain distinction
    
    # Optional colorbar configuration (currently disabled for clean appearance)
    # cb = fig.colorbar(im)
    # cb.ax.tick_params(labelsize=20)
    
    # Remove all axes and ticks for clean scientific presentation
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.tick_params(which='both', size=0, labelsize=0)

    # Save high-resolution figure with timestep information
    plt.savefig(figure_path + f"_ts{step*30}.png", dpi=400, bbox_inches='tight')


if __name__ == '__main__':
    """
    Main Execution: Medium-Scale Polycrystalline Delta Parameter Study
    
    This section orchestrates comprehensive visualization of medium-scale
    polycrystalline microstructures across systematic delta parameter variations.
    It analyzes the evolution of 512-grain oriented systems toward final states
    with approximately 10 grains, providing insight into anisotropy effects
    on grain growth kinetics and final microstructure characteristics.
    
    Delta Parameter Study Design:
    ----------------------------
    - Delta values: [0.0, 0.2, 0.4, 0.6, 0.8, 0.95] for comprehensive sensitivity
    - Initial system: 512 oriented grains with [1,0,0] crystallographic reference
    - Target criterion: Evolution to ~10 final grains for convergence analysis
    - Processing variations: 8-16 core optimization studies
    
    Grain Growth Analysis:
    ---------------------
    - Target grain count: ~10 grains for final state analysis
    - Timestep optimization: Variable based on delta-dependent growth kinetics
    - Growth rate mapping: Delta parameter effect on coarsening kinetics
    - Size distribution: Grain size evolution tracking capabilities
    
    Computational Configuration:
    ---------------------------
    - Multi-core processing: 8-16 cores (delta=0.6 uses 8-core for comparison)
    - Energy method: Average energy calculation (aveE)
    - System parameters: kt=0.66, J=1, m=2
    - Random seed: 56689 for reproducible polycrystalline structures
    - Crystallographic reference: [1,0,0] direction
    
    Timestep Selection Strategy:
    ---------------------------
    - Delta-dependent optimization for 10-grain target states
    - Growth kinetics vary significantly with anisotropy strength
    - Each delta value requires specific timestep for convergence
    - Final times encoded as: timestep × 30 for temporal reference
    
    Analysis Parameters:
    -------------------
    - Bin width: 0.16 for grain size distribution analysis
    - Size range: [-0.5, 3.5] for comprehensive distribution coverage
    - Logarithmic size coordination for statistical analysis
    - Target-based convergence criterion for consistent comparisons
    
    Scientific Applications:
    -----------------------
    - Delta parameter sensitivity quantification in oriented systems
    - Grain growth kinetics analysis under varying anisotropy conditions
    - Computational efficiency studies across multi-core configurations
    - Medium-scale microstructure evolution documentation
    - Algorithm validation for intermediate grain count systems
    """
    
    # Data location for medium-scale polycrystalline simulation results
    npy_file_folder = "/Users/lin.yang/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_poly_multiCoreCompare/results/"
    
    # Delta parameter definitions for systematic anisotropy sensitivity analysis
    circle_energy_000 = "0.0"   # Isotropic baseline (no anisotropy effects)
    circle_energy_020 = "0.2"   # Low anisotropy regime
    circle_energy_040 = "0.4"   # Moderate anisotropy effects
    circle_energy_060 = "0.6"   # Intermediate anisotropy (8-core processing)
    circle_energy_080 = "0.8"   # High anisotropy effects
    circle_energy_095 = "0.95"  # Maximum anisotropy regime

    # Construct standardized file names for delta parameter study
    # Common configuration: 512 grains, aveE method, kt=0.66, [1,0,0] reference
    # Variable: delta values and processing core count (mostly 16-core, 8-core for delta=0.6)
    npy_file_name_aniso_000 = f"p_ori_ave_aveE_512_multiCore16_delta{circle_energy_000}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_020 = f"p_ori_ave_aveE_512_multiCore16_delta{circle_energy_020}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_040 = f"p_ori_ave_aveE_512_multiCore16_delta{circle_energy_040}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_060 = f"p_ori_ave_aveE_512_multiCore8_delta{circle_energy_060}_m2_J1_refer_1_0_0_seed56689_kt066.npy"  # 8-core variant
    npy_file_name_aniso_080 = f"p_ori_ave_aveE_512_multiCore16_delta{circle_energy_080}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_095 = f"p_ori_ave_aveE_512_multiCore16_delta{circle_energy_095}_m2_J1_refer_1_0_0_seed56689_kt066.npy"

    # Load polycrystalline microstructure evolution data for all delta values
    print("Loading medium-scale polycrystalline delta parameter study data...")
    npy_file_aniso_000 = np.load(npy_file_folder + npy_file_name_aniso_000)
    npy_file_aniso_020 = np.load(npy_file_folder + npy_file_name_aniso_020)
    npy_file_aniso_040 = np.load(npy_file_folder + npy_file_name_aniso_040)
    npy_file_aniso_060 = np.load(npy_file_folder + npy_file_name_aniso_060)
    npy_file_aniso_080 = np.load(npy_file_folder + npy_file_name_aniso_080)
    npy_file_aniso_095 = np.load(npy_file_folder + npy_file_name_aniso_095)
    
    # Display dataset dimensions for verification
    print(f"The 000 data size is: {npy_file_aniso_000.shape}")
    print(f"The 020 data size is: {npy_file_aniso_020.shape}")
    print(f"The 040 data size is: {npy_file_aniso_040.shape}")
    print(f"The 060 data size is: {npy_file_aniso_060.shape}")
    print(f"The 080 data size is: {npy_file_aniso_080.shape}")
    print(f"The 095 data size is: {npy_file_aniso_095.shape}")
    print("READING DATA DONE")

    # System initialization parameters
    initial_grain_num = 512             # Starting number of oriented grains
    step_num = npy_file_aniso_000.shape[0]  # Total simulation timesteps

    # Grain size distribution analysis parameters
    bin_width = 0.16        # Bin width for grain size distribution analysis
    x_limit = [-0.5, 3.5]   # Size range for comprehensive distribution coverage
    bin_num = round((abs(x_limit[0]) + abs(x_limit[1])) / bin_width)  # Number of bins
    size_coordination = np.linspace((x_limit[0] + bin_width/2), 
                                   (x_limit[1] - bin_width/2), bin_num)  # Bin centers

    # Delta-dependent timestep selection for 10-grain target states
    # Each delta value exhibits different growth kinetics requiring optimization
    # Final times: timestep × 30 for temporal reference
    special_step_distribution_000 = 89   # 2670/30 = 89 timesteps → ~10 grains (isotropic)
    special_step_distribution_020 = 75   # 2250/30 = 75 timesteps → ~10 grains (low anisotropy)
    special_step_distribution_040 = 116  # 3480/30 = 116 timesteps → ~10 grains (moderate anisotropy)
    special_step_distribution_060 = 106  # 3180/30 = 106 timesteps → ~10 grains (intermediate, 8-core)
    special_step_distribution_080 = 105  # 3150/30 = 105 timesteps → ~10 grains (high anisotropy)
    special_step_distribution_095 = 64   # 1920/30 = 64 timesteps → ~10 grains (maximum anisotropy)

    # Generate systematic polycrystalline microstructure visualizations
    print("Generating polycrystalline delta parameter study visualizations...")
    figure_path = current_path + "/figures/microstructure_poly"
    
    # Create rainbow-colormap visualizations for each delta parameter
    # Extract grain ID data (first feature dimension) for microstructure visualization
    plot_structure_figure(special_step_distribution_000, npy_file_aniso_000[:,:,:,0], figure_path + "_000")
    plot_structure_figure(special_step_distribution_020, npy_file_aniso_020[:,:,:,0], figure_path + "_020")
    plot_structure_figure(special_step_distribution_040, npy_file_aniso_040[:,:,:,0], figure_path + "_040")
    plot_structure_figure(special_step_distribution_060, npy_file_aniso_060[:,:,:,0], figure_path + "_060")
    plot_structure_figure(special_step_distribution_080, npy_file_aniso_080[:,:,:,0], figure_path + "_080")
    plot_structure_figure(special_step_distribution_095, npy_file_aniso_095[:,:,:,0], figure_path + "_095")
    
    print("Polycrystalline delta parameter study complete!")










