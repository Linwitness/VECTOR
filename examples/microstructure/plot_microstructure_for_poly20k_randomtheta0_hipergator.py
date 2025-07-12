#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Polycrystalline Microstructure Visualization for Large-Scale Grain Growth Analysis
====================================================================================

This module provides specialized visualization capabilities for analyzing large-scale
polycrystalline microstructures obtained from SPPARKS Monte Carlo simulations. It
focuses on comparative analysis of different energy calculation methods and their
effects on grain growth evolution in systems with 20,000 initial grains.

Scientific Background:
- Large-scale 2D grain growth simulation visualization
- Comparative analysis of energy calculation methodologies
- High-quality microstructure imaging for scientific publication
- Temporal evolution analysis of polycrystalline systems

Key Features:
- Multi-method energy comparison (ave, sum, consMin, consMax, min, max)
- High-resolution microstructure plotting (400 DPI)
- Consistent color mapping across different energy methods
- Clean visualization without axes for publication quality
- HiPerGator cluster data processing optimization

Research Applications:
- Monte Carlo grain growth simulation analysis
- Energy method validation and comparison studies
- Large-scale microstructure evolution documentation
- Scientific publication figure generation

Technical Specifications:
- Initial grain count: 20,000 grains
- Domain size: Determined by SPPARKS simulation parameters
- Energy methods: 6 different calculation approaches
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
sys.path.append(current_path+'/../calculate_tangent/')

def plot_structure_figure(step, structure_figure, figure_path):
    """
    Generate High-Quality Microstructure Visualization
    
    This function creates publication-quality microstructure plots with consistent
    color mapping and clean formatting suitable for scientific publications.
    
    Parameters:
    -----------
    step : int
        Simulation timestep to visualize
    structure_figure : ndarray
        4D array containing microstructure evolution data (time, x, y, features)
    figure_path : str
        Base path for output figure (energy method suffix will be appended)
        
    Algorithm Details:
    -----------------
    - Uses initial timestep for consistent color range normalization
    - Applies 90-degree rotation for proper orientation
    - Removes all axes and ticks for clean scientific presentation
    - Saves at 400 DPI for high-quality publication figures
    - Uses rainbow colormap for maximum grain distinction
    
    Visualization Features:
    ----------------------
    - Consistent color range across all energy methods
    - No interpolation for pixel-perfect grain boundary representation
    - Tight bounding box to eliminate white space
    - Timestep encoding in filename for temporal tracking
    
    Scientific Applications:
    -----------------------
    - Comparative microstructure evolution analysis
    - Energy method effect visualization
    - Publication-quality figure generation
    - Large-scale grain growth documentation
    """
    # Close any existing plots to prevent memory issues
    plt.close()
    fig, ax = plt.subplots()

    # Extract microstructure data for visualization
    cv_initial = np.squeeze(structure_figure[0])    # Initial state for color normalization
    cv0 = np.squeeze(structure_figure[step])        # Current timestep data
    cv0 = np.rot90(cv0, 1)                         # Rotate for proper orientation

    # Create microstructure plot with consistent color mapping
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
    Main Execution: Comparative Energy Method Visualization Pipeline
    ===============================================================
    
    This script performs systematic visualization of large-scale polycrystalline
    microstructures evolved using different energy calculation methods in SPPARKS.
    It generates comparative figures to analyze the effects of energy methodology
    on grain growth evolution patterns.
    
    Analysis Framework:
    ------------------
    1. Load microstructure evolution data for 6 different energy methods
    2. Select specific timesteps showing approximately 2000 remaining grains
    3. Generate high-quality visualizations for comparative analysis
    4. Save publication-ready figures with consistent formatting
    
    Energy Methods Analyzed:
    -----------------------
    - ave: Average-based energy calculation
    - consMin: Conservative minimum energy approach
    - sum: Summation-based energy calculation
    - min: Minimum energy methodology
    - max: Maximum energy methodology  
    - consMax: Conservative maximum energy approach
    
    Scientific Objectives:
    ---------------------
    - Compare microstructural evolution under different energy calculations
    - Document grain growth patterns for method validation
    - Generate publication-quality comparative figures
    - Analyze energy method effects on final grain structures
    
    HiPerGator Integration:
    ----------------------
    - Optimized for cluster file system access
    - Handles large-scale simulation data efficiently
    - Processes multiple energy methods systematically
    """
    
    # HiPerGator cluster data directory configuration
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/"
    
    # Define energy calculation method identifiers
    TJ_energy_type_ave = "ave"           # Average energy calculation
    TJ_energy_type_consMin = "consMin"   # Conservative minimum approach
    TJ_energy_type_sum = "sum"           # Summation-based calculation
    TJ_energy_type_min = "min"           # Minimum energy methodology
    TJ_energy_type_max = "max"           # Maximum energy methodology
    TJ_energy_type_consMax = "consMax"   # Conservative maximum approach

    # Generate standardized filenames for each energy method
    # Format: p_randomtheta0_{energy_type}E_20000_delta0.6_J1_refer_1_0_0_seed56689_kt0.66.npy
    # Parameters: randomtheta0 (random initial orientations), 20000 grains, delta=0.6, kt=0.66
    npy_file_name_aniso_ave = f"p_randomtheta0_{TJ_energy_type_ave}E_20000_delta0.6_J1_refer_1_0_0_seed56689_kt0.66.npy"
    npy_file_name_aniso_consMin = f"p_randomtheta0_{TJ_energy_type_consMin}E_20000_delta0.6_J1_refer_1_0_0_seed56689_kt0.66.npy"
    npy_file_name_aniso_sum = f"p_randomtheta0_{TJ_energy_type_sum}E_20000_delta0.6_J1_refer_1_0_0_seed56689_kt0.66.npy"
    npy_file_name_aniso_min = f"p_randomtheta0_{TJ_energy_type_min}E_20000_delta0.6_J1_refer_1_0_0_seed56689_kt0.66.npy"
    npy_file_name_aniso_max = f"p_randomtheta0_{TJ_energy_type_max}E_20000_delta0.6_J1_refer_1_0_0_seed56689_kt0.66.npy"
    npy_file_name_aniso_consMax = f"p_randomtheta0_{TJ_energy_type_consMax}E_20000_delta0.6_J1_refer_1_0_0_seed56689_kt0.66.npy"

    # Load microstructure evolution data for all energy methods
    # Each array contains temporal evolution of 2D microstructure
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)
    npy_file_aniso_consMin = np.load(npy_file_folder + npy_file_name_aniso_consMin)
    npy_file_aniso_sum = np.load(npy_file_folder + npy_file_name_aniso_sum)
    npy_file_aniso_min = np.load(npy_file_folder + npy_file_name_aniso_min)
    npy_file_aniso_max = np.load(npy_file_folder + npy_file_name_aniso_max)
    npy_file_aniso_consMax = np.load(npy_file_folder + npy_file_name_aniso_consMax)
    
    # Report data dimensions for verification
    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    print(f"The consMin data size is: {npy_file_aniso_consMin.shape}")
    print(f"The sum data size is: {npy_file_aniso_sum.shape}")
    print(f"The min data size is: {npy_file_aniso_min.shape}")
    print(f"The max data size is: {npy_file_aniso_max.shape}")
    print(f"The consMax data size is: {npy_file_aniso_consMax.shape}")
    print("READING DATA DONE")

    # Simulation parameters and analysis configuration
    initial_grain_num = 20000                    # Starting number of grains
    step_num = npy_file_aniso_ave.shape[0]      # Total number of timesteps

    # Select specific timesteps for visualization (calibrated to show ~2000 grains)
    # These timesteps represent intermediate evolution stages for comparison
    special_step_distribution_ave = 12      # Timestep for ave method (target: 2000 grains)
    special_step_distribution_consMin = 12  # Timestep for consMin method
    special_step_distribution_sum = 12      # Timestep for sum method
    special_step_distribution_min = 22      # Timestep for min method (slower evolution)
    special_step_distribution_max = 12      # Timestep for max method
    special_step_distribution_consMax = 13  # Timestep for consMax method

    # Alternative timesteps for final evolution stages (currently commented)
    # These would show more advanced grain growth (fewer, larger grains)
    #special_step_distribution_ave = 334     # Late-stage evolution
    #special_step_distribution_consMin = 334 # Late-stage evolution
    #special_step_distribution_sum = 334     # Late-stage evolution
    #special_step_distribution_min = 334     # Late-stage evolution
    #special_step_distribution_max = 334     # Late-stage evolution
    #special_step_distribution_consMax = 334 # Late-stage evolution

    # Generate comparative microstructure visualizations
    # Base path for output figures
    figure_path = current_path + "/figures/microstructure_randomtheta0_poly20k"
    
    # Create visualization for each energy method at selected timesteps
    # Each method produces a figure showing its characteristic grain structure
    plot_structure_figure(special_step_distribution_min, npy_file_aniso_min[:,:,:,0], figure_path + "_min")
    plot_structure_figure(special_step_distribution_max, npy_file_aniso_max[:,:,:,0], figure_path + "_max")
    plot_structure_figure(special_step_distribution_ave, npy_file_aniso_ave[:,:,:,0], figure_path + "_ave")
    plot_structure_figure(special_step_distribution_sum, npy_file_aniso_sum[:,:,:,0], figure_path + "_sum")
    plot_structure_figure(special_step_distribution_consMin, npy_file_aniso_consMin[:,:,:,0], figure_path + "_consmin")
    plot_structure_figure(special_step_distribution_consMax, npy_file_aniso_consMax[:,:,:,0], figure_path + "_consmax")










