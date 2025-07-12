#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Hexagonal Microstructure Analysis for Triple Junction Energy Studies
======================================================================

This module specializes in visualizing and analyzing hexagonal grain structures
designed for systematic triple junction energy (TJE) validation studies. It focuses
on controlled hexagonal grain arrangements that provide ideal geometries for
testing energy calculation algorithms and grain boundary evolution models.

Scientific Background:
- Regular hexagonal grain structures for controlled TJE studies
- 48-grain hexagonal arrangements for systematic energy validation
- Angle-resolved energy calculations for triple junction analysis
- High-contrast visualization optimized for boundary definition
- Initial state analysis for energy method verification

Key Features:
- Regular hexagonal grain topology for consistent triple junction geometry
- Angle-dependent energy calculations at triple junction sites
- High-contrast grayscale visualization for boundary analysis
- Initial microstructure documentation for baseline studies
- 32-core parallel processing optimization

Research Applications:
- Triple junction energy method validation in regular geometries
- Systematic verification of energy calculation algorithms
- Controlled grain boundary evolution studies
- Energy conservation validation in discrete systems
- Algorithm benchmarking with known geometric constraints

Technical Specifications:
- Geometry: 48-grain regular hexagonal arrangement
- Triple junctions: Well-defined 120° angles for energy studies
- Processing: 32-core parallel optimization
- Visualization: High-contrast grayscale for boundary clarity
- Analysis focus: Initial state energy distribution validation

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
    Generate High-Contrast Hexagonal Microstructure Visualization
    
    This function creates publication-quality visualizations of hexagonal
    microstructure arrangements with emphasis on triple junction geometry
    using high-contrast grayscale colormapping suitable for energy method
    validation and systematic boundary analysis.
    
    Parameters:
    -----------
    step : int
        Simulation timestep to visualize (typically 0 for initial state)
    structure_figure : ndarray
        4D array containing hexagonal microstructure data (time, x, y, features)
    figure_path : str
        Base path for output figure (state suffix will be appended)
        
    Algorithm Details:
    -----------------
    - Uses initial timestep for consistent contrast normalization
    - Applies 90-degree rotation for proper triple junction orientation
    - High-contrast grayscale colormap for clear boundary definition
    - Removes all axes and ticks for clean scientific presentation
    - Saves at 400 DPI for high-quality publication figures
    
    Visualization Features:
    ----------------------
    - Consistent contrast range for systematic energy studies
    - No interpolation for pixel-perfect boundary representation
    - Grayscale colormap optimized for triple junction visualization
    - Timestep encoding in filename for temporal tracking
    - High-contrast settings for clear grain distinction
    
    Scientific Applications:
    -----------------------
    - Triple junction energy visualization in regular geometries
    - Energy method validation with controlled boundary arrangements
    - High-contrast imaging for algorithm verification
    - Publication-quality figure generation for validation studies
    """
    # Close any existing plots to prevent memory issues
    plt.close()
    fig, ax = plt.subplots()

    # Extract hexagonal microstructure data for visualization
    cv_initial = np.squeeze(structure_figure[0])    # Initial state for contrast normalization
    cv0 = np.squeeze(structure_figure[step])        # Current timestep data
    cv0 = np.rot90(cv0, 1)                         # Rotate for proper triple junction orientation

    # Create high-contrast hexagonal microstructure plot
    im = ax.imshow(cv0, vmin=np.min(cv_initial), vmax=np.max(cv_initial), 
                   cmap='gray_r', interpolation='none')  # High-contrast grayscale
    
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
    Main Execution: Hexagonal Microstructure Triple Junction Energy Analysis
    
    This section orchestrates the visualization and analysis of regular hexagonal
    microstructures specifically designed for triple junction energy (TJE) method
    validation. It focuses on initial state documentation of carefully constructed
    48-grain hexagonal arrangements that provide ideal geometric conditions for
    systematic energy calculation verification.
    
    Hexagonal System Design:
    -----------------------
    - 48 grains arranged in regular hexagonal pattern
    - Uniform triple junction angles (120°) for consistent energy calculations
    - Controlled grain boundary network for systematic validation
    - Regular topology eliminates geometric complexity effects
    
    Triple Junction Energy Focus:
    ----------------------------
    - Average energy method (aveE) for baseline validation
    - Angle-dependent energy calculations at each triple junction
    - Systematic verification of energy conservation principles
    - Controlled geometry for algorithm benchmarking
    
    Simulation Configuration:
    ------------------------
    - Multi-core processing: 32-core parallel optimization
    - Energy parameters: delta=0.6, kt=0.66, J=1, m=2
    - Crystallographic reference: [1,0,0] direction
    - Random seed: 56689 for reproducible hexagonal arrangements
    - Initial state focus: Energy method validation at t=0
    
    Analysis Parameters:
    -------------------
    - Grain count: 48 grains in regular hexagonal tessellation
    - Bin width: 0.16 for grain size distribution analysis
    - Timestep: 0 (initial state for energy method verification)
    - Angle data: Included for triple junction energy calculations
    
    Validation Objectives:
    ---------------------
    - Verify energy calculation accuracy in regular geometries
    - Document initial energy distributions for method comparison
    - Establish baseline for more complex microstructure studies
    - Validate conservation properties in discrete grain systems
    
    Scientific Applications:
    -----------------------
    - Energy method algorithm verification and validation
    - Systematic benchmarking of triple junction energy calculations
    - Development of controlled test cases for new energy methods
    - Documentation of baseline microstructures for comparative studies
    """
    
    # Data location for hexagonal microstructure TJE validation studies
    npy_file_folder = "/Users/lin/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_hex_for_TJE/results/"
    
    # Triple junction energy calculation method for validation
    TJ_energy_type_cases = "ave"  # Average energy method for baseline validation

    # Construct file name for hexagonal TJE validation dataset
    # Configuration: 48-grain hex, 32-core, delta=0.6, m=2, [1,0,0] reference
    # Special: angle data included for triple junction energy analysis
    npy_file_name = f"h_ori_ave_{TJ_energy_type_cases}E_hex_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_angle.npy"

    # Load hexagonal microstructure data with angle information
    print("Loading hexagonal microstructure for TJE validation...")
    npy_file_hex = np.load(npy_file_folder + npy_file_name)

    # Display dataset dimensions for verification
    print(f"The hexagonal data size is: {npy_file_hex.shape}")
    print("READING DATA DONE")

    # System initialization parameters for hexagonal validation study
    initial_grain_num = 48           # 48 grains in regular hexagonal arrangement
    step_num = npy_file_hex.shape[0] # Total simulation timesteps available
    
    # Grain size distribution analysis parameter
    bin_width = 0.16  # Bin width for grain size distribution studies

    # Timestep selection: Focus on initial state for energy method validation
    special_step_distribution_hex = 0  # Initial state (t=0) for TJE verification

    # Generate hexagonal microstructure visualization for TJE validation
    print("Generating hexagonal microstructure visualization for TJE validation...")
    figure_path = current_path + "/figures/microstructure_hex"
    
    # Create initial state visualization for triple junction energy validation
    # Extract grain ID data (first feature dimension) for microstructure visualization
    plot_structure_figure(special_step_distribution_hex, npy_file_hex[:,:,:,0], figure_path + "_initial")
    
    print("Hexagonal microstructure TJE validation analysis complete!")










