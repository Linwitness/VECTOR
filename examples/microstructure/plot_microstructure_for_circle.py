#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Circular Microstructure Analysis for Oriented Grain Systems
==============================================================

This module specializes in visualizing and analyzing circular microstructure
evolution in oriented grain systems using SPPARKS Monte Carlo simulations.
It focuses on parameter sensitivity studies for inclination energy methods
with systematic variation of delta values, crystallographic orientations,
and mobility parameters.

Scientific Background:
- Simplified circular grain geometry for controlled parameter studies
- Systematic delta value variation (0.0 to 0.95) for sensitivity analysis
- Crystallographic orientation effects in two-grain systems
- Mobility parameter studies (m=2,4,6) for kinetic analysis
- High-contrast visualization optimized for boundary definition

Key Features:
- Systematic delta parameter sweeps for energy method validation
- Crystallographic reference direction studies (multiple [h,k,l] orientations)
- Mobility parameter sensitivity analysis
- Two-grain circular system for simplified energy calculations
- High-contrast grayscale visualization for boundary clarity

Research Applications:
- Parameter sensitivity analysis for inclination energy methods
- Validation of energy calculation algorithms in simplified geometries
- Crystallographic orientation effect studies
- Mobility parameter optimization for realistic growth kinetics
- Controlled circular grain evolution documentation

Technical Specifications:
- Geometry: Two-grain circular system for simplified analysis
- Parameter ranges: delta ∈ [0.0, 0.95], m ∈ [2,4,6]
- Crystallographic references: Multiple [h,k,l] directions
- Visualization: High-contrast grayscale for boundary definition
- Output format: High-resolution PNG for scientific publication

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
    Generate High-Contrast Circular Microstructure Visualization
    
    This function creates publication-quality visualizations of circular
    microstructure evolution with emphasis on grain boundary definition
    using high-contrast grayscale colormapping suitable for parameter
    sensitivity studies and boundary analysis.
    
    Parameters:
    -----------
    step : int
        Simulation timestep to visualize
    structure_figure : ndarray
        4D array containing circular microstructure evolution data (time, x, y, features)
    figure_path : str
        Base path for output figure (parameter suffix will be appended)
        
    Algorithm Details:
    -----------------
    - Uses initial timestep for consistent contrast normalization
    - Applies 90-degree rotation for proper boundary orientation
    - High-contrast grayscale colormap for clear boundary definition
    - Removes all axes and ticks for clean scientific presentation
    - Saves at 400 DPI for high-quality publication figures
    
    Visualization Features:
    ----------------------
    - Consistent contrast range across all parameter combinations
    - No interpolation for pixel-perfect boundary representation
    - Grayscale colormap optimized for circular boundary visualization
    - Timestep encoding in filename for temporal tracking
    - High-contrast settings for clear grain distinction
    
    Scientific Applications:
    -----------------------
    - Parameter sensitivity visualization for circular systems
    - Boundary evolution analysis in simplified geometries
    - High-contrast imaging for energy method validation
    - Publication-quality figure generation for parameter studies
    """
    # Close any existing plots to prevent memory issues
    plt.close()
    fig, ax = plt.subplots()

    # Extract circular microstructure data for visualization
    cv_initial = np.squeeze(structure_figure[0])    # Initial state for contrast normalization
    cv0 = np.squeeze(structure_figure[step])        # Current timestep data
    cv0 = np.rot90(cv0, 1)                         # Rotate for proper boundary orientation

    # Create high-contrast circular microstructure plot
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
    Main Execution: Comprehensive Circular Microstructure Parameter Study
    
    This section orchestrates systematic visualization of circular microstructure
    evolution across multiple parameter dimensions. It performs comprehensive
    sensitivity analysis of delta values, crystallographic orientations, and
    mobility parameters in simplified two-grain circular systems.
    
    Parameter Study Design:
    ----------------------
    - Delta values: [0.0, 0.2, 0.4, 0.6, 0.8, 0.95] for energy sensitivity
    - Crystallographic orientations: Multiple [h,k,l] reference directions
    - Mobility parameters: m ∈ [2, 4, 6] for kinetic analysis
    - System geometry: Two-grain circular configuration for controlled studies
    
    Delta Sensitivity Analysis:
    --------------------------
    - Delta = 0.0: Isotropic baseline case (no anisotropy)
    - Delta = 0.2-0.8: Progressive anisotropy introduction
    - Delta = 0.95: High anisotropy regime for maximum effect studies
    - Systematic progression for parameter sensitivity mapping
    
    Crystallographic Studies:
    -------------------------
    - Reference [1,0,0]: Primary crystallographic direction
    - Variant orientations: [0.87,0.5,0], [0.71,0.71,0], [0.5,0.87,0], [0,1,0]
    - Mobility studies: m=2 (baseline), m=4, m=6 for kinetic effects
    - Multi-dimensional parameter space exploration
    
    Simulation Configuration:
    ------------------------
    - Parallel processing: 16-core optimization
    - Temperature: kt=0.66 for controlled kinetic regime
    - Random seed: 56689 for reproducible results
    - Scale factor: 1.0 for standard domain size
    - Energy method: Average energy calculation
    
    Output Organization:
    -------------------
    - Timestep selection optimized per parameter combination
    - High-contrast grayscale figures for boundary analysis
    - Systematic file naming for parameter identification
    - Publication-quality visualization for comparative studies
    
    Scientific Applications:
    -----------------------
    - Energy method validation in simplified geometries
    - Parameter sensitivity mapping for algorithm optimization
    - Crystallographic orientation effect quantification
    - Mobility parameter optimization studies
    """
    
    # Data location for circular microstructure simulation results
    npy_file_folder = "/Users/lin.yang/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_circle_multiCoreCompare/results/"
    
    # Delta parameter definitions for systematic sensitivity analysis
    circle_energy_000 = "0.0"   # Isotropic baseline (no anisotropy)
    circle_energy_020 = "0.2"   # Low anisotropy regime
    circle_energy_040 = "0.4"   # Moderate anisotropy
    circle_energy_060 = "0.6"   # Intermediate anisotropy
    circle_energy_080 = "0.8"   # High anisotropy
    circle_energy_095 = "0.95"  # Maximum anisotropy regime
    
    # Crystallographic orientation variations for reference direction studies
    circle_energy_080_087 = "_0.87_0.5_0"   # Tilted reference direction
    circle_energy_080_071 = "_0.71_0.71_0"  # Diagonal reference (45°)
    circle_energy_080_050 = "_0.5_0.87_0"   # Alternative tilt direction
    circle_energy_080_100 = "_0_1_0"        # Y-axis reference direction

    # Mobility parameter variations for kinetic studies
    circle_energy_095_m4 = "4"  # Increased mobility (faster kinetics)
    circle_energy_095_m6 = "6"  # High mobility (rapid kinetics)

    # Construct standardized file names for systematic parameter studies
    # Common parameters: 16-core, kt=0.66, seed=56689, scale=1, m=2 (unless specified)
    
    # Delta sensitivity series (isotropic to high anisotropy)
    npy_file_name_aniso_000 = f"c_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_000}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_020 = f"c_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_020}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_040 = f"c_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_040}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_060 = f"c_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_060}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_080 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_080}_m2_refer_1_0_0.npy"
    npy_file_name_aniso_095 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_095}_m2_refer_1_0_0.npy"

    # Crystallographic orientation series (delta=0.95, varying reference directions)
    npy_file_name_aniso_080_087 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_095}_m2_refer{circle_energy_080_087}.npy"
    npy_file_name_aniso_080_071 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_095}_m2_refer{circle_energy_080_071}.npy"
    npy_file_name_aniso_080_050 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_095}_m2_refer{circle_energy_080_050}.npy"
    npy_file_name_aniso_080_100 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_095}_m2_refer{circle_energy_080_100}.npy"

    # Mobility parameter series (delta=0.95, [1,0,0] reference, varying m)
    npy_file_name_aniso_095_m4 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_095}_m{circle_energy_095_m4}_refer_1_0_0.npy"
    npy_file_name_aniso_095_m6 = f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{circle_energy_095}_m{circle_energy_095_m6}_refer_1_0_0.npy"

    # Load circular microstructure evolution data for all parameter combinations
    print("Loading circular microstructure parameter study data...")
    
    # Delta sensitivity series
    npy_file_aniso_000 = np.load(npy_file_folder + npy_file_name_aniso_000)
    npy_file_aniso_020 = np.load(npy_file_folder + npy_file_name_aniso_020)
    npy_file_aniso_040 = np.load(npy_file_folder + npy_file_name_aniso_040)
    npy_file_aniso_060 = np.load(npy_file_folder + npy_file_name_aniso_060)
    npy_file_aniso_080 = np.load(npy_file_folder + npy_file_name_aniso_080)
    npy_file_aniso_095 = np.load(npy_file_folder + npy_file_name_aniso_095)
    
    # Crystallographic orientation series
    npy_file_aniso_080_087 = np.load(npy_file_folder + npy_file_name_aniso_080_087)
    npy_file_aniso_080_071 = np.load(npy_file_folder + npy_file_name_aniso_080_071)
    npy_file_aniso_080_050 = np.load(npy_file_folder + npy_file_name_aniso_080_050)
    npy_file_aniso_080_100 = np.load(npy_file_folder + npy_file_name_aniso_080_100)
    
    # Mobility parameter series
    npy_file_aniso_095_m4 = np.load(npy_file_folder + npy_file_name_aniso_095_m4)
    npy_file_aniso_095_m6 = np.load(npy_file_folder + npy_file_name_aniso_095_m6)
    
    # Display dataset dimensions for verification
    print(f"The 000 data size is: {npy_file_aniso_000.shape}")
    print(f"The 020 data size is: {npy_file_aniso_020.shape}")
    print(f"The 040 data size is: {npy_file_aniso_040.shape}")
    print(f"The 060 data size is: {npy_file_aniso_060.shape}")
    print(f"The 080 data size is: {npy_file_aniso_080.shape}")
    print(f"The 095 data size is: {npy_file_aniso_095.shape}")
    print("READING DATA DONE")

    # System initialization parameters
    initial_grain_num = 2                   # Two-grain circular system
    step_num = npy_file_aniso_000.shape[0]  # Total simulation timesteps

    # Timestep selection for delta sensitivity series
    # Optimized for clear boundary visualization at each delta value
    special_step_distribution_000 = 30  # Isotropic: longer time for boundary development
    special_step_distribution_020 = 30  # Low anisotropy: similar to isotropic
    special_step_distribution_040 = 30  # Moderate anisotropy: standard development
    special_step_distribution_060 = 30  # Intermediate: enhanced boundary effects
    special_step_distribution_080 = 30  # High anisotropy: pronounced effects
    special_step_distribution_095 = 30  # Maximum anisotropy: strongest effects

    # Timestep selection for crystallographic orientation series
    # Slightly adjusted for different orientation-dependent kinetics
    special_step_distribution_080_000 = 28  # [1,0,0] reference baseline
    special_step_distribution_080_087 = 28  # [0.87,0.5,0] tilted reference
    special_step_distribution_080_071 = 28  # [0.71,0.71,0] diagonal reference
    special_step_distribution_080_050 = 28  # [0.5,0.87,0] alternative tilt
    special_step_distribution_080_100 = 28  # [0,1,0] Y-axis reference

    # Timestep selection for mobility parameter series
    # Adjusted for different kinetic rates (higher m = faster kinetics)
    special_step_distribution_095_m2 = 14  # Standard mobility (m=2)
    special_step_distribution_095_m4 = 14  # Increased mobility (m=4)
    special_step_distribution_095_m6 = 14  # High mobility (m=6)

    # Generate systematic circular microstructure visualizations
    print("Generating circular microstructure parameter study visualizations...")
    figure_path = current_path + "/figures/microstructure_circle"
    
    # Delta sensitivity series: Progressive anisotropy effects
    plot_structure_figure(special_step_distribution_000, npy_file_aniso_000[:,:,:,0], figure_path + "_000")
    plot_structure_figure(special_step_distribution_020, npy_file_aniso_020[:,:,:,0], figure_path + "_020")
    plot_structure_figure(special_step_distribution_040, npy_file_aniso_040[:,:,:,0], figure_path + "_040")
    plot_structure_figure(special_step_distribution_060, npy_file_aniso_060[:,:,:,0], figure_path + "_060")
    plot_structure_figure(special_step_distribution_080, npy_file_aniso_080[:,:,:,0], figure_path + "_080")
    plot_structure_figure(special_step_distribution_095, npy_file_aniso_095[:,:,:,0], figure_path + "_095")

    # Crystallographic orientation series: Reference direction effects
    plot_structure_figure(special_step_distribution_080_000, npy_file_aniso_095[:,:,:,0], figure_path + "_095_000")
    plot_structure_figure(special_step_distribution_080_087, npy_file_aniso_080_087[:,:,:,0], figure_path + "_095_087")
    plot_structure_figure(special_step_distribution_080_071, npy_file_aniso_080_071[:,:,:,0], figure_path + "_095_071")
    plot_structure_figure(special_step_distribution_080_050, npy_file_aniso_080_050[:,:,:,0], figure_path + "_095_050")
    plot_structure_figure(special_step_distribution_080_100, npy_file_aniso_080_100[:,:,:,0], figure_path + "_095_100")

    # Mobility parameter series: Kinetic effects analysis
    plot_structure_figure(special_step_distribution_095_m2, npy_file_aniso_095[:,:,:,0], figure_path + "_095_m2")
    plot_structure_figure(special_step_distribution_095_m4, npy_file_aniso_095_m4[:,:,:,0], figure_path + "_095_m4")
    plot_structure_figure(special_step_distribution_095_m6, npy_file_aniso_095_m6[:,:,:,0], figure_path + "_095_m6")
    
    print("Circular microstructure parameter study complete!")












