#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRAIN BOUNDARY INCLINATION CALCULATION FOR PRIMME VALIDATION
============================================================

This script calculates grain boundary inclination vectors from phase field
simulation data for PRIMME (Parallel Runge-Kutta-based Implicit Multi-physics
Multi-scale Engine) validation studies. The script processes HDF5 microstructure
data and generates inclination vector fields using bilinear smoothing algorithms.

Key Functionality:
1. Load phase field simulation data from HDF5 files
2. Extract microstructures at specified timesteps
3. Apply bilinear smoothing to calculate inclination vectors  
4. Normalize and coordinate-transform inclination data
5. Save processed inclination fields for comparison analysis

Scientific Purpose:
- Generate reference inclination data for PRIMME validation
- Process phase field simulation results for quantitative comparison
- Calculate grain boundary normal vectors for orientation analysis
- Enable statistical comparison of grain boundary character distributions

Data Pipeline:
HDF5 Input → Microstructure Extraction → Smoothing Algorithm → Inclination Vectors → NPY Output

Created on Thu Sep 30 14:55:28 2021
@author: lin.yang
"""

import os
current_path = os.getcwd()
import sys
sys.path.append(current_path)
sys.path.append('./../../.')
import numpy as np
import math
from itertools import repeat
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

# VECTOR framework modules for grain boundary analysis
import myInput
import PACKAGE_MP_Linear as smooth     # 2D Bilinear smoothing algorithm for inclination calculation

if __name__ == '__main__':
    """
    MAIN EXECUTION: PHASE FIELD INCLINATION PROCESSING
    =================================================
    
    This script processes a specific phase field simulation dataset to calculate
    grain boundary inclination vectors for PRIMME validation studies. The analysis
    focuses on large-scale 2D polycrystalline microstructures with statistical
    grain populations suitable for quantitative comparison.
    
    Target Dataset:
    - Domain: 2400×2400 lattice sites (high resolution)
    - Grains: 20,000 grain population (statistical significance)
    - Timesteps: 300, 1600 (early and evolved states)
    - Method: Phase field simulation with grain growth
    """

    # =================================================================
    # HDF5 DATA LOADING AND EXTRACTION
    # =================================================================
    
    # Input phase field simulation data
    input_name = "input/phasefiled_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1)_kt(0.66)_cut(0).h5"
    output_name = "output/phasefiled_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1)_kt(0.66)_cut(0)_inclination_"
    
    print("Loading phase field simulation data from HDF5 file...")
    print(f"Input file: {input_name}")
    
    # Load HDF5 file containing phase field simulation results
    f = h5py.File(input_name, 'r')
    
    # Extract all datasets from HDF5 file structure
    # HDF5 structure: simulation_group -> dataset_name -> data_array
    print("Extracting datasets from HDF5 file...")
    for simu in f.keys():
        print(f"  Processing simulation group: {simu}")
        DataContainer = f.get(simu)
        
        for dataset in DataContainer.keys():
            tmpdata = DataContainer[dataset]
            # Clean dataset names (replace spaces with underscores for Python variable names)
            dataset_clean = dataset.replace(' ', '_')
            # Create global variables for each dataset
            globals()[dataset_clean] = tmpdata
            print(f"    Loaded dataset: {dataset} -> {dataset_clean}")
    
    f.close()  # Close HDF5 file
    
    # =================================================================
    # MICROSTRUCTURE PARAMETERS AND INITIALIZATION
    # =================================================================
    
    # Extract key parameters from loaded data
    steps, nz, nx, ny = ims_id.shape  # ims_id: microstructure ID array
    ng = len(euler_angles)            # euler_angles: crystallographic orientations
    
    print(f"\nMicrostructure parameters:")
    print(f"  Time steps: {steps}")
    print(f"  Domain dimensions: {nx} × {ny} (nz={nz})")
    print(f"  Total grains: {ng}")
    print(f"  Data shape: {ims_id.shape}")
    
    # Initialize storage array for all processed inclination data
    # Shape: (timesteps, 3, nx, ny) where dimension 1 contains:
    # [0,:,:] = microstructure, [1,:,:] = x-component, [2,:,:] = y-component
    main_matrix = np.zeros((steps, 3, nx, ny))
    
    # =================================================================
    # INCLINATION CALCULATION FOR SPECIFIED TIMESTEPS
    # =================================================================
    
    # Process specific timesteps for analysis (early evolution and final state)
    target_timesteps = [300, 1600]
    print(f"\nProcessing inclination vectors for timesteps: {target_timesteps}")
    
    for i in tqdm(target_timesteps, desc="Processing timesteps"):
        print(f"\n--- Processing timestep {i} ---")
        
        # Extract microstructure at current timestep
        microstructure = ims_id[i, :]           # Extract 3D slice
        microstructure = np.squeeze(microstructure)  # Remove singleton dimensions
        
        print(f"  Microstructure shape: {microstructure.shape}")
        print(f"  Grain ID range: {np.min(microstructure)} to {np.max(microstructure)}")
        
        # Initialize gradient field for smoothing algorithm
        R = np.zeros((nx, ny, 2))  # Gradient field [x-component, y-component]

        # =================================================================
        # BILINEAR SMOOTHING ALGORITHM EXECUTION
        # =================================================================
        
        print("  Executing bilinear smoothing algorithm...")
        
        # Algorithm parameters for optimal convergence
        cores = 8          # CPU cores for parallel processing
        loop_times = 5     # Smoothing iterations for convergence
        
        # Initialize smoothing algorithm class
        # Parameters: (nx, ny, ng, cores, loop_times, microstructure, gradient_field, param1, param2)
        test1 = smooth.linear_class(nx, ny, ng, cores, loop_times, microstructure, R, 0, False)
        
        # Execute inclination calculation using bilinear smoothing
        test1.linear_main('inclination')
        
        # Extract smoothed field P containing inclination vectors
        P = test1.get_P()  # Shape: (3, nx, ny)
        
        print(f"  Smoothed field shape: {np.array(P).shape}")
        
        # =================================================================
        # PERFORMANCE METRICS (OPTIONAL REPORTING)
        # =================================================================
        
        # Uncomment for performance analysis
        # print(f'  Loop iterations: {test1.loop_times}')
        # print(f'  Total runtime: {test1.running_time:.2f} seconds')
        # print(f'  Core runtime: {test1.running_coreTime:.2f} seconds')

        # =================================================================
        # INCLINATION VECTOR PROCESSING AND NORMALIZATION
        # =================================================================
        
        print("  Processing and normalizing inclination vectors...")
        
        # Convert to numpy array for manipulation
        P_final = np.array(P)
        
        # Coordinate transformation for proper inclination representation
        # Standard convention: swap and negate components for correct orientation
        P_final[1] = -P[2]  # x-component = -original_y_component  
        P_final[2] = P[1]   # y-component = original_x_component
        
        # Normalize inclination vectors to unit length
        # Magnitude calculation: sqrt(x² + y²)
        magnitude = (P_final[1]**2 + P_final[2]**2)**0.5
        
        # Avoid division by zero: set zero-magnitude vectors to zero
        mask = magnitude != 0
        P_final[1][mask] = P_final[1][mask] / magnitude[mask]
        P_final[2][mask] = P_final[2][mask] / magnitude[mask]
        
        # Handle NaN values (replace with zero)
        P_final = np.nan_to_num(P_final)
        
        print(f"  Normalized vector range: [{np.min(P_final[1:]):.3f}, {np.max(P_final[1:]):.3f}]")
        
        # Store processed data in main matrix
        main_matrix[i] = P_final
        
        # =================================================================
        # DATA OUTPUT AND STORAGE
        # =================================================================
        
        # Save processed inclination data for current timestep
        output_filename = output_name + f"step{i}"
        np.save(output_filename, P_final)
        print(f"  Saved inclination data: {output_filename}.npy")
        
    # =================================================================
    # COMPLETION SUMMARY AND DATA DESCRIPTION
    # =================================================================
    
    print("\n" + "="*60)
    print("INCLINATION CALCULATION COMPLETED!")
    print("="*60)
    print(f"Processed timesteps: {target_timesteps}")
    print(f"Output directory: {os.path.dirname(output_name)}")
    print(f"Files generated: {len(target_timesteps)} .npy files")
    
    print(f"\nData matrix organization:")
    print(f"  main_matrix[time_step, component, x, y]")
    print(f"  - time_step: Simulation timestep index")
    print(f"  - component: [0=microstructure, 1=x_vector, 2=y_vector]")
    print(f"  - x, y: Spatial coordinates in microstructure")
    
    print(f"\nData specifications:")
    print(f"  - Inclination vectors are unit-normalized")
    print(f"  - Coordinate system: standard crystallographic convention")
    print(f"  - NaN values replaced with zeros")
    print(f"  - Compatible with comparative analysis scripts")
    
    print(f"\nNext steps:")
    print("1. Load inclination data in comparison scripts")
    print("2. Calculate grain boundary character distributions")
    print("3. Generate polar plots for method validation")
    print("4. Perform statistical analysis of PRIMME accuracy")
    
    # Final data summary comment for reference
    # main_matrix[time_step, inclination_axis, x, y] is the matrix storing all inclination data
    # The first index is the time step index 
    # The second index is the inclination axis (0=microstructure, 1=x-vector, 2=y-vector)
    # The third index is the x-axis spatial coordinate
    # The fourth index is the y-axis spatial coordinate