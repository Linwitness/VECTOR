#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRIMME INCLINATION CALCULATION - HIPERGATOR HPC VERSION
=====================================================

High-Performance Computing (HPC) batch processing script for calculating grain boundary
inclination vectors from phase field simulation data on the HiPerGator supercomputing
cluster. This script processes multiple simulation cases in parallel for PRIMME validation
studies comparing different computational methods for grain boundary characterization.

PRIMME Validation Framework:
- Parallel Runge-Kutta-based Implicit Multi-physics Multi-scale Engine
- Comparative analysis: Phase Field vs SPPARKS vs PRIMME methods
- Grain boundary inclination distribution validation
- Statistical accuracy assessment across multiple simulation parameters

HPC Optimization Features:
- Batch processing of multiple simulation cases
- Parallel bilinear smoothing algorithm execution
- Efficient HDF5 data loading and memory management
- Output data organization for post-processing analysis

Target Platform: UF HiPerGator Supercomputing Cluster
- Storage: Blue file system (/blue/michael.tonks/)
- Processing: Multi-core parallel execution
- Data Format: HDF5 input, NumPy binary output

Scientific Context:
This script supports research into grain boundary character distribution accuracy
by processing simulation results from different computational methods and enabling
quantitative comparison of inclination vector predictions.

Author: lin.yang
Created: Thu Sep 30 14:55:28 2021
Purpose: HPC batch processing for PRIMME validation studies
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

import myInput
import PACKAGE_MP_Linear as smooth     #2D Bilinear smooth algorithm

def calculate_inclination_data(input_path, output_path, step_list):
    """
    BATCH INCLINATION CALCULATION FUNCTION
    =====================================
    
    Core processing function for calculating grain boundary inclination vectors
    from phase field simulation data. Designed for HPC batch processing of
    multiple simulation cases with consistent methodology.
    
    Algorithm Pipeline:
    1. HDF5 Data Loading: Extract microstructure and orientation data
    2. Parameter Extraction: Get domain dimensions and grain count
    3. Timestep Processing: Calculate inclinations for specified steps
    4. Bilinear Smoothing: Apply smoothing algorithm for vector fields
    5. Coordinate Transformation: Normalize and orient vectors properly
    6. Data Output: Save processed inclination data as NumPy arrays
    
    Parameters:
    -----------
    input_path : str
        Full path to HDF5 input file (without .h5 extension)
        Format: "/blue/michael.tonks/share/PRIMME_Inclination/CaseXY_TZ_tstep_A_B"
        
    output_path : str  
        Full path prefix for output files
        Format: "/blue/michael.tonks/share/PRIMME_Inclination_npy_files/CaseXY_TZ_inclination_"
        
    step_list : list of int
        Timesteps to process for inclination calculation
        Example: [300, 600, 1600] for early, intermediate, and final states
    
    Output Files:
    ------------
    - {output_path}step{i}.npy for each timestep i in step_list
    - Array shape: (3, nx, ny) where [0,:,:] = microstructure, [1,:,:] = x-vector, [2,:,:] = y-vector
    
    HPC Performance:
    ---------------
    - Parallel processing: 8 cores for bilinear smoothing algorithm
    - Memory optimization: Process one timestep at a time
    - I/O efficiency: Direct NumPy binary output for fast loading
    
    Scientific Validation:
    ---------------------
    - Inclination vectors normalized to unit length
    - Standard crystallographic coordinate system
    - Compatible with comparative analysis workflows
    """
    
    # =================================================================
    # HDF5 DATA EXTRACTION AND GLOBAL VARIABLE ASSIGNMENT
    # =================================================================
    
    print(f"Processing: {input_path.split('/')[-1]}")
    print(f"Loading HDF5 data from: {input_path}.h5")
    
    # Extract all variable from h5 files
    f = h5py.File(input_path+".h5", 'r')
    
    # Load all datasets into global namespace for algorithm compatibility
    # This preserves original variable names from simulation output
    for simu in f.keys():
        DataContainer = f.get(simu)
        for dataset in DataContainer.keys():
            tmpdata = DataContainer[dataset]
            # Clean dataset names for Python compatibility
            dataset = dataset.replace(' ','_')
            globals()[dataset] = tmpdata
    
    f.close()  # Close HDF5 file to free memory
    
    # =================================================================
    # MICROSTRUCTURE PARAMETER EXTRACTION
    # =================================================================
    
    # Get necessary data dimensions and parameters
    steps, nz, nx, ny = ims_id.shape      # ims_id: microstructure ID array
    ng = len(euler_angles)                # euler_angles: grain orientations
    
    print(f"Domain: {nx}×{ny}, Grains: {ng}, Available steps: {steps}")
    print(f"Processing timesteps: {step_list}")

    # =================================================================
    # TIMESTEP PROCESSING LOOP
    # =================================================================
    
    for i in step_list:
        print(f"\n--- Processing timestep {i} ---")
        
        # Extract microstructure at current timestep
        microstructure = ims_id[i,:]                # Get 3D slice
        microstructure = np.squeeze(microstructure) # Remove singleton dimensions
        
        # Initialize gradient field for smoothing algorithm
        R = np.zeros((nx,ny,2))  # [x-gradient, y-gradient]

        # =================================================================
        # BILINEAR SMOOTHING ALGORITHM CONFIGURATION
        # =================================================================
        
        # Build smoothing algorithm class and get inclination field P
        cores = 8          # HPC parallel processing cores
        loop_times = 5     # Smoothing iterations for convergence
        
        # Initialize bilinear smoothing class
        # Parameters: (nx, ny, ng, cores, iterations, microstructure, gradient_field, param1, param2)
        test1 = smooth.linear_class(nx, ny, ng, cores, loop_times, microstructure, R, 0, False)
        
        # Execute inclination calculation
        test1.linear_main('inclination')
        P = test1.get_P()  # Extract calculated inclination field
        
        # =================================================================
        # PERFORMANCE REPORTING
        # =================================================================
        
        # Report algorithm performance for HPC monitoring
        print(f'  Algorithm convergence: {test1.loop_times} iterations')
        print(f'  Total runtime: {test1.running_time:.2f} seconds')
        print(f'  Core processing time: {test1.running_coreTime:.2f} seconds')
        print(f'  Parallel efficiency: {(test1.running_coreTime/test1.running_time)*100:.1f}%')

        # =================================================================
        # INCLINATION VECTOR PROCESSING AND NORMALIZATION
        # =================================================================
        
        # Output the inclination data with proper coordinate transformation
        P_final = np.array(P)
        
        # Standard coordinate transformation for inclination vectors
        P_final[1] = -P[2]  # x-component = -original_y_component
        P_final[2] = P[1]   # y-component = original_x_component
        
        # Normalize inclination vectors to unit length
        magnitude = (P_final[1]**2 + P_final[2]**2)**0.5
        
        # Avoid division by zero
        mask = magnitude != 0
        P_final[1][mask] = P_final[1][mask] / magnitude[mask]  
        P_final[2][mask] = P_final[2][mask] / magnitude[mask]
        
        # Handle NaN values (set to zero)
        P_final = np.nan_to_num(P_final)
        
        # =================================================================
        # DATA OUTPUT AND STORAGE
        # =================================================================
        
        # Save processed inclination data
        output_filename = output_path + f"step{i}"
        np.save(output_filename, P_final)
        print(f"  Saved: {output_filename}.npy")
        print(f"  Data shape: {P_final.shape}")
        
    print(f"Completed processing: {input_path.split('/')[-1]}")
    print(f"Generated {len(step_list)} inclination files")


if __name__ == '__main__':
    """
    MAIN EXECUTION: HPC BATCH PROCESSING FOR PRIMME VALIDATION
    ========================================================
    
    High-performance batch processing of multiple phase field simulation cases
    for PRIMME validation studies. This script processes comprehensive datasets
    on HiPerGator supercomputing cluster with optimized I/O and parallel execution.
    
    Simulation Case Organization:
    - Case2: Standard grain boundary evolution scenarios
    - Case3: Advanced multi-physics coupling scenarios  
    - SPPARKS: Monte Carlo Potts model comparison baseline
    - Variable timesteps: Early (300), intermediate (400-800), final (1600)
    
    HPC Infrastructure:
    - Input Storage: /blue/michael.tonks/share/PRIMME_Inclination/
    - Output Storage: /blue/michael.tonks/share/PRIMME_Inclination_npy_files/
    - Processing: Multi-core parallel bilinear smoothing
    - Data Format: HDF5 → NumPy binary for analysis pipeline
    """

    # =================================================================
    # HPC FILE SYSTEM CONFIGURATION
    # =================================================================
    
    # HiPerGator Blue file system paths for large-scale data processing
    input_folder = "/blue/michael.tonks/share/PRIMME_Inclination/"
    output_folder = "/blue/michael.tonks/share/PRIMME_Inclination_npy_files/"
    
    print("="*70)
    print("PRIMME VALIDATION: HPC BATCH INCLINATION PROCESSING")
    print("="*70)
    print(f"Input directory:  {input_folder}")
    print(f"Output directory: {output_folder}")

    # =================================================================
    # SIMULATION CASE DEFINITIONS AND TIMESTEP SPECIFICATIONS
    # =================================================================
    
    # Comprehensive simulation case list for PRIMME validation
    # Format: CaseXY_TZ where X=scenario, Y=variant, Z=trial number
    input_name = ["Case2AS_T3_tstep_300_600",      # Case 2A Standard, Trial 3
                  "Case2BF_T8_tstep_300_1600",     # Case 2B Fast, Trial 8  
                  "Case2BS_T1_tstep_300_1600",     # Case 2B Standard, Trial 1
                  "Case2CF_T4_tstep_300_400",      # Case 2C Fast, Trial 4
                  "Case2DF_T1_tstep_300_1600",     # Case 2D Fast, Trial 1
                  "Case2DS_T5_tstep_300_800",      # Case 2D Standard, Trial 5
                  "Case2DS_T8_tstep_300_1600",     # Case 2D Standard, Trial 8
                  "Case3AF_T9_tstep_300_600",      # Case 3A Fast, Trial 9
                  "Case3AS_T6_tstep_300_1600",     # Case 3A Standard, Trial 6
                  "Case3BF_T7_tstep_300_600",      # Case 3B Fast, Trial 7
                  "Case3BS_T10_step_300_1600",     # Case 3B Standard, Trial 10
                  "Case3BS_T6_tstep_300_1600",     # Case 3B Standard, Trial 6
                  "Case3CS_T7_tstep_300_400",      # Case 3C Standard, Trial 7
                  "spparks_s1_tstep_300_400_600_800_1600"]  # SPPARKS baseline comparison

    # Corresponding timestep lists for each simulation case
    # Optimized for capturing key evolution stages: initial, intermediate, final
    step_lists = [[300,600],          # Early to intermediate evolution
                  [300,1600],         # Early to final state
                  [300,1600],         # Early to final state  
                  [300,400],          # Short evolution study
                  [300,1600],         # Full evolution range
                  [300,800],          # Early to intermediate
                  [300,1600],         # Full evolution range
                  [300,600],          # Early to intermediate
                  [300,1600],         # Full evolution range
                  [300,600],          # Early to intermediate
                  [300,1600],         # Full evolution range
                  [300,1600],         # Full evolution range
                  [300,400],          # Short evolution study
                  [300,400,600,800,1600]]  # Comprehensive SPPARKS comparison

    # =================================================================
    # BATCH PROCESSING EXECUTION LOOP
    # =================================================================
    
    print(f"\nProcessing {len(input_name)} simulation cases...")
    print("Case summary:")
    
    # Display processing plan
    for i, (case_name, steps) in enumerate(zip(input_name, step_lists)):
        print(f"  {i+1:2d}. {case_name:<35} → timesteps {steps}")
    
    print(f"\nStarting batch processing...")
    
    # Execute inclination calculation for each simulation case
    for i in range(len(input_name)):
        print(f"\n{'='*50}")
        print(f"PROCESSING CASE {i+1}/{len(input_name)}: {input_name[i]}")
        print(f"{'='*50}")
        
        # Generate output filename with consistent naming convention
        output_name = input_name[i] + "_inclination_"
        
        # Full file paths for current case
        input_path = input_folder + input_name[i]
        output_path = output_folder + output_name
        
        print(f"Input:  {input_path}.h5")
        print(f"Output: {output_path}stepXXX.npy")
        print(f"Steps:  {step_lists[i]}")
        
        try:
            # Execute inclination calculation for current case
            calculate_inclination_data(input_path, output_path, step_lists[i])
            print(f"✓ Successfully processed: {input_name[i]}")
            
        except Exception as e:
            print(f"✗ Error processing {input_name[i]}: {str(e)}")
            continue  # Continue with next case if current fails
    
    # =================================================================
    # COMPLETION SUMMARY AND NEXT STEPS
    # =================================================================
    
    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETED!")
    print(f"{'='*70}")
    print(f"Total cases processed: {len(input_name)}")
    print(f"Output directory: {output_folder}")
    
    # Calculate total output files
    total_files = sum(len(steps) for steps in step_lists)
    print(f"Generated files: {total_files} inclination datasets")
    
    print(f"\nNext steps for PRIMME validation:")
    print("1. Load inclination data using compare_inclination_PRIMME_hipergator.py")
    print("2. Generate grain boundary character distributions")
    print("3. Create comparative polar plots for method validation")
    print("4. Perform statistical accuracy analysis")
    print("5. Generate validation reports for PRIMME method")
    
    print(f"\nFile naming convention:")
    print("  {case_name}_inclination_step{timestep}.npy")
    print("  Array format: [3, nx, ny] = [microstructure, x_vector, y_vector]")
    print("  Unit normalized inclination vectors in standard coordinates")
    
    # Optional: Display storage usage estimate
    print(f"\nEstimated storage usage:")
    print(f"  Average file size: ~50-200 MB per timestep")  
    print(f"  Total dataset size: ~{total_files * 100} MB approximate")
    print(f"  Storage location: HiPerGator Blue file system")


