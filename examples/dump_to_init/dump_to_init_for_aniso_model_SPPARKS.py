#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPPARKS ANISOTROPIC MODEL PREPROCESSING SCRIPT
==============================================

Production script for converting SPPARKS simulation dump files to initialization
format with neighbor connectivity generation. This script processes large-scale
3D polycrystalline microstructures for anisotropic grain boundary energy simulations.

Key Features:
- High-performance dump file to init conversion
- Preservation of crystallographic orientations during microstructure evolution
- Multiprocessed neighbor list generation for 3D periodic domains
- Support for HPC cluster file systems and large datasets

Workflow:
1. Load original Euler angle assignments from initialization file
2. Extract evolved grain structure from SPPARKS dump file at specified timestep
3. Generate neighbor connectivity files for anisotropic energy calculations
4. Output SPPARKS-compatible files for subsequent simulations

Created on Mon Jul 31 14:33:57 2023
@author: Lin
"""

import os
current_path = os.getcwd()
import numpy as np
from numpy import seterr
seterr(all='raise')  # Enable strict numerical error handling
import matplotlib.pyplot as plt
import math
from tqdm import tqdm  # Progress tracking for large file operations
import sys
sys.path.append(current_path+'/../../')

# VECTOR framework modules for microstructure analysis
import myInput
import post_processing
import PACKAGE_MP_3DLinear as linear3d

if __name__ == '__main__':
    """
    MAIN EXECUTION: PRODUCTION DUMP-TO-INIT CONVERSION
    =================================================
    
    This script processes a specific SPPARKS simulation run to extract the
    final microstructure state and generate initialization files for subsequent
    analysis or re-simulation with anisotropic grain boundary energy models.
    
    Target Simulation:
    - 3D polycrystalline domain: 450x450x450 sites
    - 5000 grains with fully anisotropic energy model  
    - Virtual inclination energy (5D orientation space)
    - Simulation: p_ori_fully5d_fz_aveE_f1.0_t1.0_450_5k
    """

    # =================================================================
    # SIMULATION AND FILE CONFIGURATION
    # =================================================================
    
    # Target timestep for microstructure extraction
    last_step = 16  # Final evolved state from SPPARKS simulation
    
    # HPC cluster storage paths for large-scale simulations
    dump_file_foler = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_poly_fully/"
    # Alternative local path for development/testing:
    # dump_file_foler = "/Users/lin/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/3d_poly_for_GG/"
    
    # SPPARKS simulation identifier with embedded parameters
    # Naming convention breakdown:
    # p_ori_fully5d: Polycrystal with full 5D orientation space
    # fz: Frozen zone constraints
    # aveE: Average energy model  
    # f1.0: Anisotropy factor = 1.0
    # t1.0: Temperature factor = 1.0
    # 450_5k: 450^3 domain with 5000 grains
    # multiCore64: 64-core parallel execution
    # J1: Job ID = 1
    # refer_1_0_0: Reference orientation [1,0,0]
    # seed56689: Random seed for reproducibility
    # kt1.95: Monte Carlo temperature kT = 1.95
    dump_file_name = f"p_ori_fully5d_fz_aveE_f1.0_t1.0_450_5k_multiCore64_J1_refer_1_0_0_seed56689_kt1.95"
    
    # Input/output directory structure
    init_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_poly/IC/"  # Original init files
    init_file_output_folder = dump_file_foler + "IC/"                                       # Processed outputs
    
    # File naming convention for processing pipeline
    init_file_name_original = f"poly_IC450_5k.init"                    # Original grain orientations
    init_file_name = f"poly_IC450_5k_s{last_step}.init"               # Converted dump data  
    init_file_name_final = f"poly_IC450_5k_s{last_step}_neighbor5.init"  # With neighbor connectivity

    # =================================================================
    # MICROSTRUCTURE PARAMETERS AND DATA LOADING
    # =================================================================
    
    # Grain population in the 3D polycrystalline microstructure
    grain_num = 5000  # Total grains in 450^3 domain
    
    # Load crystallographic orientations from original initialization file
    # These Euler angles define the anisotropic grain boundary energy landscape
    print("Loading original Euler angle assignments...")
    euler_angle_array = post_processing.init2EAarray(
        init_file_folder + init_file_name_original, 
        grain_num
    )
    print(f"✓ Loaded orientations for {grain_num} grains")

    # =================================================================
    # DUMP FILE TO INIT CONVERSION
    # =================================================================
    
    # Convert SPPARKS dump file from timestep 16 to initialization format
    # This preserves the evolved grain structure while mapping original orientations
    print(f"Converting dump file from timestep {last_step}...")
    dump_file_name_0 = dump_file_foler + dump_file_name + f".dump.{int(last_step)}"
    
    box_size, entry_length = post_processing.output_init_from_dump(
        dump_file_name_0,                           # Input: SPPARKS dump file
        euler_angle_array,                          # Orientation mapping
        init_file_output_folder + init_file_name    # Output: Converted init file
    )
    size_x, size_y, size_z = box_size
    print(f"✓ Converted {entry_length} entries for {size_x}x{size_y}x{size_z} domain")

    # =================================================================
    # NEIGHBOR CONNECTIVITY GENERATION
    # =================================================================
    
    # Generate neighbor lists for anisotropic grain boundary energy calculations
    # interval = 5 provides sufficient range for realistic GB energy interactions
    interval = 5  # Neighbor interaction range
    
    print(f"Generating neighbor connectivity with interval={interval}...")
    print(f"Expected neighbors per site: {(2*interval+3)**3-1}")
    
    # Use multiprocessed algorithm for efficient 3D neighbor list generation
    output_neighbr_init = post_processing.output_init_neighbor_from_init_mp(
        interval,                                           # Neighbor range
        box_size,                                          # Domain dimensions
        init_file_output_folder + init_file_name,         # Input: Converted init
        init_file_output_folder + init_file_name_final    # Output: With neighbors
    )
    
    # Alternative single-threaded version for debugging:
    # output_neighbr_init = output_init_neighbor_from_init(
    #     interval, 
    #     box_size, 
    #     init_file_folder + init_file_name, 
    #     init_file_folder + init_file_name_final + "_test"
    # )
    
    # =================================================================
    # COMPLETION STATUS AND FILE COMPRESSION
    # =================================================================
    
    print(f"Neighbor file generation status: {output_neighbr_init}")
    
    if output_neighbr_init:
        print("\n" + "="*60)
        print("CONVERSION PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"✓ Input dump:  {dump_file_name_0}")
        print(f"✓ Output init: {init_file_output_folder + init_file_name}")  
        print(f"✓ Neighbor file: {init_file_output_folder + init_file_name_final}")
        print(f"\nDomain: {size_x}x{size_y}x{size_z} sites")
        print(f"Grains: {grain_num}")
        print(f"Timestep: {last_step}")
        print("\nRecommended next step:")
        print("# Compress large neighbor file for storage/transfer")
        print(f"# pigz -p 128 -k {init_file_name_final}")
    else:
        print("\n" + "="*60) 
        print("ERROR: CONVERSION PIPELINE FAILED!")
        print("="*60)
        print("Check file paths, permissions, and available disk space.")
    
    # Optional: Parallel compression command for large files
    # Uncomment to automatically compress the output neighbor file
    # pigz -p 128 -k poly_IC450_5k_s16_neighbor5.init

