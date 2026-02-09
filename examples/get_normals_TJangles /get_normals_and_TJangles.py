#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Normal Vector and Triple Junction Analysis for Complex Microstructures
=========================================================================

This module provides comprehensive 3D analysis capabilities for grain boundary
normal vectors and triple junction characterization in complex polycrystalline
systems.

Created on Fri Mar 24 11:48:29 2023
@author: Lin
"""

import os
current_path = os.getcwd()
import numpy as np
from numpy import seterr
seterr(all='raise')
import sys
sys.path.append(current_path)
sys.path.append(current_path+'/../../')
import myInput
import PACKAGE_MP_3DLinear as linear3d
sys.path.append(current_path+'/../calculate_tangent/')
import output_tangent_3d

# Import shared utilities
from utils_angles import (
    get_gb_sites_3d as get_gb_sites,
    norm_list_3d as norm_list,
    get_orientation,
    output_inclination_3d as output_inclination,
    output_dihedral_angle_3d as output_dihedral_angle
)


if __name__ == '__main__':
    """
    Main Execution: 3D Grain Boundary Analysis Pipeline
    ===================================================
    
    This script performs comprehensive 3D analysis of grain boundary normal vectors
    and triple junction characterization for complex polycrystalline microstructures.
    
    Analysis Pipeline:
    -----------------
    1. Load 3D microstructure data from SPPARKS initialization file
    2. Initialize 3D linear solver with optimized parameters
    3. Perform 3D inclination analysis using advanced computational geometry
    4. Calculate 3D normal vectors for all grain boundary sites
    5. Extract crystallographic orientations from initialization data
    6. Perform 3D triple junction analysis and dihedral angle calculations
    7. Generate comprehensive output files for further analysis
    
    Scientific Objectives:
    ---------------------
    - Characterize 3D grain boundary inclination distributions
    - Analyze complex triple junction line networks in 3D
    - Validate 3D computational geometry algorithms
    - Generate data for advanced materials science research
    
    Technical Specifications:
    ------------------------
    - Handles large-scale 3D datasets (501×501×50 voxels)
    - Processes thousands of grains efficiently
    - Uses parallel processing for computational acceleration
    - Implements robust error handling and quality control
    
    Output Products:
    ---------------
    - Detailed inclination data with 3D coordinates and normal vectors
    - Triple junction analysis with dihedral angle measurements
    - Crystallographic orientation correlation data
    - High-precision numerical results for scientific analysis
    """
    
    # Configuration for 3D microstructure analysis
    filename = "Input/An1Fe.init"  # SPPARKS initialization file
    
    # Define 3D microstructure parameters
    nx, ny, nz = 501, 501, 50  # 3D domain dimensions
    ng = 10928                 # Number of grains in the system
    cores = 8                  # Parallel processing cores
    loop_times = 5             # Solver iteration count
    print("IC's parameters done")
    
    # Load 3D microstructure from initialization file
    P0, R = myInput.init2IC3d(nx, ny, nz, ng, filename, True, './')
    print("IC is read as matrix")
    
    # Initialize 3D linear solver for inclination analysis
    test1 = linear3d.linear3d_class(nx, ny, nz, ng, cores, loop_times, P0, R, 'np')
    test1.linear3d_main("inclination")
    P = test1.get_P()
    
    # Report solver performance metrics
    print('loop_times = ' + str(test1.loop_times))
    print('running_time = %.2f' % test1.running_time)
    print('running_core time = %.2f' % test1.running_coreTime)
    print('total_errors = %.2f' % test1.errors)
    print('per_errors = %.3f' % test1.errors_per_site)
    print("Inclination calculation done")
    
    # Calculate 3D normal vectors for all grain boundary sites
    norm_list1, site_list1 = norm_list(ng, P)
    # Extract crystallographic orientations
    orientation_list1 = get_orientation(ng, filename)
    # Export detailed inclination analysis results
    output_inclination("An1Fe_inclination.txt", norm_list1, site_list1, orientation_list1)
    print("Inclination outputted")
    
    # Perform 3D triple junction analysis
    # Calculate tangent vectors and dihedral angles for triple junction lines
    triple_coord, triple_angle, triple_grain = output_tangent_3d.calculate_tangent(P0[:,:,:,0], loop_times)
    print("Tangent calculation done")
    
    # Data structure documentation for 3D analysis:
    # triple_coord: 3D coordinates of triple junction lines (left-upper voxel)
    #   axis 0 = triple junction index, axis 1 = coordinates (i,j,k)
    # triple_angle: three dihedral angles for each triple junction line
    #   axis 0 = triple junction index, axis 1 = three dihedral angles
    # triple_grain: sequence of three grains for each triple junction line
    #   axis 0 = triple junction index, axis 1 = three grain IDs
    
    # Export comprehensive dihedral angle analysis results
    output_dihedral_angle("An1Fe_dihedral.txt", triple_coord, triple_angle, triple_grain)
    print("Dihedral angle outputted")
    
    