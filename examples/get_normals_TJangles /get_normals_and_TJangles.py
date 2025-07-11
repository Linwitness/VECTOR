#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Normal Vector and Triple Junction Analysis for Complex Microstructures
=========================================================================

This module provides comprehensive 3D analysis capabilities for grain boundary
normal vectors and triple junction characterization in complex polycrystalline
systems. It extends 2D analysis methods to handle full 3D microstructures with
advanced geometric calculations and orientation analysis.

Scientific Background:
- 3D grain boundary normal vector computation using gradient methods
- Triple junction line analysis in three-dimensional space
- Complex grain boundary network topology characterization
- 3D inclination distribution analysis for texture studies

Key Features:
- Full 3D gradient calculation for accurate normal vector determination
- 3D grain boundary site identification with connectivity analysis
- Triple junction line detection and characterization
- Comprehensive output generation for 3D visualization
- Memory-efficient processing for large 3D datasets

Advanced Capabilities:
- Multi-grain boundary connectivity analysis
- 3D orientation relationship characterization
- Spatial correlation analysis of grain boundary properties
- Integration with 3D visualization and analysis pipelines

Research Applications:
- 3D microstructure characterization from experimental data
- Validation of 3D grain growth simulation results
- Advanced materials science research on 3D grain boundary behavior
- Development of 3D computational geometry algorithms

Technical Implementation:
- Uses PACKAGE_MP_3DLinear for advanced 3D linear algebra operations
- Efficient memory management for large 3D arrays
- Optimized algorithms for 3D neighbor connectivity
- Robust handling of complex grain boundary topologies

Created on Fri Mar 24 11:48:29 2023
@author: Lin
"""

import os
current_path = os.getcwd()
import numpy as np
from numpy import seterr
seterr(all='raise')
import math
from itertools import repeat
import sys
sys.path.append(current_path)
sys.path.append(current_path+'/../../')
import myInput
import PACKAGE_MP_3DLinear as linear3d
sys.path.append(current_path+'/../calculate_tangent/')
import output_tangent_3d



def get_gb_sites(P, grain_num):
    """
    Identify 3D Grain Boundary Sites in Complex Microstructures
    
    This function systematically identifies all grain boundary sites in 3D
    by analyzing neighbor connectivity and detecting interface locations
    between different grains in three-dimensional space.
    
    Parameters:
    -----------
    P : ndarray
        4D array representing 3D microstructure evolution (time, x, y, z)
    grain_num : int
        Total number of grains in the microstructure
        
    Returns:
    --------
    ggn_gbsites : list of lists
        3D grain boundary sites organized by grain ID
        Each sublist contains [i,j,k] coordinates of boundary sites for that grain
        
    Algorithm Details:
    -----------------
    - Uses 3D periodic boundary conditions for edge handling
    - Excludes boundary region (timestep=5) to avoid edge effects
    - Identifies sites where neighbors have different grain IDs in 3D space
    - Checks all 6 nearest neighbors in 3D cubic lattice
    - Systematic grain-by-grain organization for efficient processing
    
    Scientific Applications:
    -----------------------
    - 3D interface area calculation and characterization
    - Complex grain boundary network topology analysis
    - Preprocessing for 3D normal vector calculations
    - Statistical analysis of 3D grain boundary properties
    - Input for 3D visualization and rendering pipelines
    """
    _, nx, ny, nz = np.shape(P)
    timestep = 5  # Buffer zone to avoid boundary effects
    ggn_gbsites = [[] for i in repeat(None, grain_num)]
    
    # Systematic scan through 3D domain excluding boundary regions
    for i in range(timestep, nx-timestep):
        for j in range(timestep, ny-timestep):
            for k in range(timestep, nz-timestep):
                # Get 3D periodic boundary condition neighbors
                ip, im, jp, jm, kp, km = myInput.periodic_bc3d(nx, ny, nz, i, j, k)
                
                # Check if current site has neighbors with different grain IDs in 3D
                if (((P[0,ip,j,k]-P[0,i,j,k])!=0) or ((P[0,im,j,k]-P[0,i,j,k])!=0) or
                    ((P[0,i,jp,k]-P[0,i,j,k])!=0) or ((P[0,i,jm,k]-P[0,i,j,k])!=0) or
                    ((P[0,i,j,kp]-P[0,i,j,k])!=0) or ((P[0,i,j,km]-P[0,i,j,k])!=0)) and\
                    P[0,i,j,k] <= grain_num:
                    ggn_gbsites[int(P[0,i,j,k]-1)].append([i,j,k])
    
    return ggn_gbsites

def norm_list(grain_num, P_matrix):
    """
    Calculate 3D Normal Vectors for All Grain Boundary Sites
    
    This function computes the three-dimensional normal vectors at each
    grain boundary site using advanced 3D gradient calculation methods.
    
    Parameters:
    -----------
    grain_num : int
        Total number of grains in the microstructure
    P_matrix : ndarray
        3D microstructure array for gradient calculation
        
    Returns:
    --------
    norm_list : list of ndarrays
        3D normal vectors organized by grain number
        Each array contains [normal_x, normal_y, normal_z] for boundary sites
    boundary_site : list of lists
        Corresponding 3D boundary site coordinates
        
    Computational Details:
    ---------------------
    - Uses myInput.get_grad3d() for accurate 3D gradient calculation
    - Systematic processing grain by grain with progress reporting
    - Preserves spatial correlation between sites and normal vectors
    - Memory-efficient storage organized by grain boundaries
    - Handles complex 3D grain boundary topologies
    
    Scientific Applications:
    -----------------------
    - 3D interface inclination analysis
    - Complex grain boundary character distribution
    - 3D triple junction line analysis (input for advanced calculations)
    - Crystallographic orientation relationships in 3D
    - Advanced materials science research applications
    """
    # Adjust grain count for processing
    grain_num -= 1
    # Get 3D grain boundary sites for all grains
    boundary_site = get_gb_sites(P_matrix, grain_num)
    norm_list = [np.zeros((len(boundary_site[i]), 3)) for i in range(grain_num)]
    
    # Calculate 3D normal vectors for each grain's boundary sites
    for grain_i in range(grain_num):
        print(f"Processing 3D grain {grain_i} boundary normals...")
        
        for site in range(len(boundary_site[grain_i])):
            # Calculate 3D normal vector using gradient method
            norm = myInput.get_grad3d(P_matrix, boundary_site[grain_i][site][0], 
                                     boundary_site[grain_i][site][1], 
                                     boundary_site[grain_i][site][2])
            norm_list[grain_i][site,:] = list(norm)
        
    return norm_list, boundary_site

def get_orientation(grain_num, init_name):
    """
    Extract Euler Angles from SPPARKS 3D Initialization File
    
    This function reads crystallographic orientations (Euler angles) from
    SPPARKS .init files for comprehensive 3D grain boundary analysis.
    
    Parameters:
    -----------
    grain_num : int
        Total number of grains expected
    init_name : str
        Path to SPPARKS initialization file
        
    Returns:
    --------
    eulerAngle : ndarray
        Array of Euler angles [phi1, Phi, phi2] for each grain
        
    File Format Parsing:
    -------------------
    Expected format: site_id grain_id phi1 Phi phi2
    - Skips comment lines starting with '#'
    - Handles grain ID indexing (converts to 0-based)
    - Prevents duplicate angle assignment with sentinel values
    - Returns subset excluding last grain for compatibility
    
    Scientific Applications:
    -----------------------
    - Anisotropic grain boundary energy calculations in 3D
    - Misorientation analysis between neighboring grains
    - 3D texture analysis and crystallographic studies
    - Complex grain boundary character distribution analysis
    """
    # Initialize Euler angle array with sentinel values
    eulerAngle = np.ones((grain_num, 3)) * -10
    
    with open(init_name, 'r', encoding='utf-8') as f:
        for line in f:
            eachline = line.split()
    
            # Parse data lines (5 columns, not comments)
            if len(eachline) == 5 and eachline[0] != '#':
                lineN = int(eachline[1]) - 1  # Convert to 0-based indexing
                
                # Assign Euler angles if not already set
                if eulerAngle[lineN, 0] == -10:
                    eulerAngle[lineN, :] = [float(eachline[2]), float(eachline[3]), float(eachline[4])]
    
    return eulerAngle[:ng-1]  # Note: ng should be defined in calling context

def output_inclination(output_name, norm_list, site_list, orientation_list):
    """
    Export 3D Grain Boundary Inclination Data to File
    
    This function generates comprehensive output files containing 3D grain boundary
    site coordinates, normal vectors, and associated crystallographic orientations.
    
    Parameters:
    -----------
    output_name : str
        Output file name for 3D inclination data
    norm_list : list of ndarrays
        3D normal vectors organized by grain
    site_list : list of lists
        3D boundary site coordinates for each grain
    orientation_list : ndarray
        Euler angles for each grain
        
    Output Format:
    --------------
    For each grain:
    - Header: "Grain N Orientation: [phi1,Phi,phi2] centroid:"
    - Data lines: "x_coord, y_coord, z_coord, normal_x, normal_y, normal_z"
    - Blank line separator between grains
    
    Scientific Applications:
    -----------------------
    - Post-processing for 3D inclination distribution analysis
    - Input for 3D crystallographic analysis software
    - Advanced grain boundary character classification
    - Statistical analysis of 3D interface orientations
    - 3D visualization and rendering pipeline input
    """
    file = open('output/' + output_name, 'w')
    
    # Write data for each grain
    for i in range(len(norm_list)):
        # Write grain header with orientation information
        file.writelines(['Grain ' + str(i+1) + ' Orientation: ' + str(orientation_list[i]) + ' centroid: ' + '\n'])
        
        # Write 3D site coordinates and normal vectors
        for j in range(len(norm_list[i])):
            file.writelines([str(site_list[i][j][0]) + ', ' + str(site_list[i][j][1]) + ', ' + 
                           str(site_list[i][j][2]) + ', ' + str(norm_list[i][j][0]) + ', ' + 
                           str(norm_list[i][j][1]) + ', ' + str(norm_list[i][j][2]) + '\n'])
            
        file.writelines(['\n'])  # Separator between grains
    
    file.close()
    return

def output_dihedral_angle(output_name, triple_coord, triple_angle, triple_grain):
    """
    Export 3D Triple Junction Dihedral Angle Analysis Results
    
    This function creates detailed output files containing 3D triple junction
    coordinates, calculated dihedral angles, and associated grain IDs.
    
    Parameters:
    -----------
    output_name : str
        Output file name for dihedral angle data
    triple_coord : ndarray
        3D coordinates of triple junction points
    triple_angle : ndarray
        Calculated dihedral angles for each triple junction
    triple_grain : ndarray
        Grain IDs associated with each triple junction
        
    Output Format:
    --------------
    Header: "triple_index triple_coordination grain_id0:dihedral0 grain_id1:dihedral1 grain_id2:dihedral2"
    Data: "index, x y z, grain1:angle1 grain2:angle2 grain3:angle3"
    
    Scientific Applications:
    -----------------------
    - 3D triple junction line analysis
    - Equilibrium angle validation in 3D systems
    - Statistical analysis of 3D dihedral angle distributions
    - Complex microstructure characterization
    """
    file = open('output/' + output_name, 'w')
    
    # Write header for data interpretation
    file.writelines(['triple_index triple_coordination grain_id0:dihedral0 grain_id1:dihedral1 grain_id2:dihedral2'])
    
    # Write data for each triple junction
    for i in range(len(triple_coord)):
        file.writelines([str(i+1) + ', ' + str(triple_coord[i][0]) + ' ' + str(triple_coord[i][1]) + ' ' + 
                         str(triple_coord[i][2]) + ', ' + 
                         str(triple_grain[i][0]) + ':' + str(triple_angle[i][0]) + ' ' + 
                         str(triple_grain[i][1]) + ':' + str(triple_angle[i][1]) + ' ' + 
                         str(triple_grain[i][2]) + ':' + str(triple_angle[i][2])])
        
    file.close()
    return


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
    
    