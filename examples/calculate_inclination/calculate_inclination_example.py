#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================
VECTOR Framework: Main Execution Script for Grain Boundary Inclination Analysis
=========================================================================================

Main execution script for VECTOR (VoxEl-based boundary inClination smooThing AlgORithms)
This script demonstrates the complete workflow for calculating grain boundary inclinations
and outputting site-specific normal vectors for each grain in a specific format.

Primary Functions:
- Execute 3D bilinear smoothing algorithm for grain boundary analysis
- Calculate normal vectors (inclinations) at each grain boundary site
- Output inclination data in structured format for each grain
- Process experimental microstructure data (DREAM.3D format)

Output Format Example:
The script generates "total_site.txt" with the following structure:
    Grain 1 Orientation: [phi1, Phi, phi2] centroid:
    x1, y1, z1, nx1, ny1, nz1
    x2, y2, z2, nx2, ny2, nz2
    ...
    
    Grain 2 Orientation: [phi1, Phi, phi2] centroid:
    x1, y1, z1, nx1, ny1, nz1
    ...

Where:
- x, y, z: Spatial coordinates of grain boundary sites
- nx, ny, nz: Normal vector components (inclination) at each site
- phi1, Phi, phi2: Euler angles defining grain crystallographic orientation

Created on Thu Sep 30 14:55:28 2021
@author: lin.yang
"""

# ================================================================================
# ENVIRONMENT SETUP AND IMPORTS
# ================================================================================
import os
current_path = os.getcwd()                      # Get current working directory for path management
sys.path.append('../../.')                      # Add parent directory to Python path for module imports
import sys
sys.path.append(current_path)                   # Add current directory to Python path
import numpy as np                              # Numerical computing for array operations
import math                                     # Mathematical functions for calculations
from itertools import repeat                    # Utility for creating repeated iterators

# ================================================================================
# VECTOR ALGORITHM IMPORTS
# ================================================================================
import myInput                                  # Input utilities and data processing functions
import PACKAGE_MP_Vertex       #2D Vertex smooth algorithm
import PACKAGE_MP_Linear     #2D Bilinear smooth algorithm
import PACKAGE_MP_AllenCahn    #2D Allen-Cahn smooth algorithm
import PACKAGE_MP_LevelSet     #2D Level Set smooth algorithm
import PACKAGE_MP_3DVertex     #3D Vertex smooth algorithm
import PACKAGE_MP_3DLinear   #3D Bilinear smooth algorithm
import PACKAGE_MP_3DAllenCahn  #3D Allen-Cahn smooth algorithm
import PACKAGE_MP_3DLevelSet   #3D Level Set smooth algorithm

#%% ================================================================================
# 3D INITIAL CONDITIONS CONFIGURATION
# ================================================================================
"""
This section demonstrates different approaches for loading 3D microstructure data:
1. Voronoi-generated microstructures for synthetic analysis
2. Experimental microstructure data from DREAM.3D format
"""

# Example 1: Voronoi-generated microstructure (commented out for demonstration)
# Demostration Voronoi 1000 grains sample with 0 timestep, 10 timestep, 50 timestep
# nx, ny, nz = 100, 100, 100
# ng = 1000
# filepath = current_path + '/Input/'
# P0,R=myInput.init2IC3d(nx,ny,nz,ng,"VoronoiIC1000.init",False,filepath)

# Example 2: Experimental DREAM.3D microstructure data (active configuration)
# Validation Dream3d 831 grains sample ("s1400poly1_t0.init") with 0 timestep
nx, ny, nz = 201, 201, 43                      # 3D domain dimensions (voxels)
ng = 831                                       # Total number of grains in microstructure
filename = "s1400poly1_t0.init"               # Input file containing experimental microstructure
P0,R=myInput.init2IC3d(nx,ny,nz,ng,filename,True)  # Load microstructure data and reference vectors



# %% ================================================================================
# 2D INITIAL CONDITIONS (ALTERNATIVE CONFIGURATIONS)
# ================================================================================
"""
Alternative 2D microstructure configurations for different analysis scenarios.
These are commented out but provide examples of various microstructure types:
- Simple 2-grain systems for algorithm validation
- Circular grain systems for geometric testing
- Voronoi polycrystalline systems for realistic analysis
- Complex and abnormal grain configurations for specialized studies
"""

# Example 2D configurations (commented out - uncomment for 2D analysis)
# nx, ny= 200,200
# ng = 2

# P0,R=myInput.init2IC(nx, ny, ng, "PolyIC.init")     # Load from polycrystalline input file
# P0,R=myInput.Complex2G_IC3d(nx,ny,nz)              # Complex 2-grain 3D system
# P0,R=myInput.Circle_IC(nx,ny)                      # Circular grain in 2D
# P0,R=myInput.Circle_IC3d(nx,ny,nz)                 # Circular grain in 3D
# P0,R=myInput.Voronoi_IC(nx,ny,ng)                  # Voronoi tessellation 2D
# P0,R=myInput.Complex2G_IC(nx,ny)                   # Complex 2-grain 2D system
# P0,R=myInput.Abnormal_IC(nx,ny)                    # Abnormal grain growth configuration
# P0,R=myInput.SmallestGrain_IC(100,100)             # Smallest grain test system

#%% ================================================================================
# VECTOR SMOOTHING ALGORITHM EXECUTION
# ================================================================================
"""
Execute the 3D Bilinear smoothing algorithm to calculate grain boundary inclinations.
This section runs the core VECTOR algorithm with specified parameters and outputs
the smoothed inclination field for subsequent analysis.

Algorithm Parameters:
- cores: Number of CPU cores for parallel processing
- loop_times: Number of iteration loops for convergence
- Algorithm: 3D Bilinear (BL3dv1) smoothing method
"""

# Algorithm execution loop with parameter variation
for cores in [8]:                               # Number of parallel processing cores
    for loop_times in range(4,5):              # Iteration range for algorithm convergence
        
        # Initialize and execute 3D Bilinear smoothing algorithm
        test1 = PACKAGE_MP_3DLinear.BL3dv1_class(nx,ny,nz,ng,cores,loop_times,P0,R,'np')
        test1.BL3dv1_main()                     # Execute the main smoothing algorithm
        P = test1.get_P()                       # Extract smoothed inclination field
    
        
        #%% Optional visualization and analysis (commented out)
        # test1.get_gb_num(1)                   # Get grain boundary count for specific grain
        # test1.get_2d_plot('DREAM3D_poly','Bilinear')  # Generate 2D visualization plot
        
        
        #%% ================================================================================
        # ALGORITHM PERFORMANCE METRICS OUTPUT
        # ================================================================================
        """
        Display algorithm performance metrics including:
        - Iteration count and convergence information
        - Total processing time and per-core processing time
        - Optional error metrics for algorithm validation
        """
          
        print('loop_times = ' + str(test1.loop_times))        # Number of iterations completed
        print('running_time = %.2f' % test1.running_time)     # Total algorithm execution time
        print('running_core time = %.2f' % test1.running_coreTime)  # Per-core processing time
        # print('total_errors = %.2f' % test1.errors)         # Total convergence errors (optional)
        # print('per_errors = %.3f' % test1.errors_per_site)  # Per-site error metrics (optional)
        print()                                               # Blank line for output formatting
        
#%% ================================================================================
# INCLINATION DATA OUTPUT FUNCTIONS
# ================================================================================
"""
This section contains specialized functions to extract and output grain boundary
inclination data in a structured format. The output provides site-specific normal
vectors for each grain boundary location.

OUTPUT FORMAT SPECIFICATION:
The generated file "total_site.txt" contains:
    Grain N Orientation: [phi1, Phi, phi2] centroid:
    x1, y1, z1, nx1, ny1, nz1
    x2, y2, z2, nx2, ny2, nz2
    ...
    
Where:
- N: Grain number (1-indexed)
- [phi1, Phi, phi2]: Euler angles in degrees (crystallographic orientation)
- x, y, z: Voxel coordinates of grain boundary sites
- nx, ny, nz: Normal vector components (grain boundary inclination)
"""

def get_gb_sites(P,grain_num):
    """
    Extract grain boundary sites for each grain in the microstructure.
    
    This function identifies all voxel locations that are on grain boundaries
    for each individual grain by checking neighboring voxels for grain ID differences.
    
    Parameters:
    -----------
    P : numpy.ndarray
        4D array containing grain IDs (timestep, x, y, z)
    grain_num : int
        Total number of grains to analyze
        
    Returns:
    --------
    ggn_gbsites : list of lists
        List containing grain boundary site coordinates for each grain
        Format: [[grain1_sites], [grain2_sites], ...] where each site is [x,y,z]
    """
    _,nx,ny,nz=np.shape(P)                      # Extract spatial dimensions from grain ID array
    timestep=5                                  # Boundary offset to avoid edge effects
    ggn_gbsites = [[] for i in repeat(None, grain_num)]  # Initialize empty lists for each grain
    
    # Scan through internal volume (excluding boundary regions)
    for i in range(timestep,nx-timestep):
        for j in range(timestep,ny-timestep):
            for k in range(timestep,nz-timestep):
                # Get periodic boundary condition neighbors
                ip,im,jp,jm,kp,km = myInput.periodic_bc3d(nx,ny,nz,i,j,k)
                
                # Check if current site is on a grain boundary
                if ( ((P[0,ip,j,k]-P[0,i,j,k])!=0) or ((P[0,im,j,k]-P[0,i,j,k])!=0) or\
                     ((P[0,i,jp,k]-P[0,i,j,k])!=0) or ((P[0,i,jm,k]-P[0,i,j,k])!=0) or\
                     ((P[0,i,j,kp]-P[0,i,j,k])!=0) or ((P[0,i,j,km]-P[0,i,j,k])!=0) ) and\
                    P[0,i,j,k] <= grain_num:    # Ensure grain ID is within valid range
                    # Add boundary site to corresponding grain's list
                    ggn_gbsites[int(P[0,i,j,k]-1)].append([i,j,k])
    return ggn_gbsites

def norm_list(grain_num, P_matrix):
    """
    Calculate normal vectors (inclinations) at each grain boundary site.
    
    This function computes the gradient-based normal vectors at all grain boundary
    sites for each grain. The normal vectors represent the local grain boundary
    inclination directions.
    
    Parameters:
    -----------
    grain_num : int
        Total number of grains in the microstructure
    P_matrix : numpy.ndarray
        4D smoothed inclination field from VECTOR algorithm
        
    Returns:
    --------
    norm_list : list of numpy.ndarray
        List containing normal vectors for each grain's boundary sites
        Format: [grain1_normals, grain2_normals, ...] where each is shape (n_sites, 3)
    boundary_site : list of lists
        Corresponding grain boundary site coordinates for each grain
    """
    # get the norm list
    grain_num -= 1                              # Convert to 0-indexed grain numbering
    boundary_site = get_gb_sites(P_matrix, grain_num)  # Extract grain boundary sites
    
    # Initialize normal vector storage for each grain
    norm_list = [np.zeros(( len(boundary_site[i]), 3 )) for i in range(grain_num)]
    
    # Calculate normal vectors for each grain's boundary sites
    for grain_i in range(grain_num):
        print(f"finish {grain_i}")             # Progress indicator for grain processing
        
        # Process each boundary site for current grain
        for site in range(len(boundary_site[grain_i])):
            # Calculate 3D gradient (normal vector) at boundary site
            norm = myInput.get_grad3d(P_matrix, boundary_site[grain_i][site][0], 
                                     boundary_site[grain_i][site][1], boundary_site[grain_i][site][2])
            norm_list[grain_i][site,:] = list(norm)  # Store normal vector components
        
    return norm_list, boundary_site

def get_orientation(grain_num, init_name ):
    """
    Extract crystallographic orientations (Euler angles) from input file.
    
    This function reads the crystallographic orientation data from the initialization
    file, extracting Euler angles that define each grain's orientation in the sample
    reference frame.
    
    Parameters:
    -----------
    grain_num : int
        Total number of grains in the microstructure
    init_name : str
        Name of the initialization file containing orientation data
        
    Returns:
    --------
    eulerAngle : numpy.ndarray
        Array of Euler angles for each grain, shape (grain_num-1, 3)
        Format: [phi1, Phi, phi2] in degrees for each grain
    """
    # read the input euler angle from *.init
    eulerAngle = np.ones((grain_num,3))*-10     # Initialize with sentinel values (-10)
    
    # Parse initialization file for Euler angle data
    with open('Input/'+init_name, 'r', encoding = 'utf-8') as f:
        for line in f:
            eachline = line.split()             # Split line into components
    
            # Process lines with grain orientation data (5 columns, not comments)
            if len(eachline) == 5 and eachline[0] != '#':
                lineN = int(eachline[1])-1      # Convert to 0-indexed grain number
                # Store Euler angles if not already processed
                if eulerAngle[lineN,0] == -10:
                    eulerAngle[lineN,:] = [float(eachline[2]), float(eachline[3]), float(eachline[4])]
    
    return eulerAngle[:ng-1]                    # Return orientations for all grains except background

def output(output_name, norm_list, site_list, orientation_list):
    """
    Generate formatted output file with inclination data for each grain.
    
    This function creates a structured text file containing grain boundary inclination
    data in a specific format that includes grain orientations and site-specific
    normal vectors.
    
    OUTPUT FILE FORMAT:
    Grain N Orientation: [phi1, Phi, phi2] centroid:
    x1, y1, z1, nx1, ny1, nz1
    x2, y2, z2, nx2, ny2, nz2
    ...
    [blank line]
    Grain N+1 Orientation: [phi1, Phi, phi2] centroid:
    ...
    
    Parameters:
    -----------
    output_name : str
        Name of output file to be created in 'output/' directory
    norm_list : list of numpy.ndarray
        Normal vectors for each grain's boundary sites
    site_list : list of lists
        Corresponding boundary site coordinates
    orientation_list : numpy.ndarray
        Euler angles for each grain
        
    Returns:
    --------
    None : Creates formatted output file in 'output/' directory
    """
    
    file = open('output/'+output_name,'w')      # Create output file in output directory
    
    # Process each grain sequentially
    for i in range(len(norm_list)):
        # Write grain header with orientation information
        file.writelines(['Grain ' + str(i+1) + ' Orientation: ' + str(orientation_list[i]) + ' centroid: ' + '\n'])
        
        # Write each boundary site with coordinates and normal vector
        for j in range(len(norm_list[i])):
            # Format: x, y, z, nx, ny, nz (coordinates and normal vector components)
            file.writelines([str(site_list[i][j][0]) + ', ' + str(site_list[i][j][1]) + ', ' + str(site_list[i][j][2]) + ', ' + str(norm_list[i][j][0]) + ', ' + str(norm_list[i][j][1]) + ', ' + str(norm_list[i][j][2]) + '\n'])
            
        # Add blank line separator between grains
        file.writelines(['\n'])
    
    file.close()                                # Close output file
    return

# ================================================================================
# MAIN EXECUTION: INCLINATION DATA EXTRACTION AND OUTPUT
# ================================================================================
"""
Execute the complete workflow to generate inclination output:
1. Calculate normal vectors at all grain boundary sites
2. Extract crystallographic orientations from input file  
3. Generate formatted output file with structured inclination data

The output file "total_site.txt" provides a comprehensive dataset of:
- Grain-by-grain organization with crystallographic orientations
- Site-specific coordinates and normal vectors
- Structured format suitable for further analysis or visualization
"""

# Step 1: Calculate normal vectors for all grain boundary sites
norm_list1, site_list1 = norm_list(ng, P)      # Extract normal vectors and site coordinates

# Step 2: Get crystallographic orientations from input file
orientation_list1 = get_orientation(ng, filename)  # Read Euler angles for each grain

# Step 3: Generate formatted output file
output("total_site.txt", norm_list1, site_list1, orientation_list1)  # Create structured inclination data file

"""
OUTPUT FILE STRUCTURE EXAMPLE:
===============================
Grain 1 Orientation: [0.123 1.456 2.789] centroid:
10, 15, 20, 0.707, 0.000, 0.707
11, 15, 20, 0.000, 1.000, 0.000
...

Grain 2 Orientation: [1.234 2.567 3.890] centroid:
25, 30, 35, -0.707, 0.707, 0.000
26, 30, 35, 0.577, 0.577, 0.577
...

Where each line contains:
x, y, z, nx, ny, nz (spatial coordinates and normal vector components)
"""




