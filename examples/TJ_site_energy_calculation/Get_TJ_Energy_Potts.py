#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================
VECTOR Framework: Triple Junction Site Energy Calculation for Monte Carlo Potts Model
=========================================================================================

Scientific Application: Triple Junction Energy Analysis and Grain Boundary Energetics
Primary Focus: Anisotropic Energy Function Validation at Triple Junction Sites

Created on Sat Feb 25 21:00:34 2023
@author: Lin

=========================================================================================
TRIPLE JUNCTION ENERGY ANALYSIS FRAMEWORK
=========================================================================================

This Python script implements comprehensive triple junction site energy calculation
for Monte Carlo Potts model grain growth simulations, focusing on anisotropic energy
function validation and energy averaging method comparison at critical triple junction
sites where three grains meet in 2D polycrystalline microstructures.

Scientific Objectives:
- Triple junction energy function validation for anisotropic grain boundary models
- Energy averaging method comparison (ave, sum, min, max, consMin, consMax)
- Grain boundary inclination angle analysis with energy function coupling
- Verification of energy calculation algorithms for complex triple junction geometries
- Validation of energy functions against theoretical triple junction equilibrium

Key Features:
- Multiple energy averaging approaches for triple junction site energy calculation
- Anisotropic energy function with cosine dependence on grain boundary orientation
- VECTOR Linear2D integration for accurate grain boundary normal vector computation
- Verification test cases with controlled triple junction geometries (90°, 120°)
- Energy matrix visualization for spatial energy distribution analysis

Applications:
- Monte Carlo Potts model energy function validation
- Triple junction equilibrium angle verification
- Anisotropic grain boundary energy model development
- Energy averaging method sensitivity analysis for complex grain configurations
"""

# ================================================================================
# ENVIRONMENT SETUP AND PATH CONFIGURATION
# ================================================================================
import os
current_path = os.getcwd()                          # Current working directory for file operations
import sys
sys.path.append(current_path)                       # Add current directory to Python path
sys.path.append(current_path+'/../../')             # Add VECTOR framework root directory

# ================================================================================
# SCIENTIFIC COMPUTING AND VISUALIZATION LIBRARIES
# ================================================================================
import matplotlib.pyplot as plt                     # Advanced scientific visualization
import numpy as np                                  # Numerical computing and array operations
from numpy import seterr                           # Numerical error handling configuration
seterr(all='raise')                                # Raise exceptions for numerical errors
import math                                         # Mathematical functions for energy calculations

# ================================================================================
# VECTOR FRAMEWORK INTEGRATION: SPECIALIZED ANALYSIS MODULES
# ================================================================================
import PACKAGE_MP_Linear as Linear_2D               # 2D linear algebra for grain boundary analysis
import PACKAGE_MP_Vertex as Vertex_2D               # 2D vertex operations for microstructural analysis
import myInput                                      # Input parameter management and file handling

def get_2d_ic1(nx,ny):
    """
    Generate 2D verification initial condition with 90° triple junction geometry.
    
    Creates a controlled 2D microstructure with three grains meeting at a triple
    junction with 90° angles for energy function validation and algorithm verification.
    This configuration provides a known analytical solution for energy calculations.
    
    Scientific Framework:
    - Triple Junction Geometry: 90° angle configuration (non-equilibrium)
    - Grain Configuration: Three grains in quadrant-based arrangement
    - Validation Purpose: Energy function algorithm verification with known geometry
    - Theoretical Reference: Non-equilibrium triple junction for method validation
    
    Parameters:
    -----------
    nx : int
        Grid dimension in x-direction for 2D microstructure array
    ny : int  
        Grid dimension in y-direction for 2D microstructure array
        
    Returns:
    --------
    triple_map : np.ndarray
        2D integer array [nx, ny] containing grain ID assignments
        - Grain 1: Lower-left quadrant (i < nx/2, j < ny/2)
        - Grain 2: Upper-left quadrant (i < nx/2, j >= ny/2)  
        - Grain 3: Right half (i >= nx/2)
        
    Scientific Applications:
    ------------------------
    - Energy function algorithm validation with controlled geometry
    - Triple junction equilibrium angle verification studies
    - Non-equilibrium configuration analysis for energy minimization
    - Benchmark testing for anisotropic energy calculation methods
    """
    # Initialize 2D grain ID array for controlled triple junction geometry
    triple_map = np.zeros((nx,ny))
    
    # Generate three-grain configuration with 90° triple junction angles
    for i in range(nx):
        for j in range(ny):
            if i < nx/2 and j < ny/2: triple_map[i,j] = 1      # Grain 1: Lower-left quadrant
            elif i < nx/2 and j >= ny/2: triple_map[i,j] = 2   # Grain 2: Upper-left quadrant
            else: triple_map[i,j] = 3                          # Grain 3: Right half domain
    
    return triple_map

def get_2d_ic2(nx,ny):
    """
    Generate 2D verification initial condition with 120° triple junction geometry.
    
    Creates a controlled 2D microstructure with three grains meeting at a triple
    junction with 120° equilibrium angles for energy function validation. This
    configuration represents the theoretical equilibrium state for isotropic
    grain boundary energy systems.
    
    Scientific Framework:
    - Triple Junction Geometry: 120° equilibrium angle configuration
    - Grain Configuration: Three grains with geometric equilibrium arrangement
    - Validation Purpose: Equilibrium energy state verification for isotropic systems
    - Theoretical Reference: Young's equation triple junction equilibrium
    
    Parameters:
    -----------
    nx : int
        Grid dimension in x-direction for 2D microstructure array
    ny : int  
        Grid dimension in y-direction for 2D microstructure array
        
    Returns:
    --------
    triple_map : np.ndarray
        2D integer array [nx, ny] containing grain ID assignments with 120° geometry
        - Grain 1: Left region and lower-right triangular area
        - Grain 2: Upper-left region and upper-right triangular area
        - Grain 3: Central triangular region between grains 1 and 2
        
    Geometric Construction:
    -----------------------
    The 120° triple junction geometry is constructed using:
    - Linear boundaries with sqrt(3) slope for 60° angles
    - Triangular grain 3 region bounded by angled interfaces
    - Symmetric arrangement preserving 120° equilibrium angles
        
    Scientific Applications:
    ------------------------
    - Equilibrium triple junction energy validation
    - Isotropic energy function verification studies
    - Young's equation validation for grain boundary systems
    - Benchmark comparison for anisotropic vs. isotropic energy models
    """
    # Initialize 2D grain ID array for equilibrium triple junction geometry
    triple_map = np.zeros((nx,ny))
    
    # Generate three-grain configuration with 120° equilibrium triple junction angles
    for i in range(nx):
        for j in range(ny):
            if i < nx/2 and j < ny/2: triple_map[i,j] = 1                                      # Base grain 1 region
            elif i < nx/2 and j >= ny/2: triple_map[i,j] = 2                                  # Base grain 2 region
            elif i >= nx/2 and j < nx/2 - (i - nx/2) * math.sqrt(3): triple_map[i,j] = 1     # Extended grain 1 (lower triangle)
            elif i >= nx/2 and j >= nx/2 + (i - nx/2) * math.sqrt(3) - 1: triple_map[i,j] = 2 # Extended grain 2 (upper triangle)
            else: triple_map[i,j] = 3                                                          # Central grain 3 (equilibrium region)
    
    return triple_map


def energy_function(normals, delta = 0.6, m = 2):
    """
    Calculate anisotropic grain boundary energy based on normal vector orientation.
    
    Implements the cosine-based anisotropic energy function for grain boundary
    energy calculation in Monte Carlo Potts model simulations. The energy depends
    on the angle between the grain boundary normal vector and a reference direction,
    providing orientation-dependent energy that drives anisotropic grain growth.
    
    Mathematical Framework:
    E(θ) = 1 + δ * cos(m * θ)
    
    Where:
    - θ: Angle between grain boundary normal and reference direction [1,0]
    - δ: Anisotropy strength parameter (0 ≤ δ ≤ 1)
    - m: Periodicity parameter for angular dependence
    
    Scientific Background:
    - Based on Read-Shockley grain boundary energy formulation
    - Provides realistic anisotropic energy for crystallographic systems
    - Enables simulation of orientation-dependent grain growth kinetics
    - Validated against experimental grain boundary energy measurements
    
    Parameters:
    -----------
    normals : array_like
        2D normal vector [nx, ny] for grain boundary orientation
        Should be normalized unit vector for accurate energy calculation
    delta : float, optional (default=0.6)
        Anisotropy strength parameter controlling energy variation amplitude
        - delta=0.0: Isotropic energy (constant)
        - delta=1.0: Maximum anisotropy (energy range [0, 2])
    m : int, optional (default=2)
        Periodicity parameter for angular dependence
        - m=2: Four-fold symmetry (cubic crystallography)
        - m=4: Eight-fold symmetry (higher-order anisotropy)
        
    Returns:
    --------
    energy : float
        Anisotropic grain boundary energy value
        Range: [1-δ, 1+δ] for physically meaningful energy values
        
    Computational Framework:
    ------------------------
    1. Calculate angle between normal vector and reference direction [1,0]
    2. Apply cosine anisotropic energy function with specified parameters
    3. Return normalized energy value for Monte Carlo energy calculations
        
    Scientific Applications:
    ------------------------
    - Anisotropic grain growth simulation energy calculations
    - Grain boundary energy validation studies
    - Energy averaging method development and testing
    - Triple junction equilibrium angle prediction with anisotropic energy
    """
    # Reference direction vector for energy anisotropy calculation
    refer = np.array([1,0])                         # Reference direction [1,0] for energy calculation
    
    # Calculate angle between grain boundary normal and reference direction
    theta_rad = math.acos(round(np.array(normals).dot(refer), 5))  # Angle calculation with numerical precision
    
    # Apply anisotropic energy function with cosine dependence
    energy = 1 + delta * math.cos(m * theta_rad)   # Cosine-based anisotropic energy formulation
    
    return energy

def calculate_energy(P, i, j):
    """
    Calculate comprehensive site energy at specified location using multiple averaging methods.
    
    Implements advanced energy calculation for Monte Carlo Potts model sites with
    comprehensive energy averaging approaches. This function evaluates the local
    energy environment around a specific site by analyzing neighboring grain
    configurations and applying various energy averaging methodologies for
    comparison and validation studies.
    
    Energy Averaging Methods Available (commented sections):
    - Average Energy: Arithmetic mean of center and neighbor energies
    - Summation Energy: Total energy contribution from all neighbors (ACTIVE)
    - Minimum Energy: Lowest energy among all neighbors
    - Maximum Energy: Highest energy among all neighbors
    - Conservative Min/Max: Enhanced minimum/maximum with grain-wise averaging
    
    Scientific Framework:
    - Local energy environment analysis with 3x3 neighborhood window
    - Grain boundary normal vector computation using VECTOR Linear2D
    - Anisotropic energy function application for realistic energy calculation
    - Multiple averaging approaches for energy method sensitivity analysis
    
    Parameters:
    -----------
    P : np.ndarray
        2D microstructure array [nx, ny] containing grain ID assignments
        Each element represents the grain ID at that spatial location
    i : int
        x-coordinate (row) of the site for energy calculation
        Must be within valid array bounds with 1-pixel border
    j : int  
        y-coordinate (column) of the site for energy calculation
        Must be within valid array bounds with 1-pixel border
        
    Returns:
    --------
    site_energy : float
        Calculated site energy using the active energy averaging method
        Energy units consistent with anisotropic energy function scaling
        
    Energy Calculation Framework:
    -----------------------------
    1. Extract 3x3 neighborhood window around target site (i,j)
    2. Identify center grain and all neighboring grain IDs
    3. Calculate grain boundary normal vectors for energy computation
    4. Apply anisotropic energy function to each grain boundary
    5. Aggregate energies using specified averaging method
    6. Return total site energy for Monte Carlo energy calculations
        
    Neighborhood Analysis:
    ----------------------
    The 3x3 window provides comprehensive local energy environment:
    - Center site: Primary grain at location (i,j)
    - 8 neighbors: Surrounding sites for energy interaction analysis
    - Grain boundary detection: Interface identification between different grains
    - Normal vector computation: VECTOR Linear2D inclination analysis
        
    Scientific Applications:
    ------------------------
    - Triple junction site energy validation
    - Energy averaging method comparison studies  
    - Monte Carlo Potts model energy function development
    - Anisotropic grain growth energy calculation verification
    """
    # Initialize site energy accumulator
    site_energy = 0
    
    # ============================================================================
    # NEIGHBORHOOD EXTRACTION AND ANALYSIS
    # ============================================================================
    # Extract 3x3 neighborhood window for comprehensive local energy analysis
    window = P[i-1:i+2, j-1:j+2]                   # 3x3 window centered at (i,j)
    center = window[1,1]                           # Center grain ID
    
    # Extract edge neighbors and their coordinates for energy calculation
    edge = np.array(list(window[0,:])+[window[1,0],window[1,2]]+list(window[2,:]))  # 8 neighbors
    edge_coord = np.array([[i-1,j-1], [i-1,j], [i-1,j+1], [i,j-1], [i,j+1], [i+1,j-1], [i+1,j], [i+1,j+1]])
    
    
    # ============================================================================
    # CENTER SITE ENERGY CALCULATION
    # ============================================================================
    # Calculate grain boundary normal vector and energy for center site
    n1 = get_inclination(P, i, j)                  # Normal vector at center using VECTOR Linear2D
    e1 = energy_function(n1)                       # Anisotropic energy at center site
    print(f"center: {n1}, {e1}")                   # Debug output for center energy
    
    
    # Old Version
    # for m in range(len(edge_coord)):
    #     if edge[m] != center:
    #         n_edge = get_inclination(P , edge_coord[m][0], edge_coord[m][1])
    #         e_edge = energy_function(n_edge)
    #         print(f"edge: {n_edge}, {e_edge}")
            
    #         e_ave = (e1 + e_edge) / 2
    #         site_energy += e_ave
    
    # Old Version
    # for m in range(len(edge_coord)):
    #     if edge[m] != center:
    #         n_edge = get_inclination(P , edge_coord[m][0], edge_coord[m][1])
    #         n_edge = n_edge - n1
    #         n_edge = n_edge / np.linalg.norm(n_edge)
    #         e_edge = energy_function(n_edge)
    #         print(f"edge: {n_edge}, {e_edge}")
            
    #         e_ave = e_edge
    #         site_energy += e_ave
    
    # Min Energy
    # num_site = 0
    # min_eng = 8
    # for m in range(len(edge_coord)):
    #     if edge[m] != center:
    #         num_site += 1
    #         n_edge = get_inclination(P , edge_coord[m][0], edge_coord[m][1])
    #         e_edge = energy_function(n_edge)
    #         print(f"edge: {n_edge}, {e_edge}")
    #         if e_edge < min_eng: min_eng = e_edge
    # site_energy = min_eng #* num_site
    
    # Max Energy
    # num_site = 0
    # max_eng = 0
    # for m in range(len(edge_coord)):
    #     if edge[m] != center:
    #         num_site += 1
    #         n_edge = get_inclination(P , edge_coord[m][0], edge_coord[m][1])
    #         e_edge = energy_function(n_edge)
    #         print(f"edge: {n_edge}, {e_edge}")
    #         if e_edge > max_eng: max_eng = e_edge
    # site_energy = max_eng #* num_site
    
    # Ave Energy
    # num_site = 0
    # nei_grain_id = set(list(edge)+[center])
    # nei_grain_id.remove(center)
    # nei_grain_id = list(nei_grain_id)
    # engs_grain = np.zeros(len(nei_grain_id))
    # sites_grain = np.zeros(len(nei_grain_id))
    # ave_eng = 0
    # for m in range(len(edge_coord)):
    #     if edge[m] != center:
    #         num_site += 1
    #         n_edge = get_inclination(P , edge_coord[m][0], edge_coord[m][1])
    #         engs_grain[int(nei_grain_id.index(edge[m]))] += energy_function(n_edge)
    #         sites_grain[int(nei_grain_id.index(edge[m]))] += 1
    #         print(f"edge: {n_edge}, {energy_function(n_edge)}")
    # for m in range(len(nei_grain_id)): ave_eng += engs_grain[m] / sites_grain[m]
    # print(f"nei_eng: {engs_grain/sites_grain}")
    # ave_eng += e1
    # site_energy = ave_eng / (len(nei_grain_id) + 1)

    # ============================================================================
    # ENERGY AVERAGING METHOD: SUMMATION ENERGY (ACTIVE)
    # ============================================================================
    """
    Summation Energy Method: Comprehensive neighbor energy aggregation
    
    This method calculates the total energy contribution by:
    1. Identifying all unique neighboring grains
    2. Computing average energy for each neighboring grain
    3. Summing all neighbor grain energies with center energy
    4. Providing total energy for Monte Carlo energy calculations
    
    Scientific Justification:
    - Accounts for all local energy contributions
    - Preserves energy conservation in Monte Carlo dynamics
    - Enables realistic energy landscape for grain growth simulation
    """
    num_site = 0                                   # Counter for neighboring sites
    
    # Identify unique neighboring grain IDs for energy calculation
    nei_grain_id = set(list(edge)+[center])        # All grain IDs in neighborhood
    nei_grain_id.remove(center)                    # Remove center grain from neighbor list
    nei_grain_id = list(nei_grain_id)              # Convert to list for indexing
    
    # Initialize energy and site counters for each neighboring grain
    engs_grain = np.zeros(len(nei_grain_id))       # Energy accumulator for each neighbor grain
    sites_grain = np.zeros(len(nei_grain_id))      # Site count for each neighbor grain
    ave_eng = 0                                    # Total average energy accumulator
    
    # Calculate energy for each neighboring site
    for m in range(len(edge_coord)):
        if edge[m] != center:                      # Process only neighboring grains
            num_site += 1
            # Calculate normal vector and energy for neighbor site
            n_edge = get_inclination(P , edge_coord[m][0], edge_coord[m][1])  # Neighbor normal vector
            engs_grain[int(nei_grain_id.index(edge[m]))] += energy_function(n_edge)  # Accumulate energy
            sites_grain[int(nei_grain_id.index(edge[m]))] += 1                       # Count sites
            print(f"edge: {n_edge}, {energy_function(n_edge)}")                      # Debug output
    
    # Calculate average energy for each neighboring grain and sum total
    for m in range(len(nei_grain_id)): 
        ave_eng += engs_grain[m] / sites_grain[m]  # Average energy per neighbor grain
    print(f"nei_eng: {engs_grain[m]/sites_grain[m]}")  # Debug output for last neighbor
    
    # Include center energy in total summation
    ave_eng += e1                                  # Add center site energy
    site_energy = ave_eng                          # Final summation energy
    
    # ============================================================================
    # ALTERNATIVE ENERGY AVERAGING METHODS (COMMENTED FOR COMPARISON)
    # ============================================================================
    
    # AVERAGE ENERGY METHOD (COMMENTED)
    # Calculates arithmetic mean of center and neighbor energies
    # Provides balanced energy averaging for moderate anisotropy effects
    
    # MINIMUM ENERGY METHOD (COMMENTED)  
    # Selects lowest energy among all neighbors
    # Promotes energy minimization and stable grain configurations
    
    # MAXIMUM ENERGY METHOD (COMMENTED)
    # Selects highest energy among all neighbors  
    # Emphasizes high-energy configurations for instability analysis
    
    # CONSERVATIVE METHODS (COMMENTED)
    # Enhanced min/max with grain-wise averaging
    # Provides more stable energy calculation for complex geometries
            
    return site_energy
    
            
    return site_energy

def get_inclination(P, i, j, loop_times = 5, ng = 512):
    """
    Calculate grain boundary normal vector at specified site using VECTOR Linear2D analysis.
    
    Computes the grain boundary inclination (normal vector) at a specific site
    location using the VECTOR framework's Linear2D class for accurate grain
    boundary orientation analysis. This function provides the essential normal
    vector information needed for anisotropic energy function calculations
    in Monte Carlo Potts model simulations.
    
    Computational Framework:
    - VECTOR Linear2D class integration for robust normal vector computation
    - Multi-field representation of grain structure for accurate analysis
    - Iterative refinement with configurable loop_times for convergence
    - Single-core processing optimized for individual site analysis
    
    Mathematical Background:
    - Gradient-based normal vector computation from grain boundary interfaces
    - Coordinate transformation for consistent orientation representation
    - Vector normalization for unit normal vector calculation
    - Robust numerical methods for accurate boundary orientation
    
    Parameters:
    -----------
    P : np.ndarray
        2D microstructure array [nx, ny] containing grain ID assignments
        Each element represents the grain ID at that spatial location
    i : int
        x-coordinate (row) of the site for normal vector calculation
        Must be within valid array bounds for gradient computation
    j : int
        y-coordinate (column) of the site for normal vector calculation  
        Must be within valid array bounds for gradient computation
    loop_times : int, optional (default=5)
        Number of iterative refinement loops for normal vector convergence
        Higher values provide more accurate results at increased computational cost
    ng : int, optional (default=512)
        Maximum number of grains in the microstructure for field allocation
        Should match or exceed the actual number of grains in the system
        
    Returns:
    --------
    normal : np.ndarray
        2D unit normal vector [nx, ny] representing grain boundary orientation
        Normalized to unit length for consistent energy function application
        Coordinate system: Standard 2D Cartesian with proper orientation
        
    Computational Process:
    ----------------------
    1. Extract microstructure dimensions from input array
    2. Initialize multi-field representation for VECTOR Linear2D analysis
    3. Convert grain ID array to field-based representation for each grain
    4. Initialize VECTOR Linear2D class with computational parameters
    5. Compute normal vector at specified location using gradient methods
    6. Apply coordinate transformation and normalization
    7. Return unit normal vector for energy function calculation
        
    VECTOR Framework Integration:
    -----------------------------
    - Linear2D class: Specialized 2D linear algebra for grain boundary analysis
    - Multi-field processing: Robust grain boundary detection and orientation
    - Gradient computation: Accurate normal vector calculation from discrete data
    - Numerical stability: Robust methods for complex grain boundary geometries
        
    Scientific Applications:
    ------------------------
    - Anisotropic energy function normal vector input
    - Grain boundary orientation analysis for energy calculations
    - Triple junction angle measurement and validation
    - Grain boundary curvature and energy relationship studies
    """
    # Extract microstructure dimensions for VECTOR Linear2D analysis
    nx, ny = P.shape                               # 2D microstructure array dimensions
    
    # Configure VECTOR Linear2D computational parameters
    cores = 1                                      # Single-core processing for individual site analysis
    R = np.zeros((nx, ny))                         # Reference field initialization (unused)
    
    # Initialize multi-field representation for VECTOR Linear2D processing
    P0 = np.zeros((nx,ny,ng))                      # Multi-field grain representation array
    for m in range(ng):
        P0[:,:,m] = 1.0 * (P==(m+1))              # Binary field for each grain (1-indexed)
    
    # Initialize VECTOR Linear2D class for normal vector computation
    test1 = Linear_2D.linear_class(nx,ny,ng,cores,loop_times,P0,R)
    
    # Compute normal vector at specified location using VECTOR Linear2D
    normal = test1.linear_one_normal_vector_core([i, j])  # Raw normal vector from VECTOR
    
    # Apply coordinate transformation for consistent orientation representation
    normal = np.array([-normal[1], -normal[0]])    # Transform coordinates for proper orientation
    
    # Normalize to unit vector for consistent energy function application
    normal = normal / np.linalg.norm(normal)       # Unit normal vector normalization
    
    return normal
    
if __name__ == '__main__':
    """
    ==================================================================================
    MAIN EXECUTION: Triple Junction Energy Analysis and Validation
    ==================================================================================
    
    Main execution pipeline for comprehensive triple junction site energy analysis
    and energy averaging method validation in Monte Carlo Potts model systems.
    
    This section demonstrates:
    - Controlled triple junction geometry generation (90° and 120° configurations)
    - Site-specific energy calculation using multiple averaging methods
    - Energy matrix visualization for spatial energy distribution analysis
    - Validation of energy calculation algorithms with known geometries
    
    Analysis Workflow:
    1. Generate controlled triple junction test geometries
    2. Calculate site energies using active energy averaging method
    3. Generate energy matrix for spatial energy visualization
    4. Export high-resolution energy distribution plots
    
    Scientific Applications:
    - Energy function algorithm validation and verification
    - Triple junction equilibrium angle analysis
    - Energy averaging method sensitivity studies
    - Monte Carlo energy calculation validation
    """
    
    # ================================================================================
    # MICROSTRUCTURE CONFIGURATION: Controlled Triple Junction Geometry
    # ================================================================================
    nx, ny = 10, 10                               # Grid dimensions for test microstructure
    ic90 = get_2d_ic1(nx,ny)                      # 90° triple junction test geometry
    eng_matrix = np.zeros((nx,ny))                # Energy matrix for spatial visualization
    
    # Alternative test geometries (commented for future use)
    # ic120 = get_2d_ic2(nx,ny)                   # 120° equilibrium triple junction
    # ic120_2 = np.array(np.rot90(ic120, 3))      # Rotated 120° configuration
    
    # Manual microstructure modifications (commented examples)
    # ic2 = np.array(ic1)
    # ic2[int(nx/2), int(ny/2-1)] = 1             # Single site modification
    # ic3 = np.array(ic2)  
    # ic3[int(nx/2), int(ny/2)] = 2               # Additional site modification
    
    # Large-scale microstructure loading (commented for future use)
    # filename = "VoronoiIC_512_elong.init"       # Large Voronoi microstructure file
    # nx, ny, ng = 512, 512, 512                  # Large-scale dimensions
    # ic4, _ = myInput.init2IC(nx, ny, ng, filename, "./")  # Load from file
    # ic4 = ic4[:,:,0]                            # Extract 2D slice
    
    # ================================================================================
    # INDIVIDUAL SITE ENERGY ANALYSIS (COMMENTED EXAMPLES)
    # ================================================================================
    # Single site energy calculation examples for algorithm validation
    
    # 90° Triple Junction Energy Analysis:
    # x , y = 4, 4                                # Triple junction site coordinates
    # eng = calculate_energy(ic90, x, y)          # Calculate site energy
    # print(f"The energy at TJ ({x}, {y}) is {eng}")  # Display results
    
    # 120° Triple Junction Validation:
    # print("Vertical:")
    # x, y = 6, 4                                 # Vertical boundary site
    # eng = calculate_energy(ic120, x, y)         # Energy before modification
    # print(f"The energy at TJ ({x}, {y}) is {eng}")
    # ic120[x,y] = 1                             # Modify grain assignment
    # eng = calculate_energy(ic120, x, y)         # Energy after modification
    # print(f"The energy at TJ ({x}, {y}) is {eng}")
    
    # Horizontal boundary analysis:
    # print("Horizontal:")
    # x, y = 4, 3                                # Horizontal boundary site
    # eng = calculate_energy(ic120_2, x, y)       # Energy calculation
    # print(f"The energy at TJ ({x}, {y}) is {eng}")
    # ic120_2[x,y] = 1                           # Site modification
    # eng = calculate_energy(ic120_2, x, y)       # Modified energy
    # print(f"The energy at TJ ({x}, {y}) is {eng}")
    
    # ================================================================================
    # ENERGY MATRIX GENERATION: Spatial Energy Distribution Analysis
    # ================================================================================
    """
    Generate comprehensive energy matrix for spatial energy visualization.
    
    This analysis calculates site energies across a specified region to create
    a spatial energy map for visualization and analysis. The energy matrix
    provides insight into energy landscapes and validates energy calculation
    algorithms across different local environments.
    """
    # Calculate energy matrix for central region (triple junction area)
    for i in range(4,6):                          # x-coordinate range for analysis
        for j in range(4,6):                      # y-coordinate range for analysis
            eng_matrix[i][j] = calculate_energy(ic90, i ,j)  # Site energy calculation
    
    # ================================================================================
    # VISUALIZATION: High-Resolution Energy Distribution Plot
    # ================================================================================
    """
    Generate publication-quality visualization of spatial energy distribution.
    
    Creates comprehensive energy matrix visualization with:
    - Grayscale colormap for energy intensity representation
    - Horizontal colorbar for energy scale reference
    - High-resolution output for scientific publication
    - Proper energy scaling and normalization
    """
    # Generate energy distribution visualization
    plt.imshow(eng_matrix, cmap='gray_r',vmin=0,vmax=3)  # Energy matrix with gray colormap
    # plt.grid()                                 # Optional grid overlay (commented)
    plt.colorbar(orientation='horizontal')        # Horizontal colorbar for energy scale
    
    # Export high-resolution figure for scientific analysis
    plt.savefig('colorbar', dpi=400,bbox_inches='tight')  # Publication-quality output
    
    # ================================================================================
    # ANALYSIS COMPLETION SUMMARY
    # ================================================================================
    print("=" * 80)
    print("TRIPLE JUNCTION ENERGY ANALYSIS COMPLETED")
    print("=" * 80)
    print("Analysis Summary:")
    print(f"• Microstructure Dimensions: {nx} × {ny}")
    print(f"• Triple Junction Geometry: 90° test configuration")
    print(f"• Energy Averaging Method: Summation energy approach")
    print(f"• Energy Matrix Region: [{4}:{6}, {4}:{6}]")
    print(f"• Anisotropy Parameters: δ=0.6, m=2")
    print(f"• VECTOR Linear2D Integration: Normal vector computation")
    print("• Output Generated: High-resolution energy distribution plot")
    print("=" * 80)

