#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VECTOR Framework Input Generation and Utility Functions
======================================================

This module provides comprehensive initial condition generators and utility functions 
for the VECTOR grain boundary analysis framework. It includes microstructure generators, 
smoothing algorithms, boundary condition handlers, and mathematical utilities for 
grain boundary normal vector calculations.

Key Functionality:
- Initial condition generators (Voronoi, Circle, Complex geometries)
- File I/O for SPPARKS simulation data (2D and 3D)
- Bilinear smoothing matrix generation for numerical stability
- Periodic and reflective boundary condition utilities
- Domain decomposition for parallel processing
- Gradient calculation and normal vector utilities

Microstructure Types Supported:
- Voronoi tessellations: Realistic polycrystalline structures
- Circular grains: Analytical validation with known solutions
- Sinusoidal interfaces: Complex mathematical boundaries
- Abnormal grain structures: Specialized grain growth studies
- File-based initialization: Real experimental microstructures

Created on Tue Feb  9 10:32:12 2021
@author: lin.yang

Dependencies:
- numpy: Numerical computations and array operations
- matplotlib: Visualization capabilities
- math: Mathematical functions and constants
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Configuration parameters (commented for user customization)
# nx,ny = 200,200    # Grid dimensions for 2D simulations
# ng = 5             # Number of grains in microstructure
# PI = 3.1415926     # Mathematical constant

#%% File I/O Functions for SPPARKS Integration

def init2IC(nx,ny,ng,filename):
    """
    Load 2D microstructure from SPPARKS initialization file.
    
    This function reads SPPARKS .init format files and converts them into 
    phase field representation suitable for VECTOR framework analysis.
    The function handles the standard SPPARKS file format with grain IDs
    and applies necessary coordinate transformations.
    
    Parameters:
    -----------
    nx, ny : int
        Grid dimensions (x-width, y-height) for the computational domain
    ng : int
        Expected number of grains in the microstructure
    filename : str
        Name of the SPPARKS initialization file to read
        
    Returns:
    --------
    fig : ndarray
        3D array with shape (nx, ny, 1) containing grain ID map
        Values represent grain identifiers (1 to ng)
    R : ndarray
        Reference solution array with shape (nx, ny, 2) for normal vectors
        Currently initialized as zeros (for future analytical reference)
        
    File Format Expected:
    --------------------
    Line 1-3: Header information (skipped)
    Line 4+: site_id grain_id (space-separated)
    
    Coordinate Transformation:
    -------------------------
    - SPPARKS uses row-major ordering
    - Function applies np.flipud() for correct orientation
    - Reshapes 1D data into 2D grid structure
    
    Usage:
    ------
    Primarily used for loading experimental microstructures or 
    continuing analysis from previous SPPARKS simulations.
    
    Example:
    --------
    grain_map, reference = init2IC(200, 200, 50, "polycrystal.init")
    """
    # Set file path (modify as needed for different systems)
    filepath = "/Users/lin.yang/Documents/SPYDER/AllenCahnSmooth/Validation/"
    
    # Initialize reference solution array (zeros for file-based input)
    R = np.zeros((nx,ny,2))
    
    # Read SPPARKS initialization file
    with open(filepath+filename, 'r') as file:
        # Skip first 3 header lines
        beginNum = 3
        fig = []
        
        while beginNum >= 0:
            line = file.readline()
            beginNum -= 1
        
        # Verify first data line starts with site ID "1"
        if line[0] != '1':
            print("Please change beginning line! " + line)
            
        # Read grain ID data from each line
        while line:
            eachline = line.split()
            # Extract grain ID (second column) from each line
            fig.append([int(eachline[1])])
            line = file.readline()
    
    # Convert to numpy array and reshape to 2D grid
    fig = np.array(fig)
    fig = fig.reshape(nx,ny)
    
    # Apply coordinate transformation for proper orientation
    fig = np.flipud(fig)  # Flip vertically for SPPARKS coordinate convention
    fig = fig[:,:,None]   # Add third dimension for phase field compatibility

    return fig,R

def init2IC3d(nx,ny,nz,ng,filename,dream3d=False,filepath="/Users/lin.yang/Documents/SPYDER/AllenCahnSmooth/Validation/"):
    """
    Load 3D microstructure from SPPARKS initialization file.
    
    This function extends the 2D file loading capability to handle 3D microstructures
    from SPPARKS simulations. It supports both standard SPPARKS format and Dream3D
    coordinate conventions through the dream3d parameter.
    
    Parameters:
    -----------
    nx, ny, nz : int
        Grid dimensions (x-width, y-height, z-depth) for the 3D computational domain
    ng : int
        Expected number of grains in the microstructure
    filename : str
        Name of the SPPARKS 3D initialization file to read
    dream3d : bool, optional
        Flag for Dream3D coordinate convention (default: False)
        If True: maintains original coordinate system
        If False: applies SPPARKS coordinate transformation
    filepath : str, optional
        Directory path containing the initialization file
        
    Returns:
    --------
    fig : ndarray
        4D array with shape (nx, ny, nz, 1) containing 3D grain ID map
    R : ndarray
        3D reference solution array with shape (nx, ny, nz, 3) for normal vectors
        
    Coordinate Transformations:
    --------------------------
    - Reads 1D site data and reshapes to 3D grid (nz, nx, ny)
    - Transposes to proper VECTOR convention (nx, ny, nz)
    - Applies flipud() for SPPARKS convention (unless dream3d=True)
    
    File Format:
    -----------
    Similar to 2D format but with 3D site indexing:
    Line 1-3: Header information
    Line 4+: site_id grain_id (1-indexed sites)
    
    Memory Considerations:
    ---------------------
    For large 3D domains (e.g., 500³), this function requires significant
    memory for storing the complete grain ID array.
    
    Usage:
    ------
    Essential for 3D grain boundary analysis and validation of 3D algorithms
    against SPPARKS simulation results.
    """
    # Default file path can be overridden via parameter
    # filepath = "/Users/lin.yang/Documents/SPYDER/AllenCahnSmooth/Validation/"
    
    # Initialize 3D reference solution array 
    R = np.zeros((nx,ny,nz,3))
    
    # Read 3D SPPARKS initialization file
    with open(filepath+filename, 'r') as file:
        # Skip first 3 header lines
        beginNum = 3
        # Pre-allocate array for better memory efficiency
        fig = np.zeros((nx*ny*nz))
        
        while beginNum >= 0:
            line = file.readline()
            beginNum -= 1
        
        # Verify first data line format
        if line[0] != '1':
            print("Please change beginning line! " + line)
            
        # Read grain ID data - using direct indexing for efficiency
        while line:
            eachline = line.split()
            # SPPARKS uses 1-indexed sites, convert to 0-indexed
            site_index = int(eachline[0]) - 1
            grain_id = int(eachline[1])
            fig[site_index] = grain_id
            
            line = file.readline()
    
    # Reshape 1D array to 3D grid with proper dimensions
    # SPPARKS ordering: (z, x, y) -> reshape accordingly
    fig = fig.reshape(nz,nx,ny)
    
    # Transpose to VECTOR framework convention: (x, y, z)
    fig = fig.transpose((1,2,0))
    
    # Apply coordinate transformation based on data source
    if dream3d:
        # Dream3D data: maintain original coordinate system
        pass
    else:
        # SPPARKS data: apply vertical flip for correct orientation
        fig = np.flipud(fig)
        
    # Add fourth dimension for phase field compatibility
    fig = fig[:,:,:,None]
    
    return fig,R

# Commented out advanced 3D file loading function for Dream3D integration
# def init2IC3d_v2(nx,ny,nz,ng):
#     """Advanced Dream3D HDF5 file loading - implementation available separately"""
#     pass

#%% Analytical Initial Condition Generators

def Circle_IC(nx,ny):
    """
    Generate circular grain initial condition with analytical normal vector solution.
    
    This function creates a simple two-grain system with a circular interface,
    providing an analytical benchmark for validating normal vector calculation
    algorithms. The circular geometry allows exact mathematical comparison
    of computed vs theoretical normal vectors.
    
    Parameters:
    -----------
    nx, ny : int
        Grid dimensions defining the computational domain size
        
    Returns:
    --------
    P : ndarray
        Phase field array with shape (nx, ny, 2) containing:
        P[:,:,0] = 1 inside circle (grain 1), 0 elsewhere
        P[:,:,1] = 1 outside circle (grain 2), 0 elsewhere
    R : ndarray
        Analytical reference normal vectors with shape (nx, ny, 2)
        R[i,j,:] = unit normal vector pointing radially outward from circle center
        
    Geometry Configuration:
    ----------------------
    - Circle center: (nx/2, ny/2) - domain center
    - Circle radius: nx/4 - quarter of domain width
    - Inner grain (grain 1): circular region
    - Outer grain (grain 2): exterior region
    
    Analytical Solution:
    -------------------
    For any point (i,j) at distance r from center:
    - Normal vector direction: radial unit vector from center
    - R[i,j,0] = (j - ny/2) / r  (x-component)
    - R[i,j,1] = (i - nx/2) / r  (y-component)
    
    Validation Usage:
    ----------------
    This configuration provides exact analytical normal vectors for
    algorithm validation. Perfect algorithms should reproduce these
    values exactly (within numerical precision).
    
    Special Handling:
    ----------------
    - Avoids division by zero at circle center (r=0)
    - Both grains share same normal vector field (interface orientation)
    
    Example:
    --------
    phases, reference = Circle_IC(200, 200)
    # Creates 200x200 domain with circular grain of radius 50
    """
    # Fixed geometry: two-grain circular system
    ng = 2
    P = np.zeros((nx,ny,ng))    # Phase field array
    R = np.zeros((nx,ny,2))     # Reference normal vector array
    
    # Generate circular interface with analytical normal vectors
    for i in range(0,nx):
        for j in range(0,ny):
            # Calculate distance from domain center
            radius = math.sqrt((j-ny/2)**2+(i-nx/2)**2)
            
            if radius < nx/4:
                # Interior: grain 1 (circular grain)
                P[i,j,0] = 1.
                # Calculate analytical normal vector (avoid division by zero)
                if radius != 0:
                    R[i,j,0] = (j-ny/2)/radius    # x-component (normalized)
                    R[i,j,1] = (i-nx/2)/radius    # y-component (normalized)
            else:
                # Exterior: grain 2 (matrix grain)
                P[i,j,1] = 1.
                # Same normal vector calculation (interface orientation)
                if radius != 0:
                    R[i,j,0] = (j-ny/2)/radius
                    R[i,j,1] = (i-nx/2)/radius

    return P,R

def QuarterCircle_IC(nx,ny):
    """
    Generate quarter-circle grain configuration for specialized validation studies.
    
    Creates a two-grain system with a quarter-circular interface in one quadrant,
    useful for testing algorithm behavior with curved boundaries and domain edges.
    
    Parameters:
    -----------
    nx, ny : int
        Grid dimensions for computational domain
        
    Returns:
    --------
    P : ndarray
        Phase field array (nx, ny, 2) with quarter-circle geometry
    R : ndarray  
        Analytical reference normal vectors (nx, ny, 2)
        
    Geometry:
    ---------
    - Quarter-circle radius: 40 pixels (fixed)
    - Location: top-left quadrant (i < nx/2, j < ny/2)
    - Grain 1: interior of quarter-circle
    - Grain 2: remainder of domain
    """
    ng = 2
    P = np.zeros((nx,ny,ng))
    R = np.zeros((nx,ny,2))
    
    for i in range(0,nx):
        for j in range(0,ny):
            radius = math.sqrt((j-ny/2)**2+(i-nx/2)**2)
            # Quarter-circle condition: radius < 40 AND in top-left quadrant
            if radius < 40 and i < nx/2 and j < ny/2:
                P[i,j,0] = 1.
                if radius != 0:
                    R[i,j,0] = (j-ny/2)/radius
                    R[i,j,1] = (i-nx/2)/radius
            else:
                P[i,j,1] = 1.
                if radius != 0:
                    R[i,j,0] = (j-ny/2)/radius
                    R[i,j,1] = (i-nx/2)/radius

    return P,R

def Circle_IC3d(nx,ny,nz,r=25):
    """
    Generate 3D spherical grain initial condition with analytical normal vector solution.
    
    Extends the 2D circular geometry to 3D spherical geometry for validating
    3D normal vector calculation algorithms. Provides exact analytical reference
    for 3D algorithm benchmarking.
    
    Parameters:
    -----------
    nx, ny, nz : int
        Grid dimensions for 3D computational domain
    r : int, optional
        Sphere radius in grid units (default: 25)
        
    Returns:
    --------
    P : ndarray
        3D phase field array (nx, ny, nz, 2) with spherical geometry
    R : ndarray
        3D analytical reference normal vectors (nx, ny, nz, 3)
        
    Geometry Configuration:
    ----------------------
    - Sphere center: (nx/2, ny/2, nz/2) - domain center
    - Sphere radius: r (parameter-controlled)
    - Inner grain: spherical region
    - Outer grain: exterior region
    
    Analytical Solution:
    -------------------
    For point (i,j,k) at distance radius from center:
    - Normal vector: radial unit vector from sphere center
    - R[i,j,k,0] = (j - ny/2) / radius  (x-component)
    - R[i,j,k,1] = (i - nx/2) / radius  (y-component)  
    - R[i,j,k,2] = (k - nz/2) / radius  (z-component)
    
    Usage:
    ------
    Essential for 3D algorithm validation and testing computational
    efficiency on large 3D domains.
    """
    ng = 2
    P = np.zeros((nx,ny,nz,ng))    # 3D phase field array
    R = np.zeros((nx,ny,nz,3))     # 3D reference normal vectors
    
    # Generate spherical interface with analytical normal vectors
    for i in range(0,nx):
        for j in range(0,ny):
            for k in range(0,nz):
                # Calculate 3D distance from domain center
                radius = math.sqrt((j-ny/2)**2+(i-nx/2)**2+(k-nz/2)**2)
                
                if radius < r:  # Interior: spherical grain
                    P[i,j,k,0] = 1.
                    # Calculate 3D analytical normal vector
                    if radius != 0:
                        R[i,j,k,0] = (j-ny/2)/radius    # x-component
                        R[i,j,k,1] = (i-nx/2)/radius    # y-component
                        R[i,j,k,2] = (k-nz/2)/radius    # z-component
                else:  # Exterior: matrix grain
                    P[i,j,k,1] = 1.
                    # Same normal vector calculation (interface orientation)
                    if radius != 0:
                        R[i,j,k,0] = (j-ny/2)/radius
                        R[i,j,k,1] = (i-nx/2)/radius
                        R[i,j,k,2] = (k-nz/2)/radius

    return P,R

def Voronoi_IC(nx,ny,ng):
    """
    Generate Voronoi tessellation microstructure with pre-computed reference solutions.
    
    This function creates realistic polycrystalline microstructures using Voronoi
    tessellation with fixed grain centers. Pre-computed reference normal vectors
    are provided for specific domain sizes to enable algorithm validation.
    
    Parameters:
    -----------
    nx, ny : int
        Grid dimensions for computational domain
    ng : int
        Number of grains in the tessellation
        
    Returns:
    --------
    P : ndarray
        Phase field array (nx, ny, ng) with Voronoi grain structure
        P[i,j,k] = 1 if site (i,j) belongs to grain k, 0 otherwise
    R : ndarray
        Reference normal vectors (nx, ny, 2) for algorithm validation
        Contains analytical or high-accuracy numerical solutions
        
    Grain Center Configurations:
    ----------------------------
    Fixed grain centers are provided for specific domain sizes:
    - 200×200 domain: 5 grains with predetermined coordinates
    - 400×400 domain: Alternative configuration (commented)
    - 100×100 domain: Smaller scale configuration (commented)
    - 50×50 domain: Minimal configuration (commented)
    
    Voronoi Algorithm:
    -----------------
    For each grid point (i,j):
    1. Calculate distance to all grain centers
    2. Assign point to nearest grain center
    3. Create phase field representation
    
    Reference Solutions:
    -------------------
    Pre-computed reference normal vectors are provided for specific
    grain boundary segments using analytical geometry calculations.
    These enable quantitative validation of computed normal vectors.
    
    Coordinate System:
    -----------------
    - i-index: vertical (row) coordinate
    - j-index: horizontal (column) coordinate
    - Grain centers specified as [i, j] coordinates
    
    Usage:
    ------
    Primary microstructure generator for algorithm development and
    validation studies. Provides realistic grain structures with
    known reference solutions for accuracy assessment.
    
    Customization:
    -------------
    Users can modify grain center coordinates or add new domain
    size configurations by following the existing pattern.
    """
    # Initialize arrays
    P = np.zeros((nx,ny,ng))    # Phase field array for all grains
    R = np.zeros((nx,ny,2))     # Reference normal vector array
    
    # Fixed grain center coordinates for reproducible results
    # Configuration optimized for 200×200 domain with 5 grains
    
    # Alternative random generation (commented for reproducibility):
    # GCoords = np.zeros((ng,2))
    # for i in range(0,ng):
    #     GCoords[i,0],GCoords[i,1]= np.random.randint(nx),np.random.randint(ny)
    
    # Grain center coordinates for different domain sizes:
    
    # Configuration for 200×200 domain with 5 grains
    GCoords = np.array([[ 36., 132.],    # Grain 0 center
                        [116.,  64.],    # Grain 1 center  
                        [ 43.,  90.],    # Grain 2 center
                        [128., 175.],    # Grain 3 center
                        [194.,  60.]])   # Grain 4 center
    
    # Alternative configurations for other domain sizes (commented):
    # 400×400 domain configuration:
    # GCoords = np.array([[ 69., 321.],
    #                     [298., 134.],
    #                     [174., 138.],
    #                     [294., 392.],
    #                     [ 69., 324.]])
    
    # 100×100 domain configuration:
    # GCoords = np.array([[20., 95.],
    #                     [27., 61.],
    #                     [37., 93.],
    #                     [65., 18.],
    #                     [25., 17.]])
    
    # 50×50 domain configuration:
    # GCoords = np.array([[ 0., 35.],
    #                     [43., 36.],
    #                     [43.,  9.],
    #                     [38., 37.],
    #                     [28., 36.]])
    
    # Voronoi tessellation algorithm: assign each point to nearest grain center
    for i in range(0,nx):
        for j in range(0,ny):
            # Initialize with distance to first grain center
            MinDist = math.sqrt((GCoords[0,1]-j)**2+(GCoords[0,0]-i)**2)
            GG = 0  # Index of nearest grain
            
            # Check all other grain centers for closer distance
            for G in range(1,ng):
                dist = math.sqrt((GCoords[G,1]-j)**2+(GCoords[G,0]-i)**2) 
                if dist < MinDist:
                    GG = G          # Update nearest grain index
                    MinDist = dist  # Update minimum distance
                    
            # Assign this point to the nearest grain
            P[i,j,GG] = 1.
            
    # Pre-computed reference normal vectors for specific grain boundary segments
    # These provide analytical or high-accuracy numerical solutions for validation
    
    # Reference solution for 200×200 domain configuration
    for i in range(0,nx):
        for j in range(0,ny):
            # Grain boundary segment 1: linear interface with slope -10/1
            if i>0 and i<=93 and j<=120 and j>104:
                # Normal vector for linear boundary: perpendicular to slope
                R[i,j,0] = -10.0/math.sqrt(101)  # x-component  
                R[i,j,1] = 1.0/math.sqrt(101)    # y-component
                
            # Special corner cases: domain boundaries
            elif i==0 and j==ny-1:
                # Top-left corner: diagonal normal
                R[i,j,0] = -math.sqrt(0.5)
                R[i,j,1] = math.sqrt(0.5)
                
            # Domain edge cases: axis-aligned normals
            elif i==0:
                # Left edge: horizontal normal
                R[i,j,0] = 0
                R[i,j,1] = 1
            elif j==ny-1:
                # Top edge: vertical normal  
                R[i,j,0] = 1
                R[i,j,1] = 0
                
            # Grain boundary segment 2: different linear interface
            elif j>=123 and j<ny-1 and i<=96 and i>=60:
                # Normal vector for second linear boundary
                R[i,j,0] = -9.0/math.sqrt(442)
                R[i,j,1] = -19.0/math.sqrt(442)
                
            # Transition region: specific corner handling
            elif (i==94 or i==95 or i==96) and (j==121 or j==122):
                # Transition zone normal vector
                R[i,j,0] = -2.0/math.sqrt(5)
                R[i,j,1] = 1.0/math.sqrt(5)
    
    # Alternative reference solutions for other domain sizes (commented):
    # Contains similar analytical solutions for 400×400, 100×100, and 50×50 domains
    # Each provides specific grain boundary normal vectors for validation
    
    # Optional: Load pre-computed reference from file
    # R = np.load('npy/voronoi_R.npy')
    
    return P,R


def Voronoi_IC3d(nx,ny,nz,ng):
    
    
    return 0


# Sin(x) IC
def Complex2G_IC(nx,ny,wavelength=20):
# =============================================================================
#     output the sin(x) initial condition, 
#     nx is the sites in x coordination, 
#     ny is the site in y coordination, 
#     ng is the grain number
# =============================================================================
    
    ng = 2
    P = np.zeros((nx,ny,ng))
    R = np.zeros((nx,ny,2))
    A = wavelength/2
     
    for i in range(0,nx):
        # slope
        slope = -1.0/(10*math.pi/A*math.cos(math.pi/A*(i+A/2)))
        length = math.sqrt(1 + slope**2)
        
        for j in range(0,ny):
            if j < ny/2 + 10*math.sin((i+A/2)*3.1415926/A):
                P[i,j,0] = 1.
                R[i,j,0] = slope/length
                R[i,j,1] = 1.0/length
            else:
                P[i,j,1] = 1.
                R[i,j,0] = slope/length
                R[i,j,1] = 1.0/length
    
    for i in range(0,nx):
        for j in range(0,ny):
            # if (i==0 and j==0) or (i==nx-1 and j==ny-1):
            #     R[i,j,0]=math.sqrt(0.5)
            #     R[i,j,1]=math.sqrt(0.5)
            # elif (i==0 and j==ny-1) or (i==nx-1 and j==0):
            #     R[i,j,0]=-math.sqrt(0.5)
            #     R[i,j,1]=math.sqrt(0.5)
            if j==0 or j==nx-1:
                R[i,j,0]=1
                R[i,j,1]=0

    return P,R

def Complex2G_IC3d(nx,ny,nz,wavelength=20):
# =============================================================================
#     output the circle initial condition, 
#     nx is the sites in x coordination, 
#     ny is the site in y coordination, 
#     nz is the site in z coordination,
#     ng is the grain number
# =============================================================================
    ng = 2
    P = np.zeros((nx,ny,nz,ng))
    R = np.zeros((nx,ny,nz,3))
    A = wavelength/2
    
    for i in range(0,nx):
        for j in range(0,ny):
            vector_i = np.array([1, 0, 5*math.pi/(2*A)*math.cos(math.pi/A*(0.5*i+A/2))])
            length_i = math.sqrt(1 + (5*math.pi/(2*A)*math.cos(math.pi/A*(0.5*i+A/2)))**2)
            vector_i = vector_i/length_i
            vector_j = np.array([0, 1, 5*math.pi/(2*A)*math.cos(math.pi/A*(0.5*j+A/2))])
            length_j = math.sqrt(1 + (5*math.pi/(2*A)*math.cos(math.pi/A*(0.5*j+A/2)))**2)
            vector_j = vector_j/length_j
            
            
            for k in range(0,nz):
                if k < nz/2 + 5*math.sin((0.5*i+A/2)*3.1415926/A) + 5*math.sin((0.5*j+A/2)*3.1415926/A):
                    P[i,j,k,0] = 1.
                    R[i,j,k,:] = np.cross(vector_i,vector_j)
                    tmp_r = R[i,j,k,:]/np.linalg.norm(R[i,j,k,:])
                    R[i,j,k,:]=[tmp_r[1],tmp_r[0],tmp_r[2]]
                    # print(f"i={i} j={j} k={k}")
                    # print(R[i,j,k,:])
                else:
                    P[i,j,k,1] = 1. 
                    R[i,j,k,:] = -np.cross(vector_i,vector_j)
                    tmp_r = R[i,j,k,:]/np.linalg.norm(R[i,j,k,:])
                    R[i,j,k,:]=[tmp_r[1],tmp_r[0],tmp_r[2]]
                    
                    
            
    for k in [0,nz-1]:
        for i in range(0,nx):
            for j in range(0,ny):
                R[i,j,k,2] = 2.*((k>nz/2)*1-0.5)
    
    return P, R
        




# A real abnormal grain growth
def Abnormal_IC(nx,ny):
    
    ng = 2
    P = np.zeros((nx,ny,ng))
    R = np.zeros((nx,ny,2))

    file=open(f"input/AG{nx}x{ny}.txt")
    lines=file.readlines()
     
    row=0
    for line in lines:
        # line = " ".join(line)
        line=line.strip().split() 
        for i in range(0,len(line)):
            P[row,i,0]=float(line[i])
            P[row,i,1]=1-float(line[i])
        row+=1

    # abnormal possible true value
    if nx==200:
        R1 = np.load('npy/ACabnormal20_R.npy')
        R2 = np.load('npy/BLabnormal04_R.npy')
        R3 = np.load('npy/LSabnormal01_R.npy')
        R4 = np.load('npy/VTabnormal03_R.npy')
        
        m=0
        for i in range(0,nx):
            for j in range(0,ny):
                if R4[i,j,1]*R1[i,j,1]+R4[i,j,0]*R1[i,j,0] < -0.7:
                    R4[i,j,0] = -R4[i,j,0]
                    R4[i,j,1] = -R4[i,j,0]
                    m+=1
                    # print("i = " + str(i) + " j = " + str(j) + " m = " + str(m))
        
        for i in range(0,nx):
            for j in range(0,ny):
                R[i,j,0] = (R1[i,j,0] + R2[i,j,0] + R3[i,j,0] + R4[i,j,0])/4
                R[i,j,1] = (R1[i,j,1] + R2[i,j,1] + R3[i,j,1] + R4[i,j,1])/4
                length = math.sqrt(R[i,j,0]**2+R[i,j,1]**2)
                if length ==0:
                    R[i,j,0] = 0
                    R[i,j,1] = 0
                else:
                    R[i,j,0] = R[i,j,0]/length
                    R[i,j,1] = R[i,j,1]/length
            
    
    return P,R
    
    
    
    # data = np.loadtxt('AG200x200.txt', dtype=np.float32, delimiter=' ')
    
    
def SmallestGrain_IC(nx,ny):
     ng = 2
     P = np.zeros((nx,ny,ng))
     
     for i in range(0,nx):
         
         for j in range(0,ny):
             
             if i==25 and j==10:
                 P[i,j,0] = 1
             elif i>=50 and i<=90 and j==10:
                 P[i,j,0] = 1
                 
             elif i>=24 and i<=25 and j>=25 and j<=26:
                 P[i,j,0] = 1
             elif i>=50 and i<=90 and j>=25 and j<=26:
                 P[i,j,0] = 1
                 
             elif i>=23 and i<=25 and j>=40 and j<=42:
                 P[i,j,0] = 1
             elif i>=50 and i<=90 and j>=40 and j<=42:
                 P[i,j,0] = 1
                 
             elif i>=22 and i<=25 and j>=60 and j<=63:
                 P[i,j,0] = 1
             elif i>=50 and i<=90 and j>=60 and j<=63:
                 P[i,j,0] = 1
                 
             elif i>=21 and i<=25 and j>=83 and j<=87:
                 P[i,j,0] = 1
             elif i>=50 and i<=90 and j>=83 and j<=87:
                 P[i,j,0] = 1
                 
             else:
                 P[i,j,1] = 1
                 
     return P


#%% Core Bilinear Smoothing Algorithm Functions

def output_linear_smoothing_matrix(iteration):
    """
    Generate bilinear smoothing kernel matrix for numerical gradient calculations.
    
    This function creates a smoothing kernel that reduces numerical noise in 
    discrete microstructure data while preserving interface geometry. The kernel
    is based on repeated application of a 3×3 bilinear filter.
    
    Parameters:
    -----------
    iteration : int
        Number of smoothing iterations to apply
        Higher values = more smoothing, larger kernel size
        
    Returns:
    --------
    ndarray : 2D smoothing kernel matrix
        Shape: (2*iteration+1, 2*iteration+1)
        Normalized weights for convolution-based smoothing
        
    Algorithm:
    ----------
    1. Start with basic 3×3 bilinear kernel:
       [[1/16, 1/8, 1/16],
        [1/8,  1/4, 1/8 ],
        [1/16, 1/8, 1/16]]
    
    2. Iteratively expand kernel by convolving with itself
    3. Each iteration increases kernel size by 2 in each direction
    4. Final kernel size: (2*iteration+1) × (2*iteration+1)
    
    Mathematical Foundation:
    -----------------------
    The bilinear kernel approximates Gaussian smoothing with discrete weights.
    Repeated convolution approaches continuous Gaussian filter in the limit.
    
    Usage:
    ------
    Essential component for bilinear smoothing algorithm, used to reduce
    numerical artifacts in gradient calculations at grain boundaries.
    
    Example:
    --------
    kernel = output_linear_smoothing_matrix(2)  # Creates 5×5 smoothing kernel
    """
    # Calculate expanded matrix size to accommodate all iterations
    matrix_length = 2*iteration+3
    
    # Initialize storage for all iteration levels
    matrix = np.zeros((iteration, matrix_length, matrix_length))
    
    # Define basic 3×3 bilinear smoothing kernel
    matrix_unit = np.array([[1/16, 1/8, 1/16], 
                           [1/8,  1/4, 1/8 ], 
                           [1/16, 1/8, 1/16]])
    
    # Set the finest level (highest iteration index) with basic kernel
    matrix[iteration-1, iteration:iteration+3, iteration:iteration+3] = matrix_unit
    
    # Iteratively build coarser levels by convolution
    for i in range(iteration-2, -1, -1):  # Work backwards from fine to coarse
        for j in range(i+1, matrix_length-i-1):
            for k in range(i+1, matrix_length-i-1):
                # Convolve previous level with basic kernel to expand smoothing
                matrix[i,j,k] += np.sum(matrix[i+1,j-1:j+2,k-1:k+2] * matrix_unit)
    
    # Return the coarsest level (full smoothing kernel) with boundary trim
    return matrix[0, 1:-1, 1:-1]

def output_linear_smoothing_matrix3D(iteration):
# =============================================================================
#     The function will output the bottom matrix,
#     the matrix can be use to calculate smoothing status.
#     bottom matrix length is 2*iteration+1
# =============================================================================
    
    matrix_length = 2*iteration+3
    sa, sb, sc, sd = 1/8, 1/16, 1/32, 1/64
    matrix = np.zeros((iteration, matrix_length, matrix_length, matrix_length))
    matrix_unit = np.array([[[sd,sc,sd],[sc,sb,sc],[sd,sc,sd]],
                            [[sc,sb,sc],[sb,sa,sb],[sc,sb,sc]],
                            [[sd,sc,sd],[sc,sb,sc],[sd,sc,sd]]])
    matrix[iteration-1,iteration:iteration+3,iteration:iteration+3,iteration:iteration+3] = matrix_unit
    
    for i in range(iteration-2,-1,-1):
        for j in range(i+1, matrix_length-i-1):
            for k in range(i+1, matrix_length-i-1):
                for p in range(i+1, matrix_length-i-1):
                    matrix[i,j,k,p] += np.sum(matrix[i+1,j-1:j+2,k-1:k+2,p-1:p+2] * matrix_unit)
    
    
    
    return matrix[0,1:-1,1:-1,1:-1]

def output_linear_vector_matrix(iteration,clip=0):
    """
    Generate gradient calculation matrices for bilinear normal vector computation.
    
    This function creates specialized convolution kernels for calculating 
    spatial gradients from smoothed microstructure data. The gradients are
    used to determine grain boundary normal vectors.
    
    Parameters:
    -----------
    iteration : int
        Smoothing kernel size parameter (from output_linear_smoothing_matrix)
    clip : int, optional
        Boundary clipping parameter to reduce kernel size (default: 0)
        
    Returns:
    --------
    matrix_i : ndarray
        Gradient kernel for i-direction (vertical) gradients
    matrix_j : ndarray  
        Gradient kernel for j-direction (horizontal) gradients
        
    Algorithm:
    ----------
    1. Generate base smoothing matrix using output_linear_smoothing_matrix()
    2. Create gradient operators by differencing smoothed values:
       - Forward difference: matrix[i+1] - matrix[i]
       - Backward difference: matrix[i] - matrix[i-1]
    3. Apply spatial offsets to create finite difference operators
    4. Clip boundaries if specified to reduce computational cost
    
    Gradient Calculation Method:
    ---------------------------
    For 2D gradient calculation:
    - ∂f/∂i ≈ [smoothed(i+1,j) - smoothed(i-1,j)] / 2
    - ∂f/∂j ≈ [smoothed(i,j+1) - smoothed(i,j-1)] / 2
    
    The matrices implement these operations as convolution kernels.
    
    Matrix Structure:
    ----------------
    - matrix_i: calculates vertical (i-direction) gradients
    - matrix_j: calculates horizontal (j-direction) gradients
    - Both matrices have same size as smoothing kernel
    
    Boundary Handling:
    -----------------
    The clip parameter allows reduction of kernel size near boundaries
    to handle edge effects in finite domains.
    
    Usage:
    ------
    Essential for converting smoothed phase field data into normal
    vector components at grain boundaries.
    
    Example:
    --------
    grad_i, grad_j = output_linear_vector_matrix(3, clip=1)
    # Creates gradient kernels for 3-iteration smoothing with 1-pixel clipping
    """
    # Calculate matrix dimensions
    matrix_length = 2*iteration+3
    
    # Initialize gradient matrices
    matrix_j = np.zeros((matrix_length, matrix_length))  # Horizontal gradients
    matrix_i = np.zeros((matrix_length, matrix_length))  # Vertical gradients
    
    # Get base smoothing matrix for gradient calculations
    smoothing_matrix = output_linear_smoothing_matrix(iteration)
    
    # Create horizontal gradient operator (j-direction)
    # Forward difference: smoothing_matrix shifted right
    matrix_j[1:-1, 2:] = smoothing_matrix
    # Backward difference: smoothing_matrix shifted left (subtracted)
    matrix_j[1:-1, 0:-2] += -smoothing_matrix
    
    # Create vertical gradient operator (i-direction)  
    # Forward difference: smoothing_matrix shifted down
    matrix_i[2:, 1:-1] = smoothing_matrix
    # Backward difference: smoothing_matrix shifted up (subtracted)
    matrix_i[0:-2, 1:-1] += -smoothing_matrix
    
    # Apply boundary clipping if specified
    matrix_i = matrix_i[clip:matrix_length-clip, clip:matrix_length-clip]
    matrix_j = matrix_j[clip:matrix_length-clip, clip:matrix_length-clip]
    
    return matrix_i, matrix_j

def output_linear_vector_matrix3D(iteration, clip=0):
# =============================================================================
#     The function will output the bottom matrix,
#     the matrix can be use to calculate vector from smoothing status.
#     bottom matrix length is 2*iteration+1
#     00 im 00
# km  jm CT jp   kp
#     00 ip 00
# =============================================================================
    
    matrix_length = 2*iteration+3
    matrix_j = np.zeros((matrix_length, matrix_length, matrix_length))
    matrix_i = np.zeros((matrix_length, matrix_length, matrix_length))
    matrix_k = np.zeros((matrix_length, matrix_length, matrix_length))
    smoothing_matrix = output_linear_smoothing_matrix3D(iteration)
    matrix_j[1:-1,2:,1:-1] = smoothing_matrix
    matrix_j[1:-1,0:-2,1:-1] += -smoothing_matrix
    matrix_i[2:,1:-1,1:-1] = smoothing_matrix
    matrix_i[0:-2,1:-1,1:-1] += -smoothing_matrix
    matrix_k[1:-1,1:-1,2:] = smoothing_matrix
    matrix_k[1:-1,1:-1,0:-2] += -smoothing_matrix
    matrix_i = matrix_i[clip:matrix_length-clip, clip:matrix_length-clip, clip:matrix_length-clip]
    matrix_j = matrix_j[clip:matrix_length-clip, clip:matrix_length-clip, clip:matrix_length-clip]
    matrix_k = matrix_k[clip:matrix_length-clip, clip:matrix_length-clip, clip:matrix_length-clip]
    
    return matrix_i, matrix_j, matrix_k

def output_smoothed_matrix(simple_test3,linear_smoothing_matrix):
# =============================================================================
#     output the smoothed final matrix
# =============================================================================
    edge = int(np.floor(np.shape(linear_smoothing_matrix)[0]/2))
    ilen,jlen = np.shape(simple_test3)
    smoothed_matrix3 = np.zeros((ilen,jlen))
    for i in range(edge,ilen-edge):
        for j in range(edge, jlen-edge):
            smoothed_matrix3[i,j] = np.sum(simple_test3[i-edge:i+edge+1,j-edge:j+edge+1]*linear_smoothing_matrix)
    
    return smoothed_matrix3


#%% Boundary Condition and Coordinate Utilities

def periodic_bc(nx,ny,i,j):
    """
    Apply periodic boundary conditions for 2D grid coordinates.
    
    This function calculates neighbor coordinates with periodic wrapping,
    essential for grain boundary analysis in domains with periodic boundaries.
    Converts edge/corner sites to have valid neighbors through domain wrapping.
    
    Parameters:
    -----------
    nx, ny : int
        Grid dimensions (x-width, y-height)
    i, j : int
        Current grid coordinates to find neighbors for
        
    Returns:
    --------
    ip, im : int
        Forward and backward neighbors in i-direction (vertical)
    jp, jm : int
        Forward and backward neighbors in j-direction (horizontal)
        
    Periodic Wrapping Rules:
    -----------------------
    - i+1 beyond nx-1 wraps to 0 (right edge → left edge)
    - i-1 below 0 wraps to nx-1 (left edge → right edge)  
    - j+1 beyond ny-1 wraps to 0 (top edge → bottom edge)
    - j-1 below 0 wraps to ny-1 (bottom edge → top edge)
    
    Usage:
    ------
    Essential for grain boundary detection and gradient calculations
    in domains with periodic boundary conditions. Ensures all sites
    have valid neighbors for finite difference operations.
    
    Example:
    --------
    ip, im, jp, jm = periodic_bc(200, 200, 0, 50)
    # For site at left edge (i=0), im wraps to 199
    """
    # Calculate forward neighbors
    ip = i + 1  # Next site in i-direction
    jp = j + 1  # Next site in j-direction
    
    # Calculate backward neighbors  
    im = i - 1  # Previous site in i-direction
    jm = j - 1  # Previous site in j-direction
    
    # Apply periodic boundary conditions
    if ip > nx - 1:
        ip = 0          # Right edge wraps to left
    if im < 0:
        im = nx - 1     # Left edge wraps to right
    if jp > ny - 1:
        jp = 0          # Top edge wraps to bottom
    if jm < 0:
        jm = ny - 1     # Bottom edge wraps to top
        
    return ip,im,jp,jm

def periodic_bc3d(nx,ny,nz,i,j,k):
    ip = i + 1
    im = i - 1
    jp = j + 1
    jm = j - 1
    kp = k + 1
    km = k - 1
    #Periodic BC
    if ip > nx - 1:
        ip = 0
    if im < 0:
        im = nx - 1
    if jp > ny - 1:
        jp = 0
    if jm < 0:
        jm = ny - 1
    if kp > nz - 1:
        kp = 0
    if km < 0:
        km = nz - 1
    return ip,im,jp,jm,kp,km

def repeat_bc3d(nx,ny,nz,i,j,k):
    ip = i + 1
    im = i - 1
    jp = j + 1
    jm = j - 1
    kp = k + 1
    km = k - 1
    #Periodic BC
    if ip > nx - 1:
        ip = nx
    if im < 0:
        im = 0
    if jp > ny - 1:
        jp = ny
    if jm < 0:
        jm = 0
    if kp > nz - 1:
        kp = nz
    if km < 0:
        km = 0
    return ip,im,jp,jm,kp,km

def filter_bc3d(nx,ny,nz,i,j,k,length):
    if i-length < 0:
        return False
    if i+length > nx-1:
        return False
    if j-length < 0:
        return False
    if j+length > ny-1:
        return False
    if k-length < 0:
        return False
    if k+length > nz-1:
        return False
    return True

def get_grad(P,i,j):
    """
    Calculate normalized gradient vector (normal vector) from phase field data.
    
    This function extracts the computed gradient components from the phase field
    array and converts them into normalized normal vectors for grain boundary
    analysis. The gradients represent interface orientations.
    
    Parameters:
    -----------
    P : ndarray
        Phase field array with shape (nx, ny, 3) where:
        P[0,:,:] = grain ID map
        P[1,:,:] = gradient x-component  
        P[2,:,:] = gradient y-component
    i, j : int
        Grid coordinates where to extract normal vector
        
    Returns:
    --------
    VecX, VecY : float
        Normalized normal vector components at site (i,j)
        Magnitude normalized to unity for direction analysis
        
    Algorithm:
    ----------
    1. Extract gradient components from phase field array
    2. Apply sign conventions: VecX = -H*DX, VecY = -H*DY
    3. Calculate vector magnitude: |Vec| = sqrt(VecX² + VecY²)
    4. Normalize to unit vector: Vec_norm = Vec / |Vec|
    5. Handle zero-gradient case (no interface) with unit scaling
    
    Mathematical Foundation:
    -----------------------
    The gradient of a phase field φ gives interface normal direction:
    ∇φ = (∂φ/∂x, ∂φ/∂y) points from low to high phase values
    
    Sign Convention:
    ---------------
    H = 1.0 (interface width parameter)
    Negative signs ensure proper normal vector orientation
    
    Special Cases:
    -------------
    - Zero gradient (VecLen = 0): returns original coordinates with unit scaling
    - Used for sites in grain interiors with no interface
    
    Usage:
    ------
    Primary function for extracting normal vectors after bilinear smoothing
    computation. Essential for grain boundary characterization and validation.
    
    Example:
    --------
    nx, ny = get_grad(phase_field, 100, 150)
    # Returns normalized normal vector at grid point (100, 150)
    """
    # Extract gradient components from phase field array
    DX = P[2,i,j]  # Gradient x-component (stored in P[2,:,:])
    DY = P[1,i,j]  # Gradient y-component (stored in P[1,:,:])
    
    # Interface width parameter (typically 1.0 for normalized calculations)
    H = 1.
    
    # Apply sign convention for proper normal vector orientation
    VecX = -H*DX   # x-component with negative sign
    VecY = -H*DY   # y-component with negative sign
    
    # Calculate vector magnitude for normalization
    VecLen = math.sqrt(VecX**2+VecY**2)
    
    # Determine normalization factor
    if VecLen == 0:
        # No gradient (grain interior): use unit scaling
        VecScale = 1
    else:
        # Interface site: normalize to unit magnitude
        VecScale = H/VecLen
    
    # Return normalized normal vector components
    # Note: VecY has additional negative sign for coordinate system consistency
    return VecScale*VecX, -VecScale*VecY

def get_grad3d(P,i,j,k):
    DX = P[2,i,j,k]
    DY = P[1,i,j,k]
    DZ = P[3,i,j,k]
    H = 1.0
    VecX = -H*DX
    VecY = -H*DY
    VecZ = -H*DZ
    VecLen = math.sqrt(VecX**2+VecY**2+VecZ**2)
    if VecLen == 0:
        VecScale = 1
    else:
        VecScale = H/VecLen
    return VecScale*VecX,-VecScale*VecY,VecScale*VecZ

def split_cores(cores, sc_d = 2):
    """
    Decompose number of CPU cores into optimal grid dimensions for parallel processing.
    
    This function factorizes the number of available CPU cores into rectangular
    grid dimensions for efficient domain decomposition in parallel algorithms.
    The decomposition minimizes communication overhead between subdomains.
    
    Parameters:
    -----------
    cores : int
        Total number of CPU cores available (must be power of 2)
    sc_d : int, optional
        Spatial dimensions for decomposition (default: 2)
        2 = 2D decomposition (length × width)
        3 = 3D decomposition (length × width × height)
        
    Returns:
    --------
    For sc_d = 2:
        sc_length, sc_width : int
            Grid dimensions for 2D domain decomposition
    For sc_d = 3:
        sc_length, sc_width, sc_height : int
            Grid dimensions for 3D domain decomposition
            
    Algorithm:
    ----------
    1. Calculate total power of 2: cores = 2^sc_p
    2. Distribute powers across dimensions as evenly as possible
    3. For 2D: length gets ceiling(sc_p/2), width gets floor(sc_p/2)
    4. For 3D: distribute sc_p across three dimensions optimally
    
    Load Balancing Strategy:
    -----------------------
    The function aims to create nearly square (2D) or cubic (3D) subdomains
    to minimize surface-to-volume ratio and reduce inter-process communication.
    
    Examples:
    ---------
    split_cores(8, 2) → returns (4, 2)   # 4×2 grid for 8 cores
    split_cores(16, 2) → returns (4, 4)  # 4×4 grid for 16 cores  
    split_cores(8, 3) → returns (2, 2, 2) # 2×2×2 grid for 8 cores
    
    Constraints:
    -----------
    - cores must be power of 2 for optimal decomposition
    - Non-power-of-2 values may lead to uneven load balancing
    
    Usage:
    ------
    Essential for multiprocessing setup in bilinear smoothing algorithm.
    Determines how computational domain is split across CPU cores.
    """
    # Calculate total power of 2
    sc_p = 0
    temp_cores = cores
    while temp_cores != 1:
        temp_cores = temp_cores/2
        sc_p += 1
        
    # Distribute powers across spatial dimensions
    if sc_d == 2:
        # 2D decomposition: length × width
        sc_length = 2**(math.ceil(sc_p/sc_d))    # Larger dimension
        sc_width = 2**(math.floor(sc_p/sc_d))    # Smaller dimension
        return sc_length, sc_width
        
    elif sc_d == 3:
        # 3D decomposition: length × width × height
        sc_length = 2**(math.ceil(sc_p/sc_d))
        sc_width = 2**(math.floor(sc_p/sc_d))
        # Calculate remaining dimension
        sc_height = int(2**sc_p/(sc_length*sc_width))
        return sc_length, sc_width, sc_height


def split_IC(split_V,cores,dimentions=2, sic_nx_order = 1, sic_ny_order = 2, sic_nz_order = 3):
    """ Split a large matrix into several small matrix based on cores num"""
    if dimentions==2:
        sic_lc, sic_wc = split_cores(cores)
    elif dimentions==3:
        sic_lc,sic_wc,sic_hc = split_cores(cores,dimentions)
    # sic_width  = nx/sic_wc
    # sic_length  = ny /sic_lc
    
    new_arrayin = np.array_split(split_V, sic_wc, axis = sic_nx_order)
    new_arrayout = []
    for arrayi in new_arrayin:
        arrayi = np.array_split(arrayi, sic_lc, axis = sic_ny_order)
        if dimentions==3:
            new_array3 = []
            for arrayj in arrayi:
                arrayj = np.array_split(arrayj, sic_hc, axis = sic_nz_order)
                new_array3.append(arrayj)
            new_arrayout.append(new_array3)
        else:
            new_arrayout.append(arrayi)
    
    return new_arrayout



#%% test
# nx=200
# ny=200

# for g in range(0,ng):
#         for i in range(0,nx):
#         # slope
#             slope = -1.0/(math.pi*math.cos(math.pi/10*(i+5)))
#             length = math.sqrt(1 + slope**2)
#             for j in range(0,ny):
#                 ip,im,jp,jm = periodic_bc(i,j)
#                 if ( ((P[ip,j,g]-P[i,j,g])!=0) or ((P[im,j,g]-P[0,i,j,g])!=0) or ((P[i,jp,g]-P[i,j,g])!=0) or ((P[i,jm,g]-P[i,j,g])!=0) )\
#                        and P[i,j,g]==1:
#                     R[i,j,0] = slope/length
#                     R[i,j,1] = 1.0/length



# P0,R=Complex2G_IC(200,200,100)
# P0,R=Circle_IC(200,200)
# P0,R=Voronoi_IC(100,100,5)
# P0,T=Abnormal_IC(200,200)
# P0=SmallestGrain_IC(100,100)

# fig1 = plt.figure(2)
# plt.imshow(P0[:,:,0], cmap='nipy_spectral', interpolation='nearest')

# i=40
# j=100
# plt.arrow(j,i,22*R[i,j,0],22*R[i,j,1],color='blue')

# arrows = []
# for i in range(0,200,1):
#     for j in range(0,200,1):
#         if j==100:
#             Dx = R[i,j,0]
#             Dy = R[i,j,1]
#             plt.arrow(j,i,12*Dx,12*Dy,color='white')

# fig,R=init2IC3d(201,201,43,831,"s1400poly1_t0.init")
# fig2,R2=init2IC3d_v2(201,201,43,831)
# fig2 = np.flipud(fig2)
# fig2,R2 = init2IC(2, 4, 8, "Simple3D.init")
# spparks's order:
# z=0
# [3,4]
# [1,2]
# z=1
# [7,8]
# [5,6]

# plt.imshow(fig2[:,:,21,0], cmap='nipy_spectral', interpolation='nearest')


# test for complex_3D
# P,R = Complex2G_IC3d(100,100,100)

# fig = plt.figure(1)
# ax3 = plt.axes(projection='3d')
# ax3.set_xlim3d(0, 100)
# ax3.set_ylim3d(0, 100)
# ax3.set_zlim3d(0, 100)

# xx = np.arange(0,100,0.5)
# yy = np.arange(0,100,0.5)
# X, Y = np.meshgrid(xx, yy)
# Z = 50 + 5*np.sin((0.5*X+5)*0.31415926) + 5*np.sin((0.5*Y+5)*0.31415926)
# ax3.plot_surface(X,Y,Z,cmap='rainbow')

# # for i in range(1,100,5):
# #     for j in range(1,100,5):
# #         for k in range(20,80):
# #             ip,im,jp,jm,kp,km = periodic_bc3d(100,100,100,i,j,k)
# #             if ( ((P[ip,j,k,0]-P[i,j,k,0])!=0) or ((P[im,j,k,0]-P[i,j,k,0])!=0) or\
# #                   ((P[i,jp,k,0]-P[i,j,k,0])!=0) or ((P[i,jm,k,0]-P[i,j,k,0])!=0) or\
# #                   ((P[i,j,kp,0]-P[i,j,k,0])!=0) or ((P[i,j,km,0]-P[i,j,k,0])!=0) ) and\
# #                     P[i,j,k,0]==1:
# #                     ax3.quiver(i,j,k, (R[i,j,k][0]),(R[i,j,k][1]),  (R[i,j,k][2]), length = 20, normalize = True)

# plt.show()


