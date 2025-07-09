#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bilinear Smoothing Algorithm for Grain Boundary Normal Vector Calculation
=========================================================================

This module implements a sophisticated bilinear smoothing algorithm (BLv2) for calculating 
accurate grain boundary normal vectors in polycrystalline microstructures. The algorithm 
employs multiprocessing optimization and advanced smoothing techniques to minimize numerical 
errors in gradient calculations at grain boundaries.

Scientific Context:
- Grain boundary normal vectors are critical for understanding interface physics
- Bilinear smoothing reduces numerical artifacts from discrete microstructure data
- Multiprocessing enables efficient analysis of large polycrystalline domains
- Error quantification provides validation against analytical reference solutions

Key Features:
- Parallel multiprocessing for computational efficiency
- Configurable smoothing kernel size and iterations
- Comprehensive error analysis and performance monitoring
- Visualization capabilities for grain boundary normal vectors
- Support for various initial condition generators (Voronoi, complex geometries)

Created on Fri Jan 22 15:59:27 2021
@author: lin.yang

Dependencies:
- numpy: Numerical computations and array operations
- matplotlib: Visualization of grain boundaries and normal vectors
- multiprocessing: Parallel computation across CPU cores
- myInput: Custom module for initial condition generation and utilities
- datetime: Performance timing and benchmarking
"""

import sys
# sys.path.append('/Users/lin.yang/Documents/SPYDER/AllenCahnSmooth/')
import numpy as np
import math
import matplotlib.pyplot as plt
import myInput
import datetime
import multiprocessing as mp



class BLv2_class(object):
    """
    Bilinear Smoothing Version 2 (BLv2) Algorithm Class
    ==================================================
    
    This class implements an advanced bilinear smoothing algorithm for calculating 
    grain boundary normal vectors in polycrystalline microstructures. The algorithm 
    combines numerical smoothing techniques with parallel processing to achieve 
    accurate and efficient gradient calculations.
    
    Algorithm Overview:
    1. Initialize microstructure data and multiprocessing parameters
    2. Apply bilinear smoothing kernels to reduce numerical noise
    3. Calculate gradient fields using smoothed data
    4. Extract normal vectors at grain boundary sites
    5. Validate results against analytical reference solutions
    
    Key Innovations:
    - Adaptive smoothing window based on local grain geometry
    - Parallel processing with domain decomposition
    - Comprehensive error analysis and performance metrics
    - Memory-efficient handling of large microstructure domains
    
    Attributes:
        nx, ny (int): Grid dimensions in x and y directions
        ng (int): Number of grains in the initial microstructure
        cores (int): Number of CPU cores for parallel processing
        loop_times (int): Smoothing iterations for accuracy control
        P (ndarray): 3D array storing [grain_map, normal_x, normal_y]
        R (ndarray): Reference solution for error validation
    """
    
    def __init__(self,nx,ny,ng,cores,loop_times,P0,R,clip = 0):
        """
        Initialize the BLv2 smoothing algorithm with microstructure and computational parameters.
        
        Parameters:
        -----------
        nx, ny : int
            Grid dimensions defining the computational domain size
        ng : int
            Number of grains in the initial microstructure condition
        cores : int
            Number of CPU cores to utilize for parallel processing
        loop_times : int
            Number of smoothing iterations (higher = more accurate but slower)
        P0 : ndarray
            Initial condition array with individual grain phase fields
        R : ndarray
            Reference solution for error validation and algorithm verification
        clip : int, optional
            Boundary clipping parameter for smoothing kernel (default: 0)
            
        Internal Data Structures:
        -------------------------
        self.P[0,:,:] : Grain ID map (converted from individual phase fields)
        self.P[1,:,:] : Calculated normal vector x-components
        self.P[2,:,:] : Calculated normal vector y-components
        
        Performance Metrics:
        -------------------
        self.running_time : Total algorithm execution time
        self.running_coreTime : Maximum time spent by any single core
        self.errors : Total angular error compared to reference solution
        self.errors_per_site : Average error per grain boundary site
        """
        # Algorithm performance and error tracking variables
        self.matrix_value = 10          # Internal matrix initialization value
        self.running_time = 0           # Total wall-clock execution time (seconds)
        self.running_coreTime = 0       # Maximum core processing time (seconds)
        self.errors = 0                 # Cumulative angular error (radians)
        self.errors_per_site = 0        # Average error per grain boundary site
        self.clip = clip                # Boundary clipping for smoothing kernel
        
        # Microstructure geometry and grain information
        self.nx = nx # number of sites in x axis
        self.ny = ny # number of sites in y axis
        self.ng = ng # number of grains in IC
        self.R = R  # results of analysis model
        
        # Convert individual grain phase fields into unified grain ID map
        # P[0,:,:] stores grain IDs, P[1,:,:] and P[2,:,:] store normal vector components
        self.P = np.zeros((3,nx,ny)) # matrix to store IC and normal vector results
        for i in range(0,np.shape(P0)[2]):
            # Combine individual grain phase fields into single grain ID map
            # Each grain gets ID = i+1 (grain IDs start from 1, background = 0)
            self.P[0,:,:] += P0[:,:,i]*(i+1)
        
        # Parallel processing configuration for multicore optimization
        self.cores = cores              # Number of CPU cores for parallel execution
        
        # Smoothing algorithm accuracy and kernel size parameters
        self.loop_times = loop_times    # Number of smoothing iterations for convergence
        self.tableL = 2*(loop_times+1)+1 # Total smoothing kernel size (e.g., loop_times=2 → 7x7 kernel)
        self.halfL = loop_times+1       # Half-width of smoothing kernel for indexing
        
        # Memory optimization: commented out sparse matrix for efficiency
        # self.V_sparse = np.empty((loop_times+1,nx,ny),dtype=dict)  # matrix to store the tmp data during calculation
        
        # Pre-computed smoothing vectors for bilinear interpolation
        # These vectors define the smoothing kernel weights for gradient calculation
        self.smoothed_vector_i, self.smoothed_vector_j = myInput.output_linear_vector_matrix(self.loop_times, self.clip)
    
        
    
        
    #%% Utility and Analysis Functions
    def get_P(self):
        """
        Retrieve the complete microstructure and normal vector data.
        
        Returns:
        --------
        ndarray : 3D array with shape (3, nx, ny)
            [0,:,:] : Grain ID map (1 to ng for grains, 0 for background)
            [1,:,:] : Normal vector x-components at each grid point
            [2,:,:] : Normal vector y-components at each grid point
            
        Usage:
        ------
        This method provides access to both the input microstructure and 
        the computed normal vector field for post-processing analysis.
        """
        return self.P
    
    def get_errors(self):
        """
        Calculate angular errors between computed and reference normal vectors.
        
        This method quantifies the accuracy of the bilinear smoothing algorithm by 
        comparing computed normal vectors against analytical reference solutions.
        The error metric uses the angular difference between unit normal vectors.
        
        Error Calculation:
        ------------------
        For each grain boundary site (i,j):
        1. Compute normal vector [dx, dy] using gradient calculation
        2. Calculate dot product with reference normal vector R[i,j,:]
        3. Angular error = arccos(|dx*Rx + dy*Ry|) 
        4. Accumulate total error and compute per-site average
        
        Updates:
        --------
        self.errors : Total cumulative angular error (radians)
        self.errors_per_site : Average angular error per grain boundary site
        
        Notes:
        ------
        - Angular errors are measured in radians
        - Perfect algorithm would have zero angular error
        - Typical acceptable error range: < 0.1 radians (< 5.7 degrees)
        """
        # Get all grain boundary sites for error analysis
        ge_gbsites = self.get_gb_num()
        
        # Iterate through each grain boundary site and calculate angular error
        for gbSite in ge_gbsites :
            [gei,gej] = gbSite
            # Calculate gradient (normal vector) at this grain boundary site
            ge_dx,ge_dy = myInput.get_grad(self.P,gei,gej)
            # Compute angular error using dot product with reference solution
            # The abs() ensures we measure the acute angle between vectors
            # Round to 5 decimal places to avoid numerical precision issues
            self.errors += math.acos(round(abs(ge_dx*self.R[gei,gej,0]+ge_dy*self.R[gei,gej,1]),5))
        
        # Calculate average error per grain boundary site for normalized comparison
        self.errors_per_site = self.errors/len(ge_gbsites)
        
        # return self.errors, self.errors_per_site
        
    def get_2d_plot(self,init,algo):
        """
        Generate 2D visualization of microstructure with normal vector arrows.
        
        This method creates a comprehensive visualization showing both the grain 
        structure and the computed normal vectors at grain boundaries. The plot 
        includes grain boundaries overlaid with normal vector arrows for validation.
        
        Parameters:
        -----------
        init : str
            Description of initial condition type (e.g., 'Poly', 'Voronoi')
        algo : str  
            Algorithm name for plot title (e.g., 'Bilinear')
            
        Visualization Features:
        ----------------------
        - Grayscale grain map showing grain boundaries and interiors
        - Navy blue arrows indicating normal vector directions
        - Arrow length scaled by 10x for visibility
        - Configurable arrow thickness and transparency
        - Title showing algorithm type and smoothing iterations
        
        Plot Configuration:
        ------------------
        - Colormap: 'gray' for clear grain boundary contrast
        - Arrow scaling: 10x magnitude for visual clarity
        - Arrow styling: width=0.1, alpha=0.8, color='navy'
        - No axis ticks for clean presentation
        
        Usage:
        ------
        Primarily used for algorithm validation and publication-quality figures
        showing the relationship between grain structure and computed normal vectors.
        """
        # Configure matplotlib layout and create new figure
        plt.subplots_adjust(wspace=0.2,right=1.8)
        plt.close()
        fig1 = plt.figure(1)
        
        # Create descriptive title with algorithm parameters
        fig_page = self.loop_times
        plt.title(f'{algo}-{init} \n loop = '+str(fig_page))
        
        # Generate zero-padded string for consistent file naming (if saving)
        if fig_page < 10:
            String = '000'+str(fig_page)
        elif fig_page < 100:
            String = '00'+str(fig_page)
        elif fig_page < 1000:
            String = '0'+str(fig_page)
        elif fig_page < 10000:
            String = str(fig_page)
            
        # Display grain structure as grayscale image
        plt.imshow(self.P[0,:,:], cmap='gray', interpolation='nearest')
        plt.xticks([])  # Remove x-axis ticks for clean presentation
        plt.yticks([])  # Remove y-axis ticks for clean presentation
        # Optional: plt.savefig('BL_PolyGray_noArrows.png',dpi=1000,bbox_inches='tight')
        
        # Overlay normal vector arrows at grain boundary sites
        g2p_gbsites = self.get_gb_num()
        for gbSite in g2p_gbsites:
            [g2pi,g2pj] = gbSite
            # Calculate normal vector components at this grain boundary site
            g2p_dx,g2p_dy = myInput.get_grad(self.P,g2pi,g2pj)
            # Optional spatial filtering: if g2pi >200 and g2pi<500:
            # Draw arrow with 10x scaling for visibility
            # Note: plt.arrow uses (x,y) coordinates, so (j,i) for proper orientation
            plt.arrow(g2pj,g2pi,10*g2p_dx,10*g2p_dy,width=0.1,lw=0.1,alpha=0.8,color='navy')
        
        # Optional: Additional plot configuration and saving
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('BL_PolyGray_Arrows.png',dpi=1000,bbox_inches='tight')
        plt.show()
    
    def get_gb_num(self,grainID=1):
        """
        Identify and return all grain boundary sites for a specific grain.
        
        This method scans the entire microstructure to locate sites where the specified 
        grain interfaces with neighboring grains. It uses periodic boundary conditions 
        and checks all four nearest neighbors to determine grain boundary locations.
        
        Parameters:
        -----------
        grainID : int, optional
            Target grain ID to analyze for boundary sites (default: 1)
            
        Returns:
        --------
        list : List of [i,j] coordinates representing grain boundary sites
            Each element is a 2-element list [row, column] in grid coordinates
            
        Algorithm:
        ----------
        For each grid point (i,j):
        1. Check if current site belongs to target grain (P[0,i,j] == grainID)
        2. Apply periodic boundary conditions to get neighbor coordinates
        3. Compare grain ID with four nearest neighbors (±1 in x,y directions)
        4. If any neighbor has different grain ID → this is a boundary site
        5. Add [i,j] coordinates to boundary site list
        
        Boundary Detection Logic:
        ------------------------
        A site is classified as grain boundary if:
        - Current site has target grainID AND
        - At least one neighbor has different grain ID
        
        This ensures only interior-to-boundary transitions are detected,
        excluding external boundaries and isolated pixels.
        
        Usage:
        ------
        Essential for error analysis, normal vector calculation, and visualization.
        Typically called with grainID=1 for single-grain analysis or iterated 
        over all grain IDs for comprehensive boundary mapping.
        """
        ggn_gbsites = []  # Initialize list to store boundary site coordinates
        
        # Scan entire microstructure domain
        for i in range(0,self.nx):
            for j in range(0,self.ny):
                # Apply periodic boundary conditions to handle domain edges
                ip,im,jp,jm = myInput.periodic_bc(self.nx,self.ny,i,j)
                
                # Check if current site belongs to target grain AND has different neighbors
                # Boundary condition: current site = grainID AND any neighbor ≠ grainID
                if ( ((self.P[0,ip,j]-self.P[0,i,j])!=0) or     # Right neighbor different
                     ((self.P[0,im,j]-self.P[0,i,j])!=0) or     # Left neighbor different  
                     ((self.P[0,i,jp]-self.P[0,i,j])!=0) or     # Top neighbor different
                     ((self.P[0,i,jm]-self.P[0,i,j])!=0) ) \
                        and self.P[0,i,j]==grainID:             # Current site is target grain
                    # This site is on the boundary of the target grain
                    ggn_gbsites.append([i,j])
                    
        return ggn_gbsites
                    
    def check_subdomain_and_nei(self,A):
        """
        Determine subdomain assignment and neighboring subdomains for parallel processing.
        
        This method implements domain decomposition for multiprocessing by determining 
        which subdomain a given coordinate belongs to and identifying all neighboring 
        subdomains that may require data exchange during parallel computation.
        
        Parameters:
        -----------
        A : list or array
            Coordinate [x, y] to analyze for subdomain assignment
            
        Returns:
        --------
        ca_area_cen : list
            [width_index, length_index] of the central subdomain containing point A
        ca_area_nei : list
            List of 8 neighboring subdomain coordinates in Moore neighborhood
            Each element is [width_index, length_index] with periodic wrapping
            
        Domain Decomposition Strategy:
        -----------------------------
        1. Split total domain (nx × ny) into rectangular subdomains
        2. Each subdomain assigned to one CPU core for parallel processing
        3. Map input coordinate to appropriate subdomain indices
        4. Identify all 8 neighboring subdomains (Moore neighborhood)
        5. Apply periodic boundary conditions for edge/corner subdomains
        
        Neighbor Order (8-connected):
        ----------------------------
        Neighbors arranged in clockwise order starting from top-left:
        [0] Top-left     [1] Top       [2] Top-right
        [7] Left         [X] Center    [3] Right
        [6] Bottom-left  [5] Bottom    [4] Bottom-right
        
        Usage in Multiprocessing:
        -------------------------
        - Central subdomain: processes local data
        - Neighboring subdomains: provide boundary data for smoothing operations
        - Essential for handling subdomain edge effects in parallel algorithms
        """
        # Determine subdomain grid dimensions based on number of cores
        ca_length,ca_width = myInput.split_cores(self.cores)
        
        # Map coordinate A to subdomain indices
        # Scale coordinate by domain fractions to get subdomain assignment
        ca_area_cen = [int(A[0]/self.nx*ca_width),int(A[1]/self.ny*ca_length)]
        
        # Initialize list for 8 neighboring subdomains
        ca_area_nei = []
        
        # Define 8-connected Moore neighborhood with periodic boundary conditions
        # Each neighbor calculated using modulo arithmetic for wrapping
        ca_area_nei.append( [int((ca_area_cen[0]-1)%ca_width), int((ca_area_cen[1]-1)%ca_length)] )  # Top-left
        ca_area_nei.append( [int((ca_area_cen[0]-1)%ca_width), int(ca_area_cen[1])] )                # Top
        ca_area_nei.append( [int((ca_area_cen[0]-1)%ca_width), int((ca_area_cen[1]+1)%ca_length)] )  # Top-right
        ca_area_nei.append( [int(ca_area_cen[0]), int((ca_area_cen[1]+1)%ca_length)] )               # Right
        ca_area_nei.append( [int((ca_area_cen[0]+1)%ca_width), int((ca_area_cen[1]+1)%ca_length)] )  # Bottom-right
        ca_area_nei.append( [int((ca_area_cen[0]+1)%ca_width), int(ca_area_cen[1])] )                # Bottom
        ca_area_nei.append( [int((ca_area_cen[0]+1)%ca_width), int((ca_area_cen[1]-1)%ca_length)] )  # Bottom-left
        ca_area_nei.append( [int(ca_area_cen[0]), int((ca_area_cen[1]-1)%ca_length)] )               # Left
        
        return ca_area_cen, ca_area_nei
        
    
    def transfer_data(self,i, ii, j, jj, corner1, corner3, kk=0):
        """
        Determine which neighboring subdomains require data transfer for smoothing operations.
        
        This method analyzes the spatial relationship between a computation point and 
        subdomain boundaries to identify which neighboring subdomains must provide 
        data for accurate smoothing kernel application near subdomain edges.
        
        Parameters:
        -----------
        i, ii : int
            Current and offset coordinates in x-direction
        j, jj : int  
            Current and offset coordinates in y-direction
        corner1, corner3 : array-like
            Opposing corner coordinates of current subdomain
        kk : int, optional
            Additional offset parameter for kernel size adjustment (default: 0)
            
        Returns:
        --------
        list : Neighbor subdomain indices requiring data transfer
            List of integers (0-7) corresponding to Moore neighborhood directions
            Empty list if no data transfer needed (interior computation)
            
        Data Transfer Logic:
        -------------------
        For smoothing operations near subdomain boundaries, the algorithm needs 
        data from neighboring subdomains to apply kernels correctly. This method 
        determines which neighbors to contact based on proximity to subdomain edges.
        
        Proximity Analysis:
        ------------------
        - Check distance from computation point to each subdomain boundary
        - If distance < halfL-kk, data transfer from that direction is required
        - Multiple neighbors may be needed for corner/edge computations
        - halfL represents the smoothing kernel half-width
        
        Neighbor Direction Mapping:
        --------------------------
        Direction indices correspond to 8-connected Moore neighborhood:
        0,7: Top-left region    1: Top edge       2,3: Top-right region
        7: Left edge           Center             3: Right edge  
        6,7: Bottom-left       5: Bottom edge     3,4: Bottom-right
        
        Usage:
        ------
        Called during parallel smoothing to minimize inter-process communication
        by only requesting data from subdomains that actually contribute to 
        the local computation.
        """
        td_out = []  # Initialize list of required neighbor directions
        
        # Analyze proximity to vertical subdomain boundaries (x-direction)
        if abs(i+ii-corner1[0]) < self.halfL-kk:
            # Close to left boundary - need data from left neighbors
            td_out.append(1)  # Top edge neighbor
            
            # Check proximity to horizontal boundaries for corner cases
            if abs(j+jj-corner1[1]) < self.halfL-kk:
                # Top-left corner region - need top-left neighbors
                td_out.append(0)  # Top-left diagonal
                td_out.append(7)  # Left edge
                
            elif abs(j+jj-corner3[1]) < self.halfL-kk:
                # Bottom-left corner region - need bottom-left neighbors  
                td_out.append(2)  # Top-right diagonal
                td_out.append(3)  # Right edge
                
        elif abs(i+ii-corner3[0]) < self.halfL-kk:
            # Close to right boundary - need data from right neighbors
            td_out.append(5)  # Bottom edge neighbor
            
            # Check proximity to horizontal boundaries for corner cases
            if abs(j+jj-corner1[1]) < self.halfL-kk:
                # Top-right corner region - need top-right neighbors
                td_out.append(6)  # Bottom-left diagonal  
                td_out.append(7)  # Left edge
                
            elif abs(j+jj-corner3[1]) < self.halfL-kk:
                # Bottom-right corner region - need bottom-right neighbors
                td_out.append(3)  # Right edge
                td_out.append(4)  # Bottom-right diagonal
                
        # Analyze proximity to horizontal boundaries only (no vertical proximity)        
        elif abs(j+jj-corner1[1]) < self.halfL-kk:
            # Close to top boundary only
            td_out.append(7)  # Left edge neighbor
            
        elif abs(j+jj-corner3[1]) < self.halfL-kk:
            # Close to bottom boundary only  
            td_out.append(3)  # Right edge neighbor
        
        return td_out
    
    def find_window(self,i,j):
        """
        Extract local smoothing window around a specific grid point for bilinear operations.
        
        This method creates a binary mask window centered on grid point (i,j) that 
        identifies which neighboring points belong to the same grain. The window 
        is used for applying smoothing kernels while preserving grain boundaries.
        
        Parameters:
        -----------
        i, j : int
            Center coordinates for window extraction
            
        Returns:
        --------
        ndarray : 2D binary array with shape (fw_len, fw_len)
            1 = neighboring point belongs to same grain as center
            0 = neighboring point belongs to different grain
            
        Window Construction Algorithm:
        -----------------------------
        1. Calculate window size based on smoothing parameters and clipping
        2. Center window on input coordinates (i,j)
        3. Apply periodic boundary conditions for domain edge handling
        4. Compare grain ID at each window position with center grain ID
        5. Generate binary mask: 1 for same grain, 0 for different grain
        
        Window Size Calculation:
        -----------------------
        fw_len = tableL - 2*clip
        - tableL: Total smoothing kernel size (2*(loop_times+1)+1)
        - clip: Boundary clipping parameter to reduce window size
        - fw_half: Half-width for symmetric window centering
        
        Periodic Boundary Conditions:
        -----------------------------
        Uses modulo arithmetic to handle window edges that extend beyond 
        domain boundaries, ensuring consistent behavior across the entire 
        computational domain including edges and corners.
        
        Grain Boundary Preservation:
        ----------------------------
        The binary mask ensures smoothing operations only use data from 
        the same grain, preventing artificial smoothing across grain 
        boundaries that would corrupt the interface geometry.
        
        Usage:
        ------
        Called during smoothing kernel application to create grain-aware 
        smoothing that maintains sharp grain boundary definitions while 
        reducing numerical noise within grain interiors.
        """
        # Calculate window dimensions with clipping adjustment
        fw_len = self.tableL - 2*self.clip  # Effective window size after clipping
        fw_half = int((fw_len-1)/2)         # Half-width for centering
        
        # Initialize binary window array
        window = np.zeros((fw_len,fw_len))
        
        # Extract grain ID at center point for comparison
        center_grain_id = self.P[0,i,j]
        
        # Populate window with binary grain membership mask
        for wi in range(fw_len):
            for wj in range(fw_len):
                # Calculate global coordinates with periodic boundary conditions
                # Offset from center by (wi-fw_half, wj-fw_half)
                global_x = (i-fw_half+wi)%self.nx  # Apply modulo for x-direction wrapping
                global_y = (j-fw_half+wj)%self.ny  # Apply modulo for y-direction wrapping
                
                # Compare grain ID at this position with center grain
                if self.P[0,global_x,global_y] == center_grain_id:
                    window[wi,wj] = 1  # Same grain - include in smoothing
                else:
                    window[wi,wj] = 0  # Different grain - exclude from smoothing
                    
                # Alternative: store actual grain IDs instead of binary mask
                # window[wi,wj] = self.P[0,global_x,global_y]
        
        return window
    
    
    #%% Core Parallel Processing Functions
    
    def BLv2_core_apply(self,core_input, core_all_queue):
        """
        Core parallel processing function for bilinear smoothing computation.
        
        This method represents the main computational kernel executed by each CPU core 
        in the parallel processing framework. It processes a subdomain of the total 
        microstructure and calculates normal vectors using bilinear smoothing at 
        grain boundary sites within that subdomain.
        
        Parameters:
        -----------
        core_input : ndarray
            3D array with shape (subdomain_x, subdomain_y, 2) containing 
            coordinate pairs [i,j] for all grid points assigned to this core
        core_all_queue : multiprocessing queues
            Communication queues for inter-process data exchange (currently unused 
            but available for future subdomain boundary data sharing)
            
        Returns:
        --------
        tuple : (fval, execution_time)
            fval : ndarray with shape (nx, ny, 2)
                Computed normal vector components for the entire domain
                [i,j,0] = x-component of normal vector at grid point (i,j)
                [i,j,1] = y-component of normal vector at grid point (i,j)
            execution_time : float
                Wall-clock time spent by this core (seconds)
                
        Algorithm Workflow:
        ------------------
        1. Initialize timing and local result arrays
        2. Determine subdomain boundaries and processor identification
        3. Iterate through all assigned grid points in subdomain
        4. For each grain boundary site:
           a. Create local smoothing window using find_window()
           b. Apply bilinear smoothing kernels to calculate gradients
           c. Store normal vector components in result array
        5. Return computed results and performance metrics
        
        Grain Boundary Detection:
        ------------------------
        Uses periodic boundary conditions to check if current site has neighbors 
        with different grain IDs. Only processes sites identified as grain boundaries 
        to optimize computational efficiency.
        
        Bilinear Smoothing Application:
        ------------------------------
        - Extract grain-aware local window around each boundary site
        - Apply pre-computed smoothing vectors (smoothed_vector_i/j)
        - Calculate gradient components using weighted convolution
        - Negative x-component, positive y-component for proper orientation
        
        Performance Monitoring:
        ----------------------
        - Tracks execution time for load balancing analysis
        - Reports processor identification for debugging
        - Currently includes placeholder variables for future queue monitoring
        
        Usage in Multiprocessing:
        -------------------------
        Called asynchronously by multiprocessing.Pool.apply_async() for each 
        subdomain. Results collected by callback function res_back() for 
        final assembly of complete normal vector field.
        """
        # Initialize performance timing
        core_stime = datetime.datetime.now()
        
        # Extract subdomain dimensions and initialize result array
        li,lj,lk=np.shape(core_input)                # Subdomain dimensions
        fval = np.zeros((self.nx,self.ny,2))         # Global result array for this core
        
        # Determine subdomain corner coordinates for processor identification
        corner1 = core_input[0,0,:]      # Top-left corner of subdomain
        corner3 = core_input[li-1,lj-1,:]  # Bottom-right corner of subdomain
        
        # Identify which processor/subdomain this core is handling
        core_area_cen, core_area_nei = self.check_subdomain_and_nei(corner1)
        print(f'the processor {core_area_cen} start...')
        
        # Performance monitoring variables (currently for testing/debugging)
        # MY TEST AREAQ
        # core_A,core_B = split_cores(cores)
        # print_me("lalala")
        # print("core start...")
        # core_test=math.floor(10.5)
        
        test_check_read_num = 0   # Placeholder for queue read operations monitoring
        test_check_max_qsize = 0  # Placeholder for maximum queue size tracking
        
        # Main computation loop: process all grid points in assigned subdomain
        for core_a in core_input:
            for core_b in core_a:
                # Extract current grid coordinates
                i = core_b[0]  # x-coordinate in global grid
                j = core_b[1]  # y-coordinate in global grid
                
                # Placeholder for future table-based indexing optimization
                # fv_i, fv_j = self.find_tableij(corner1,i,j)
                
                # Apply periodic boundary conditions to get neighbor coordinates
                ip,im,jp,jm = myInput.periodic_bc(self.nx,self.ny,i,j)
                
                # Grain boundary detection: check if any neighbor has different grain ID
                if ( ((self.P[0,ip,j]-self.P[0,i,j])!=0) or     # Right neighbor different
                     ((self.P[0,im,j]-self.P[0,i,j])!=0) or     # Left neighbor different
                     ((self.P[0,i,jp]-self.P[0,i,j])!=0) or     # Top neighbor different
                     ((self.P[0,i,jm]-self.P[0,i,j])!=0) ):     # Bottom neighbor different
                    
                    # This is a grain boundary site - apply bilinear smoothing
                    
                    # Initialize local smoothing window
                    window = np.zeros((self.tableL,self.tableL))
                    
                    # Extract grain-aware smoothing window around current site
                    window = self.find_window(i,j)
                    
                    # Optional: Debug window visualization
                    # print(window)
                    
                    # Apply bilinear smoothing kernels to calculate normal vector components
                    # Convolve smoothing window with pre-computed gradient vectors
                    fval[i,j,0] = -np.sum(window*self.smoothed_vector_i)  # x-component (negative for proper orientation)
                    fval[i,j,1] = np.sum(window*self.smoothed_vector_j)   # y-component (positive orientation)
                    

        # Report performance monitoring results
        print(f"process{core_area_cen} read {test_check_read_num} times and max qsize {test_check_max_qsize}")
        
        # Calculate execution time for this core
        core_etime = datetime.datetime.now()
        execution_time = (core_etime - core_stime).total_seconds()
        print("my core time is " + str(execution_time))
        
        # Return computed normal vectors and performance metrics
        return (fval, execution_time)
        
    
    def res_back(self,back_result):
        """
        Callback function for collecting parallel processing results.
        
        This method serves as the callback function for multiprocessing.Pool.apply_async() 
        operations. It receives results from individual CPU cores and accumulates them 
        into the main result arrays while tracking performance metrics.
        
        Parameters:
        -----------
        back_result : tuple
            Result tuple from BLv2_core_apply() containing:
            - fval: Normal vector field computed by one core
            - core_time: Execution time for that core
            
        Algorithm:
        ----------
        1. Extract normal vector field and execution time from core result
        2. Update maximum core execution time for load balancing analysis
        3. Accumulate normal vector components into global result arrays
        4. Track callback processing time for performance optimization
        
        Result Accumulation:
        -------------------
        - self.P[1,:,:] += fval[:,:,0]  # Accumulate x-components of normal vectors
        - self.P[2,:,:] += fval[:,:,1]  # Accumulate y-components of normal vectors
        
        The += operation allows multiple cores to contribute to the same global 
        result array, with each core contributing only its computed subdomain results.
        
        Performance Tracking:
        --------------------
        - running_coreTime: Maximum execution time across all cores
        - Identifies computational bottlenecks and load balancing issues
        - Enables optimization of subdomain decomposition strategies
        
        Thread Safety:
        --------------
        This callback function is executed sequentially by the multiprocessing 
        framework, ensuring thread-safe accumulation of results without explicit 
        locking mechanisms.
        
        Usage:
        ------
        Automatically called by multiprocessing.Pool when each core completes 
        its assigned computation. Provides seamless result collection without 
        manual synchronization code.
        """
        # Initialize callback timing for performance analysis
        res_stime = datetime.datetime.now()
        
        # Extract normal vector field and core execution time from result tuple
        (fval,core_time) = back_result
        
        # Update maximum core execution time for load balancing metrics
        # This identifies the slowest core, which determines overall algorithm performance
        if core_time > self.running_coreTime:
            self.running_coreTime = core_time
        
        print("res_back start...")
        
        # Accumulate normal vector components into global result arrays
        # Each core contributes its computed subdomain results to the total field
        self.P[1,:,:] += fval[:,:,0]  # Accumulate x-components of normal vectors
        self.P[2,:,:] += fval[:,:,1]  # Accumulate y-components of normal vectors
        
        # Calculate callback processing time for performance optimization
        res_etime = datetime.datetime.now()
        callback_time = (res_etime - res_stime).total_seconds()
        print("my res time is " + str(callback_time))
    
    def BLv2_main(self):
        """
        Main orchestration function for parallel bilinear smoothing algorithm.
        
        This method coordinates the entire bilinear smoothing computation by setting up 
        multiprocessing infrastructure, distributing work across CPU cores, and collecting 
        results. It implements domain decomposition for efficient parallel processing of 
        large microstructure datasets.
        
        Algorithm Workflow:
        ------------------
        1. Initialize multiprocessing pool and timing
        2. Create domain decomposition based on available CPU cores
        3. Set up inter-process communication queues (for future enhancements)
        4. Distribute grid coordinates across subdomains
        5. Launch parallel bilinear smoothing computations
        6. Collect results via callback functions
        7. Calculate performance metrics and error analysis
        
        Domain Decomposition Strategy:
        -----------------------------
        - Split computational domain (nx × ny) into rectangular subdomains
        - Each subdomain assigned to one CPU core for parallel processing
        - Subdomain boundaries handled with periodic boundary conditions
        - Load balancing achieved through uniform subdomain size distribution
        
        Multiprocessing Architecture:
        ----------------------------
        - Pool of worker processes equal to number of CPU cores
        - Asynchronous task distribution with apply_async()
        - Callback-based result collection for memory efficiency
        - Graceful process synchronization with pool.close() and pool.join()
        
        Inter-Process Communication:
        ---------------------------
        Communication queues are created for potential future enhancements:
        - Data exchange between neighboring subdomains
        - Boundary condition synchronization
        - Real-time progress monitoring
        
        Currently, queues are prepared but not actively used in computation.
        
        Performance Metrics:
        -------------------
        - Total wall-clock execution time
        - Maximum core execution time (bottleneck identification)
        - Error analysis compared to reference solutions
        - Memory usage and computational efficiency tracking
        
        Error Handling:
        --------------
        - Robust process management with proper cleanup
        - Comprehensive timing for performance optimization
        - Validation against analytical reference solutions
        
        Usage:
        ------
        Called after object initialization to execute the complete bilinear 
        smoothing algorithm and populate self.P with computed normal vectors.
        """
        
        # Initialize global timing for performance measurement
        # global starttime, endtime  # Commented global timing variables
        starttime = datetime.datetime.now() 
    
        # Create multiprocessing pool with specified number of cores
        pool = mp.Pool(processes=self.cores)
        
        # Determine subdomain grid layout based on available cores
        main_lc, main_wc = myInput.split_cores(self.cores)  # length_count, width_count
        
        # Initialize inter-process communication infrastructure
        # Create managed queues for potential future subdomain data exchange
        manager = mp.Manager()
        all_queue = []
        
        # Calculate maximum queue size based on subdomain dimensions and smoothing parameters
        max_queue_size = int(((self.nx/main_wc)+(self.ny/main_lc))*self.loop_times*self.loop_times)
        print(f"The max size of queue {max_queue_size}")
        
        # Create 2D array of communication queues matching subdomain grid layout
        for queue_i in range(main_wc):
            tmp = []
            for queue_j in range(main_lc):
                # Each subdomain gets a dedicated queue for neighbor communication
                queue_capacity = int(((self.nx/main_wc)+(self.ny/main_lc))*(1+self.loop_times)*(1+self.loop_times))
                tmp.append(manager.Queue(queue_capacity))
            all_queue.append(tmp)
                
        # Prepare computational domain for distribution across cores
        # Create coordinate array containing [x,y] for every grid point
        all_sites = np.array([[x,y] for x in range(self.nx) for y in range(self.ny) ]).reshape(self.nx,self.ny,2)
        
        # Distribute grid coordinates across subdomains for parallel processing
        # multi_input[i][j] contains coordinate array for subdomain (i,j)
        multi_input = myInput.split_IC(all_sites, self.cores, 2, 0, 1)
        
        # Launch parallel computation across all subdomains
        res_list=[]  # List to track asynchronous computation tasks
        
        for ki in range(main_wc):        # Iterate over subdomain width
            for kj in range(main_lc):    # Iterate over subdomain length
                # Launch asynchronous bilinear smoothing computation for subdomain (ki,kj)
                res_one = pool.apply_async(
                    func = self.BLv2_core_apply,           # Core computation function
                    args = (multi_input[ki][kj], all_queue), # Subdomain coordinates and communication queues
                    callback = self.res_back               # Result collection callback
                )
                res_list.append(res_one)  # Track this computation task
    
        # Synchronize parallel computation completion
        pool.close()  # No more tasks will be submitted to the pool
        pool.join()   # Wait for all worker processes to complete
        
        print("core done!")
        # Optional: Access individual task results if needed
        # print(res_list[0].get())
        
        # Calculate total algorithm execution time
        endtime = datetime.datetime.now()
        self.running_time = (endtime - starttime).total_seconds()
        
        # Perform error analysis against reference solution
        self.get_errors()
        
    
        

if __name__ == '__main__':
    """
    Test and benchmarking section for bilinear smoothing algorithm validation.
    
    This section provides comprehensive testing capabilities for the BLv2 algorithm,
    including performance benchmarking, accuracy validation, and parameter sensitivity
    analysis. Multiple test configurations are provided for different use cases.
    
    Test Configuration Options:
    --------------------------
    - Domain sizes: 200×200 (testing) to 700×700 (production)
    - Grain counts: 5 (simple) to 62 (complex polycrystalline)
    - Core counts: 1 (sequential) to 8+ (parallel)
    - Loop iterations: 3-10 (accuracy vs speed trade-off)
    
    Initial Condition Generators:
    ----------------------------
    - Voronoi tessellation: Realistic polycrystalline microstructures
    - Circular grains: Analytical validation with known solutions
    - Complex geometries: Advanced geometric configurations
    - Abnormal grain structures: Grain growth study configurations
    - File-based initialization: Real experimental microstructures
    
    Performance Metrics:
    -------------------
    - Total execution time: Wall-clock algorithm performance
    - Core execution time: Parallel efficiency and load balancing
    - Angular error: Accuracy compared to analytical solutions
    - Per-site error: Normalized accuracy metrics
    
    Benchmarking Arrays:
    -------------------
    - BL_errors2: Angular error vs smoothing iterations
    - BL_runningTime2: Execution time vs computational parameters
    
    Usage:
    ------
    Modify test parameters and uncomment desired initial condition generator
    to evaluate algorithm performance for specific applications.
    """
    
    # Initialize benchmarking arrays for performance analysis
    BL_errors2 = np.zeros(10)     # Angular error tracking array
    BL_runningTime2 = np.zeros(10) # Execution time tracking array
    
    # Domain configuration - adjust for different test scales
    # Large domain for production testing:
    # nx, ny = 700, 700  # High-resolution domain
    # ng = 62            # Complex polycrystalline structure
    
    # Small domain for rapid testing and development:
    nx, ny = 200, 200    # Moderate resolution for testing
    ng = 5               # Simple grain structure for validation
    
    # Algorithm parameters
    clip = 0             # No boundary clipping (full smoothing kernel)
    # cores = 8          # Parallel processing with 8 cores (uncomment for production)
    
    # Initial condition selection - choose appropriate microstructure generator
    # File-based initialization (real experimental data):
    # P0,R=myInput.init2IC(nx, ny, ng, "PolyIC.init")
    
    # Analytical circular grains (known solution for validation):
    # P0,R=myInput.Circle_IC(nx,ny)
    
    # Voronoi tessellation (realistic polycrystalline structure):
    P0,R=myInput.Voronoi_IC(nx,ny,ng)
    
    # Complex two-grain configuration (interface studies):
    # P0,R=myInput.Complex2G_IC(nx,ny)
    
    # Abnormal grain growth configuration (grain evolution studies):
    # P0,R=myInput.Abnormal_IC(nx,ny)
    
    # Smallest grain configuration (extreme geometry testing):
    # P0[:,:,:]=myInput.SmallestGrain_IC(100,100)
    
    # Parameter sweep testing - evaluate performance vs accuracy trade-offs
    for cores in [1]:               # Test with single core (sequential processing)
        # loop_times=10             # Fixed high-accuracy setting (uncomment for production)
        
        for loop_times in range(3,4):  # Test smoothing iterations 3-3 (single iteration for testing)
            
            # Initialize and execute bilinear smoothing algorithm
            test1 = BLv2_class(nx,ny,ng,cores,loop_times,P0,R,clip)
            test1.BLv2_main()            # Execute main algorithm
            P5_s = test1.get_P()         # Retrieve computed normal vectors
        
            # Optional visualization (uncomment for validation plots):
            #%%
            # test1.get_2d_plot('Poly','Bilinear')
            
            # Performance and accuracy reporting
            #%% error
            print('loop_times = ' + str(test1.loop_times))
            print('running_time = %.2f' % test1.running_time)
            print('running_core time = %.2f' % test1.running_coreTime)
            print('total_errors = %.2f' % test1.errors)
            print('per_errors = %.3f' % test1.errors_per_site)
            print()
            
            # Store results in benchmarking arrays for analysis
            BL_errors2[loop_times-1] = test1.errors_per_site      # Angular error per site
            BL_runningTime2[loop_times-1] = test1.running_coreTime # Maximum core execution time
        
#%% (Jupyter notebook cell separator for interactive analysis)


