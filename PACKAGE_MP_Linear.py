#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Smoothing Method Implementation for Interface Analysis

This module implements a linear smoothing method for calculating grain boundary
normal vectors and curvature in polycrystalline materials. The method uses
linear filtering with these features:

1. Linear Filtering:
   - Applies weighted averaging to smooth interface properties
   - Uses variable kernel sizes for multi-scale analysis
   - Preserves sharp features at grain junctions

2. Normal Vector Calculation:
   - Computes gradients from smoothed interface data
   - Handles multiple grain boundaries efficiently
   - Provides consistent normals at triple junctions

3. Curvature Calculation:
   - Uses second derivatives of smoothed data
   - Handles multiple length scales
   - Maintains numerical stability at interfaces

Key Features:
- Parallel implementation for large datasets
- Configurable smoothing parameters
- Efficient sparse matrix operations
- Error calculation against analytical solutions

VARIABLE NAMING CONVENTIONS
============================

Core Data Structures:
    P[0,:,:] : Grain ID at each site
    P[1,:,:] : x-component of normal vector
    P[2,:,:] : y-component of normal vector
    C[0,:,:] : Grain ID (curvature calculation)
    C[1,:,:] : Curvature values

Grid Parameters:
    nx, ny : Grid dimensions (number of sites)
    i, j : Site coordinates [0, nx) and [0, ny)
    ip, im, jp, jm : Periodic neighbor indices

Finite Difference Stencil (5x5):
    I_ij : Value at stencil position [i,j]
    I22 : Center point at (2,2)
    I12, I32 : Vertical neighbors (±1 in i)
    I21, I23 : Horizontal neighbors (±1 in j)

    Stencil layout:
        [0,0] [0,1] [0,2] [0,3] [0,4]     i-2
        [1,0] [1,1] [1,2] [1,3] [1,4]     i-1
        [2,0] [2,1] [2,2] [2,3] [2,4]  ←  i (center)
        [3,0] [3,1] [3,2] [3,3] [3,4]     i+1
        [4,0] [4,1] [4,2] [4,3] [4,4]     i+2
                         ↑
                    j-2 j-1 j j+1 j+2

Derivatives (after calculation):
    phi_x, phi_y : First derivatives ∂φ/∂x, ∂φ/∂y
    phi_xx, phi_yy : Second derivatives ∂²φ/∂x², ∂²φ/∂y²
    phi_xy : Mixed second derivative ∂²φ/∂x∂y

Smoothing Parameters:
    loop_times : Smoothing window half-width (typical: 3-10)
    tableL : Full window size = 2*(loop_times+1)+1
    halfL : Half window size = loop_times+1
    clip : Number of boundary layers to exclude (typical: 0-2)

Parallel Processing:
    cores : Number of CPU cores to use
    core_area_cen : [width_id, length_id] of current subdomain
    core_area_nei : List of 8 neighboring subdomain IDs

For complete variable definitions, see GLOSSARY.md

Author: Lin Yang
"""

import os
current_path = os.getcwd()+'/'
import sys
sys.path.append(current_path)
import numpy as np
import math
import myInput
import datetime
import multiprocessing as mp
from PACKAGE_MP_Base2D import Base2D


class linear_class(Base2D):
    """Linear smoothing algorithm implementation.

    This class implements linear smoothing to calculate normal vectors and curvature
    at grain boundaries. The algorithm uses a sliding window approach with weighted
    averaging to smooth boundaries and compute geometric properties.

    Attributes:
        P (ndarray): Phase field array storing microstructure and normal vectors
        C (ndarray): Array storing calculated curvature values
        nx (int): Number of grid points in x direction
        ny (int): Number of grid points in y direction
        ng (int): Number of grains
        cores (int): Number of CPU cores for parallel processing
        loop_times (int): Number of smoothing iterations
        clip (int): Number of boundary elements to ignore
        errors (float): Accumulated angle errors
        errors_per_site (float): Average error per boundary site
        running_time (float): Total computation time
        running_coreTime (float): Maximum core processing time
    """

    def __init__(self,nx,ny,ng,cores,loop_times,P0,R,clip=0,verification_system = True, curvature_sign = False):
        """Initialize the linear smoothing algorithm.

        Args:
            nx (int): Number of grid points in x direction
            ny (int): Number of grid points in y direction
            ng (int): Number of grains in the system
            cores (int): Number of CPU cores for parallel processing
            loop_times (int): Size of smoothing window
            P0 (ndarray): Initial microstructure configuration
            R (ndarray): Reference/analytical solution for validation
            clip (int): Number of boundary layers to ignore
            verification_system (bool): Enable validation against analytical solution
            curvature_sign (bool): Calculate signed curvature values
        """
        super().__init__(nx, ny, ng, cores, P0, R, clip, verification_system, curvature_sign)

        # Smoothing parameters
        self.loop_times = loop_times
        self.tableL = 2*(loop_times+1)+1  # Table length for repeated calcs
        self.tableL_curv = 2*(loop_times+2)+1
        self.halfL = loop_times+1

        # Get smoothing matrices
        self.smoothed_vector_i, self.smoothed_vector_j = myInput.output_linear_vector_matrix(
            self.loop_times, self.clip)

    def get_2d_plot(self, init, algo):
        """Generate 2D visualization of microstructure with normal vectors.

        Args:
            init (str): Name of initial condition
            algo (str): Name of algorithm used
        """
        super().get_2d_plot(init, algo, self.loop_times, arrow_scale=30, cmap='gray',
                            save_prefix='BL', filter_range=(200, 500))

    def find_window(self, i, j, fw_len):
        """Extract binary grain membership window around a point.

        Creates a square window centered at (i,j) indicating which sites
        belong to the same grain as the center point. Used for local
        smoothing operations that respect grain boundaries.

        Args:
            i (int): x coordinate of center point
            j (int): y coordinate of center point
            fw_len (int): Window side length (must be odd)

        Returns:
            ndarray: Binary matrix of shape (fw_len, fw_len) where:
                - 1 = site belongs to same grain as center
                - 0 = site belongs to different grain

        Example:
            For a 5x5 window at a grain boundary:

            [[0, 0, 1, 1, 1],     grain 2 | grain 1
             [0, 1, 1, 1, 1],     --------+--------
             [1, 1, 1, 1, 1],  ←  Center (i,j) in grain 1
             [1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1]]

        Notes:
            - Automatically handles periodic boundary conditions
            - Window extends ±(fw_len-1)/2 in each direction
            - Used as weight mask for linear smoothing operations
            - Sites outside the grain get zero weight in smoothing
        """
        fw_half = int((fw_len-1)/2)
        window = np.zeros((fw_len,fw_len))

        for wi in range(fw_len):
            for wj in range(fw_len):
                global_x = (i-fw_half+wi)%self.nx
                global_y = (j-fw_half+wj)%self.ny
                if self.P[0,global_x,global_y] == self.P[0,i,j]:
                    window[wi,wj] = 1
                else:
                    window[wi,wj] = 0

        return window

    def calculate_curvature(self, matrix):
        """Calculate mean curvature from smoothed normal vector field.

        Computes mean curvature κ using second derivatives of the smoothed
        interface function via central finite differences on a 5x5 stencil.

        Args:
            matrix (ndarray): 5x5 array of smoothed normal vector values

        Returns:
            float: Mean curvature κ in units of 1/voxel. Returns 0 if
                  gradient magnitude is zero (flat region).

        Mathematical Formula:
        --------------------
        Mean curvature is calculated as:

            κ = (φ_x² φ_yy - 2φ_x φ_y φ_xy + φ_y² φ_xx) / (φ_x² + φ_y²)^(3/2)

        Where:
            φ_x, φ_y     = First derivatives (gradient components)
            φ_xx, φ_yy   = Second derivatives (Hessian diagonal)
            φ_xy         = Mixed second derivative (Hessian off-diagonal)

        This is equivalent to κ = ∇·n̂ where n̂ = ∇φ/|∇φ| is the unit normal.

        Finite Difference Stencil:
        --------------------------
        The 5x5 matrix indices map to spatial positions:

            [0,0] [0,1] [0,2] [0,3] [0,4]     i-2
            [1,0] [1,1] [1,2] [1,3] [1,4]     i-1
            [2,0] [2,1] [2,2] [2,3] [2,4]  ←  i (center)
            [3,0] [3,1] [3,2] [3,3] [3,4]     i+1
            [4,0] [4,1] [4,2] [4,3] [4,4]     i+2
                             ↑
                        j-2 j-1 j j+1 j+2

        Variable Naming Convention:
            I_ij = matrix[i,j] = value at stencil position

        First derivatives (central differences, h=1):
            φ_x = (φ[i+1,j] - φ[i-1,j]) / 2 = (I32 - I12) / 2
            φ_y = (φ[i,j+1] - φ[i,j-1]) / 2 = (I23 - I21) / 2

        Second derivatives:
            φ_xx = (φ[i+1,j] - 2φ[i,j] + φ[i-1,j]) / 1²
                 = ((I42-I22)/2 - (I22-I02)/2) / 2
            φ_yy = (φ[i,j+1] - 2φ[i,j] + φ[i,j-1]) / 1²
            φ_xy = (φ[i+1,j+1] - φ[i+1,j-1] - φ[i-1,j+1] + φ[i-1,j-1]) / 4

        Notes:
            - Grid spacing h = 1 (unit voxel)
            - Uses 5-point stencil for 2nd order accuracy
            - Returns absolute value unless curvature_sign=True
        """
        # Extract values from 5x5 stencil (only 13 points needed for 2nd derivatives)
        I02 = matrix[0,2]  # i-2, j
        I11 = matrix[1,1]  # i-1, j-1
        I12 = matrix[1,2]  # i-1, j
        I13 = matrix[1,3]  # i-1, j+1
        I20 = matrix[2,0]  # i, j-2
        I21 = matrix[2,1]  # i, j-1
        I22 = matrix[2,2]  # i, j (center)
        I23 = matrix[2,3]  # i, j+1
        I24 = matrix[2,4]  # i, j+2
        I31 = matrix[3,1]  # i+1, j-1
        I32 = matrix[3,2]  # i+1, j
        I33 = matrix[3,3]  # i+1, j+1
        I42 = matrix[4,2]  # i+2, j

        # First derivatives (central differences, spacing h=1)
        phi_x = (I32 - I12) / 2  # ∂φ/∂x
        phi_y = (I23 - I21) / 2  # ∂φ/∂y

        # Intermediate first derivatives for second derivative calculation
        phi_x_at_i_minus = (I22 - I02) / 2  # ∂φ/∂x at i-1
        phi_x_at_i_plus = (I42 - I22) / 2   # ∂φ/∂x at i+1
        phi_y_at_j_minus = (I22 - I20) / 2  # ∂φ/∂y at j-1
        phi_y_at_j_plus = (I24 - I22) / 2   # ∂φ/∂y at j+1
        phi_xy_at_i_minus = (I13 - I11) / 2 # ∂φ/∂y at i-1
        phi_xy_at_i_plus = (I33 - I31) / 2  # ∂φ/∂y at i+1

        # Second derivatives (central differences of first derivatives)
        phi_xx = (phi_x_at_i_plus - phi_x_at_i_minus) / 2  # ∂²φ/∂x²
        phi_yy = (phi_y_at_j_plus - phi_y_at_j_minus) / 2  # ∂²φ/∂y²
        phi_xy = (phi_xy_at_i_plus - phi_xy_at_i_minus) / 2  # ∂²φ/∂x∂y

        # Check for zero gradient (flat region)
        grad_magnitude_sq = phi_x**2 + phi_y**2
        if grad_magnitude_sq == 0:
            return 0

        # Mean curvature formula: κ = (φ_x² φ_yy - 2φ_x φ_y φ_xy + φ_y² φ_xx) / (φ_x² + φ_y²)^(3/2)
        numerator = phi_x**2 * phi_yy - 2*phi_x*phi_y*phi_xy + phi_y**2 * phi_xx
        denominator = grad_magnitude_sq**1.5

        if self.curvature_sign:
            return -numerator / denominator  # Signed curvature
        else:
            return abs(numerator / denominator)  # Absolute curvature
    #%%
    # Core
    def linear_curvature_core(self,core_input):
        """Core function for curvature calculation.

        Implements linear smoothing and calculates curvature
        using second derivatives of smoothed data.

        Args:
            core_input: Input data for this subdomain

        Returns:
            tuple: Calculated curvature values and timing information
        """
        core_stime = datetime.datetime.now()
        li,lj,lk = np.shape(core_input)
        fval = np.zeros((self.nx,self.ny,1))

        corner1 = core_input[0,0,:]
        corner3 = core_input[li-1,lj-1,:]

        # Get core area and neighbors
        core_area_cen, core_area_nei = self.check_subdomain_and_nei(corner1)
        if self.verification_system:
            print(f'the processor {core_area_cen} start...')

        test_check_read_num = 0
        test_check_max_qsize = 0

        # Process each point in subdomain
        for core_a in core_input:
            for core_b in core_a:
                i = core_b[0]
                j = core_b[1]

                # Check if point is on grain boundary
                if myInput.is_grain_boundary(self.P, i, j, self.nx, self.ny):

                    window = self.find_window(i,j,self.tableL_curv - 2*self.clip)
                    smoothed_matrix = myInput.output_smoothed_matrix(window, myInput.output_linear_smoothing_matrix(self.loop_times))[self.loop_times:-self.loop_times,self.loop_times:-self.loop_times]

                    # Calculate curvature
                    fval[i,j,0] = self.calculate_curvature(smoothed_matrix)

        core_etime = datetime.datetime.now()
        if self.verification_system:
            print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval,(core_etime - core_stime).total_seconds())

    def linear_one_normal_vector_core(self, core_input):
        """Calculate normal vector for a single grain boundary site.

        Applies linear smoothing to a local window around one grain boundary
        point and computes the interface normal from smoothed gradients.

        Args:
            core_input (list): [i, j] coordinates of the site to process

        Returns:
            ndarray: Normal vector [nx, ny] components. Returns [0, 0] if
                    not on grain boundary.

        Notes:
            This function checks all 8 neighbors (including diagonals) to ensure
            site is truly on grain boundary before calculation.
        """
        i = core_input[0]
        j = core_input[1]
        # fv_i, fv_j = self.find_tableij(corner1,i,j)

        window = np.zeros((self.tableL,self.tableL))
        ip,im,jp,jm = myInput.periodic_bc(self.nx,self.ny,i,j)
        # Check if point is on grain boundary (including diagonal neighbors)
        if ( ((self.P[0,ip,j]-self.P[0,i,j])!=0) or
             ((self.P[0,im,j]-self.P[0,i,j])!=0) or
             ((self.P[0,i,jp]-self.P[0,i,j])!=0) or
             ((self.P[0,i,jm]-self.P[0,i,j])!=0) or
             ((self.P[0,ip,jp]-self.P[0,i,j])!=0) or
             ((self.P[0,im,jp]-self.P[0,i,j])!=0) or
             ((self.P[0,ip,jm]-self.P[0,i,j])!=0) or
             ((self.P[0,im,jm]-self.P[0,i,j])!=0) ):


            window = self.find_window(i,j,self.tableL - 2*self.clip)

        return np.array([-np.sum(window*self.smoothed_vector_i), np.sum(window*self.smoothed_vector_j)])

    def linear_normal_vector_core(self,core_input):
        """Core function for normal vector calculation.

        Implements linear smoothing and calculates interface normals
        using central differences.

        Args:
            core_input: Subset of points to process

        Returns:
            tuple: (Normal vector array, Computation time)
        """
        core_stime = datetime.datetime.now()
        li,lj,lk = np.shape(core_input)
        fval = np.zeros((self.nx,self.ny,2))

        corner1 = core_input[0,0,:]
        corner3 = core_input[li-1,lj-1,:]

        # Get core area and neighbors
        core_area_cen, core_area_nei = self.check_subdomain_and_nei(corner1)
        if self.verification_system:
            print(f'the processor {core_area_cen} start...')

        test_check_read_num = 0
        test_check_max_qsize = 0

        # Process each point in subdomain
        for core_a in core_input:
            for core_b in core_a:
                i = core_b[0]
                j = core_b[1]

                # Check if point is on grain boundary
                if myInput.is_grain_boundary(self.P, i, j, self.nx, self.ny):

                    window = self.find_window(i,j,self.tableL - 2*self.clip)
                    # print(window)

                    # Calculate normal vector components using smoothing matrices
                    fval[i,j,0] = -np.sum(window*self.smoothed_vector_i)
                    fval[i,j,1] = np.sum(window*self.smoothed_vector_j)

        core_etime = datetime.datetime.now()
        if self.verification_system:
            print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval,(core_etime - core_stime).total_seconds())

    def linear_main(self, purpose="inclination"):
        """Main execution function for linear smoothing algorithm.

        Controls the overall workflow including:
        - Parallel processing setup
        - Smoothing operations
        - Normal vector calculation
        - Error calculation

        Args:
            purpose (str): Type of calculation ("inclination" or "curvature")
        """
        self.run_main(self.linear_normal_vector_core, self.linear_curvature_core,
                      purpose=purpose, verbose=self.verification_system)
        if purpose == "inclination":
            self.get_errors()
        elif purpose == "curvature":
            self.get_curvature_errors()




if __name__ == '__main__':


    nx, ny = 50, 50
    ng = 2
    # cores = 8
    max_iteration = 5
    radius = 20
    filename_save = f"examples/curvature_calculation/BL_Curvature_R{radius}_Iteration_1_{max_iteration}"

    BL_errors =np.zeros(max_iteration)
    BL_runningTime = np.zeros(max_iteration)

    # P0,R=myInput.init2IC(nx, ny, ng, "PolyIC.init")
    P0,R=myInput.Circle_IC(nx,ny,radius)
    # P0,R=myInput.Voronoi_IC(nx,ny,ng)
    # P0,R=myInput.Complex2G_IC(nx,ny)
    # P0,R=myInput.Abnormal_IC(nx,ny)
    # P0[:,:,:]=myInput.SmallestGrain_IC(100,100)

    for cores in [1]:
        # loop_times=10
        for loop_times in range(4,max_iteration):


            test1 = linear_class(nx,ny,ng,cores,loop_times,P0,R)
            # test1.linear_main()
            # P = test1.get_P()

            test1.linear_main("curvature")
            C_ln = test1.get_C()


            #%%
            # test1.get_2d_plot('Poly','Bilinear')


            #%% error

            print('loop_times = ' + str(test1.loop_times))
            print('running_time = %.2f' % test1.running_time)
            print('running_core time = %.2f' % test1.running_coreTime)
            print('total_errors = %.2f' % test1.errors)
            print('per_errors = %.3f' % test1.errors_per_site)
            print()


            BL_errors[loop_times-1] = test1.errors_per_site
            BL_runningTime[loop_times-1] = test1.running_coreTime

    # np.savez(filename_save, BL_errors=BL_errors, BL_runningTime=BL_runningTime)
