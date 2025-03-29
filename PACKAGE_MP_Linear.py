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

Author: Lin Yang
"""

import os
current_path = os.getcwd()+'/'
import sys
sys.path.append(current_path)
import numpy as np
import math
import matplotlib.pyplot as plt
import myInput
import datetime
import multiprocessing as mp


class linear_class(object):
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
        # Runtime tracking
        self.running_time = 0  # Total execution time
        self.running_coreTime = 0  # Core calculation time
        self.errors = 0  # Accumulated angle errors
        self.errors_per_site = 0  # Average error per GB site
        self.clip = clip  # Boundary clipping

        # Grid parameters
        self.nx = nx  # Number of sites in x axis
        self.ny = ny  # Number of sites in y axis  
        self.ng = ng  # Number of grains
        self.R = R   # Reference normal vectors

        # Initialize result arrays
        self.P = np.zeros((3,nx,ny))  # Stores IC and normal vectors
        self.C = np.zeros((2,nx,ny))  # Stores curvature
        
        # Convert individual grain maps to single map
        if len(P0.shape) == 2:
            self.P[0,:,:] = np.array(P0)
            self.C[0,:,:] = np.array(P0)
        else:
            for i in range(0,np.shape(P0)[2]):
                self.P[0,:,:] += P0[:,:,i]*(i+1)
                self.C[0,:,:] += P0[:,:,i]*(i+1)

        # Parallel processing parameters
        self.cores = cores

        # Smoothing parameters
        self.loop_times = loop_times
        self.tableL = 2*(loop_times+1)+1  # Table length for repeated calcs
        self.tableL_curv = 2*(loop_times+2)+1
        self.halfL = loop_times+1

        # Get smoothing matrices
        self.smoothed_vector_i, self.smoothed_vector_j = myInput.output_linear_vector_matrix(
            self.loop_times, self.clip)
        self.verification_system = verification_system
        self.curvature_sign = curvature_sign

    def get_P(self):
        """Get the phase field and normal vector results.
        
        Returns:
            ndarray: Matrix containing grain structure and normal vectors. Has shape (3,nx,ny) where:
                    - P[0,:,:] contains grain IDs
                    - P[1:3,:,:] contains normal vector components
        """
        return self.P

    def get_C(self):
        """Get the curvature calculation results.
        
        Returns:
            ndarray: Matrix containing curvature values. Has shape (2,nx,ny) where:
                    - C[0,:,:] contains grain IDs
                    - C[1,:,:] contains curvature values
        """
        return self.C

    def get_errors(self):
        """Calculate error between calculated and reference normal vectors.
        
        Computes angular difference between calculated normal vectors and reference
        values at each grain boundary site.
        """
        ge_gbsites = self.get_gb_list()
        for gbSite in ge_gbsites:
            [gei,gej] = gbSite
            ge_dx,ge_dy = myInput.get_grad(self.P,gei,gej)
            # Calculate angle between vectors using dot product
            self.errors += math.acos(round(abs(ge_dx*self.R[gei,gej,0]+ge_dy*self.R[gei,gej,1]),5))

        if len(ge_gbsites) > 0:
            self.errors_per_site = self.errors/len(ge_gbsites)
        else:
            self.errors_per_site = 0

    def get_curvature_errors(self):
        """Calculate error between calculated and reference curvature values.
        
        For each grain boundary site, computes difference between calculated
        curvature and reference value.
        """
        gce_gbsites = self.get_gb_list()
        for gbSite in gce_gbsites:
            [gcei,gcej] = gbSite
            self.errors += abs(self.R[gcei,gcej,2] - self.C[1,gcei,gcej])

        if len(gce_gbsites)!=0:
            self.errors_per_site = self.errors/len(gce_gbsites)
        else:
            self.errors_per_site = 0

    def get_2d_plot(self,init,algo):
        """Generate 2D visualization of microstructure with normal vectors.
        
        Args:
            init (str): Name of initial condition
            algo (str): Name of algorithm used
        """
        plt.subplots_adjust(wspace=0.2,right=1.8)
        plt.close()
        fig1 = plt.figure(1)
        fig_page = self.loop_times
        plt.title(f'{algo}-{init} \n loop = '+str(fig_page))
        if fig_page < 10:
            String = '000'+str(fig_page)
        elif fig_page < 100:
            String = '00'+str(fig_page)
        elif fig_page < 1000:
            String = '0'+str(fig_page)
        elif fig_page < 10000:
            String = str(fig_page)
        plt.imshow(self.P[0,:,:], cmap='gray', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        plt.savefig('BL_PolyGray_noArrows.png',dpi=1000,bbox_inches='tight')

        g2p_gbsites = self.get_gb_list()
        for gbSite in g2p_gbsites:
            [g2pi,g2pj] = gbSite
            g2p_dx,g2p_dy = myInput.get_grad(self.P,g2pi,g2pj)
            if g2pi >200 and g2pi<500:
                plt.arrow(g2pj,g2pi,30*g2p_dx,30*g2p_dy,width=0.1,lw=0.1,alpha=0.8,color='navy')

        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('BL_PolyGray_Arrows.png',dpi=1000,bbox_inches='tight')

    def get_gb_list(self,grainID=1):
        """Get list of grain boundary sites.
        
        Args:
            grainID (int): ID of grain to find boundaries for
            
        Returns:
            list: List of [i,j] coordinates of boundary sites
        """
        ggn_gbsites = []
        for i in range(0,self.nx):
            for j in range(0,self.ny):
                ip,im,jp,jm = myInput.periodic_bc(self.nx,self.ny,i,j)
                # Check if any neighbors have different grain ID
                if (((self.P[0,ip,j]-self.P[0,i,j])!=0) or
                    ((self.P[0,im,j]-self.P[0,i,j])!=0) or
                    ((self.P[0,i,jp]-self.P[0,i,j])!=0) or
                    ((self.P[0,i,jm]-self.P[0,i,j])!=0)) and self.P[0,i,j]==grainID:
                    ggn_gbsites.append([i,j])
        return ggn_gbsites

    def get_all_gb_list(self):
        gagn_gbsites = [[] for _ in range(int(self.ng))]
        for i in range(0,self.nx):
            for j in range(0,self.ny):
                ip,im,jp,jm = myInput.periodic_bc(self.nx,self.ny,i,j)
                if ( ((self.P[0,ip,j]-self.P[0,i,j])!=0) or
                     ((self.P[0,im,j]-self.P[0,i,j])!=0) or
                     ((self.P[0,i,jp]-self.P[0,i,j])!=0) or
                     ((self.P[0,i,jm]-self.P[0,i,j])!=0) ):
                    gagn_gbsites[int(self.P[0,i,j]-1)].append([i,j])
        return gagn_gbsites

    def check_subdomain_and_nei(self,A):
        ca_length,ca_width = myInput.split_cores(self.cores)
        ca_area_cen = [int(A[0]/self.nx*ca_width),int(A[1]/self.ny*ca_length)]
        ca_area_nei = []
        ca_area_nei.append( [int((ca_area_cen[0]-1)%ca_width), int((ca_area_cen[1]-1)%ca_length)] )
        ca_area_nei.append( [int((ca_area_cen[0]-1)%ca_width), int(ca_area_cen[1])] )
        ca_area_nei.append( [int((ca_area_cen[0]-1)%ca_width), int((ca_area_cen[1]+1)%ca_length)] )
        ca_area_nei.append( [int(ca_area_cen[0]), int((ca_area_cen[1]+1)%ca_length)] )
        ca_area_nei.append( [int((ca_area_cen[0]+1)%ca_width), int((ca_area_cen[1]+1)%ca_length)] )
        ca_area_nei.append( [int((ca_area_cen[0]+1)%ca_width), int(ca_area_cen[1])] )
        ca_area_nei.append( [int((ca_area_cen[0]+1)%ca_width), int((ca_area_cen[1]-1)%ca_length)] )
        ca_area_nei.append( [int(ca_area_cen[0]), int((ca_area_cen[1]-1)%ca_length)] )

        return ca_area_cen, ca_area_nei

    def find_window(self,i,j,fw_len):
        """Find the window matrix around a point for smoothing calculations.
        
        Args:
            i (int): x coordinate of center point
            j (int): y coordinate of center point
            fw_len (int): Width/height of window
            
        Returns:
            ndarray: Binary window matrix indicating grain membership
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

    def calculate_curvature(self,matrix):
        """Calculate curvature from normal vectors.
        
        Args:
            matrix (ndarray): Normal vector field
            
        Returns:
            float: Local curvature value
        """
        I02 = matrix[0,2]
        I11 = matrix[1,1]
        I12 = matrix[1,2]
        I13 = matrix[1,3]
        I20 = matrix[2,0]
        I21 = matrix[2,1]
        I22 = matrix[2,2]
        I23 = matrix[2,3]
        I24 = matrix[2,4]
        I31 = matrix[3,1]
        I32 = matrix[3,2]
        I33 = matrix[3,3]
        I42 = matrix[4,2]

        # calculate the improve or decrease of each site
        Ii = (I32-I12)/2 #
        Ij = (I23-I21)/2 #

        Imi = (I22-I02)/2 #
        Ipi = (I42-I22)/2 #
        Imj = (I22-I20)/2 #
        Ipj = (I24-I22)/2 #
        Imij = (I13-I11)/2 #
        Ipij = (I33-I31)/2 #

        Iii = (Ipi-Imi)/2 #
        Ijj = (Ipj-Imj)/2 #
        Iij = (Ipij-Imij)/2 #

        if (Ii**2 + Ij**2) == 0:
            return 0

        if self.curvature_sign:
            return -(Ii**2 * Ijj - 2*Ii*Ij*Iij + Ij**2 * Iii) / (Ii**2 + Ij**2)**1.5
        else:
            return abs(Ii**2 * Ijj - 2*Ii*Ij*Iij + Ij**2 * Iii) / (Ii**2 + Ij**2)**1.5
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
                ip,im,jp,jm = myInput.periodic_bc(self.nx,self.ny,i,j)
                if (((self.P[0,ip,j]-self.P[0,i,j])!=0) or
                    ((self.P[0,im,j]-self.P[0,i,j])!=0) or
                    ((self.P[0,i,jp]-self.P[0,i,j])!=0) or
                    ((self.P[0,i,jm]-self.P[0,i,j])!=0)):

                    window = self.find_window(i,j,self.tableL_curv - 2*self.clip)
                    smoothed_matrix = myInput.output_smoothed_matrix(window, myInput.output_linear_smoothing_matrix(self.loop_times))[self.loop_times:-self.loop_times,self.loop_times:-self.loop_times]
                    
                    # Calculate curvature
                    fval[i,j,0] = self.calculate_curvature(smoothed_matrix)

        core_etime = datetime.datetime.now()
        if self.verification_system:
            print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval,(core_etime - core_stime).total_seconds())

    def linear_one_normal_vector_core(self,core_input):
        # Return the normal vector of one specific pixel:
        i = core_input[0]
        j = core_input[1]
        # fv_i, fv_j = self.find_tableij(corner1,i,j)

        window = np.zeros((self.tableL,self.tableL))
        ip,im,jp,jm = myInput.periodic_bc(self.nx,self.ny,i,j)
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
                ip,im,jp,jm = myInput.periodic_bc(self.nx,self.ny,i,j)
                if (((self.P[0,ip,j]-self.P[0,i,j])!=0) or
                    ((self.P[0,im,j]-self.P[0,i,j])!=0) or
                    ((self.P[0,i,jp]-self.P[0,i,j])!=0) or
                    ((self.P[0,i,jm]-self.P[0,i,j])!=0)):

                    window = self.find_window(i,j,self.tableL - 2*self.clip)
                    # print(window)
                    
                    # Calculate normal vector components using smoothing matrices
                    fval[i,j,0] = -np.sum(window*self.smoothed_vector_i)
                    fval[i,j,1] = np.sum(window*self.smoothed_vector_j)

        core_etime = datetime.datetime.now()
        if self.verification_system:
            print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval,(core_etime - core_stime).total_seconds())

    def res_back(self,back_result):
        res_stime = datetime.datetime.now()
        (fval,core_time) = back_result
        if core_time > self.running_coreTime:
            self.running_coreTime = core_time

        if self.verification_system == True: print("res_back start...")
        if fval.shape[2] == 1:
            self.C[1,:,:] += fval[:,:,0]
        elif fval.shape[2] == 2:
            self.P[1,:,:] += fval[:,:,0]
            self.P[2,:,:] += fval[:,:,1]
        res_etime = datetime.datetime.now()
        if self.verification_system == True: print("my res time is " + str((res_etime - res_stime).total_seconds()))

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
        starttime = datetime.datetime.now()

        # Setup parallel processing
        pool = mp.Pool(processes=self.cores)
        main_lc, main_wc = myInput.split_cores(self.cores)

        # Split domain for parallel processing
        all_sites = np.array([[x,y] for x in range(self.nx) for y in range(self.ny)]).reshape(self.nx,self.ny,2)
        multi_input = myInput.split_IC(all_sites, self.cores,2, 0,1)

        # Run calculations in parallel
        res_list = []
        if purpose == "inclination":
            for ki in range(main_wc):
                for kj in range(main_lc):
                    res_one = pool.apply_async(
                        func=self.linear_normal_vector_core, 
                        args=(multi_input[ki][kj],),
                        callback=self.res_back)
                    res_list.append(res_one)
        elif purpose == "curvature":
            for ki in range(main_wc):
                for kj in range(main_lc):
                    res_one = pool.apply_async(
                        func=self.linear_curvature_core,
                        args=(multi_input[ki][kj],),
                        callback=self.res_back)
                    res_list.append(res_one)

        # Wait for all processes to complete
        pool.close()
        pool.join()

        if self.verification_system:
            print("core done!")

        # Calculate timing and errors
        endtime = datetime.datetime.now()
        self.running_time = (endtime - starttime).total_seconds()
        
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
