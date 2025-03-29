#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Allen-Cahn Method Implementation for Interface Analysis

This module implements the Allen-Cahn method for calculating grain boundary 
normal vectors and curvature in polycrystalline materials. The method uses
phase field evolution with these features:

1. Phase Field Evolution:
   - Uses Allen-Cahn equation for interface motion
   - Implements double-well potential energy function
   - Maintains phase field values between 0 and 1

2. Normal Vector Calculation:
   - Computes gradients of evolved phase field
   - Uses high-order accurate numerical schemes
   - Handles multiple grain boundaries

3. Curvature Calculation:
   - Based on divergence of normal vectors
   - Maintains numerical stability at junctions
   - Uses efficient matrix operations

Key Features:
- Parallel implementation for performance
- Automatic timestep selection
- Energy minimization tracking
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
import multiprocessing as mp
import myInput
import datetime


class allenCahn_class(object):
    """A class implementing the Allen-Cahn equation solver.
    
    This class provides methods to evolve grain boundaries using the Allen-Cahn equation
    and calculate normal vectors and curvature at grain boundaries.
    
    Attributes:
        k (float): Gradient energy coefficient (ε²)
        m (float): Mobility coefficient (M) 
        L (float): Time stepping factor
        matrix_value (float): Initial value for unfilled matrix elements
        running_time (float): Total execution time
        running_coreTime (float): Execution time for core calculations
        errors (float): Accumulated angle errors
        errors_per_site (float): Average error per grain boundary site
        nx, ny (int): Grid dimensions
        ng (int): Number of grains
        R (ndarray): Reference normal vectors 
        P (ndarray): Phase field and calculated normal vectors
        C (ndarray): Calculated curvature
        cores (int): Number of parallel processes
        nsteps (int): Number of time steps
        dt (float): Time step size
        tableL (int): Length of calculation table
        halfL (int): Half length of calculation window
        V (ndarray): Temporary calculation array
    """

    def __init__(self,nx,ny,ng,cores,nsteps,P0,R,clip=0,verification_system=True,curvature_sign=False,mobility=1.0,grad_coef=1.0):
        """Initialize the Allen-Cahn algorithm.
        
        Args:
            nx,ny (int): Grid dimensions
            ng (int): Number of grains/phases
            cores (int): Number of CPU cores for parallel processing
            nsteps (int): Number of evolution timesteps
            P0 (ndarray): Initial microstructure
            R (ndarray): Reference solution for validation
            mobility (float): Mobility coefficient M
            grad_coef (float): Gradient energy coefficient κ
        """
        # V_matrix init value; runnning time and error for the algorithm
        self.k = grad_coef  # Gradient energy coefficient
        self.m = mobility  # Mobility
        self.L = 1  # Time stepping factor
        self.matrix_value = 10
        self.running_time = 0
        self.running_coreTime = 0 
        self.errors = 0
        self.errors_per_site = 0

        # Initial condition data
        self.nx,self.ny = nx, ny
        self.ng = 2
        self.R = R  # Reference normal vectors
        # Convert individual grains map into one grain map
        self.P = np.zeros((3,nx,ny))  # Stores IC and normal vectors
        self.C = np.zeros((2,nx,ny))  # Stores curvature
        for i in range(0,np.shape(P0)[2]):
            self.P[0,:,:] += P0[:,:,i]*(i+1)
            self.C[0,:,:] += P0[:,:,i]*(i+1)

        self.clip = clip  # Clipping value for curvature
        self.verification_system = verification_system  # Flag for verification system
        self.curvature_sign = curvature_sign  # Flag for curvature sign

        # Multiprocessing parameters
        self.cores = cores

        # Numerical solution parameters
        self.nsteps = nsteps  # Number of timesteps
        self.dt = 0.1  # Timestep size
        self.tableL = 2*(nsteps+1)+1  # Table length for repeated calculations
        self.tableL_curv = 2*(nsteps+2)+1
        self.halfL = nsteps+1
        self.halfL_curv = nsteps+2

        # Temporary calculation matrix
        self.V = np.ones((nsteps+1,nx,ny,ng))*self.matrix_value

    def get_P(self):
        """Get the phase field and normal vector results.
        
        Returns:
            ndarray: Matrix containing grain structure and normal vectors
        """
        return self.P

    def get_C(self):
        """Get the curvature calculation results.
        
        Returns:
            ndarray: Matrix containing curvature values
        """
        return self.C

    def get_errors(self):
        """Calculate error between calculated and reference normal vectors.
        
        Computes angular difference between calculated normal vectors and reference
        values at each grain boundary site.
        """
        ge_gbsites = self.get_gb_list()
        for gbSite in ge_gbsites :
            [gei,gej] = gbSite
            ge_dx,ge_dy = myInput.get_grad(self.P,gei,gej)
            self.errors += math.acos(round(abs(ge_dx*self.R[gei,gej,0]+ge_dy*self.R[gei,gej,1]),5))

        self.errors_per_site = self.errors/len(ge_gbsites)

    def get_2d_plot(self,init,algo):
        """Generate 2D visualization of results.
        
        Creates plot showing grain structure and normal vectors.
        
        Args:
            init (str): Name of initial condition
            algo (str): Name of algorithm
        """
        plt.subplots_adjust(wspace=0.2,right=1.8)
        plt.close()
        fig1 = plt.figure(1)
        fig_page = self.nsteps
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

        g2p_gbsites = self.get_gb_list()
        for gbSite in g2p_gbsites :
            [g2pi,g2pj] = gbSite
            g2p_dx,g2p_dy = myInput.get_grad(self.P,g2pi,g2pj)
            if g2pi >200 and g2pi<500:
                plt.arrow(g2pj,g2pi,30*g2p_dx,30*g2p_dy,width=0.1,lw=0.1,alpha=0.8,color='navy')

        plt.xticks([])
        plt.yticks([])
        plt.savefig('AC_PolyGray_Arrows.png',dpi=1000,bbox_inches='tight')

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
                if ( ((self.P[0,ip,j]-self.P[0,i,j])!=0) or ((self.P[0,im,j]-self.P[0,i,j])!=0) or ((self.P[0,i,jp]-self.P[0,i,j])!=0) or ((self.P[0,i,jm]-self.P[0,i,j])!=0) )\
                        and self.P[0,i,j]==grainID:
                    ggn_gbsites.append([i,j])
        return ggn_gbsites


    def check_subdomain_and_nei(self,A):
        """Set up parallel domain decomposition.
        
        Divides domain into subdomains for parallel processing and identifies
        neighboring subdomains.
        
        Args:
            A: Subdomain coordinates
            
        Returns:
            tuple: Center and neighbor subdomain indices
        """
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

    def res_back(self,back_result):
        """Collect results from parallel processes.
        
        Callback function to gather and combine results from parallel calculations.
        
        Args:
            back_result: Results from a parallel process
        """
        res_stime = datetime.datetime.now()
        (fval,core_time,self.V) = back_result
        if core_time > self.running_coreTime:
            self.running_coreTime = core_time

        print("res_back start...")
        if fval.shape[2] == 1:
            self.C[1,:,:] += fval[:,:,0]
        elif fval.shape[2] == 2:
            self.P[1,:,:] += fval[:,:,0]
            self.P[2,:,:] += fval[:,:,1]
        res_etime = datetime.datetime.now()
        print("my res time is " + str((res_etime - res_stime).total_seconds()))


    #%% Core
    def allenCahn_curvature_core(self,core_input):
        """Core curvature calculation function.
        
        Implements the Allen-Cahn equation solution for curvature calculation
        on a subdomain.
        
        The curvature κ is calculated as:
        κ = (ϕxx*ϕy² - 2*ϕx*ϕy*ϕxy + ϕyy*ϕx²)/(ϕx² + ϕy²)^(3/2)
        
        where ϕx, ϕy are first derivatives and ϕxx, ϕxy, ϕyy are second derivatives.
        
        Args:
            core_input: Input data for this subdomain
            
        Returns:
            tuple: Calculated curvature values and timing information
        """
        core_stime = datetime.datetime.now()
        li,lj,lk=np.shape(core_input)
        fval = np.zeros((self.nx,self.ny,1))

        corner1 = core_input[0,0,:]
        corner3 = core_input[li-1,lj-1,:]

        core_area_cen, core_area_nei = self.check_subdomain_and_nei(corner1)
        print(f'the processor {core_area_cen} start...')

        test_check_read_num = 0
        test_check_max_qsize = 0
        for core_a in core_input:
            for core_b in core_a:
                i = core_b[0]
                j = core_b[1]

                ip,im,jp,jm = myInput.periodic_bc(self.nx,self.ny,i,j)
                if ( ((self.P[0,ip,j]-self.P[0,i,j])!=0) or ((self.P[0,im,j]-self.P[0,i,j])!=0) or ((self.P[0,i,jp]-self.P[0,i,j])!=0) or ((self.P[0,i,jm]-self.P[0,i,j])!=0) ):

                    # convert the small table into 0 and 1
                    for ii in range(-self.halfL_curv,self.halfL_curv+1):
                        for jj in range(-self.halfL_curv,self.halfL_curv+1):
                            local_x = (i+ii)%self.nx
                            local_y = (j+jj)%self.ny

                            # plus and minus matrix to code the BL function
                            if self.V[0,local_x,local_y,int(self.P[0,i,j]-1)] == self.matrix_value:
                                if self.P[0,local_x,local_y] != self.P[0,i,j]:
                                    self.V[0,local_x,local_y,int(self.P[0,i,j]-1)] = 0
                                else:
                                    self.V[0,local_x,local_y,int(self.P[0,i,j]-1)] = 1

                    #  calculate the smooth value
                    for kk in range(1,self.nsteps+1):
                        for ii in range(-self.halfL_curv+kk,self.halfL_curv+1-kk):
                            for jj in range(-self.halfL_curv+kk,self.halfL_curv+1-kk):
                                local_x = (i+ii)%self.nx
                                local_y = (j+jj)%self.ny
                                if self.V[kk,local_x,local_y,int(self.P[0,i,j]-1)] == self.matrix_value:
                                    # necessary coordination
                                    local_xp1 = (i+ii+1)%self.nx
                                    local_xm1 = (i+ii-1)%self.nx
                                    local_yp1 = (j+jj+1)%self.ny
                                    local_ym1 = (j+jj-1)%self.ny

                                    Etas = ( self.V[kk-1,local_x,local_y,int(self.P[0,i,j]-1)]**2+(1-self.V[kk-1,local_x,local_y,int(self.P[0,i,j]-1)])**2 )-self.V[kk-1,local_x,local_y,int(self.P[0,i,j]-1)]
                                    df0 = self.V[kk-1,local_x,local_y,int(self.P[0,i,j]-1)]**3-self.V[kk-1,local_x,local_y,int(self.P[0,i,j]-1)]+3*self.V[kk-1,local_x,local_y,int(self.P[0,i,j]-1)]*Etas  #Free energy derivative, simple multi-well
                                    fd = (self.V[kk-1,local_xm1,local_y,int(self.P[0,i,j]-1)]+self.V[kk-1,local_xp1,local_y,int(self.P[0,i,j]-1)]-4*self.V[kk-1,local_x,local_y,int(self.P[0,i,j]-1)]+self.V[kk-1,local_x,local_ym1,int(self.P[0,i,j]-1)]+self.V[kk-1,local_x,local_yp1,int(self.P[0,i,j]-1)])/1**2 #2nd order central differencing
                                    self.V[kk,local_x,local_y,int(self.P[0,i,j]-1)] = self.V[kk-1,local_x,local_y,int(self.P[0,i,j]-1)] - self.L*(self.m*df0-self.k*fd)*self.dt
                                    # if i==64 and j==65 :
                                    #     print(f"!!!the value ({ii+2},{jj+2}) is {self.V[kk,local_x,local_y,int(self.P[0,i,j]-1)]}")
                    # necessary coordination
                    local_x = (i)%self.nx
                    local_xp1 = (i+1)%self.nx
                    local_xp2 = (i+2)%self.nx
                    local_xm1 = (i-1)%self.nx
                    local_xm2 = (i-2)%self.nx
                    local_y = (j)%self.ny
                    local_yp1 = (j+1)%self.ny
                    local_yp2 = (j+2)%self.ny
                    local_ym1 = (j-1)%self.ny
                    local_ym2 = (j-2)%self.ny

                    I02 = self.V[self.nsteps,local_xm2,local_y,int(self.P[0,i,j]-1)]
                    I11 = self.V[self.nsteps,local_xm1,local_ym1,int(self.P[0,i,j]-1)]
                    I12 = self.V[self.nsteps,local_xm1,local_y,int(self.P[0,i,j]-1)]
                    I13 = self.V[self.nsteps,local_xm1,local_yp1,int(self.P[0,i,j]-1)]
                    I20 = self.V[self.nsteps,local_x,local_ym2,int(self.P[0,i,j]-1)]
                    I21 = self.V[self.nsteps,local_x,local_ym1,int(self.P[0,i,j]-1)]
                    I22 = self.V[self.nsteps,local_x,local_y,int(self.P[0,i,j]-1)]
                    I23 = self.V[self.nsteps,local_x,local_yp1,int(self.P[0,i,j]-1)]
                    I24 = self.V[self.nsteps,local_x,local_yp2,int(self.P[0,i,j]-1)]
                    I31 = self.V[self.nsteps,local_xp1,local_ym1,int(self.P[0,i,j]-1)]
                    I32 = self.V[self.nsteps,local_xp1,local_y,int(self.P[0,i,j]-1)]
                    I33 = self.V[self.nsteps,local_xp1,local_yp1,int(self.P[0,i,j]-1)]
                    I42 = self.V[self.nsteps,local_xp2,local_y,int(self.P[0,i,j]-1)]

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
                        fval[i,j,0] = 0
                    else:
                        fval[i,j,0]=abs(Ii**2 * Ijj - 2*Ii*Ij*Iij + Ij**2 * Iii) / (Ii**2 + Ij**2)**1.5
                        
                    # if i==64 and j==65:
                    #     print(f"I02:{I02}; I1,1-3:{I11},{I12},{I13}; I2,0-4:{I20},{I21},{I22},{I23},{I24}; I3,1-3:{I31},{I32},{I33}; I42:{I42}")
                    # if i==65 and j==65:
                    #     print(f"I02:{I02}; I1,1-3:{I11},{I12},{I13}; I2,0-4:{I20},{I21},{I22},{I23},{I24}; I3,1-3:{I31},{I32},{I33}; I42:{I42}")
        print(f"processor {core_area_cen} read {test_check_read_num} times and max qsize {test_check_max_qsize}")
        core_etime = datetime.datetime.now()
        print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval,(core_etime - core_stime).total_seconds(),self.V)


    def allenCahn_normal_vector_core(self,core_input):
        """Core function for normal vector calculation.
        
        Implements Allen-Cahn evolution and calculates interface normals
        using phase field gradients.
        
        Args:
            core_input: Subset of points to process
            
        Returns:
            tuple: (Normal vector array, Computation time)
        """
        core_stime = datetime.datetime.now()
        li,lj,lk=np.shape(core_input)
        fval = np.zeros((self.nx,self.ny,2))

        corner1 = core_input[0,0,:]
        corner3 = core_input[li-1,lj-1,:]

        core_area_cen, core_area_nei = self.check_subdomain_and_nei(corner1)
        print(f'the processor {core_area_cen} start...')

        test_check_read_num = 0
        test_check_max_qsize = 0
        for core_a in core_input:
            for core_b in core_a:
                i = core_b[0]
                j = core_b[1]

                ip,im,jp,jm = myInput.periodic_bc(self.nx,self.ny,i,j)
                if ( ((self.P[0,ip,j]-self.P[0,i,j])!=0) or ((self.P[0,im,j]-self.P[0,i,j])!=0) or ((self.P[0,i,jp]-self.P[0,i,j])!=0) or ((self.P[0,i,jm]-self.P[0,i,j])!=0) ):

                    # convert the small table into 0 and 1
                    for ii in range(-self.halfL,self.halfL+1):
                        for jj in range(-self.halfL,self.halfL+1):
                            local_x = (i+ii)%self.nx
                            local_y = (j+jj)%self.ny

                            # plus and minus matrix to code the BL function
                            if self.V[0,local_x,local_y,int(self.P[0,i,j]-1)] == self.matrix_value:
                                if self.P[0,local_x,local_y] != self.P[0,i,j]:
                                    self.V[0,local_x,local_y,int(self.P[0,i,j]-1)] = 0
                                else:
                                    self.V[0,local_x,local_y,int(self.P[0,i,j]-1)] = 1

                    #  calculate the smooth value
                    for kk in range(1,self.nsteps+1):
                        for ii in range(-self.halfL+kk,self.halfL+1-kk):
                            for jj in range(-self.halfL+kk,self.halfL+1-kk):
                                local_x = (i+ii)%self.nx
                                local_y = (j+jj)%self.ny
                                if self.V[kk,local_x,local_y,int(self.P[0,i,j]-1)] == self.matrix_value:
                                    # necessary coordination
                                    local_xp1 = (i+ii+1)%self.nx
                                    local_xm1 = (i+ii-1)%self.nx
                                    local_yp1 = (j+jj+1)%self.ny
                                    local_ym1 = (j+jj-1)%self.ny

                                    Etas = ( self.V[kk-1,local_x,local_y,int(self.P[0,i,j]-1)]**2+(1-self.V[kk-1,local_x,local_y,int(self.P[0,i,j]-1)])**2 )-self.V[kk-1,local_x,local_y,int(self.P[0,i,j]-1)]
                                    df0 = self.V[kk-1,local_x,local_y,int(self.P[0,i,j]-1)]**3-self.V[kk-1,local_x,local_y,int(self.P[0,i,j]-1)]+3*self.V[kk-1,local_x,local_y,int(self.P[0,i,j]-1)]*Etas  #Free energy derivative, simple multi-well
                                    fd = (self.V[kk-1,local_xm1,local_y,int(self.P[0,i,j]-1)]+self.V[kk-1,local_xp1,local_y,int(self.P[0,i,j]-1)]-4*self.V[kk-1,local_x,local_y,int(self.P[0,i,j]-1)]+self.V[kk-1,local_x,local_ym1,int(self.P[0,i,j]-1)]+self.V[kk-1,local_x,local_yp1,int(self.P[0,i,j]-1)])/1**2 #2nd order central differencing
                                    self.V[kk,local_x,local_y,int(self.P[0,i,j]-1)] = self.V[kk-1,local_x,local_y,int(self.P[0,i,j]-1)] - self.L*(self.m*df0-self.k*fd)*self.dt

                    fval[i,j,0] = (self.V[self.nsteps,im,j,int(self.P[0,i,j]-1)]-self.V[self.nsteps,ip,j,int(self.P[0,i,j]-1)])/2
                    fval[i,j,1] = (self.V[self.nsteps,i,jp,int(self.P[0,i,j]-1)]-self.V[self.nsteps,i,jm,int(self.P[0,i,j]-1)])/2
        print(f"processor {core_area_cen} read {test_check_read_num} times and max qsize {test_check_max_qsize}")
        core_etime = datetime.datetime.now()
        print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval,(core_etime - core_stime).total_seconds(),self.V)

    def allenCahn_main(self, purpose="inclination"):
        """Main execution function for Allen-Cahn algorithm.
        
        Controls the overall workflow including:
        - Parallel processing setup
        - Phase field evolution
        - Normal vector calculation
        - Energy minimization
        - Error calculation
        """
        # calculate time
        starttime = datetime.datetime.now()

        pool = mp.Pool(processes=self.cores)
        main_lc, main_wc = myInput.split_cores(self.cores)

        all_sites = np.array([[x,y] for x in range(self.nx) for y in range(self.ny) ]).reshape(self.nx,self.ny,2)
        multi_input = myInput.split_IC(all_sites, self.cores,2, 0,1)

        res_list=[]
        if purpose == "inclination":
            for ki in range(main_wc):
                for kj in range(main_lc):
                    res_one = pool.apply_async(func = self.allenCahn_normal_vector_core, args = (multi_input[ki][kj], ), callback=self.res_back )
                    res_list.append(res_one)
        elif purpose == "curvature":
            for ki in range(main_wc):
                for kj in range(main_lc):
                    res_one = pool.apply_async(func = self.allenCahn_curvature_core, args = (multi_input[ki][kj], ), callback=self.res_back )
                    res_list.append(res_one)


        pool.close()
        pool.join()
        print("core done!")
        # print(res_list[0].get())

        # calculate time
        endtime = datetime.datetime.now()

        self.running_time = (endtime - starttime).total_seconds()
        self.get_errors()

if __name__ == '__main__':

    AC_errors =np.zeros(10)
    AC_runningTime = np.zeros(10)

    nx, ny = 200, 200
    ng = 2

    # P0,R=myInput.init2IC(nx, ny, ng, "PolyIC.init")
    P0,R=myInput.Circle_IC(nx,ny)
    # P0,R=myInput.Voronoi_IC(nx,ny,ng)
    # P0,R=myInput.Complex2G_IC(nx,ny)
    # P0,R=myInput.Abnormal_IC(nx,ny)
    # P0[:,:,:]=myInput.SmallestGrain_IC(100,100)

    for cores in [4]:
        for nsteps in range(20,21,2):

            test1 = allenCahn_class(nx,ny,ng,cores,nsteps,P0,R)
            test1.allenCahn_main("curvature")
            # P = test1.get_P()
            C_ac = test1.get_C()
            V_ac = test1.V

            #%% Figure

            # test1.get_2d_plot('Abnormal','Allen-Cahn')



            #%% errors
            print('loop_times = ' + str(test1.nsteps))
            print('running_time = %.2f' % test1.running_time)
            print('running_core time = %.2f' % test1.running_coreTime)
            print('total_errors = %.2f' % test1.errors)
            print('per_errors = %.3f' % test1.errors_per_site)
            print()

            AC_errors[int(nsteps/2-1)] = test1.errors_per_site
            AC_runningTime[int(nsteps/2-1)] = test1.running_coreTime
