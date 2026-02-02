#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Level Set Method Implementation for Interface Analysis

This module implements the level set method for calculating grain boundary
normal vectors and curvature in polycrystalline materials. The method uses
signed distance functions with these features:

1. Level Set Function:
   - Represents interfaces as zero contours
   - Maintains signed distance property
   - Implements efficient reinitialization

2. Normal Vector Calculation:
   - Uses spatial derivatives of level set function
   - Provides accurate interface normals
   - Handles multiple grain boundaries

3. Curvature Calculation:
   - Based on level set derivatives
   - Maintains numerical stability
   - Handles complex junctions

Key Features:
- Parallel implementation for performance
- Automatic reinitialization
- Memory efficient operations
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
from PACKAGE_MP_Base2D import Base2D

class levelSet_class(Base2D):
    """Level set method implementation for interface analysis.

    This class implements the level set method to calculate normal vectors and
    curvature at grain boundaries. The method uses an implicit representation
    of interfaces through a signed distance function.

    Key attributes:
        P: Phase field array storing microstructure and normal vectors
        C: Array storing calculated curvature values
        V: Level set function values
        nsteps: Number of time steps for evolution
        dt: Time step size for evolution
    """

    def __init__(self,nx,ny,ng,cores,nsteps,P0,R,clip=0,verification_system=True,curvature_sign=False):
        """Initialize the level set algorithm.

        Args:
            nx,ny (int): Grid dimensions
            ng (int): Number of grains
            cores (int): Number of CPU cores for parallel processing
            nsteps (int): Number of evolution timesteps
            P0 (ndarray): Initial microstructure
            R (ndarray): Reference solution for validation
        """
        super().__init__(nx, ny, ng, cores, P0, R, clip, verification_system, curvature_sign)

        # Level set specific parameters
        self.matrix_value = 10  # Initial value for level set function
        self.nsteps = nsteps  # Number of evolution time steps
        self.dt = 1  # Time step size
        self.tableL = 2*(2*nsteps+1)+1  # Window size for normal vectors
        self.tableL_curv = 2*(2*nsteps+2)+1  # Window size for curvature
        self.halfL = 2*nsteps+1  # Half window size for normal vectors
        self.halfL_curv = 2*nsteps+2  # Half window size for curvature

        # Initialize level set function
        self.V = np.ones((nsteps+1,nx,ny))*self.matrix_value
        self.myTable = np.ones((nx,ny))

    def res_back(self, back_result):
        self.res_back_with_V(back_result)

    def get_2d_plot(self, init, algo):
        super().get_2d_plot(init, algo, self.nsteps, arrow_scale=40, cmap='gray',
                            save_prefix='LS', filter_range=(200, 500))
        plt.savefig('LS_PolyGray_Arrows.png', dpi=1000, bbox_inches='tight')

    def Neighbors(self,arr,x,y,n):
        ''' Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]'''
        arr=np.roll(np.roll(arr,shift=-x+math.floor(n/2),axis=0),shift=-y+math.floor(n/2),axis=1)
        return arr[:n,:n]

    def find_distance(self,i,j,d): # let d=2
        """Calculate signed distance from point to interface.

        Uses scanning algorithm to find closest interface point.

        Args:
            i,j (int): Current point coordinates
            d (int): Search radius

        Returns:
            float: Signed distance to nearest interface
        """
        smallTable = self.Neighbors(self.P[0,:,:],i,j,d*2+1)

        # two layer list
        layerList = [[d-1,d], [d+1,d], [d,d-1], [d,d+1],
                     [d-1,d-1], [d-1,d+1], [d+1,d-1], [d+1,d+1],
                     [d,d-2], [d,d+2], [d-2,d], [d+2,d],
                     [d+1,d-2], [d-1,d-2], [d-2,d-1], [d-2,d+1], [d-1,d+2], [d+1,d+2], [d+2,d+1], [d+2,d-1],
                     [d-2,d-2], [d-2,d+2], [d+2,d+2], [d+2,d-2],
                     [d-3,d], [d+3,d], [d,d-3], [d,d+3],
                     [d+1,d-3], [d-1,d-3], [d-3,d-1], [d-3,d+1], [d-1,d+3], [d+1,d+3], [d+3,d+1], [d+3,d-1],
                     [d+2,d-3], [d-2,d-3], [d-3,d-2], [d-3,d+2], [d-2,d+3], [d+2,d+3], [d+3,d+2], [d+3,d-2],
                     [d-3,d-3], [d+3,d-3], [d+3,d+3], [d-3,d+3]
                     ]

        for k in layerList[:(2*d+1)**2-1]:
            if smallTable[k[0],k[1]] != smallTable[d,d]:
                return math.sqrt((d-k[0])**2 + (d-k[1])**2)

        return d*math.sqrt(2)

    #%% Smooth Core Site-based Stored data

    def levelSet_curvature_core(self,core_input):
        """Core curvature calculation function using level set method.

        Implements level set evolution and curvature calculation on a subdomain.
        Uses higher-order accurate derivatives to compute mean curvature.

        Args:
            core_input: Input data for this subdomain

        Returns:
            tuple: Calculated curvature values, timing and level set function
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

                if myInput.is_grain_boundary(self.P, i, j, self.nx, self.ny):

                    # convert the small table into distance LS function
                    for ii in range(-self.halfL_curv,self.halfL_curv+1):
                        for jj in range(-self.halfL_curv,self.halfL_curv+1):
                            local_x = (i+ii)%self.nx
                            local_y = (j+jj)%self.ny

                            # plus and minus matrix to code the LS function
                            if self.P[0,local_x,local_y] != self.P[0,i,j]:
                                self.myTable[local_x,local_y] = -1
                            else:
                                self.myTable[local_x,local_y] = 1

                            if self.V[0,local_x,local_y] == self.matrix_value:
                                self.V[0,local_x,local_y] = self.find_distance(local_x,local_y,2)

                    #  calculate the smooth value
                    for kk in range(1,self.nsteps+1):
                        for ii in range(-self.halfL_curv+kk*2,self.halfL_curv+1-kk*2):
                            for jj in range(-self.halfL_curv+kk*2,self.halfL_curv+1-kk*2):
                                local_x = (i+ii)%self.nx
                                local_y = (j+jj)%self.ny
                                if self.V[kk,local_x,local_y] == self.matrix_value:
                                    # necessary coordination
                                    local_xp1 = (i+ii+1)%self.nx
                                    local_xp2 = (i+ii+2)%self.nx
                                    local_xm1 = (i+ii-1)%self.nx
                                    local_xm2 = (i+ii-2)%self.nx
                                    local_yp1 = (j+jj+1)%self.ny
                                    local_yp2 = (j+jj+2)%self.ny
                                    local_ym1 = (j+jj-1)%self.ny
                                    local_ym2 = (j+jj-2)%self.ny

                                    # myTable=np.ones((5,5))
                                    # myTable[:,:]=Neighbors(P[0,:,:],local_x,local_y,5)
                                    # myTable[myTable[:,:]!=P[0,i,j]]=-1
                                    I02 = self.myTable[local_xm2,local_y]*self.V[kk-1,local_xm2,local_y]
                                    I11 = self.myTable[local_xm1,local_ym1]*self.V[kk-1,local_xm1,local_ym1]
                                    I12 = self.myTable[local_xm1,local_y]*self.V[kk-1,local_xm1,local_y]
                                    I13 = self.myTable[local_xm1,local_yp1]*self.V[kk-1,local_xm1,local_yp1]
                                    I20 = self.myTable[local_x,local_ym2]*self.V[kk-1,local_x,local_ym2]
                                    I21 = self.myTable[local_x,local_ym1]*self.V[kk-1,local_x,local_ym1]
                                    I22 = self.myTable[local_x,local_y]*self.V[kk-1,local_x,local_y]
                                    I23 = self.myTable[local_x,local_yp1]*self.V[kk-1,local_x,local_yp1]
                                    I24 = self.myTable[local_x,local_yp2]*self.V[kk-1,local_x,local_yp2]
                                    I31 = self.myTable[local_xp1,local_ym1]*self.V[kk-1,local_xp1,local_ym1]
                                    I32 = self.myTable[local_xp1,local_y]*self.V[kk-1,local_xp1,local_y]
                                    I33 = self.myTable[local_xp1,local_yp1]*self.V[kk-1,local_xp1,local_yp1]
                                    I42 = self.myTable[local_xp2,local_y]*self.V[kk-1,local_xp2,local_y]


                                    # Calculate the improvement to level set value at this site
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

                                    # coded status: V[:,:,:], decoded status: myTable[:,:]*V[:,:,:]
                                    if (Ii**2 + Ij**2) == 0:
                                        self.V[kk,local_x,local_y] = self.V[kk-1,local_x,local_y]
                                    else:
                                        self.V[kk,local_x,local_y] = self.V[kk-1,local_x,local_y] + self.myTable[local_x,local_y]*self.dt*(Ii**2 * Ijj - 2*Ii*Ij*Iij + Ij**2 * Iii) / (Ii**2 + Ij**2)
                                        # print(self.dt*(Ii**2 * Ijj - 2*Ii*Ij*Iij + Ij**2 * Iii) / (Ii**2 + Ij**2))
                                        # self.V[kk,local_x,local_y] = self.V[kk-1,local_x,local_y] + self.myTable[local_x,local_y]*self.dt* -math.sqrt(Ii**2 + Ij**2)

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

                    # decode the V matrix
                    I02 = self.myTable[local_xm2,local_y]*self.V[self.nsteps,local_xm2,local_y]
                    I11 = self.myTable[local_xm1,local_ym1]*self.V[self.nsteps,local_xm1,local_ym1]
                    I12 = self.myTable[local_xm1,local_y]*self.V[self.nsteps,local_xm1,local_y]
                    I13 = self.myTable[local_xm1,local_yp1]*self.V[self.nsteps,local_xm1,local_yp1]
                    I20 = self.myTable[local_x,local_ym2]*self.V[self.nsteps,local_x,local_ym2]
                    I21 = self.myTable[local_x,local_ym1]*self.V[self.nsteps,local_x,local_ym1]
                    I22 = self.myTable[local_x,local_y]*self.V[self.nsteps,local_x,local_y]
                    I23 = self.myTable[local_x,local_yp1]*self.V[self.nsteps,local_x,local_yp1]
                    I24 = self.myTable[local_x,local_yp2]*self.V[self.nsteps,local_x,local_yp2]
                    I31 = self.myTable[local_xp1,local_ym1]*self.V[self.nsteps,local_xp1,local_ym1]
                    I32 = self.myTable[local_xp1,local_y]*self.V[self.nsteps,local_xp1,local_y]
                    I33 = self.myTable[local_xp1,local_yp1]*self.V[self.nsteps,local_xp1,local_yp1]
                    I42 = self.myTable[local_xp2,local_y]*self.V[self.nsteps,local_xp2,local_y]

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

        print(f"processor {core_area_cen} read {test_check_read_num} times and max qsize {test_check_max_qsize}")
        core_etime = datetime.datetime.now()
        print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval,(core_etime - core_stime).total_seconds(),self.V)

    def levelSet_normal_vector_core(self,core_input):
        """Core function for normal vector calculation.

        Implements evolution of level set function and calculation of
        normal vectors through spatial derivatives.

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

                if myInput.is_grain_boundary(self.P, i, j, self.nx, self.ny):

                    # convert the small table into distance LS function
                    for ii in range(-self.halfL,self.halfL+1):
                        for jj in range(-self.halfL,self.halfL+1):
                            local_x = (i+ii)%self.nx
                            local_y = (j+jj)%self.ny

                            # plus and minus matrix to code the LS function
                            if self.P[0,local_x,local_y] != self.P[0,i,j]:
                                self.myTable[local_x,local_y] = -1
                            else:
                                self.myTable[local_x,local_y] = 1

                            if self.V[0,local_x,local_y] == self.matrix_value:
                                self.V[0,local_x,local_y] = self.find_distance(local_x,local_y,2)

                    #  calculate the smooth value
                    for kk in range(1,self.nsteps+1):
                        for ii in range(-self.halfL+kk*2,self.halfL+1-kk*2):
                            for jj in range(-self.halfL+kk*2,self.halfL+1-kk*2):
                                local_x = (i+ii)%self.nx
                                local_y = (j+jj)%self.ny
                                if self.V[kk,local_x,local_y] == self.matrix_value:
                                    # necessary coordination
                                    local_xp1 = (i+ii+1)%self.nx
                                    local_xp2 = (i+ii+2)%self.nx
                                    local_xm1 = (i+ii-1)%self.nx
                                    local_xm2 = (i+ii-2)%self.nx
                                    local_yp1 = (j+jj+1)%self.ny
                                    local_yp2 = (j+jj+2)%self.ny
                                    local_ym1 = (j+jj-1)%self.ny
                                    local_ym2 = (j+jj-2)%self.ny

                                    # myTable=np.ones((5,5))
                                    # myTable[:,:]=Neighbors(P[0,:,:],local_x,local_y,5)
                                    # myTable[myTable[:,:]!=P[0,i,j]]=-1
                                    I02 = self.myTable[local_xm2,local_y]*self.V[kk-1,local_xm2,local_y]
                                    I11 = self.myTable[local_xm1,local_ym1]*self.V[kk-1,local_xm1,local_ym1]
                                    I12 = self.myTable[local_xm1,local_y]*self.V[kk-1,local_xm1,local_y]
                                    I13 = self.myTable[local_xm1,local_yp1]*self.V[kk-1,local_xm1,local_yp1]
                                    I20 = self.myTable[local_x,local_ym2]*self.V[kk-1,local_x,local_ym2]
                                    I21 = self.myTable[local_x,local_ym1]*self.V[kk-1,local_x,local_ym1]
                                    I22 = self.myTable[local_x,local_y]*self.V[kk-1,local_x,local_y]
                                    I23 = self.myTable[local_x,local_yp1]*self.V[kk-1,local_x,local_yp1]
                                    I24 = self.myTable[local_x,local_yp2]*self.V[kk-1,local_x,local_yp2]
                                    I31 = self.myTable[local_xp1,local_ym1]*self.V[kk-1,local_xp1,local_ym1]
                                    I32 = self.myTable[local_xp1,local_y]*self.V[kk-1,local_xp1,local_y]
                                    I33 = self.myTable[local_xp1,local_yp1]*self.V[kk-1,local_xp1,local_yp1]
                                    I42 = self.myTable[local_xp2,local_y]*self.V[kk-1,local_xp2,local_y]


                                    # Calculate the improvement to level set value at this site
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

                                    # coded status: V[:,:,:], decoded status: myTable[:,:]*V[:,:,:]
                                    if (Ii**2 + Ij**2) == 0:
                                        self.V[kk,local_x,local_y] = self.V[kk-1,local_x,local_y]
                                    else:
                                        self.V[kk,local_x,local_y] = self.V[kk-1,local_x,local_y] + self.myTable[local_x,local_y]*self.dt*(Ii**2 * Ijj - 2*Ii*Ij*Iij + Ij**2 * Iii) / (Ii**2 + Ij**2)
                                        # print(self.dt*(Ii**2 * Ijj - 2*Ii*Ij*Iij + Ij**2 * Iii) / (Ii**2 + Ij**2))
                                        # self.V[kk,local_x,local_y] = self.V[kk-1,local_x,local_y] + self.myTable[local_x,local_y]*self.dt* -math.sqrt(Ii**2 + Ij**2)

                    # decode the V matrix
                    if ((self.P[0,ip,j]-self.P[0,i,j])!=0):
                        dw = -1
                    else:
                        dw = 1
                    if ((self.P[0,im,j]-self.P[0,i,j])!=0):
                        up = -1
                    else:
                        up = 1
                    if ((self.P[0,i,jp]-self.P[0,i,j])!=0):
                        rt = -1
                    else:
                        rt = 1
                    if ((self.P[0,i,jm]-self.P[0,i,j])!=0):
                        lt = -1
                    else:
                        lt = 1

                    fval[i,j,0]=(up*self.V[self.nsteps,im,j]-dw*self.V[self.nsteps,ip,j])/2
                    fval[i,j,1]=(rt*self.V[self.nsteps,i,jp]-lt*self.V[self.nsteps,i,jm])/2

        print(f"processor {core_area_cen} read {test_check_read_num} times and max qsize {test_check_max_qsize}")
        core_etime = datetime.datetime.now()
        print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval,(core_etime - core_stime).total_seconds(),self.V)

    def levelSet_main(self, purpose="inclination"):
        """Main function to run the level set algorithm.

        Sets up parallel processing and manages the overall calculation for
        either normal vectors or curvature.

        Args:
            purpose: Either "inclination" for normal vectors or
                    "curvature" for curvature calculation
        """
        self.run_main(self.levelSet_normal_vector_core, self.levelSet_curvature_core,
                      purpose=purpose, callback=self.res_back)
        self.get_errors()

if __name__ == '__main__':

    LS_errors =np.zeros(10)
    LS_runningTime = np.zeros(10)

    nx, ny = 200,200
    ng = 2
    # cores = 2

    # P0,R=myInput.init2IC(nx, ny, ng, "PolyIC.init")
    P0,R=myInput.Circle_IC(nx,ny)
    # P0,R=myInput.Voronoi_IC(nx,ny,ng)
    # P0,R=myInput.Complex2G_IC(nx,ny)
    # P0,R=myInput.Abnormal_IC(nx,ny)
    # P0[:,:,:]=myInput.SmallestGrain_IC(100,100)
    for cores in [4]:

        for nsteps in range(10,11):
            test1 = levelSet_class(nx,ny,ng,cores,nsteps,P0,R)
            test1.levelSet_main("curvature")
            C_ls = test1.get_C()

        #%% plot the figure2D
            # test1.get_2d_plot('Poly', 'Level-Set')

            #%% error
            print('loop_times = ' + str(test1.nsteps))
            print('running_time = %.2f' % test1.running_time)
            print('running_core time = %.2f' % test1.running_coreTime)
            print('total_errors = %.2f' % test1.errors)
            print('per_errors = %.3f' % test1.errors_per_site)
            print()

            LS_errors[nsteps-1] = test1.errors_per_site
            LS_runningTime[nsteps-1] = test1.running_coreTime
