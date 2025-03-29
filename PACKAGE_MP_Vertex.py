#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vertex Method Implementation for Interface Analysis

This module implements the vertex method for calculating grain boundary 
normal vectors and curvature in polycrystalline materials. The vertex method
uses local geometric information to track boundary points and compute:

1. Normal vectors by constructing local boundary segments
2. Curvature by fitting circles to sets of boundary points

Key features:
- Direct tracking of boundary points
- Local geometric calculations
- Parallel implementation for efficiency
- Supports both 2D and 3D interfaces

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

class vertex_class(object):
    """Vertex method implementation for interface analysis.
    
    This class implements the vertex method to calculate normal vectors and 
    curvature at grain boundaries. The method tracks interface points directly
    and uses local geometric constructions.
    
    Key attributes:
        P: Phase field array storing microstructure and normal vectors
        C: Array storing calculated curvature values
        interval: Number of boundary points to use in calculations
    """

    def __init__(self,nx,ny,ng,cores,interval,P0,R,clip=0,verification_system=True,curvature_sign=False):
        """Initialize the vertex method algorithm.
        
        Args:
            nx, ny: Grid dimensions
            ng: Number of grains
            cores: Number of parallel processes 
            interval: Number of boundary points for calculations
            P0: Initial phase field configuration
            R: Reference normal vectors
        """
        # Runtime tracking
        self.running_time = 0   # Total execution time
        self.running_coreTime = 0  # Core calculation time 
        self.errors = 0  # Accumulated errors
        self.errors_per_site = 0  # Average error per site

        # Grid parameters
        self.nx = nx  # Number of sites in x axis
        self.ny = ny  # Number of sites in y axis
        self.ng = ng  # Number of grains
        self.R = R    # Reference normal vectors
        self.clip = clip  # Clipping value for curvature calculation
        self.verification_system = verification_system
        self.curvature_sign = curvature_sign  # Sign convention for curvature

        # Initialize result arrays
        self.P = np.zeros((3,nx,ny))  # Stores IC and normal vectors
        self.C = np.zeros((2,nx,ny))  # Stores curvature
        
        # Convert individual grain maps to single map
        for i in range(0,np.shape(P0)[2]):
            self.P[0,:,:] += P0[:,:,i]*(i+1)
            self.C[0,:,:] += P0[:,:,i]*(i+1)

        # Algorithm parameters
        self.cores = cores  # Number of parallel processes
        self.interval = interval  # Number of boundary points to use

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
        for gbSite in ge_gbsites:
            [gei,gej] = gbSite
            ge_dx,ge_dy = myInput.get_grad(self.P,gei,gej)
            # Note sign convention for comparison with reference
            self.errors += math.acos(round(abs(-ge_dx*self.R[gei,gej,0]+ge_dy*self.R[gei,gej,1]),5))
            
        self.errors_per_site = self.errors/len(ge_gbsites)

    def get_curvature_errors(self):
        """Calculate error between calculated and reference curvature values.
        
        For each grain boundary site, computes difference between calculated
        curvature and reference value.
        """
        gce_gbsites = self.get_gb_list()
        for gceSite in gce_gbsites:
            [gcei, gcej] = gceSite
            self.errors += abs(self.R[gcei, gcej, 2] - self.C[1, gcei, gcej])
            
        self.errors_per_site = self.errors/len(gce_gbsites)

    def get_2d_plot(self,init,algo):
        m = 35
        n = 2
        plt.close()
        plt.subplots_adjust(wspace=0.2,right=1.8)
        fig1 = plt.figure(1)
        fig_page = self.interval
        plt.title(f'{algo}-{init} \n half interval = '+str(fig_page))
        if fig_page < 10:
            String = '000'+str(fig_page)
        elif fig_page < 100:
            String = '00'+str(fig_page)
        elif fig_page < 1000:
            String = '0'+str(fig_page)
        elif fig_page < 10000:
            String = str(fig_page)
        plt.imshow(self.P[0,:,:], cmap='nipy_spectral', interpolation='nearest')

        g2p_gbsites = self.get_gb_list()
        for gbSite in g2p_gbsites:
            [g2pi,g2pj] = gbSite
            if m != 0:
                if self.P[0,g2pi+m,g2pj+n]==self.P[0,g2pi,g2pj]:
                    anchor = np.array([g2pi+m,g2pj+n])
                elif self.P[0,g2pi-m,g2pj+n]==self.P[0,g2pi,g2pj]:
                    anchor = np.array([g2pi-m,g2pj+n])
                elif self.P[0,g2pi+m,g2pj-n]==self.P[0,g2pi,g2pj]:
                    anchor = np.array([g2pi+m,g2pj-n])
                else:
                    anchor = np.array([g2pi-m,g2pj-n])
                m=0
            if np.dot( (anchor - np.array([g2pi,g2pj])), (np.array([self.P[1,g2pi,g2pj],self.P[2,g2pi,g2pj]])) ) >= 0:
                g2p_prof = -1
            else:
                g2p_prof = 1

            g2p_dx,g2p_dy = myInput.get_grad(self.P,g2pi,g2pj)
            plt.arrow(g2pj,g2pi,-g2p_prof*10*g2p_dx,g2p_prof*10*g2p_dy,width=0.1,lw=0.1,alpha=0.8,color='navy')

        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'VT_{init}_{algo}_{String}.png',dpi=1000,bbox_inches='tight')

    def get_gb_list(self,grainID=1):
        """Find grain boundary sites.
        
        Identifies voxels at grain boundaries by checking neighboring voxels
        for grain ID changes.
        
        Args:
            grainID: ID of grain to find boundaries for (default=1)
            
        Returns:
            list: Coordinates of grain boundary sites
        """
        ggn_gbsites = []
        for i in range(0,self.nx):
            for j in range(0,self.ny):
                ip,im,jp,jm = myInput.periodic_bc(self.nx,self.ny,i,j)
                if ( ((self.P[0,ip,j]-self.P[0,i,j])!=0) or
                     ((self.P[0,im,j]-self.P[0,i,j])!=0) or
                     ((self.P[0,i,jp]-self.P[0,i,j])!=0) or
                     ((self.P[0,i,jm]-self.P[0,i,j])!=0) ) and\
                     self.P[0,i,j]==grainID:
                    ggn_gbsites.append([i,j])
        return ggn_gbsites

    def find_circle(self,A,B,C):
        """Find circle passing through three points.
        
        Calculates center and radius of circle passing through three given points
        using circumcenter formula.
        
        Args:
            A,B,C: Three 2D points to fit circle through
            
        Returns:
            tuple: Circle center coordinates and radius
        """
        a = A[1]-B[1]
        b = A[0]-B[0]
        c = A[1]-C[1]
        d = A[0]-C[0]
        e = ((A[1]**2-B[1]**2)-(B[0]**2-A[0]**2))/2.0
        f = ((A[1]**2-C[1]**2)-(C[0]**2-A[0]**2))/2.0
        x0 = -(d*e-b*f)/(b*c-a*d)
        y0 = -(a*f-c*e)/(b*c-a*d)
        r = math.sqrt((A[1]-x0)**2+(A[0]-y0)**2)
        return x0,y0,r

    def find_fittingCircle(self,array):
        """Fit circle to set of boundary points.
        
        Takes a set of boundary points and finds best fit circle by averaging
        circles through point triplets.
        
        Args:
            array: List of boundary point coordinates
            
        Returns:
            tuple: Average circle center and radius
        """
        K_mat = []
        Y_mat = []

        for point in array[1:]:
            K_mat.append([point[0],point[1],1])
            Y_mat.append([point[0]**2+point[1]**2])

        K_mat = np.mat(K_mat)
        Y_mat = np.mat(Y_mat)
        K_mat_tra = K_mat.T
        X_mat = (K_mat_tra*K_mat).I * K_mat_tra * Y_mat
        fit_x0 = X_mat[0]/2
        fit_y0 = X_mat[1]/2
        fit_R = math.sqrt(X_mat[2]+fit_x0**2+fit_y0**2)

        return fit_y0, fit_x0, fit_R

    def check_collinear(self,A,B,C):
        """Check if three points are collinear.
        
        Args:
            A,B,C: Three points to check
            
        Returns:
            bool: True if points are collinear
        """
        a = A[1]-B[1]
        b = A[0]-B[0]
        c = A[1]-C[1]
        d = A[0]-C[0]
        if a*d==b*c:
            return True
        else:
            return False


    # loop from any value in one list
    def starting_with(self,arr, start_index):
         # use xrange instead of range in python 3
         for i in range(start_index, len(arr)):
            yield arr[i]
         for i in range(start_index):
            yield arr[i]

    # find the boundary
    def find_connect(self,A):
        ip,im,jp,jm = myInput.periodic_bc(self.nx,self.ny,A[0],A[1])
        output = []

        leftListC = [ [A[0],jm],[im,jm],[im,A[1]],[im,jp],[A[0],jp],[ip,jp],[ip,A[1]],[ip,jm] ]
        leftListA = [ [A[0],jm],[ip,jm],[ip,A[1]],[ip,jp],[A[0],jp],[im,jp],[im,A[1]],[im,jm] ]

        if self.P[0,A[0],jm] != self.P[0,A[0],A[1]]:
            for ti in self.starting_with(leftListC, leftListC.index([A[0],jm]) ):
                if self.P[0,ti[0],ti[1]] == self.P[0,A[0],A[1]]:
                    output.append(ti)
                    break
            for ti in self.starting_with(leftListA, leftListA.index([A[0],jm]) ):
                if self.P[0,ti[0],ti[1]] == self.P[0,A[0],A[1]]:
                    output.append(ti)
                    break

        elif self.P[0,A[0],jp] != self.P[0,A[0],A[1]]:
            for ti in self.starting_with(leftListC, leftListC.index([A[0],jp]) ):
                if self.P[0,ti[0],ti[1]] == self.P[0,A[0],A[1]]:
                    output.append(ti)
                    break
            for ti in self.starting_with(leftListA, leftListA.index([A[0],jp]) ):
                if self.P[0,ti[0],ti[1]] == self.P[0,A[0],A[1]]:
                    output.append(ti)
                    break

        elif self.P[0,im,A[1]] != self.P[0,A[0],A[1]]:
            for ti in self.starting_with(leftListC, leftListC.index([im,A[1]]) ):
                if self.P[0,ti[0],ti[1]] == self.P[0,A[0],A[1]]:
                    output.append(ti)
                    break
            for ti in self.starting_with(leftListA, leftListA.index([im,A[1]]) ):
                if self.P[0,ti[0],ti[1]] == self.P[0,A[0],A[1]]:
                    output.append(ti)
                    break

        elif self.P[0,ip,A[1]] != self.P[0,A[0],A[1]]:
            for ti in self.starting_with(leftListC, leftListC.index([ip,A[1]]) ):
                if self.P[0,ti[0],ti[1]] == self.P[0,A[0],A[1]]:
                    output.append(ti)
                    break
            for ti in self.starting_with(leftListA, leftListA.index([ip,A[1]]) ):
                if self.P[0,ti[0],ti[1]] == self.P[0,A[0],A[1]]:
                    output.append(ti)
                    break
        return output


    def find_connect_single(self,A,B,clockwise):
        ip,im,jp,jm = myInput.periodic_bc(self.nx,self.ny,A[0],A[1])
        output = []

        theListC = [ [A[0],jm],[im,jm],[im,A[1]],[im,jp],[A[0],jp],[ip,jp],[ip,A[1]],[ip,jm] ]
        theListA = [ [A[0],jm],[ip,jm],[ip,A[1]],[ip,jp],[A[0],jp],[im,jp],[im,A[1]],[im,jm] ]

        if clockwise == 1:
            for ci in self.starting_with(theListC, theListC.index(B)+1 ):
                if self.P[0,ci[0],ci[1]] == self.P[0,A[0],A[1]]:
                    output.append(ci)
                    break

        elif clockwise == 0:
            for ci in self.starting_with(theListA, theListA.index(B)+1 ):
                if self.P[0,ci[0],ci[1]] == self.P[0,A[0],A[1]]:
                    output.append(ci)
                    break

        if len(output) != 1:
            print("connect_single != 1 !!")

        return output

    #%% Core base
    def vertex_curvature_core(self,core_input):
        """Core curvature calculation function using vertex method.
        
        Implements vertex method curvature calculation on a subdomain.
        Fits circles to sets of boundary points to compute curvature.
        
        Args:
            core_input: Input data for this subdomain
            
        Returns:
            tuple: Calculated curvature values and timing information
        """
        core_stime = datetime.datetime.now()
        li,lj,lk=np.shape(core_input)
        fval = np.zeros((self.nx,self.ny,1))


        for core_a in core_input:
            for core_b in core_a:
                i = core_b[0]
                j = core_b[1]

                stored_boun = [] # store all the boundary by sequence

                ip,im,jp,jm = myInput.periodic_bc(self.nx,self.ny,i,j)
                if ( ((self.P[0,ip,j]-self.P[0,i,j])!=0) or ((self.P[0,im,j]-self.P[0,i,j])!=0) or ((self.P[0,i,jp]-self.P[0,i,j])!=0) or ((self.P[0,i,jm]-self.P[0,i,j])!=0) ):
                    stored_boun.append([i,j])
                    stored_boun.append([i,j])
                    direc = self.find_connect([i,j])

                    if direc == []:
                        print("exception1")
                        return -1

                    # remove the exception case: don't care when 3 neighbor, only site with 1 neighbor are important
                    if direc[0] == direc[1]:
                        inc_i = i-direc[0][0]
                        inc_j = j-direc[0][1]
                        vec_len = math.sqrt(inc_i**2+inc_j**2)
                        H = 1.0
                        fval[i,j,0] = 2  # max curvature in the IC, r=0.5 voxel, so k=1/r=2

                    direc_a = direc[0]
                    direc_b = direc[1]
                    stored_boun.append(direc_a)
                    stored_boun.append(direc_b)
                    #remove the exception case when we only need 1 interval
                    if self.interval == 1:
                        pass
                    else:
                        for k in range(2,self.interval+1):
                            # for clockwise direction
                            next_point = self.find_connect_single(direc_a,stored_boun[-4],1)
                            direc_a = next_point[0]
                            stored_boun.append(direc_a)

                            next_point = self.find_connect_single(direc_b,stored_boun[-4],0)
                            direc_b = next_point[0]
                            stored_boun.append(direc_b)

                        if len(stored_boun) != (self.interval+1)*2:
                            print("stored_boun cannot get right length " + str((self.interval+1)*2-len(stored_boun)) )

                    # two boundary case: line and circle
                    if self.check_collinear(stored_boun[0],stored_boun[-1],stored_boun[-2]):
                        inc_i = stored_boun[-2][0]-stored_boun[-1][0]
                        inc_j = stored_boun[-2][1]-stored_boun[-1][1]
                        vec_len = math.sqrt(inc_i**2+inc_j**2)
                        H = 1.0
                        fval[i,j,0] = 0
                    else:
                        x0,y0,radius = self.find_circle(stored_boun[-1],stored_boun[0],stored_boun[-2])

                        fval[i,j,0] = 1/radius
        core_etime = datetime.datetime.now()
        print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval,(core_etime - core_stime).total_seconds())


    def vertex_normal_vector_core(self,core_input):
        """Core normal vector calculation function using vertex method.
        
        Implements vertex method normal vector calculation on a subdomain.
        Uses local boundary geometry to compute normal vectors.
        
        Args:
            core_input: Input data for this subdomain
            
        Returns:
            tuple: Calculated normal vectors and timing information
        """
        core_stime = datetime.datetime.now()
        li,lj,lk=np.shape(core_input)
        fval = np.zeros((self.nx,self.ny,2))


        for core_a in core_input:
            for core_b in core_a:
                i = core_b[0]
                j = core_b[1]

                stored_boun = [] # store all the boundary by sequence

                ip,im,jp,jm = myInput.periodic_bc(self.nx,self.ny,i,j)
                if ( ((self.P[0,ip,j]-self.P[0,i,j])!=0) or ((self.P[0,im,j]-self.P[0,i,j])!=0) or ((self.P[0,i,jp]-self.P[0,i,j])!=0) or ((self.P[0,i,jm]-self.P[0,i,j])!=0) ):
                    stored_boun.append([i,j])
                    stored_boun.append([i,j])
                    direc = self.find_connect([i,j])

                    if direc == []:
                        print("exception1")
                        return -1

                    # remove the exception case: don't care when 3 neighbor, only site with 1 neighbor are important
                    if direc[0] == direc[1]:
                        inc_i = i-direc[0][0]
                        inc_j = j-direc[0][1]
                        vec_len = math.sqrt(inc_i**2+inc_j**2)
                        H = 1.0
                        fval[i,j,0] = -H*inc_j/vec_len
                        fval[i,j,1] = H*inc_i/vec_len

                    direc_a = direc[0]
                    direc_b = direc[1]
                    stored_boun.append(direc_a)
                    stored_boun.append(direc_b)
                    #remove the exception case when we only need 1 interval
                    if self.interval == 1:
                        pass
                    else:
                        for k in range(2,self.interval+1):
                            # for clockwise direction
                            next_point = self.find_connect_single(direc_a,stored_boun[-4],1)
                            direc_a = next_point[0]
                            stored_boun.append(direc_a)

                            next_point = self.find_connect_single(direc_b,stored_boun[-4],0)
                            direc_b = next_point[0]
                            stored_boun.append(direc_b)

                        if len(stored_boun) != (self.interval+1)*2:
                            print("stored_boun cannot get right length " + str((self.interval+1)*2-len(stored_boun)) )

                    # two boundary case: line and circle
                    if self.check_collinear(stored_boun[0],stored_boun[-1],stored_boun[-2]):
                        inc_i = stored_boun[-2][0]-stored_boun[-1][0]
                        inc_j = stored_boun[-2][1]-stored_boun[-1][1]
                        vec_len = math.sqrt(inc_i**2+inc_j**2)
                        H = 1.0
                        fval[i,j,0] = -H*inc_j/vec_len
                        fval[i,j,1] = H*inc_i/vec_len
                    else:
                        x0,y0,radius = self.find_circle(stored_boun[-1],stored_boun[0],stored_boun[-2])

                        fval[i,j,0] = -(y0-stored_boun[0][0])/radius#(stored_boun[0][1]-x0)/radius
                        fval[i,j,1] = (stored_boun[0][1]-x0)/radius#-(y0-stored_boun[0][0])/radius
        core_etime = datetime.datetime.now()
        print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval,(core_etime - core_stime).total_seconds())

    def vertex_main(self, purpose="inclination"):
        """Main function to run the vertex algorithm.
        
        Sets up parallel processing and manages the overall calculation for
        either normal vectors or curvature.
        
        Args:
            purpose: Either "inclination" for normal vectors or 
                    "curvature" for curvature calculation
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
                    print(f'processor [{ki},{kj}] starting...')
                    res_one = pool.apply_async(
                        func=self.vertex_normal_vector_core,
                        args=(multi_input[ki][kj],),
                        callback=self.res_back)
                    res_list.append(res_one)
        elif purpose == "curvature":
            for ki in range(main_wc):
                for kj in range(main_lc):
                    print(f'processor [{ki},{kj}] starting...')
                    res_one = pool.apply_async(
                        func=self.vertex_curvature_core,
                        args=(multi_input[ki][kj],),
                        callback=self.res_back)
                    res_list.append(res_one)

        # Wait for all processes to complete
        pool.close()
        pool.join()
        print("core done!")
        # print(res_list[0].get())

        # Calculate timing and errors
        endtime = datetime.datetime.now()
        self.running_time = (endtime - starttime).total_seconds()
        
        if purpose == "inclination":
            self.get_errors()
        elif purpose == "curvature":
            self.get_curvature_errors()

    def res_back(self,back_result):
        res_stime = datetime.datetime.now()
        (fval,core_time) = back_result
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



#%%
if __name__ == '__main__':

    # VT_errors =np.zeros(10)
    # VT_runningTime = np.zeros(10)
    # VT2_errors =np.zeros(10)
    # VT2_runningTime = np.zeros(10)
    # ctx = mp.get_context('fork')
    interval = range(11,12)
    nx, ny = 200, 200
    ng = 2
    cores = [4]

    # P0,R=myInput.init2IC(nx, ny, ng, "PolyIC.init")
    P0,R=myInput.Circle_IC(nx,ny, 5)
    # P0,R=myInput.Voronoi_IC(nx,ny,ng)
    # P0,R=myInput.Complex2G_IC(nx,ny)
    # P0,R=myInput.Abnormal_IC(nx,ny)
    # P0[:,:,:]=myInput.SmallestGrain_IC(100,100)
    for ci in cores:
        for inti in interval:
            test1 = vertex_class(nx,ny,ng,ci,inti,P0,R)
            test1.vertex_main("curvature")
            # P = test1.get_P()
            C_vt = test1.get_C()

            #%% figure
            # test1.get_2d_plot('Poly', 'Vertex')

            #%% error


            print('loop_times = ' + str(test1.interval))
            print('running_time = %.2f' % test1.running_time)
            print('running_core time = %.2f' % test1.running_coreTime)
            print('total_errors = %.2f' % test1.errors)
            print('per_errors = %.3f' % test1.errors_per_site)
            print()

            # VT_errors[inti-1] = test1.errors_per_site
            # VT_runningTime[inti-1] = test1.running_coreTime
