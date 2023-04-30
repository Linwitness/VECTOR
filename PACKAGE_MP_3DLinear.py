#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:29:17 2021

@author: lin.yang
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

# we use sparse data struture (doi:10.1088/0965-0393/14/7/007) to store temporary matrix V. The size is nsteps*nx*ny*nz dict

class linear3d_class(object):

    def __init__(self,nx,ny,nz,ng,cores,loop_times,P0,R,bc,clip=0):
        # V_matrix init value; runnning time and error for the algorithm
        self.matrix_value = 10
        self.running_time = 0
        self.running_coreTime = 0
        self.errors = 0
        self.errors_per_site = 0
        self.clip = clip

        # initial condition data
        self.nx = nx # number of sites in x axis
        self.ny = ny # number of sites in y axis
        self.nz = nz
        self.ng = ng # number of grains in IC
        self.R = R  # results of analysis model
        # convert individual grains map into one grain map
        self.P = np.zeros((4,nx,ny,nz)) # matrix to store IC and normal vector results
        self.C = np.zeros((2,nx,ny,nz)) # curvature result matrix
        for i in range(0,np.shape(P0)[3]):
            self.P[0,:,:,:] += P0[:,:,:,i]*(i+1)
            self.C[0,:,:,:] += P0[:,:,:,i]*(i+1)
        self.bc = bc

        # data for multiprocessing
        self.cores = cores

        # data for accuracy
        self.loop_times = loop_times
        self.tableL = 2*(loop_times+1)+1 # when repeatting two times, the table length will be 7 (7 by 7 table)
        self.tableL_curv = 2*(loop_times+2)+1
        self.halfL = loop_times+1

        # self.V_sparse = np.empty((loop_times+1,nx,ny,nz),dtype=dict)

        # some attributes
        # linear smoothing matrix
        self.smoothed_vector_i, self.smoothed_vector_j, self.smoothed_vector_k = myInput.output_linear_vector_matrix3D(self.loop_times, self.clip)

    #%% Functions
    def get_P(self):
        # Outout the result matrix, first level is microstructure,
        # last two layers are normal vectors
        return self.P

    def get_C(self):
        # Get curvature matrix
        return self.C

    def get_errors(self):
        ge_gbsites = self.get_gb_list()
        for gbSite in ge_gbsites :
            [gei,gej,gek] = gbSite
            ge_dx,ge_dy,ge_dz = myInput.get_grad3d(self.P,gei,gej,gek)
            self.errors += math.acos(round(abs(ge_dx*self.R[gei,gej,gek,0]+ge_dy*self.R[gei,gej,gek,1]+ge_dz*self.R[gei,gej,gek,2]),5))

        if len(ge_gbsites) > 0: self.errors_per_site = self.errors/len(ge_gbsites)
        else: self.errors_per_site = 0

    def get_curvature_errors(self):
        gce_gbsites = self.get_gb_list()
        for gceSite in gce_gbsites :
            [gcei,gcej,gcek] = gceSite
            self.errors += abs(self.R[gcei, gcej, gcek, 3] - self.C[1, gcei, gcej, gcek])

        self.errors_per_site = self.errors/len(gce_gbsites)


    def get_2d_plot(self,init,algo,z_surface = 0):
        z_surface = int(self.nz/2)
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
        plt.imshow(self.P[0,:,:,z_surface], cmap='nipy_spectral', interpolation='nearest')

        g2p_gbsites = self.get_gb_list()
        for gbSite in g2p_gbsites:
            [g2pi,g2pj,g2pk] = gbSite
            if g2pk==z_surface:

                g2p_dx,g2p_dy,g2p_dz = myInput.get_grad3d(self.P,g2pi,g2pj,g2pk)
                plt.arrow(g2pj,g2pi,10*g2p_dx,10*g2p_dy,width=0.1,lw=0.1,alpha=0.8,color='navy')

        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'{init}-{algo}.{String}.png',dpi=1000,bbox_inches='tight')

    def get_gb_list(self,grainID=1):
        ggn_gbsites = []
        if self.bc == 'np':
            edge_l = self.halfL
        else:
            edge_l = 0
        for i in range(0+edge_l,self.nx-edge_l):
            for j in range(0+edge_l,self.ny-edge_l):
                for k in range(0+edge_l,self.nz-edge_l):
                    ip,im,jp,jm,kp,km = myInput.periodic_bc3d(self.nx,self.ny,self.nz,i,j,k)
                    if ( ((self.P[0,ip,j,k]-self.P[0,i,j,k])!=0) or ((self.P[0,im,j,k]-self.P[0,i,j,k])!=0) or\
                         ((self.P[0,i,jp,k]-self.P[0,i,j,k])!=0) or ((self.P[0,i,jm,k]-self.P[0,i,j,k])!=0) or\
                         ((self.P[0,i,j,kp]-self.P[0,i,j,k])!=0) or ((self.P[0,i,j,km]-self.P[0,i,j,k])!=0) ):#\
                           #and self.P[0,i,j,k]==grainID:
                        ggn_gbsites.append([i,j,k])
        return ggn_gbsites

    def get_all_gb_list(self):
        gagn_gbsites = [[] for _ in range(int(self.ng))]
        for i in range(0,self.nx):
            for j in range(0,self.ny):
                for k in range(0,self.nz):
                    ip,im,jp,jm,kp,km = myInput.periodic_bc3d(self.nx,self.ny,self.nz,i,j,k)
                    if ( ((self.P[0,ip,j,k]-self.P[0,i,j,k])!=0) or ((self.P[0,im,j,k]-self.P[0,i,j,k])!=0) or\
                         ((self.P[0,i,jp,k]-self.P[0,i,j,k])!=0) or ((self.P[0,i,jm,k]-self.P[0,i,j,k])!=0) or\
                         ((self.P[0,i,j,kp]-self.P[0,i,j,k])!=0) or ((self.P[0,i,j,km]-self.P[0,i,j,k])!=0) ):
                        gagn_gbsites[int(self.P[0,i,j,k]-1)].append([i,j,k])
        return gagn_gbsites

    def find_window(self,i,j,k,fw_len):
        # fw_len = self.tableL - 2*self.clip
        fw_half = int((fw_len-1)/2)
        window = np.zeros((fw_len,fw_len,fw_len))

        for wi in range(fw_len):
            for wj in range(fw_len):
                for wk in range(fw_len):
                    global_x = (i-fw_half+wi)%self.nx
                    global_y = (j-fw_half+wj)%self.ny
                    global_z = (k-fw_half+wk)%self.nz
                    if self.P[0,global_x,global_y,global_z] == self.P[0,i,j,k]:
                        window[wi,wj,wk] = 1
                    else:
                        window[wi,wj,wk] = 0
                    # window[wi,wj] = self.P[0,global_x,global_y]

        return window

    def calculate_curvature(self,matrix):
        # calculate curvature based on 5x5 smoothed matrix
        I022 = matrix[0][2][2]
        I112 = matrix[1][1][2]
        I122 = matrix[1][2][2]
        I132 = matrix[1][3][2]
        I202 = matrix[2][0][2]
        I212 = matrix[2][1][2]
        I222 = matrix[2][2][2]
        I232 = matrix[2][3][2]
        I242 = matrix[2][4][2]
        I312 = matrix[3][1][2]
        I322 = matrix[3][2][2]
        I332 = matrix[3][3][2]
        I422 = matrix[4][2][2]

        I220 = matrix[2][2][0]
        I111 = matrix[1][1][1]
        I121 = matrix[1][2][1]
        I131 = matrix[1][3][1]
        I211 = matrix[2][1][1]
        I221 = matrix[2][2][1]
        I231 = matrix[2][3][1]
        I311 = matrix[3][1][1]
        I321 = matrix[3][2][1]
        I331 = matrix[3][3][1]

        I113 = matrix[1][1][3]
        I123 = matrix[1][2][3]
        I133 = matrix[1][3][3]
        I213 = matrix[2][1][3]
        I223 = matrix[2][2][3]
        I233 = matrix[2][3][3]
        I313 = matrix[3][1][3]
        I323 = matrix[3][2][3]
        I333 = matrix[3][3][3]
        I224 = matrix[2][2][4]

        # calculate teh improve or decrease of each site
        Ii = (I322-I122)/2
        Ij = (I232-I212)/2
        Ik = (I223-I221)/2

        Imi = (I222-I022)/2
        Ipi = (I422-I222)/2
        Imj = (I222-I202)/2
        Ipj = (I242-I222)/2
        Imk = (I222-I220)/2
        Ipk = (I224-I222)/2
        Imij = (I132-I112)/2
        Ipij = (I332-I312)/2
        Imik = (I321-I121)/2
        Ipik = (I323-I123)/2
        Imjk = (I231-I211)/2
        Ipjk = (I233-I213)/2

        Iii = (Ipi-Imi)/2
        Ijj = (Ipj-Imj)/2
        Ikk = (Ipk-Imk)/2
        Iij = (Ipij-Imij)/2
        Iik = (Ipik-Imik)/2
        Ijk = (Ipjk-Imjk)/2

        if (Ii**2 + Ij**2 + Ik**2) == 0: return 0
        return abs((Ij**2+Ik**2)*Iii + (Ik**2+Ii**2)*Ijj + (Ii**2+Ij**2)*Ikk - 2*Ii*Ij*Iij - 2*Ij*Ik*Ijk - 2*Ik*Ii*Iik) / (2*(Ii**2 + Ij**2 + Ik**2)**(1.5))

    #%%
    # Core
    def linear3d_curvature_core(self,core_input, core_all_queue):
        # Return the curvature of each boundary pixel:
        # 1. find the smoothed matrix
        # 2. calculate mean curvature based on the smoothed matrix by equation
        #    H = ((Ij**2+Ik**2)*Iii + (Ik**2+Ii**2)*Ijj + (Ii**2+Ij**2)*Ikk - 2*Ii*Ij*Iij - 2*Ij*Ik*Ijk - 2*Ik*Ii*Iik) / (2*(Ii**2 + Ij**2 + Ik**2)**(1.5))
        core_stime = datetime.datetime.now()
        li,lj,lk,lp=np.shape(core_input)
        fval = np.zeros((self.nx,self.ny,self.nz,1))

        for core_a in core_input:
            for core_b in core_a:
                for core_c in core_b:
                    i = core_c[0]
                    j = core_c[1]
                    k = core_c[2]

                    ip,im,jp,jm,kp,km = myInput.periodic_bc3d(self.nx,self.ny,self.nz,i,j,k)
                    if ( ((self.P[0,ip,j,k]-self.P[0,i,j,k])!=0) or ((self.P[0,im,j,k]-self.P[0,i,j,k])!=0) or
                         ((self.P[0,i,jp,k]-self.P[0,i,j,k])!=0) or ((self.P[0,i,jm,k]-self.P[0,i,j,k])!=0) or
                         ((self.P[0,i,j,kp]-self.P[0,i,j,k])!=0) or ((self.P[0,i,j,km]-self.P[0,i,j,k])!=0) ):

                        window = np.zeros((self.tableL_curv,self.tableL_curv,self.tableL_curv))
                        window = self.find_window(i,j,k,self.tableL_curv - 2*self.clip)
                        smoothed_matrix = myInput.output_smoothed_matrix3D(window, myInput.output_linear_smoothing_matrix3D(self.loop_times))[self.loop_times:-self.loop_times,self.loop_times:-self.loop_times,self.loop_times:-self.loop_times]

                        # Error check for smoothed matrix
                        if (smoothed_matrix.shape != (5,5,5)):
                            print(f"The smoothed matrix is not correct: {smoothed_matrix.shape}")


                        # print(window)

                        # if i==64 and j==65:
                        #     signal = 1
                        # else:
                        #     signal=0
                        fval[i,j,k,0] = self.calculate_curvature(smoothed_matrix)


        core_etime = datetime.datetime.now()
        print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval,(core_etime - core_stime).total_seconds())

    def linear3d_normal_vector_core(self,core_input, core_all_queue):
        core_stime = datetime.datetime.now()
        li,lj,lk,lp=np.shape(core_input)
        fval = np.zeros((self.nx,self.ny,self.nz,3))

        for core_a in core_input:
            for core_b in core_a:
                for core_c in core_b:
                    i = core_c[0]
                    j = core_c[1]
                    k = core_c[2]

                    # check the boundary condition
                    if self.bc == 'np':
                        if not myInput.filter_bc3d(self.nx, self.ny, self.nz, i, j, k, self.halfL):
                            continue
                    else:
                        pass

                    ip,im,jp,jm,kp,km = myInput.periodic_bc3d(self.nx,self.ny,self.nz,i,j,k)
                    if ( ((self.P[0,ip,j,k]-self.P[0,i,j,k])!=0) or ((self.P[0,im,j,k]-self.P[0,i,j,k])!=0) or
                         ((self.P[0,i,jp,k]-self.P[0,i,j,k])!=0) or ((self.P[0,i,jm,k]-self.P[0,i,j,k])!=0) or
                         ((self.P[0,i,j,kp]-self.P[0,i,j,k])!=0) or ((self.P[0,i,j,km]-self.P[0,i,j,k])!=0) ):

                        window = np.zeros((self.tableL,self.tableL,self.tableL))
                        window = self.find_window(i,j,k,self.tableL - 2*self.clip)
                        # print(window)

                        fval[i,j,k,0] = -np.sum(window*self.smoothed_vector_i)
                        fval[i,j,k,1] = np.sum(window*self.smoothed_vector_j)
                        fval[i,j,k,2] = np.sum(window*self.smoothed_vector_k)


        core_etime = datetime.datetime.now()
        print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval,(core_etime - core_stime).total_seconds())


    def res_back(self,back_result):
        res_stime = datetime.datetime.now()
        (fval,core_time) = back_result
        if core_time > self.running_coreTime:
            self.running_coreTime = core_time

        print("res_back start...")
        if fval.shape[3] == 1:
            self.C[1,:,:,:] += fval[:,:,:,0]
        elif fval.shape[3] == 3:
            self.P[1,:,:,:] += fval[:,:,:,0]
            self.P[2,:,:,:] += fval[:,:,:,1]
            self.P[3,:,:,:] += fval[:,:,:,2]
        res_etime = datetime.datetime.now()
        print("my res time is " + str((res_etime - res_stime).total_seconds()))

    def linear3d_main(self, purpose ="inclination"):

        # global starttime, endtime
        starttime = datetime.datetime.now()

        pool = mp.Pool(processes=self.cores)
        main_lc, main_wc, main_hc = myInput.split_cores(self.cores,3)

        # create all the queue
        manager = mp.Manager()
        all_queue = []
        print("")
        for queue_i in range(main_wc):
            tmp1 = []
            for queue_j in range(main_lc):
                tmp2 = []
                for queue_k in range(main_hc):
                    tmp2.append(manager.Queue(int(((self.nx/main_wc)+(self.ny/main_lc))*(1+self.loop_times)*(1+self.loop_times))))
                tmp1.append(tmp2)
            all_queue.append(tmp1)


        all_sites = np.array([[x,y,z] for x in range(self.nx) for y in range(self.ny) for z in range(self.nz) ]).reshape(self.nx,self.ny,self.nz,3)
        multi_input = myInput.split_IC(all_sites, self.cores,3, 0,1,2)

        res_list=[]
        if purpose == "inclination":
            for mpi in range(main_wc):
                for mpj in range(main_lc):
                    for mpk in range(main_hc):
                        res_one = pool.apply_async(func = self.linear3d_normal_vector_core, args = (multi_input[mpi][mpj][mpk], all_queue, ), callback=self.res_back )
                        res_list.append(res_one)
        elif purpose == "curvature":
            for mpi in range(main_wc):
                for mpj in range(main_lc):
                    for mpk in range(main_hc):
                        res_one = pool.apply_async(func = self.linear3d_curvature_core, args = (multi_input[mpi][mpj][mpk], all_queue, ), callback=self.res_back )
                        res_list.append(res_one)


        pool.close()
        pool.join()

        print("core done!")
        # print(res_list[0].get())

        # calculate time
        endtime = datetime.datetime.now()

        self.running_time = (endtime - starttime).total_seconds()
        if purpose == "inclination":
            self.get_errors()
        elif purpose == "curvature":
            self.get_curvature_errors()


#%% Simple tests
if __name__ == '__main__':
    BL3d2_errors = np.zeros(10)
    BL3d2_runningTime = np.zeros(10)

    # Demostration Voronoi 1000 grains sample with 0 timestep, 10 timestep, 50 timestep
    # nx, ny, nz = 100, 100, 100
    # ng = 1000
    # filepath = '/Users/lin.yang/projects/SPPARKS-AGG/examples/agg/Voronoi/1000grs100sts/'
    # P0,R=myInput.init2IC3d(nx,ny,nz,ng,"VoronoiIC1000.init",False,filepath)

    # Demonstration Dream3d 5432 grains sample ("s1400_t0.init") with 0 timestep
    # nx, ny, nz = 500,  500, 50
    # ng = 5432
    # P0,R=myInput.init2IC3d(nx,ny,nz,ng,"s1400_t0.init",True)

    # Validation Dream3d 831 grains sample ("s1400poly1_t0.init") with 0 timestep
    # nx, ny, nz = 201, 201, 43
    # ng = 831
    # P0,R=myInput.init2IC3d(nx,ny,nz,ng,"s1400poly1_t0.init",True)

    # Sample ICs
    nx, ny, nz = 100, 100, 100
    ng = 2

    # P0,R=myInput.init2IC(nx, ny, ng, "PolyIC.init")
    # P0,R = myInput.Complex2G_IC3d(nx,ny,nz)
    # P0,R=myInput.Circle_IC(nx,ny)
    P0,R=myInput.Circle_IC3d(nx,ny,nz)
    # P0,R=myInput.Voronoi_IC(nx,ny,ng)
    # P0,R=myInput.Complex2G_IC(nx,ny)
    # P0[:,:,:],R=myInput.Abnormal_IC(nx,ny)
    # P0[:,:,:]=myInput.SmallestGrain_IC(100,100)

    for cores in [8]:
        for loop_times in range(1,2):


            test1 = linear3d_class(nx,ny,nz,ng,cores,loop_times,P0,R,'np')
            test1.linear3d_main("curvature")
            C_ln = test1.get_C()


            #%%
            # test1.get_gb_list(5432)
            # test1.get_2d_plot('DREAM3D_poly','Bilinear')


            #%% error

            print('loop_times = ' + str(test1.loop_times))
            print('running_time = %.2f' % test1.running_time)
            print('running_core time = %.2f' % test1.running_coreTime)
            print('total_errors = %.2f' % test1.errors)
            print('per_errors = %.3f' % test1.errors_per_site)
            print()

            BL3d2_errors[loop_times-1] = test1.errors_per_site
            BL3d2_runningTime[loop_times-1] = test1.running_coreTime
