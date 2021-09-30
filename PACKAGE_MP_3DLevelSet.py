#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 18:09:28 2021

@author: lin.yang
"""


import sys
sys.path.append('/Users/lin.yang/Documents/SPYDER/AllenCahnSmooth/')
import numpy as np
import math
import matplotlib.pyplot as plt
import myInput
import datetime
import multiprocessing as mp

class LS3dv1_class(object):
    
    def __init__(self,nx,ny,nz,ng,cores,nsteps,P0,R,bc,switch = False):
        # V_matrix init value; runnning time and error for the algorithm
        self.matrix_value = 10
        self.running_time = 0
        self.running_coreTime = 0
        self.errors = 0
        self.errors_per_site = 0
        self.switch = switch
        
        # initial condition data
        self.nx = nx # number of sites in x axis
        self.ny = ny # number of sites in y axis
        self.nz = nz
        self.ng = ng # number of grains in IC
        self.R = R  # results of analysis model
        # convert individual grains map into one grain map
        self.P = np.zeros((4,nx,ny,nz)) # matrix to store IC and normal vector results
        for i in range(0,np.shape(P0)[3]):
            self.P[0,:,:,:] += P0[:,:,:,i]*(i+1)
        self.bc = bc
        
        # data for multiprocessing
        self.cores = cores
        
        # data for accuracy
        self.nsteps = nsteps #Number of timesteps
        self.dt = 1 #Timestep size
        self.tableL = 2*(2*nsteps+1)+1 # smallest table is 7by7
        self.halfL = 2*nsteps+1
        
        # temporary matrix to increase efficiency 
        self.V = np.ones((nsteps+1,nx,ny,nz))*self.matrix_value
        self.myTable = np.ones((nx,ny,nz))
    
    #%% Function
    def get_P(self):
        return self.P
    
    def get_errors(self):
        ge_gbsites = self.get_gb_num()
        for gbSite in ge_gbsites :
            [gei,gej,gek] = gbSite
            ge_dx,ge_dy,ge_dz = myInput.get_grad3d(self.P,gei,gej,gek)
            self.errors += math.acos(round(abs(ge_dx*self.R[gei,gej,gek,0]+ge_dy*self.R[gei,gej,gek,1]+ge_dz*self.R[gei,gej,gek,2]),5))
        
        self.errors_per_site = self.errors/len(ge_gbsites)
        
    def get_2d_plot(self,init,algo,z_surface = 0):
        for z_surface in range(0,self.nz,10):
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
            plt.imshow(self.P[0,:,:,z_surface], cmap='nipy_spectral', interpolation='nearest')
            
            g2p_gbsites = self.get_gb_num()
            for gbSite in g2p_gbsites:
                [g2pi,g2pj,g2pk] = gbSite
                if g2pk==z_surface:
                    g2p_dx,g2p_dy,g2p_dz = myInput.get_grad3d(self.P,g2pi,g2pj,g2pk)
                    plt.arrow(g2pj,g2pi,10*g2p_dx,10*g2p_dy,width=0.1,lw=0.1,alpha=0.8,color='navy')
            
            plt.xticks([])
            plt.yticks([])
            plt.savefig(f'{init}-{algo}.{String}.{z_surface/self.nz}.png',dpi=1000,bbox_inches='tight')

    def get_gb_num(self,grainID=1):
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
                         ((self.P[0,i,j,kp]-self.P[0,i,j,k])!=0) or ((self.P[0,i,j,km]-self.P[0,i,j,k])!=0) )\
                           and self.P[0,i,j,k]==grainID:
                        ggn_gbsites.append([i,j,k])
        return ggn_gbsites
    
            
    def Neighbors(self,arr,x,y,z,n):
        ''' Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]'''
        arr=np.roll(np.roll(np.roll(arr,shift=-x+math.floor(n/2),axis=0),shift=-y+math.floor(n/2),axis=1),shift=-z+math.floor(n/2),axis=2)
        return arr[:n,:n,:n]
    
    def find_distance(self,i,j,k,d): # let d=2
        smallTable = self.Neighbors(self.P[0,:,:,:],i,j,k,d*2+1)
        
        # two layer list
        layerList = [[d-1,d,d], [d+1,d,d], [d,d-1,d], [d,d+1,d], [d,d,d-1], [d,d,d+1],
                     [d-1,d-1,d], [d-1,d+1,d], [d+1,d-1,d], [d+1,d+1,d], [d-1,d,d-1], [d-1,d,d+1], [d+1,d,d-1], [d+1,d,d+1], [d,d-1,d-1], [d,d-1,d+1], [d,d+1,d-1], [d,d+1,d+1],
                     [d-1,d-1,d-1], [d-1,d+1,d-1], [d+1,d-1,d-1], [d+1,d+1,d-1], [d-1,d-1,d+1], [d-1,d+1,d+1], [d+1,d-1,d+1], [d+1,d+1,d+1],
                     [d,d-2,d], [d,d+2,d], [d-2,d,d], [d+2,d,d], [d,d,d-2], [d,d,d+2],
                     [d-1,d,d-2], [d+1,d,d-2], [d,d-1,d-2], [d,d+1,d-2], [d-1,d,d+2], [d+1,d,d+2], [d,d-1,d+2], [d,d+1,d+2], [d-1,d-2,d], [d+1,d-2,d], [d,d-2,d-1], [d,d-2,d+1], [d-1,d+2,d], [d+1,d+2,d], [d,d+2,d-1], [d,d+2,d+1], [d-2,d-1,d], [d-2,d+1,d], [d-2,d,d-1], [d-2,d,d+1], [d+2,d-1,d], [d+2,d+1,d], [d+2,d,d-1], [d+2,d,d+1],
                     [d-1,d-1,d-2], [d-1,d+1,d-2], [d+1,d-1,d-2], [d+1,d+1,d-2], [d-1,d-1,d+2], [d-1,d+1,d+2], [d+1,d-1,d+2], [d+1,d+1,d+2], [d-1,d-2,d-1], [d-1,d-2,d+1], [d+1,d-2,d-1], [d+1,d-2,d+1], [d-1,d+2,d-1], [d-1,d+2,d+1], [d+1,d+2,d-1], [d+1,d+2,d+1], [d-2,d-1,d-1], [d-2,d-1,d+1], [d-2,d+1,d-1], [d-2,d+1,d+1], [d+2,d-1,d-1], [d+2,d-1,d+1], [d+2,d+1,d-1], [d+2,d+1,d+1],
                     [d-2,d-2,d], [d-2,d+2,d], [d+2,d+2,d], [d+2,d-2,d], [d-2,d,d-2], [d-2,d,d+2], [d+2,d,d+2], [d+2,d,d-2], [d,d-2,d-2], [d,d-2,d+2], [d,d+2,d+2], [d,d+2,d-2],
                     [d-2,d-2,d-1], [d-2,d+2,d-1], [d+2,d+2,d-1], [d+2,d-2,d-1], [d-2,d-1,d-2], [d-2,d-1,d+2], [d+2,d-1,d+2], [d+2,d-1,d-2], [d-1,d-2,d-2], [d-1,d-2,d+2], [d-1,d+2,d+2], [d-1,d+2,d-2], [d-2,d-2,d+1], [d-2,d+2,d+1], [d+2,d+2,d+1], [d+2,d-2,d+1], [d-2,d+1,d-2], [d-2,d+1,d+2], [d+2,d+1,d+2], [d+2,d+1,d-2], [d+1,d-2,d-2], [d+1,d-2,d+2], [d+1,d+2,d+2], [d+1,d+2,d-2],
                     [d-2,d-2,d-2], [d-2,d+2,d-2], [d+2,d+2,d-2], [d+2,d-2,d-2], [d-2,d-2,d+2], [d-2,d+2,d+2], [d+2,d+2,d+2], [d+2,d-2,d+2]
                     ]
        
        for k in layerList[:(2*d+1)**3-1]:
            if smallTable[k[0],k[1],k[2]] != smallTable[d,d,d]:
                return math.sqrt((d-k[0])**2 + (d-k[1])**2 + (d-k[2])**2)
        
        return d*math.sqrt(3)
    
    #%% Smooth Core Site-based Stored data
    
    def res_back(self,back_result):
        res_stime = datetime.datetime.now()
        (fval,core_time,self.V) = back_result
        if core_time > self.running_coreTime:
            self.running_coreTime = core_time
        
        print("res_back start...")
        self.P[1,:,:,:] += fval[:,:,:,0]
        self.P[2,:,:,:] += fval[:,:,:,1]
        self.P[3,:,:,:] += fval[:,:,:,2]
        res_etime = datetime.datetime.now()
        print("my res time is " + str((res_etime - res_stime).total_seconds()))
        
    def LS3dv1_core_apply(self,core_input, core_all_queue):
        core_stime = datetime.datetime.now()
        li,lj,lk, lp=np.shape(core_input)
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
                
                        # convert the small table into distance LS function
                        for ii in range(-self.halfL,self.halfL+1):
                            for jj in range(-self.halfL,self.halfL+1):
                                for kk in range(-self.halfL,self.halfL+1):
                                    local_x = (i+ii)%self.nx
                                    local_y = (j+jj)%self.ny
                                    local_z = (k+kk)%self.nz
                                
                                    # plus and minus matrix to code the LS function
                                    if self.P[0,local_x,local_y,local_z] != self.P[0,i,j,k]:
                                        self.myTable[local_x,local_y,local_z] = -1
                                    else:
                                        self.myTable[local_x,local_y,local_z] = 1
                                        
                                    if self.V[0,local_x,local_y,local_z] == self.matrix_value:
                                        self.V[0,local_x,local_y,local_z] = self.find_distance(local_x,local_y,local_z,2)
                                    
                        #  calculate the smooth value
                        for lps in range(1,self.nsteps+1):
                            for ii in range(-self.halfL+lps*2,self.halfL+1-lps*2):
                                for jj in range(-self.halfL+lps*2,self.halfL+1-lps*2):
                                    for kk in range(-self.halfL+lps*2,self.halfL+1-lps*2):
                                        local_x = (i+ii)%self.nx
                                        local_y = (j+jj)%self.ny
                                        local_z = (k+kk)%self.nz
                                        if self.V[lps,local_x,local_y,local_z] == self.matrix_value:
                                            # necessary coordination
                                            local_xp1 = (i+ii+1)%self.nx
                                            local_xp2 = (i+ii+2)%self.nx
                                            local_xm1 = (i+ii-1)%self.nx
                                            local_xm2 = (i+ii-2)%self.nx
                                            local_yp1 = (j+jj+1)%self.ny
                                            local_yp2 = (j+jj+2)%self.ny
                                            local_ym1 = (j+jj-1)%self.ny
                                            local_ym2 = (j+jj-2)%self.ny
                                            local_zp1 = (k+kk+1)%self.nz
                                            local_zp2 = (k+kk+2)%self.nz
                                            local_zm1 = (k+kk-1)%self.nz
                                            local_zm2 = (k+kk-2)%self.nz
                                            
                                            I022 = self.myTable[local_xm2,local_y,local_z]*self.V[lps-1,local_xm2,local_y,local_z]
                                            I112 = self.myTable[local_xm1,local_ym1,local_z]*self.V[lps-1,local_xm1,local_ym1,local_z]
                                            I122 = self.myTable[local_xm1,local_y,local_z]*self.V[lps-1,local_xm1,local_y,local_z]
                                            I132 = self.myTable[local_xm1,local_yp1,local_z]*self.V[lps-1,local_xm1,local_yp1,local_z]
                                            I202 = self.myTable[local_x,local_ym2,local_z]*self.V[lps-1,local_x,local_ym2,local_z]
                                            I212 = self.myTable[local_x,local_ym1,local_z]*self.V[lps-1,local_x,local_ym1,local_z]
                                            I222 = self.myTable[local_x,local_y,local_z]*self.V[lps-1,local_x,local_y,local_z]
                                            I232 = self.myTable[local_x,local_yp1,local_z]*self.V[lps-1,local_x,local_yp1,local_z]
                                            I242 = self.myTable[local_x,local_yp2,local_z]*self.V[lps-1,local_x,local_yp2,local_z]
                                            I312 = self.myTable[local_xp1,local_ym1,local_z]*self.V[lps-1,local_xp1,local_ym1,local_z]
                                            I322 = self.myTable[local_xp1,local_y,local_z]*self.V[lps-1,local_xp1,local_y,local_z]
                                            I332 = self.myTable[local_xp1,local_yp1,local_z]*self.V[lps-1,local_xp1,local_yp1,local_z]
                                            I422 = self.myTable[local_xp2,local_y,local_z]*self.V[lps-1,local_xp2,local_y,local_z]
                                            
                                            I220 = self.myTable[local_x,local_y,local_zm2]*self.V[lps-1,local_x,local_y,local_zm2]
                                            I111 = self.myTable[local_xm1,local_ym1,local_zm1]*self.V[lps-1,local_xm1,local_ym1,local_zm1]
                                            I121 = self.myTable[local_xm1,local_y,local_zm1]*self.V[lps-1,local_xm1,local_y,local_zm1]
                                            I131 = self.myTable[local_xm1,local_yp1,local_zm1]*self.V[lps-1,local_xm1,local_yp1,local_zm1]
                                            I211 = self.myTable[local_x,local_ym1,local_zm1]*self.V[lps-1,local_x,local_ym1,local_zm1]
                                            I221 = self.myTable[local_x,local_y,local_zm1]*self.V[lps-1,local_x,local_y,local_zm1]
                                            I231 = self.myTable[local_x,local_yp1,local_zm1]*self.V[lps-1,local_x,local_yp1,local_zm1]
                                            I311 = self.myTable[local_xp1,local_ym1,local_zm1]*self.V[lps-1,local_xp1,local_ym1,local_zm1]
                                            I321 = self.myTable[local_xp1,local_y,local_zm1]*self.V[lps-1,local_xp1,local_y,local_zm1]
                                            I331 = self.myTable[local_xp1,local_yp1,local_zm1]*self.V[lps-1,local_xp1,local_yp1,local_zm1]
                                            
                                            I113 = self.myTable[local_xm1,local_ym1,local_zp1]*self.V[lps-1,local_xm1,local_ym1,local_zp1]
                                            I123 = self.myTable[local_xm1,local_y,local_zp1]*self.V[lps-1,local_xm1,local_y,local_zp1]
                                            I133 = self.myTable[local_xm1,local_yp1,local_zp1]*self.V[lps-1,local_xm1,local_yp1,local_zp1]
                                            I213 = self.myTable[local_x,local_ym1,local_zp1]*self.V[lps-1,local_x,local_ym1,local_zp1]
                                            I223 = self.myTable[local_x,local_y,local_zp1]*self.V[lps-1,local_x,local_y,local_zp1]
                                            I233 = self.myTable[local_x,local_yp1,local_zp1]*self.V[lps-1,local_x,local_yp1,local_zp1]
                                            I313 = self.myTable[local_xp1,local_ym1,local_zp1]*self.V[lps-1,local_xp1,local_ym1,local_zp1]
                                            I323 = self.myTable[local_xp1,local_y,local_zp1]*self.V[lps-1,local_xp1,local_y,local_zp1]
                                            I333 = self.myTable[local_xp1,local_yp1,local_zp1]*self.V[lps-1,local_xp1,local_yp1,local_zp1]
                                            I224 = self.myTable[local_x,local_y,local_zp2]*self.V[lps-1,local_x,local_y,local_zp2]
                                            
                                            
                                                
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
                                            
                                            # coded status: V[:,:,:], decoded status: myTable[:,:]*V[:,:,:]
                                            if (Ii**2 + Ij**2 + Ik**2) == 0:
                                                self.V[lps,local_x,local_y,local_z] = self.V[lps-1,local_x,local_y,local_z]
                                            else:
                                                self.V[lps,local_x,local_y,local_z] = self.V[lps-1,local_x,local_y,local_z] + self.myTable[local_x,local_y,local_z]*self.dt*( (Ij**2+Ik**2)*Iii + (Ik**2+Ii**2)*Ijj + (Ii**2+Ij**2)*Ikk - 2*Ii*Ij*Iij - 2*Ij*Ik*Ijk - 2*Ik*Ii*Iik ) / (2*(Ii**2 + Ij**2 + Ik**2)**(1))
                                        
                        # decode the V matrix
                        if ((self.P[0,ip,j,k]-self.P[0,i,j,k])!=0):
                            dw = -1
                        else:
                            dw = 1
                        if ((self.P[0,im,j,k]-self.P[0,i,j,k])!=0):
                            up = -1
                        else:
                            up = 1
                        if ((self.P[0,i,jp,k]-self.P[0,i,j,k])!=0):
                            rt = -1
                        else:
                            rt = 1
                        if ((self.P[0,i,jm,k]-self.P[0,i,j,k])!=0):
                            lt = -1
                        else:
                            lt = 1
                        if ((self.P[0,i,j,km]-self.P[0,i,j,k])!=0):
                            fr = -1
                        else:
                            fr = 1
                        if ((self.P[0,i,j,kp]-self.P[0,i,j,k])!=0):
                            bk = -1
                        else:
                            bk = 1
                            
                        fval[i,j,k,0]=(up*self.V[self.nsteps,im,j,k]-dw*self.V[self.nsteps,ip,j,k])/2
                        fval[i,j,k,1]=(rt*self.V[self.nsteps,i,jp,k]-lt*self.V[self.nsteps,i,jm,k])/2
                        fval[i,j,k,2]=(bk*self.V[self.nsteps,i,j,kp]-fr*self.V[self.nsteps,i,j,km])/2
                    
        core_etime = datetime.datetime.now()
        print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval,(core_etime - core_stime).total_seconds(),self.V)
    
    def LS3dv1_main(self):
        # calculate time
        starttime = datetime.datetime.now()
        
        pool = mp.Pool(processes=self.cores)
        main_lc, main_wc, main_hc = myInput.split_cores(self.cores,3)
        
        # create all the queue
        manager = mp.Manager()
        all_queue = []
        print(f"The max size of queue {int(((self.nx/main_wc)+(self.ny/main_lc))*self.nsteps*self.nsteps)}")
        for queue_i in range(main_wc):
            tmp1 = []
            for queue_j in range(main_lc):
                tmp2 = []
                for queue_k in range(main_hc):
                    tmp2.append(manager.Queue(int(((self.nx/main_wc)+(self.ny/main_lc))*(1+self.nsteps)*(1+self.nsteps))))
                tmp1.append(tmp2)
            all_queue.append(tmp1)
        
        all_sites = np.array([[x,y,z] for x in range(self.nx) for y in range(self.ny) for z in range(self.nz) ]).reshape(self.nx,self.ny,self.nz,3)
        multi_input = myInput.split_IC(all_sites, self.cores,3, 0,1,2)
        
        res_list=[]
        for mpi in range(main_wc):
            for mpj in range(main_lc):
                for mpk in range(main_hc):
                    res_one = pool.apply_async(func = self.LS3dv1_core_apply, args = (multi_input[mpi][mpj][mpk], all_queue, ), callback=self.res_back )
                    res_list.append(res_one)
    
        pool.close()
        pool.join()
        print("core done!")
        # print(res_list[0].get())
    
        # calculate time
        endtime = datetime.datetime.now() 
        
        self.running_time = (endtime - starttime).total_seconds()
        # self.get_errors()

if __name__ == '__main__':

    LS3d_errors =np.zeros(10)
    LS3d_runningTime = np.zeros(10)
    
    nx, ny, nz = 100,100,100
    ng = 2
    # cores = 2
    
    # P0,R=myInput.init2IC(nx, ny, ng, "PolyIC.init")
    # P0,R=myInput.Circle_IC(nx,ny)
    P0,R=myInput.Circle_IC3d(nx,ny,nz)
    # P0,R = myInput.Complex2G_IC3d(nx,ny,nz)
    # P0[:,:,:],R=myInput.Voronoi_IC(nx,ny,ng)
    # P0[:,:,:],R=myInput.Complex2G_IC(nx,ny)
    # P0[:,:,:],R=myInput.Abnormal_IC(nx,ny)
    # P0[:,:,:]=myInput.SmallestGrain_IC(100,100)
    for cores in [8]:
    
        for nsteps in range(1,11): 
            test1 = LS3dv1_class(nx,ny,nz,ng,cores,nsteps,P0,R,'np')
            test1.LS3dv1_main()
            P = test1.get_P()
            V = test1.V
        
        #%% plot the figure2D
            test1.get_2d_plot('Poly', 'Level-Set')
            
            #%% error
            print('loop_times = ' + str(test1.nsteps))
            print('running_time = %.2f' % test1.running_time)
            print('running_core time = %.2f' % test1.running_coreTime)
            print('total_errors = %.2f' % test1.errors)
            print('per_errors = %.3f' % test1.errors_per_site)
            print()
            
            LS3d_errors[nsteps-1] = test1.errors_per_site
            LS3d_runningTime[nsteps-1] = test1.running_coreTime





