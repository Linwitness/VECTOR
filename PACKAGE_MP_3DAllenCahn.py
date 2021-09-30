#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:13:02 2021

@author: lin.yang
"""


import sys
sys.path.append('/Users/lin.yang/Documents/SPYDER/AllenCahnSmooth/')
import numpy as np
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import myInput
import datetime


class AC3dv1_class(object):
    
    def __init__(self,nx,ny,nz,ng,cores,nsteps,P0,R,switch=False):
        # V_matrix init value; runnning time and error for the algorithm
        self.k = 1
        self.m = 1
        self.L = 1
        self.matrix_value = 10
        self.running_time = 0
        self.running_coreTime = 0
        self.errors = 0
        self.errors_per_site = 0
        self.switch = switch
        
        # initial condition data
        self.nx,self.ny,self.nz = nx, ny, nz
        self.ng = 2
        self.R = R  # results of analysis model
        # convert individual grains map into one grain map
        self.P = np.zeros((4,nx,ny,nz)) # matrix to store IC and normal vector results
        for i in range(0,np.shape(P0)[3]):
            self.P[0,:,:,:] += P0[:,:,:,i]*(i+1)
        
        # data for multiprocessing
        self.cores = cores
        
        # data for accuracy
        self.nsteps = nsteps #Number of timesteps
        self.dt = 0.1 #Timestep size
        self.tableL = 2*(nsteps+1)+1 # when repeatting two times, the table length will be 7 (7 by 7 table)
        self.halfL = nsteps+1
        
        # temporary matrix to increase efficiency 
        self.V = np.ones((nsteps+1,nx,ny,nz,ng))*self.matrix_value
        
        

    
    
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
        z_surface = int(self.nz/2)
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
        plt.savefig(f'{init}-{algo}.{String}.png',dpi=1000,bbox_inches='tight')
    
    def get_gb_num(self,grainID=1):
        ggn_gbsites = []  
        edge_l = 1
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
    
    
    #%% Core  
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
        
    def AC3dv1_core_apply(self,core_input, core_all_queue):
        core_stime = datetime.datetime.now()
        li,lj,lk,lp=np.shape(core_input)
        fval = np.zeros((self.nx,self.ny,self.nz,3))
        
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
        
                    
                        # convert the small table into 0 and 1
                        for ii in range(-self.halfL,self.halfL+1):
                            for jj in range(-self.halfL,self.halfL+1):
                                for kk in range(-self.halfL,self.halfL+1):
                                    local_x = (i+ii)%self.nx
                                    local_y = (j+jj)%self.ny
                                    local_z = (k+kk)%self.nz
                                
                                    # plus and minus matrix to code the BL function
                                    if self.V[0,local_x,local_y,local_z,int(self.P[0,i,j,k]-1)] == self.matrix_value:
                                        if self.P[0,local_x,local_y,local_z] != self.P[0,i,j,k]:
                                            self.V[0,local_x,local_y,local_z,int(self.P[0,i,j,k]-1)] = 0
                                        else:
                                            self.V[0,local_x,local_y,local_z,int(self.P[0,i,j,k]-1)] = 1
                                    
                        #  calculate the smooth value
                        for lps in range(1,self.nsteps+1):
                            for ii in range(-self.halfL+lps,self.halfL+1-lps):
                                for jj in range(-self.halfL+lps,self.halfL+1-lps):
                                    for kk in range(-self.halfL+lps,self.halfL+1-lps):
                                        local_x = (i+ii)%self.nx
                                        local_y = (j+jj)%self.ny
                                        local_z = (k+kk)%self.nz
                                        if self.V[lps,local_x,local_y,local_z,int(self.P[0,i,j,k]-1)] == self.matrix_value:
                                            # necessary coordination
                                            local_xp1 = (i+ii+1)%self.nx
                                            local_xm1 = (i+ii-1)%self.nx
                                            local_yp1 = (j+jj+1)%self.ny
                                            local_ym1 = (j+jj-1)%self.ny
                                            local_zp1 = (k+kk+1)%self.ny
                                            local_zm1 = (k+kk-1)%self.ny
                                            
                                            Etas = ( self.V[lps-1,local_x,local_y,local_z,int(self.P[0,i,j,k]-1)]**2+(1-self.V[lps-1,local_x,local_y,local_z,int(self.P[0,i,j,k]-1)])**2 )-self.V[lps-1,local_x,local_y,local_z,int(self.P[0,i,j,k]-1)]
                                            df0 = self.V[lps-1,local_x,local_y,local_z,int(self.P[0,i,j,k]-1)]**3-self.V[lps-1,local_x,local_y,local_z,int(self.P[0,i,j,k]-1)]+3*self.V[lps-1,local_x,local_y,local_z,int(self.P[0,i,j,k]-1)]*Etas  #Free energy derivative, simple multi-well
                                            fd = (self.V[lps-1,local_xm1,local_y,local_z,int(self.P[0,i,j,k]-1)]+self.V[lps-1,local_xp1,local_y,local_z,int(self.P[0,i,j,k]-1)] +\
                                                  self.V[lps-1,local_x,local_ym1,local_z,int(self.P[0,i,j,k]-1)]+self.V[lps-1,local_x,local_yp1,local_z,int(self.P[0,i,j,k]-1)] +\
                                                  self.V[lps-1,local_x,local_y,local_zm1,int(self.P[0,i,j,k]-1)]+self.V[lps-1,local_x,local_y,local_zp1,int(self.P[0,i,j,k]-1)] -\
                                                  6*self.V[lps-1,local_x,local_y,local_z,int(self.P[0,i,j,k]-1)] )/1**2 #2nd order central differencing
                                            self.V[lps,local_x,local_y,local_z,int(self.P[0,i,j,k]-1)] = self.V[lps-1,local_x,local_y,local_z,int(self.P[0,i,j,k]-1)] - self.L*(self.m*df0-self.k*fd)*self.dt
                                        
                        fval[i,j,k,0] = (self.V[self.nsteps,im,j,k,int(self.P[0,i,j,k]-1)]-self.V[self.nsteps,ip,j,k,int(self.P[0,i,j,k]-1)])/2
                        fval[i,j,k,1] = (self.V[self.nsteps,i,jp,k,int(self.P[0,i,j,k]-1)]-self.V[self.nsteps,i,jm,k,int(self.P[0,i,j,k]-1)])/2
                        fval[i,j,k,2] = (self.V[self.nsteps,i,j,kp,int(self.P[0,i,j,k]-1)]-self.V[self.nsteps,i,j,km,int(self.P[0,i,j,k]-1)])/2
        core_etime = datetime.datetime.now()
        print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval,(core_etime - core_stime).total_seconds(),self.V)
    
    def AC3dv1_main(self):
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
                    # print(f"{mpi} {mpj} {mpk}")
                    res_one = pool.apply_async(func = self.AC3dv1_core_apply, args = (multi_input[mpi][mpj][mpk], all_queue, ), callback=self.res_back )
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
    
    AC3d_errors =np.zeros(10)
    AC3d_runningTime = np.zeros(10)
    
    nx, ny, nz = 100, 100, 100
    ng = 2
    
    # P0,R=myInput.init2IC(nx, ny, ng, "PolyIC.init")
    # P0,R=myInput.Circle_IC(nx,ny)
    # P0,R=myInput.Circle_IC3d(nx,ny,nz)
    P0,R = myInput.Complex2G_IC3d(nx,ny,nz)
    # P0[:,:,:],R=myInput.Voronoi_IC(nx,ny,ng)
    # P0[:,:,:],R=myInput.Complex2G_IC(nx,ny)
    # P0[:,:,:],R=myInput.Abnormal_IC(nx,ny)
    # P0[:,:,:]=myInput.SmallestGrain_IC(100,100)
    
    for cores in [8]:
        for nsteps in range(2,21,2):
            
            test1 = AC3dv1_class(nx,ny,nz,ng,cores,nsteps,P0,R)
            test1.AC3dv1_main()
            P = test1.get_P()
            V = test1.V
        
            #%% Figure
                        
            test1.get_2d_plot('Poly','Allen-Cahn')
                        
        
            
            #%% errors
            print('loop_times = ' + str(test1.nsteps))
            print('running_time = %.2f' % test1.running_time)
            print('running_core time = %.2f' % test1.running_coreTime)
            print('total_errors = %.2f' % test1.errors)
            print('per_errors = %.3f' % test1.errors_per_site)
            print()
            
            AC3d_errors[int(nsteps/2-1)] = test1.errors_per_site
            AC3d_runningTime[int(nsteps/2-1)] = test1.running_coreTime