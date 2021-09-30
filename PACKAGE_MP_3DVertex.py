#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 19:54:13 2021

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

class VT3dv1_class(object):
    def __init__(self,nx,ny,nz,ng,cores,interval,P0,R):
        # V_matrix init value; runnning time and error for the algorithm
        self.running_time = 0
        self.running_coreTime = 0
        self.errors = 0
        self.errors_per_site = 0
        
        # initial condition data
        self.nx,self.ny,self.nz = nx, ny, nz
        self.ng = ng
        self.R = R  # results of analysis model
        # convert individual grains map into one grain map
        self.P = np.zeros((4,nx,ny,nz)) # matrix to store IC and normal vector results
        for i in range(0,np.shape(P0)[3]):
            self.P[0,:,:,:] += P0[:,:,:,i]*(i+1)
        
        # data for multiprocessing
        self.cores = cores
        
        # data for accuracy
        self.interval = interval #Number of timesteps
    
    
    
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
        # z_surface = int(self.nz/2)
        z_surface = int(40)
        m = 35
        n = 0
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
        plt.imshow(self.P[0,:,:,z_surface], cmap='nipy_spectral', interpolation='nearest')
        
        g2p_gbsites = self.get_gb_num()
        for gbSite in g2p_gbsites:
            [g2pi,g2pj,g2pk] = gbSite
            if g2pk==z_surface:
                if m != 0:
                    if self.P[0,g2pi+m,g2pj+n,g2pk]==self.P[0,g2pi,g2pj,g2pk]:
                        anchor = np.array([g2pi+m,g2pj+n,g2pk])
                    elif self.P[0,g2pi-m,g2pj+n,g2pk]==self.P[0,g2pi,g2pj,g2pk]:
                        anchor = np.array([g2pi-m,g2pj+n,g2pk])
                    elif self.P[0,g2pi+m,g2pj-n,g2pk]==self.P[0,g2pi,g2pj,g2pk]:
                        anchor = np.array([g2pi+m,g2pj-n,g2pk])
                    else:
                        anchor = np.array([g2pi-m,g2pj-n,g2pk])
                    m=0
                if np.dot( (anchor - np.array([g2pi,g2pj,g2pk])), (np.array([self.P[1,g2pi,g2pj,g2pk],self.P[2,g2pi,g2pj,g2pk],self.P[3,g2pi,g2pj,g2pk]])) ) >= 0:
                    g2p_prof = -1
                else:
                    g2p_prof = 1
                    
                g2p_dx,g2p_dy,g2p_dz = myInput.get_grad3d(self.P,g2pi,g2pj,g2pk)
                plt.arrow(g2pj,g2pi,g2p_prof*10*g2p_dx,g2p_prof*10*g2p_dy,width=0.1,lw=0.1,alpha=0.8,color='navy')
        
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
        
    def check_coplane(self,array):
        array2 = [None]*len(array)
        for arri in range(0,len(array)):
            array2[arri] = [int(x) for x in array[arri].split(',')]
            
        line1 = [(array2[0][0]-array2[1][0]), (array2[0][1]-array2[1][1]), (array2[0][2]-array2[1][2])]
        line2 = [(array2[0][0]-array2[2][0]), (array2[0][1]-array2[2][1]), (array2[0][2]-array2[2][2])]
        line3 = [(array2[0][0]-array2[3][0]), (array2[0][1]-array2[3][1]), (array2[0][2]-array2[3][2])]
        if abs(np.dot(line3,np.cross(line1,line2))) <= 1e-6:
            if len(array2) <= 4:
                return True
            else:
                totalline = [line1,line2,line3]
                for cci in range(0,len(array2)-4):
                    totalline.append([(array2[0][0]-array2[cci+4][0]), (array2[0][1]-array2[cci+4][1]), (array2[0][2]-array2[cci+4][2])])
                    if abs(np.dot(totalline[-1],np.cross(totalline[-2],totalline[-3]))) > 1e-6:
                        return False
                return True
                
        else:
            return False
    
    def find_crossLine(self,array):
        array2 = [None]*len(array)
        for arri in range(0,len(array)):
            array2[arri] = [int(x) for x in array[arri].split(',')]
            
        line1 = [(array2[0][0]-array2[1][0]), (array2[0][1]-array2[1][1]), (array2[0][2]-array2[1][2])]
        line2 = [(array2[0][0]-array2[2][0]), (array2[0][1]-array2[2][1]), (array2[0][2]-array2[2][2])]
        
        if np.linalg.norm(np.cross(line1,line2)) <= 1e-6:
            totalline = [line1,line2]
            for fci in range(0,len(array2)-3):
                totalline.append([(array2[0][0]-array2[fci+3][0]), (array2[0][1]-array2[fci+3][1]), (array2[0][2]-array2[fci+3][2])])
                if np.linalg.norm(np.cross(totalline[-1],totalline[-2])) >= 1e-6:
                    return np.cross(totalline[-1],totalline[-2])/np.linalg.norm(np.cross(totalline[-1],totalline[-2]))
            return print("All points in one line!!!!")
        else:
            return np.cross(line2,line1)/np.linalg.norm(np.cross(line1,line2))
        
    
    
    def find_fittingSphere(self,array):
        # print(array)
        array2 = [None]*len(array)
        for arri in range(0,len(array)):
            array2[arri] = [int(x) for x in array[arri].split(',')]
        
        K_mat = []
        Y_mat = []
        
        for point in array2:
            K_mat.append([-2*point[0],-2*point[1],-2*point[2],1])
            Y_mat.append([-point[0]**2-point[1]**2-point[2]**2])
        
        K_mat = np.mat(K_mat)
        Y_mat = np.mat(Y_mat)
        K_mat_tra = K_mat.T
        X_mat = (K_mat_tra*K_mat).I * K_mat_tra * Y_mat
        fit_x0 = X_mat[0]
        fit_y0 = X_mat[1]
        fit_z0 = X_mat[2]
        fit_R = math.sqrt(fit_x0**2+fit_y0**2+fit_z0**2-X_mat[3])
        
        return fit_y0, fit_x0, fit_z0, fit_R
        


    def find_neighbor_sites(self,ori_A):
        
        A = [int(x) for x in ori_A.split(',')]
        ip,im,jp,jm,kp,km = myInput.periodic_bc3d(self.nx,self.ny,self.nz,A[0],A[1],A[2])
        output = []
        
        for fi in [-1,0,1]:
            for fj in [-1,0,1]:
                for fk in [-1,0,1]:
                    local_i = (A[0]+fi)%self.ny
                    local_j = (A[1]+fj)%self.nx
                    local_k = (A[2]+fk)%self.nz
                    fip,fim,fjp,fjm,fkp,fkm = myInput.periodic_bc3d(self.nx,self.ny,self.nz,local_i,local_j,local_k)
                    if (self.P[0,local_i,local_j,local_k] == self.P[0,A[0],A[1],A[2]]) and \
                       ( ((self.P[0,fip,local_j,local_k]-self.P[0,local_i,local_j,local_k])!=0) or ((self.P[0,fim,local_j,local_k]-self.P[0,local_i,local_j,local_k])!=0) or \
                         ((self.P[0,local_i,fjp,local_k]-self.P[0,local_i,local_j,local_k])!=0) or ((self.P[0,local_i,fjm,local_k]-self.P[0,local_i,local_j,local_k])!=0) or \
                         ((self.P[0,local_i,local_j,fkp]-self.P[0,local_i,local_j,local_k])!=0) or ((self.P[0,local_i,local_j,fkm]-self.P[0,local_i,local_j,local_k])!=0) ):
                        output.append(f"{local_i},{local_j},{local_k}")
                        
        output = list(set(output)-set([f"{A[0]},{A[1]},{A[2]}"]))
        
        return output
    
    
    #%% Core base
    
    def VT3dv1_core(self,core_input):
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
                        
                        # collect all data
                        stored_boun = [[f"{i},{j},{k}"]]
                        
                        for surr_i in range(0,self.interval):
                            # surr_i:  all sites in each "circle" on surface
                            surr_sites = stored_boun[surr_i]
                            next_cir = []
                            for surr_j in surr_sites:
                                # surr_j: each site on the "circle"
                                next_cir.extend(self.find_neighbor_sites(surr_j))
                            
                            # remove all all sites on otehr "circle"
                            for surr_k in range(0,surr_i+1):
                                # surr_k: the number order of previous level
                                next_cir = list(set(next_cir)-set(stored_boun[surr_k]))
                            
                            if len(next_cir) == 0:
                                pass
                            # elif len(next_cir) < 4:
                            #     pass
                            
                            stored_boun.append(next_cir)
                        # print(stored_boun)
                        # choose the data you need
                        fitting_data = stored_boun[0]
                        fitting_data.extend(stored_boun[-1])
                        
                        # if we dont have enought points
                        if len(fitting_data) < 4:
                            fitting_data = stored_boun[0]
                            for fdi in range(1,len(stored_boun)):
                                fitting_data.extend(stored_boun[fdi])
                            
                            if len(fitting_data) < 4:
                                fval[i,j,k,0], fval[i,j,k,1], fval[i,j,k,2] = 1,0,0
                                break
                        
                        # print(fitting_data)
                        # calculation for plane and hump
                        if self.check_coplane(fitting_data):
                            # calculay=te normal vector on one plane
                            [fval[i,j,k,0], fval[i,j,k,1], fval[i,j,k,2]] = self.find_crossLine(fitting_data)
                            
                        else:
                            # print(fitting_data)
                            x0,y0,z0,radius = self.find_fittingSphere(fitting_data)
                            fval[i,j,k,0] = (i-y0)/radius
                            fval[i,j,k,1] = (x0-j)/radius
                            fval[i,j,k,2] = (z0-k)/radius
                                
                            
                        
                    
        core_etime = datetime.datetime.now()
        print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval,(core_etime - core_stime).total_seconds())
    
    def VT3dv1_main(self):
        # calculate time
        starttime = datetime.datetime.now()
        
        pool = mp.Pool(processes=self.cores)
        main_lc, main_wc, main_hc = myInput.split_cores(self.cores,3)
        
        all_sites = np.array([[x,y,z] for x in range(self.nx) for y in range(self.ny) for z in range(self.nz) ]).reshape(self.nx,self.ny,self.nz,3)
        multi_input = myInput.split_IC(all_sites, self.cores,3, 0,1,2)
        
        res_list=[]        
        for mpi in range(main_wc):
            for mpj in range(main_lc):
                for mpk in range(main_hc):
                    res_one = pool.apply_async(func = self.VT3dv1_core, args = (multi_input[mpi][mpj][mpk], ), callback=self.res_back )
                    res_list.append(res_one)
        
        pool.close()
        pool.join()
        print("core done!")
        # print(res_list[0].get())
    
        # calculate time
        endtime = datetime.datetime.now() 
        
        self.running_time = (endtime - starttime).total_seconds()
        # self.get_errors()
    
    def res_back(self,back_result):
        res_stime = datetime.datetime.now()
        (fval,core_time) = back_result
        if core_time > self.running_coreTime:
            self.running_coreTime = core_time
        
        print("res_back start...")
        self.P[1,:,:,:] += fval[:,:,:,0]
        self.P[2,:,:,:] += fval[:,:,:,1]
        self.P[3,:,:,:] += fval[:,:,:,2]
        res_etime = datetime.datetime.now()
        print("my res time is " + str((res_etime - res_stime).total_seconds()))
        
        
        
#%%
if __name__ == '__main__':
    
    VT3d_errors =np.zeros(10)
    VT3d_runningTime = np.zeros(10)
    # ctx = mp.get_context('fork')
    interval = range(1,11)
    nx, ny, nz = 100, 100, 100
    ng = 2
    cores = [8]
    
    # P0,R=myInput.init2IC(nx, ny, ng, "PolyIC.init")
    # P0,R=myInput.Circle_IC(nx,ny)
    P0,R=myInput.Circle_IC3d(nx,ny,nz)
    # P0,R = myInput.Complex2G_IC3d(nx,ny,nz)
    # P0,R=myInput.Voronoi_IC(nx,ny,ng)
    # P0,R=myInput.Complex2G_IC(nx,ny)
    # P0[:,:,:],R=myInput.Abnormal_IC(nx,ny)
    # P0[:,:,:]=myInput.SmallestGrain_IC(100,100)
    for ci in cores:
        for inti in interval:
            test1 = VT3dv1_class(nx,ny,nz,ng,ci,inti,P0,R)
            test1.VT3dv1_main()
            P = test1.get_P()
            
            #%% figure
            test1.get_2d_plot('Poly', 'Vertex')
            
            #%% error
            
                    
            print('loop_times = ' + str(test1.interval))
            print('running_time = %.2f' % test1.running_time)
            print('running_core time = %.2f' % test1.running_coreTime)
            print('total_errors = %.2f' % test1.errors)
            print('per_errors = %.3f' % test1.errors_per_site)
            print()
        
            VT3d_errors[inti-1] = test1.errors_per_site
            VT3d_runningTime[inti-1] = test1.running_coreTime