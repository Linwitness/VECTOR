#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 15:59:27 2021

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


class BLv2_class(object):
    
    def __init__(self,nx,ny,ng,cores,loop_times,P0,R,switch = False):
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
        self.ng = ng # number of grains in IC
        self.R = R  # results of analysis model
        # convert individual grains map into one grain map
        self.P = np.zeros((3,nx,ny)) # matrix to store IC and normal vector results
        for i in range(0,np.shape(P0)[2]):
            self.P[0,:,:] += P0[:,:,i]*(i+1)
        
        # data for multiprocessing
        self.cores = cores
        
        # data for accuracy
        self.loop_times = loop_times
        self.tableL = 2*(loop_times+1)+1 # when repeatting two times, the table length will be 7 (7 by 7 table)
        self.halfL = loop_times+1
        
        # temporary matrix to increase efficiency  
        self.V_sparse = np.empty((loop_times+1,nx,ny),dtype=dict)  # matrix to store the tmp data during calculation
        
        # some attributes
        # arrows
        
        
    
        
    
        
    #%% Functions
    def get_P(self):
        return self.P
    
    def get_errors(self):
        ge_gbsites = self.get_gb_num()
        for gbSite in ge_gbsites :
            [gei,gej] = gbSite
            ge_dx,ge_dy = myInput.get_grad(self.P,gei,gej)
            self.errors += math.acos(round(abs(ge_dx*self.R[gei,gej,0]+ge_dy*self.R[gei,gej,1]),5))
        
        self.errors_per_site = self.errors/len(ge_gbsites)
        
        # return self.errors, self.errors_per_site
        
    def get_2d_plot(self,init,algo):
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
        
        g2p_gbsites = self.get_gb_num()
        for gbSite in g2p_gbsites:
            [g2pi,g2pj] = gbSite
            g2p_dx,g2p_dy = myInput.get_grad(self.P,g2pi,g2pj)
            if g2pi >200 and g2pi<500:
                plt.arrow(g2pj,g2pi,30*g2p_dx,30*g2p_dy,width=0.1,lw=0.1,alpha=0.8,color='navy')
        
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('BL_PolyGray_Arrows.png',dpi=1000,bbox_inches='tight')
    
    def get_gb_num(self,grainID=1):
        ggn_gbsites = []  
        for i in range(0,self.nx):
            for j in range(0,self.ny):
                ip,im,jp,jm = myInput.periodic_bc(self.nx,self.ny,i,j)
                if ( ((self.P[0,ip,j]-self.P[0,i,j])!=0) or ((self.P[0,im,j]-self.P[0,i,j])!=0) or ((self.P[0,i,jp]-self.P[0,i,j])!=0) or ((self.P[0,i,jm]-self.P[0,i,j])!=0) )\
                        and self.P[0,i,j]==grainID:
                    ggn_gbsites.append([i,j])
        return ggn_gbsites
                    
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
        
    
    def transfer_data(self,i, ii, j, jj, corner1, corner3, kk=0):
        td_out = []
        
        if abs(i+ii-corner1[0]) < self.halfL-kk:
            # go 1
            td_out.append(1)
            
            if abs(j+jj-corner1[1]) < self.halfL-kk:
                #go 0 and 7
                td_out.append(0)
                td_out.append(7)
                
            elif abs(j+jj-corner3[1]) < self.halfL-kk:
                #go 2 and 3
                td_out.append(2)
                td_out.append(3)
                
        elif abs(i+ii-corner3[0]) < self.halfL-kk:
            # go 5
            td_out.append(5)
            
            if abs(j+jj-corner1[1]) < self.halfL-kk:
                #go 6 and 7
                td_out.append(6)
                td_out.append(7)
                
            elif abs(j+jj-corner3[1]) < self.halfL-kk:
                # go 4 and 3
                td_out.append(3)
                td_out.append(4)
                
        elif abs(j+jj-corner1[1]) < self.halfL-kk:
            # go 7
            td_out.append(7)
            
        elif abs(j+jj-corner3[1]) < self.halfL-kk:
            # go 3
            td_out.append(3)
        
        return td_out
    
    # def find_tableij(self,corner,i,j):
    #     return abs(i-corner[0]), abs(j-corner[1])
    
    
    #%%
    # Core
    
    def BLv2_core_apply(self,core_input, core_all_queue):
        core_stime = datetime.datetime.now()
        li,lj,lk=np.shape(core_input)
        fval = np.zeros((self.nx,self.ny,2))
        
        corner1 = core_input[0,0,:]
        corner3 = core_input[li-1,lj-1,:]
        
        core_area_cen, core_area_nei = self.check_subdomain_and_nei(corner1)
        print(f'the processor {core_area_cen} start...')
        
        # MY TEST AREAQ
        # core_A,core_B = split_cores(cores)
        # print_me("lalala")
        # print("core start...")
        # core_test=math.floor(10.5)
        
        test_check_read_num = 0
        test_check_max_qsize = 0
        for core_a in core_input:
            for core_b in core_a:
                i = core_b[0]
                j = core_b[1]
                # fv_i, fv_j = self.find_tableij(corner1,i,j)
                
                # if self.switch:
                #     #  check fi the queue of the area is enmty of not
                #     while not core_all_queue[core_area_cen[0]][core_area_cen[1]].empty():
                        
                #         core_tmp_qsize = core_all_queue[core_area_cen[0]][core_area_cen[1]].qsize()
                #         if test_check_max_qsize < core_tmp_qsize:
                #             test_check_max_qsize = core_tmp_qsize
                            
                #         trs_data = core_all_queue[core_area_cen[0]][core_area_cen[1]].get()
                #         if self.V[trs_data[0],trs_data[1],trs_data[2],trs_data[3]] == self.matrix_value:
                #             self.V[trs_data[0],trs_data[1],trs_data[2],trs_data[3]] = trs_data[4]
                #             test_check_read_num+=1
                #             # if trs_data[0] != 0:
    
                
                ip,im,jp,jm = myInput.periodic_bc(self.nx,self.ny,i,j)
                if ( ((self.P[0,ip,j]-self.P[0,i,j])!=0) or ((self.P[0,im,j]-self.P[0,i,j])!=0) or ((self.P[0,i,jp]-self.P[0,i,j])!=0) or ((self.P[0,i,jm]-self.P[0,i,j])!=0) ):
                    
                    # convert the small table into 0 and 1
                    for ii in range(-self.halfL,self.halfL+1):
                        for jj in range(-self.halfL,self.halfL+1):
                            local_x = (i+ii)%self.nx
                            local_y = (j+jj)%self.ny
                            
                            if type(self.V_sparse[0,local_x,local_y]) != dict:
                                self.V_sparse[0,local_x,local_y] = {}
                            
                            # plus and minus matrix to code the BL function
                            if int(self.P[0,i,j]) not in self.V_sparse[0,local_x,local_y]:
                                if self.P[0,local_x,local_y] != self.P[0,i,j]:
                                    self.V_sparse[0,local_x,local_y] = {int(self.P[0,i,j]):0}
                                else:
                                    self.V_sparse[0,local_x,local_y] = {int(self.P[0,i,j]):1}
                                
                                # if self.switch:
                                #     #  which queue the data need to send
                                #     target_queue = self.transfer_data(i,ii,j,jj,corner1,corner3)
                                #     if len(target_queue) > 0:
                                #         for tqi in target_queue:
                                #             core_all_queue[core_area_nei[tqi][0]][core_area_nei[tqi][1]].put([0,local_x,local_y,int(self.P[0,i,j]-1), self.V[0,local_x,local_y,int(self.P[0,i,j]-1)]])
    
                    
                    #  calculate the smooth value
                    for kk in range(1,self.loop_times+1):
                        for ii in range(-self.halfL+kk,self.halfL+1-kk):
                            for jj in range(-self.halfL+kk,self.halfL+1-kk):
                                local_x = (i+ii)%self.nx
                                local_y = (j+jj)%self.ny
                                
                                if type(self.V_sparse[kk,local_x,local_y]) != dict:
                                    self.V_sparse[kk,local_x,local_y] = {}
                                
                                if int(self.P[0,i,j]) not in self.V_sparse[kk,local_x,local_y]:
                                    # necessary coordination
                                    local_xp1 = (i+ii+1)%self.nx
                                    local_xm1 = (i+ii-1)%self.nx
                                    local_yp1 = (j+jj+1)%self.ny
                                    local_ym1 = (j+jj-1)%self.ny
                                    
                                    self.V_sparse[kk,local_x,local_y][int(self.P[0,i,j])] = 1./4*self.V_sparse[kk-1,local_x,local_y][int(self.P[0,i,j])] + \
                                                                            1./8*(self.V_sparse[kk-1,local_xm1,local_y][int(self.P[0,i,j])]+self.V_sparse[kk-1,local_xp1,local_y][int(self.P[0,i,j])]+self.V_sparse[kk-1,local_x,local_ym1][int(self.P[0,i,j])]+self.V_sparse[kk-1,local_x,local_yp1][int(self.P[0,i,j])]) + \
                                                                            1./16*(self.V_sparse[kk-1,local_xm1,local_ym1][int(self.P[0,i,j])]+self.V_sparse[kk-1,local_xm1,local_yp1][int(self.P[0,i,j])]+self.V_sparse[kk-1,local_xp1,local_ym1][int(self.P[0,i,j])]+self.V_sparse[kk-1,local_xp1,local_yp1][int(self.P[0,i,j])])
                                    
                                    # if self.switch:
                                    #     #  which queue the data need to send
                                    #     target_queue = self.transfer_data(i,ii,j,jj,corner1,corner3,kk)
                                    #     if len(target_queue) > 0:
                                    #         for tqi in target_queue:
                                    #             core_all_queue[core_area_nei[tqi][0]][core_area_nei[tqi][1]].put([kk,local_x,local_y,int(self.P[0,i,j]-1), self.V[kk,local_x,local_y,int(self.P[0,i,j]-1)]])
                                    
                                # else:
                                #     print("ohHa ohHa " + str(os.getpid()) + " parents " + str(os.getppid()))
                    
    
                                
                    fval[i,j,0]=(self.V_sparse[self.loop_times,im,j][int(self.P[0,i,j])]-self.V_sparse[self.loop_times,ip,j][int(self.P[0,i,j])])/2
                    fval[i,j,1]=(self.V_sparse[self.loop_times,i,jp][int(self.P[0,i,j])]-self.V_sparse[self.loop_times,i,jm][int(self.P[0,i,j])])/2
                    # fval[fv_i,fv_j,2] = i
                    # fval[fv_i,fv_j,3] = j
        print(f"process{core_area_cen} read {test_check_read_num} times and max qsize {test_check_max_qsize}")
        core_etime = datetime.datetime.now()
        print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval,(core_etime - core_stime).total_seconds())
        
    # def res_back(self,back_result):
    #     res_stime = datetime.datetime.now()
    #     (fval,core_time) = back_result
    #     if core_time > self.running_coreTime:
    #         self.running_coreTime = core_time
        
    #     print("res_back start...")
    #     rb_i, rb_j, _ = np.shape(fval)
    #     for i in range(rb_i):
    #         for j in range(rb_j):
    #             self.P[1,int(fval[i,j,2]),int(fval[i,j,3])] = fval[i,j,0]
    #             self.P[2,int(fval[i,j,2]),int(fval[i,j,3])] = fval[i,j,1]  
    #     res_etime = datetime.datetime.now()
    #     print("my res time is " + str((res_etime - res_stime).total_seconds()))
    
    def res_back(self,back_result):
        res_stime = datetime.datetime.now()
        (fval,core_time) = back_result
        if core_time > self.running_coreTime:
            self.running_coreTime = core_time
        
        print("res_back start...")
        self.P[1,:,:] += fval[:,:,0]
        self.P[2,:,:] += fval[:,:,1]
        res_etime = datetime.datetime.now()
        print("my res time is " + str((res_etime - res_stime).total_seconds()))
    
    def BLv2_main(self):
        
        # global starttime, endtime
        starttime = datetime.datetime.now() 
    
        pool = mp.Pool(processes=self.cores)
        main_lc, main_wc = myInput.split_cores(self.cores)
        
        # create all the queue
        manager = mp.Manager()
        all_queue = []
        print(f"The max size of queue {int(((self.nx/main_wc)+(self.ny/main_lc))*self.loop_times*self.loop_times)}")
        for queue_i in range(main_wc):
            tmp = []
            for queue_j in range(main_lc):
                tmp.append(manager.Queue(int(((self.nx/main_wc)+(self.ny/main_lc))*(1+self.loop_times)*(1+self.loop_times))))
            all_queue.append(tmp)
                
        
        all_sites = np.array([[x,y] for x in range(self.nx) for y in range(self.ny) ]).reshape(self.nx,self.ny,2)
        multi_input = myInput.split_IC(all_sites, self.cores,2, 0,1)
        
        res_list=[]
        for ki in range(main_wc):
            for kj in range(main_lc):
                res_one = pool.apply_async(func = self.BLv2_core_apply, args = (multi_input[ki][kj], all_queue, ), callback=self.res_back )
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
    BL_errors =np.zeros(10)
    BL_runningTime = np.zeros(10)
    
    nx, ny = 200, 200
    ng = 5
    # cores = 8
    
    # P0,R=myInput.init2IC(nx, ny, ng, "PolyIC.init")
    # P0,R=myInput.Circle_IC(nx,ny)
    P0,R=myInput.Voronoi_IC(nx,ny,ng)
    # P0,R=myInput.Complex2G_IC(nx,ny)
    # P0,R=myInput.Abnormal_IC(nx,ny)
    # P0[:,:,:]=myInput.SmallestGrain_IC(100,100)
    
    for cores in [1]:
        # loop_times=10
        for loop_times in range(1,11):
            
            
            test1 = BLv2_class(nx,ny,ng,cores,loop_times,P0,R)
            test1.BLv2_main()
            P = test1.get_P()
        
            
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
        
