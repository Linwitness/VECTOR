#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 15:59:27 2021

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


class linear_class(object):

    def __init__(self,nx,ny,ng,cores,loop_times,P0,R,clip = 0):
        # V_matrix init value; runnning time and error for the algorithm
        self.running_time = 0
        self.running_coreTime = 0
        self.errors = 0
        self.errors_per_site = 0
        self.clip = clip

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
        # self.V_sparse = np.empty((loop_times+1,nx,ny),dtype=dict)  # matrix to store the tmp data during calculation

        # some attributes
        # linear smoothing matrix
        self.smoothed_vector_i, self.smoothed_vector_j = myInput.output_linear_vector_matrix(self.loop_times, self.clip)

    #%% Functions
    def get_P(self):
        # Outout the result matrix, first level is microstructure,
        # last two layers are normal vectors
        return self.P

    def get_errors(self):
        ge_gbsites = self.get_gb_list()
        for gbSite in ge_gbsites :
            [gei,gej] = gbSite
            ge_dx,ge_dy = myInput.get_grad(self.P,gei,gej)
            self.errors += math.acos(round(abs(ge_dx*self.R[gei,gej,0]+ge_dy*self.R[gei,gej,1]),5))

        self.errors_per_site = self.errors/len(ge_gbsites)

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




    def find_window(self,i,j):
        fw_len = self.tableL - 2*self.clip
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
                # window[wi,wj] = self.P[0,global_x,global_y]

        return window

    #%%
    # Core

    def linear_normal_vector_core(self,core_input, core_all_queue):
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
                # fv_i, fv_j = self.find_tableij(corner1,i,j)


                ip,im,jp,jm = myInput.periodic_bc(self.nx,self.ny,i,j)
                if ( ((self.P[0,ip,j]-self.P[0,i,j])!=0) or ((self.P[0,im,j]-self.P[0,i,j])!=0) or ((self.P[0,i,jp]-self.P[0,i,j])!=0) or ((self.P[0,i,jm]-self.P[0,i,j])!=0) ):

                    window = np.zeros((self.tableL,self.tableL))
                    window = self.find_window(i,j)
                    # print(window)

                    fval[i,j,0] = -np.sum(window*self.smoothed_vector_i)
                    fval[i,j,1] = np.sum(window*self.smoothed_vector_j)

        print(f"process{core_area_cen} read {test_check_read_num} times and max qsize {test_check_max_qsize}")
        core_etime = datetime.datetime.now()
        print("my core time is " + str((core_etime - core_stime).total_seconds()))
        return (fval,(core_etime - core_stime).total_seconds())

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

    def linear_main(self):

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
                res_one = pool.apply_async(func = self.linear_normal_vector_core, args = (multi_input[ki][kj], all_queue, ), callback=self.res_back )
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
    BL_errors =np.zeros(10)
    BL_runningTime = np.zeros(10)

    nx, ny = 200, 200
    ng = 2
    # cores = 8

    # P0,R=myInput.init2IC(nx, ny, ng, "PolyIC.init")
    P0,R=myInput.Circle_IC(nx,ny)
    # P0,R=myInput.Voronoi_IC(nx,ny,ng)
    # P0,R=myInput.Complex2G_IC(nx,ny)
    # P0,R=myInput.Abnormal_IC(nx,ny)
    # P0[:,:,:]=myInput.SmallestGrain_IC(100,100)

    for cores in [4]:
        # loop_times=10
        for loop_times in range(5,6):


            test1 = linear_class(nx,ny,ng,cores,loop_times,P0,R)
            test1.linear_main()
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
