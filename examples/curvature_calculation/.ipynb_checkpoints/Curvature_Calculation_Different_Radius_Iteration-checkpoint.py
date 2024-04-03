#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 17:46:02 2022

@author: Lin
"""

import os
current_path = os.getcwd()+'/'
import sys
sys.path.append(current_path)
sys.path.append(current_path+'../../')
import numpy as np
import math

import PACKAGE_MP_Linear as Linear_2D
import PACKAGE_MP_Vertex as Vertex_2D
import PACKAGE_MP_3DLinear as Linear_3D
import PACKAGE_MP_3DVertex as Vertex_3D
import myInput

def test_2d():
    nx, ny = 200, 200
    ng = 2
    cores = 8
    max_iteration = 20
    radius = 20
    filename_save = f"examples/curvature_calculation/BL_Curvature_R{radius}_Iteration_1_{max_iteration}"
    
    BL_errors =np.zeros(max_iteration)
    BL_runningTime = np.zeros(max_iteration)

    # P0,R=myInput.init2IC(nx, ny, ng, "PolyIC.init")
    P0,R=myInput.Circle_IC(nx,ny,radius)
    # P0,R=myInput.Voronoi_IC(nx,ny,ng)

    for cores in [cores]:
        # loop_times=10
        for loop_times in range(1,max_iteration):


            test1 = Linear_2D.linear_class(nx,ny,ng,cores,loop_times,P0,R)
            # test1.linear_main()
            # P = test1.get_P()

            test1.linear_main("curvature")
            C_ln = test1.get_C()

            # error

            print('loop_times = ' + str(test1.loop_times))
            print('running_time = %.2f' % test1.running_time)
            print('running_core time = %.2f' % test1.running_coreTime)
            print('total_errors = %.2f' % test1.errors)
            print('per_errors = %.3f' % test1.errors_per_site)
            print()


            BL_errors[loop_times-1] = test1.errors_per_site
            BL_runningTime[loop_times-1] = test1.running_coreTime
            
    np.savez(filename_save, BL_errors=BL_errors, BL_runningTime=BL_runningTime)

def test_vertex_2d():
    nx, ny = 200, 200
    ng = 2
    cores = 8
    max_iteration = 20
    radius = 80
    filename_save = f"./VT_Curvature_R{radius}_Iteration_1_{max_iteration}"
    
    VT_errors =np.zeros(max_iteration)
    VT_runningTime = np.zeros(max_iteration)

    # P0,R=myInput.init2IC(nx, ny, ng, "PolyIC.init")
    P0,R=myInput.Circle_IC(nx,ny,radius)
    # P0,R=myInput.Voronoi_IC(nx,ny,ng)

    for cores in [cores]:
        # interval=10
        for interval in range(1,max_iteration):


            test1 = Vertex_2D.vertex_class(nx,ny,ng,cores,interval,P0,R)
            # test1.linear_main()
            # P = test1.get_P()

            test1.vertex_main("curvature")
            C_ln = test1.get_C()

            # error

            print('interval = ' + str(test1.interval))
            print('running_time = %.2f' % test1.running_time)
            print('running_core time = %.2f' % test1.running_coreTime)
            print('total_errors = %.2f' % test1.errors)
            print('per_errors = %.3f' % test1.errors_per_site)
            print()


            VT_errors[interval-1] = test1.errors_per_site
            VT_runningTime[interval-1] = test1.running_coreTime
            
    np.savez(filename_save, VT_errors=VT_errors, VT_runningTime=VT_runningTime)


def test_3d():
    nx, ny, nz = 200, 200, 200
    ng = 2
    cores = 8
    max_iteration = 20
    radius = [5, 20, 50,80]
    
    for r in radius:
        filename_save = f"./BL3D_Curvature_R{r}_Iteration_1_{max_iteration}"
        
        BL_errors =np.zeros(max_iteration)
        BL_runningTime = np.zeros(max_iteration)
    
        # P0,R=myInput.init2IC(nx, ny, ng, "PolyIC.init")
        P0,R=myInput.Circle_IC3d(nx,ny,nz,r)
        # P0,R=myInput.Voronoi_IC(nx,ny,ng)
    
        for cores in [cores]:
            # loop_times=10
            for loop_times in range(1,max_iteration):
    
    
                test1 = Linear_3D.linear3d_class(nx,ny,nz,ng,cores,loop_times,P0,R,'np')
                # test1.linear_main()
                # P = test1.get_P()
    
                test1.linear3d_main("curvature")
                C_ln = test1.get_C()
    
                # error
    
                print('loop_times = ' + str(test1.loop_times))
                print('running_time = %.2f' % test1.running_time)
                print('running_core time = %.2f' % test1.running_coreTime)
                print('total_errors = %.2f' % test1.errors)
                print('per_errors = %.3f' % test1.errors_per_site)
                print()
    
    
                BL_errors[loop_times-1] = test1.errors_per_site
                BL_runningTime[loop_times-1] = test1.running_coreTime
                
        np.savez(filename_save, BL_errors=BL_errors, BL_runningTime=BL_runningTime)

def test_vertex_3d():
    nx, ny, nz = 200, 200, 200
    ng = 2
    cores = 8
    max_iteration = 20
    radius = [5, 20, 50, 80, 2, 1]
    
    for r in radius:
        filename_save = f"./VT3D_Curvature_R{r}_Iteration_1_{max_iteration}"
        
        VT_errors =np.zeros(max_iteration)
        VT_runningTime = np.zeros(max_iteration)
    
        # P0,R=myInput.init2IC(nx, ny, ng, "PolyIC.init")
        P0,R=myInput.Circle_IC3d(nx,ny,nz,r)
        # P0,R=myInput.Voronoi_IC(nx,ny,ng)
    
        for cores in [cores]:
            # loop_times=10
            for interval in range(1,max_iteration):
    
    
                test1 = Vertex_3D.vertex3d_class(nx,ny,nz,ng,cores,interval,P0,R)
                # test1.linear_main()
                # P = test1.get_P()
    
                test1.vertex3d_main("curvature")
                C_ln = test1.get_C()
    
                # error
    
                print('loop_times = ' + str(test1.interval))
                print('running_time = %.2f' % test1.running_time)
                print('running_core time = %.2f' % test1.running_coreTime)
                print('total_errors = %.2f' % test1.errors)
                print('per_errors = %.3f' % test1.errors_per_site)
                print()
    
    
                VT_errors[interval-1] = test1.errors_per_site
                VT_runningTime[interval-1] = test1.running_coreTime
                
        np.savez(filename_save, VT_errors=VT_errors, VT_runningTime=VT_runningTime)
        
def test_vertex_3dComplex():
    nx, ny, nz = 200, 200, 200
    ng = 2
    cores = 8
    max_iteration = 20
    wave = [5, 20, 50, 80, 2, 1]
    
    for w in wave:
        filename_save = f"./VT3DComp_Curvature_wave{w}_Iteration_1_{max_iteration}"
        
        VT_errors =np.zeros(max_iteration)
        VT_runningTime = np.zeros(max_iteration)
    
        # P0,R=myInput.init2IC(nx, ny, ng, "PolyIC.init")
        # P0,R=myInput.Circle_IC3d(nx,ny,nz,r)
        P0,R=myInput.Complex2G_IC3d(nx,ny,nz,w)
        # P0,R=myInput.Voronoi_IC(nx,ny,ng)
    
        for cores in [cores]:
            # loop_times=10
            for interval in range(1,max_iteration):
    
    
                test1 = Vertex_3D.vertex3d_class(nx,ny,nz,ng,cores,interval,P0,R)
                # test1.linear_main()
                # P = test1.get_P()
    
                test1.vertex3d_main("curvature")
                C_ln = test1.get_C()
    
                # error
    
                print('loop_times = ' + str(test1.interval))
                print('running_time = %.2f' % test1.running_time)
                print('running_core time = %.2f' % test1.running_coreTime)
                print('total_errors = %.2f' % test1.errors)
                print('per_errors = %.3f' % test1.errors_per_site)
                print()
    
    
                VT_errors[interval-1] = test1.errors_per_site
                VT_runningTime[interval-1] = test1.running_coreTime
                
        np.savez(filename_save, VT_errors=VT_errors, VT_runningTime=VT_runningTime)




if __name__ == '__main__':
    
    test_vertex_3dComplex()
    
    