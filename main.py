#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 14:55:28 2021

@author: lin.yang
"""

import os
current_path = os.getcwd()
import sys
sys.path.append(current_path)
import numpy as np
import math

import myInput
import PACKAGE_MP_Vertex       #2D Vertex smooth algorithm
import PACKAGE_MP_Bilinear     #2D Bilinear smooth algorithm
import PACKAGE_MP_AllenCahn    #2D Allen-Cahn smooth algorithm
import PACKAGE_MP_LevelSet     #2D Level Set smooth algorithm
import PACKAGE_MP_3DVertex     #3D Vertex smooth algorithm
import PACKAGE_MP_3DBilinear   #3D Bilinear smooth algorithm
import PACKAGE_MP_3DAllenCahn  #3D Allen-Cahn smooth algorithm
import PACKAGE_MP_3DLevelSet   #3D Level Set smooth algorithm

#%% 3D initial conditions

# Demostration Voronoi 1000 grains sample with 0 timestep, 10 timestep, 50 timestep
# nx, ny, nz = 100, 100, 100
# ng = 1000
# filepath = current_path + '/Input/'
# P0,R=myInput.init2IC3d(nx,ny,nz,ng,"VoronoiIC1000.init",False,filepath)


# Validation Dream3d 831 grains sample ("s1400poly1_t0.init") with 0 timestep
nx, ny, nz = 201, 201, 43
ng = 831
P0,R=myInput.init2IC3d(nx,ny,nz,ng,"s1400poly1_t0.init",True)



# %% 2D initial conditions

# nx, ny= 200,200
# ng = 2

# P0,R=myInput.init2IC(nx, ny, ng, "PolyIC.init")
# P0,R=myInput.Complex2G_IC3d(nx,ny,nz)
# P0,R=myInput.Circle_IC(nx,ny)
# P0,R=myInput.Circle_IC3d(nx,ny,nz)
# P0,R=myInput.Voronoi_IC(nx,ny,ng)
# P0,R=myInput.Complex2G_IC(nx,ny)
# P0,R=myInput.Abnormal_IC(nx,ny)
# P0,R=myInput.SmallestGrain_IC(100,100)

#%% Start the algorithms

for cores in [8]:
    for loop_times in range(4,5):
        
        
        test1 = PACKAGE_MP_3DBilinear.BL3dv1_class(nx,ny,nz,ng,cores,loop_times,P0,R,'np')
        test1.BL3dv1_main()
        P = test1.get_P()
    
        
        #%%
        # test1.get_gb_num(1)
        # test1.get_2d_plot('DREAM3D_poly','Bilinear')
        
        
        #%% Running time
          
        print('loop_times = ' + str(test1.loop_times))
        print('running_time = %.2f' % test1.running_time)
        print('running_core time = %.2f' % test1.running_coreTime)
        # print('total_errors = %.2f' % test1.errors)
        # print('per_errors = %.3f' % test1.errors_per_site)
        print()
        
#%% Output the inclination data


          