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
from itertools import repeat

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
filename = "s1400poly1_t0.init"
P0,R=myInput.init2IC3d(nx,ny,nz,ng,filename,True)



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

def get_gb_sites(P,grain_num):
    _,nx,ny,nz=np.shape(P)
    timestep=5
    ggn_gbsites = [[] for i in repeat(None, grain_num)]
    for i in range(timestep,nx-timestep):
        for j in range(timestep,ny-timestep):
            for k in range(timestep,nz-timestep):
                ip,im,jp,jm,kp,km = myInput.periodic_bc3d(nx,ny,nz,i,j,k)
                if ( ((P[0,ip,j,k]-P[0,i,j,k])!=0) or ((P[0,im,j,k]-P[0,i,j,k])!=0) or\
                     ((P[0,i,jp,k]-P[0,i,j,k])!=0) or ((P[0,i,jm,k]-P[0,i,j,k])!=0) or\
                     ((P[0,i,j,kp]-P[0,i,j,k])!=0) or ((P[0,i,j,km]-P[0,i,j,k])!=0) ) and\
                    P[0,i,j,k] <= grain_num:
                    ggn_gbsites[int(P[0,i,j,k]-1)].append([i,j,k])
    return ggn_gbsites

def norm_list(grain_num, P_matrix):
    # get the norm list
    grain_num -= 1
    boundary_site = get_gb_sites(P_matrix, grain_num)
    norm_list = [np.zeros(( len(boundary_site[i]), 3 )) for i in range(grain_num)]
    for grain_i in range(grain_num):
        print(f"finish {grain_i}")
        
        for site in range(len(boundary_site[grain_i])):
            norm = myInput.get_grad3d(P_matrix, boundary_site[grain_i][site][0], boundary_site[grain_i][site][1], boundary_site[grain_i][site][2])
            norm_list[grain_i][site,:] = list(norm)
        
    return norm_list, boundary_site

def get_orientation(grain_num, init_name ):
    # read the input euler angle from *.init
    eulerAngle = np.ones((grain_num,3))*-10
    with open('Input/'+init_name, 'r', encoding = 'utf-8') as f:
        for line in f:
            eachline = line.split()
    
            if len(eachline) == 5 and eachline[0] != '#':
                lineN = int(eachline[1])-1
                if eulerAngle[lineN,0] == -10:
                    eulerAngle[lineN,:] = [float(eachline[2]), float(eachline[3]), float(eachline[4])]
    return eulerAngle[:ng-1]

def output(output_name, norm_list, site_list, orientation_list):
    
    file = open('output/'+output_name,'w')
    for i in range(len(norm_list)):
        file.writelines(['Grain ' + str(i+1) + ' Orientation: ' + str(orientation_list[i]) + ' centroid: ' + '\n'])
        
        for j in range(len(norm_list[i])):
            file.writelines([str(site_list[i][j][0]) + ', ' + str(site_list[i][j][1]) + ', ' + str(site_list[i][j][2]) + ', ' + str(norm_list[i][j][0]) + ', ' + str(norm_list[i][j][1]) + ', ' + str(norm_list[i][j][2]) + '\n'])
            
        file.writelines(['\n'])
    
    file.close()
    return

norm_list1, site_list1 = norm_list(ng, P)
orientation_list1 = get_orientation(ng, filename)
output("total_site.txt", norm_list1, site_list1, orientation_list1)




