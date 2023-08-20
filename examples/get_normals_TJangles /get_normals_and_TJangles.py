#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:48:29 2023

@author: Lin
"""

import os
current_path = os.getcwd()
import numpy as np
from numpy import seterr
seterr(all='raise')
import math
from itertools import repeat
import sys
sys.path.append(current_path)
sys.path.append(current_path+'/../../')
import myInput
import PACKAGE_MP_3DLinear as linear3d
sys.path.append(current_path+'/../calculate_tangent/')
import output_tangent_3d



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
    with open(init_name, 'r', encoding = 'utf-8') as f:
        for line in f:
            eachline = line.split()
    
            if len(eachline) == 5 and eachline[0] != '#':
                lineN = int(eachline[1])-1
                if eulerAngle[lineN,0] == -10:
                    eulerAngle[lineN,:] = [float(eachline[2]), float(eachline[3]), float(eachline[4])]
    return eulerAngle[:ng-1]

def output_inclination(output_name, norm_list, site_list, orientation_list):
    
    file = open('output/'+output_name,'w')
    for i in range(len(norm_list)):
        file.writelines(['Grain ' + str(i+1) + ' Orientation: ' + str(orientation_list[i]) + ' centroid: ' + '\n'])
        
        for j in range(len(norm_list[i])):
            file.writelines([str(site_list[i][j][0]) + ', ' + str(site_list[i][j][1]) + ', ' + str(site_list[i][j][2]) + ', ' + str(norm_list[i][j][0]) + ', ' + str(norm_list[i][j][1]) + ', ' + str(norm_list[i][j][2]) + '\n'])
            
        file.writelines(['\n'])
    
    file.close()
    return

def output_dihedral_angle(output_name, triple_coord, triple_angle, triple_grain):
    file = open('output/'+output_name,'w')
    file.writelines(['triple_index triple_coordination grain_id0:dihedral0 grain_id1:dihedral1 grain_id2:dihedral2'])
    for i in range(len(triple_coord)):
        file.writelines([str(i+1) + ', ' + str(triple_coord[i][0]) + ' ' + str(triple_coord[i][1]) + ' ' + str(triple_coord[i][2]) + ', ' + \
                         str(triple_grain[i][0]) + ':' + str(triple_angle[i][0]) + ' ' + \
                         str(triple_grain[i][1]) + ':' + str(triple_angle[i][1]) + ' ' + \
                         str(triple_grain[i][2]) + ':' + str(triple_angle[i][2])])
        
    file.close()
    return


if __name__ == '__main__':
    filename = "Input/An1Fe.init"
    
    # Get IC (501, 501, 50)
    nx, ny, nz = 501, 501, 50
    ng = 10928
    cores = 8
    loop_times = 5
    print("IC's parameters done")
    
    P0,R = myInput.init2IC3d(nx, ny, nz, ng, filename, True, './')
    print("IC is read as matrix")
    
    test1 = linear3d.linear3d_class(nx,ny,nz,ng,cores,loop_times,P0,R,'np')
    test1.linear3d_main("inclination")
    P = test1.get_P()
    print('loop_times = ' + str(test1.loop_times))
    print('running_time = %.2f' % test1.running_time)
    print('running_core time = %.2f' % test1.running_coreTime)
    print('total_errors = %.2f' % test1.errors)
    print('per_errors = %.3f' % test1.errors_per_site)
    print("Inclinatin calculation done")
    
    norm_list1, site_list1 = norm_list(ng, P)
    orientation_list1 = get_orientation(ng, filename)
    output_inclination("An1Fe_inclination.txt", norm_list1, site_list1, orientation_list1)
    print("Inclination outputted")
    
    
    # Calculate the tanget for corresponding triple junction
    triple_coord, triple_angle, triple_grain = output_tangent_3d.calculate_tangent(P0[:,:,:,0], loop_times)
    print("tanget calculation done")
    # The triple_coord is the coordination of the triple junction(left-upper voxel), axis 0 is the index of triple, axis 1 is the coordination (i,j,k)
    # The triple_angle saves the three dihedral angle for corresponding triple junction, 
    #   axis 0 is the index of triple, 
    #   axis 1 is the three dihedral angle
    # The triple_grain saves the sequence of three grains for corresponding triplr point
    #   axis 0 is the index of the triple,
    #   axis 1 is the three grains
    
    output_dihedral_angle("An1Fe_dihedral.txt", triple_coord, triple_angle, triple_grain)
    print("Dihedral angle outputted")
    
    