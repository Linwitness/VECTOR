#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 16:51:24 2022

@author: lin.yang
"""

import os
current_path = os.getcwd()
import numpy as np
from numpy import seterr
seterr(all='raise')
import math
import myInput

def find_window_3d(P,i,j,k,iteration,refer_id):
    # Find the windows around the voxel i,j,k, the size depend on iteration
    nx,ny,nz=P.shape
    tableL=2*(iteration+1)+1
    fw_len = tableL
    fw_half = int((fw_len-1)/2)
    window = np.zeros((fw_len,fw_len,fw_len))
    
    for wi in range(fw_len):
        for wj in range(fw_len):
            for wk in range(fw_len):
                global_x = (i-fw_half+wi)%nx
                global_y = (j-fw_half+wj)%ny
                global_z = (k-fw_half+wj)%nz
                if P[global_x,global_y,global_z] == refer_id:
                    window[wi,wj,wk] = 1
                else:
                    window[wi,wj,wk] = 0
    
    return window

def find_normal_structure_3d(P,i,j,k,iteration,refer_id):
    # A basic structure to calculate normals
    smoothed_vector_i, smoothed_vector_j, smoothed_vector_k = myInput.output_linear_vector_matrix3D(iteration)
    
    a = (
        np.sum(find_window_3d(P,i,j,k,iteration,refer_id) * smoothed_vector_i) + 
        np.sum(find_window_3d(P,i+1,j,k,iteration,refer_id) * smoothed_vector_i) +  
        np.sum(find_window_3d(P,i,j+1,k,iteration,refer_id) * smoothed_vector_i) + 
        np.sum(find_window_3d(P,i+1,j+1,k,iteration,refer_id) * smoothed_vector_i) + 
        np.sum(find_window_3d(P,i,j,k+1,iteration,refer_id) * smoothed_vector_i) + 
        np.sum(find_window_3d(P,i+1,j,k+1,iteration,refer_id) * smoothed_vector_i) +  
        np.sum(find_window_3d(P,i,j+1,k+1,iteration,refer_id) * smoothed_vector_i) + 
        np.sum(find_window_3d(P,i+1,j+1,k+1,iteration,refer_id) * smoothed_vector_i)
        ) / 8
    b = (
        np.sum(find_window_3d(P,i,j,k,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window_3d(P,i+1,j,k,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window_3d(P,i,j+1,k,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window_3d(P,i+1,j+1,k,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window_3d(P,i,j,k+1,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window_3d(P,i+1,j,k+1,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window_3d(P,i,j+1,k+1,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window_3d(P,i+1,j+1,k+1,iteration,refer_id) * smoothed_vector_j)
        ) / 8
    c = (
        np.sum(find_window_3d(P,i,j,k,iteration,refer_id) * smoothed_vector_k) + 
        np.sum(find_window_3d(P,i+1,j,k,iteration,refer_id) * smoothed_vector_k) + 
        np.sum(find_window_3d(P,i,j+1,k,iteration,refer_id) * smoothed_vector_k) + 
        np.sum(find_window_3d(P,i+1,j+1,k,iteration,refer_id) * smoothed_vector_k) + 
        np.sum(find_window_3d(P,i,j,k+1,iteration,refer_id) * smoothed_vector_k) + 
        np.sum(find_window_3d(P,i+1,j,k+1,iteration,refer_id) * smoothed_vector_k) + 
        np.sum(find_window_3d(P,i,j+1,k+1,iteration,refer_id) * smoothed_vector_k) + 
        np.sum(find_window_3d(P,i+1,j+1,k+1,iteration,refer_id) * smoothed_vector_k)
        ) / 8
    return a, b, c

def find_normal_3d(P,i,j,k,nei_flat,iteration):
    # Calculate the nomals for all the eight voxels in the triple line and assign them to 3 grains
    tri_norm = np.zeros((8,3))
    tri_norm_final = np.zeros((3,3))
    tri_grains = np.zeros(3)
    
    for fi in range(len(nei_flat)):
        tri_norm[fi,0], tri_norm[fi,1], tri_norm[fi,2] = find_normal_structure_3d(P,i,j,k,iteration,nei_flat[fi])
    
    tri_grains = np.array(list(set(nei_flat)))
    
    for fi in range(len(nei_flat)):
        grain_sequence = np.where(tri_grains==nei_flat[fi])[0][0]
        tri_norm_final[grain_sequence] += tri_norm[fi]
    # tri_norm[0,0], tri_norm[0,1], tri_norm[0,2] = find_normal_structure_3d(P,i,j,k,iteration,P[i,j,k])
    # tri_norm[1,0], tri_norm[1,1], tri_norm[1,2] = find_normal_structure_3d(P,i,j,k,iteration,P[i,j+1,k])
    # tri_norm[2,0], tri_norm[2,1], tri_norm[2,2] = find_normal_structure_3d(P,i,j,k,iteration,P[i+1,j+1,k])
    # tri_norm[3,0], tri_norm[3,1], tri_norm[3,2] = find_normal_structure_3d(P,i,j,k,iteration,P[i+1,j,k])
    # tri_norm[4,0], tri_norm[4,1], tri_norm[4,2] = find_normal_structure_3d(P,i,j,k,iteration,P[i,j,k+1])
    # tri_norm[5,0], tri_norm[5,1], tri_norm[5,2] = find_normal_structure_3d(P,i,j,k,iteration,P[i,j+1,k+1])
    # tri_norm[6,0], tri_norm[6,1], tri_norm[6,2] = find_normal_structure_3d(P,i,j,k,iteration,P[i+1,j+1,k+1])
    # tri_norm[7,0], tri_norm[7,1], tri_norm[7,2] = find_normal_structure_3d(P,i,j,k,iteration,P[i+1,j,k+1])
    # tri_grains = np.array([P[i,j+1], P[i+1,j+1], P[i+1,j]])
    
    for ni in range(len(tri_norm_final)):
        try:
            tri_norm_final[ni] = tri_norm_final[ni]/np.linalg.norm(tri_norm_final[ni])
        except:
            print(tri_norm[ni])
            print(f"{i},{j},{k}")
            tri_norm_final[ni] = np.zeros(3)
        
    
    return tri_norm_final, tri_grains

def find_angle_3d(each_normal):
    # Find the 3 dihedral angles based on the normals of three neighboring grains
    tri_angle = np.zeros(3)
    
    tri_angle[0] = 180 / np.pi * (2 * np.pi - 2 * math.acos(round(np.dot(each_normal[1], each_normal[2]),5)))
    tri_angle[1] = 180 / np.pi * (2 * np.pi - 2 * math.acos(round(np.dot(each_normal[0], each_normal[2]),5)))
    tri_angle[2] = 180 / np.pi * (2 * np.pi - 2 * math.acos(round(np.dot(each_normal[0], each_normal[1]),5)))
    
    
    # A way to ignore the issue of some special triple angle
    # if abs(sum(tri_angle)-360) > 5:
        # print()
        # print(sum(tri_angle))
        # print(tri_angle)
        # print(each_normal)
        # tri_angle[2] = 360 - tri_angle[0] - tri_angle[1]
    
    return tri_angle

def read_3d_input(filename,nx,ny,nz):
    # Keep the 3d input macrostructure
    
    triple_map = np.zeros((nx,ny,nz))
    f = open(filename)
    line = f.readline()
    line = f.readline()
    # print(line)
    while line:
        each_element = line.split()
        i = int(each_element[0])-1
        j = int(each_element[1])-1
        k = int(each_element[2])-1
        triple_map[i,j,k] = int(each_element[7])
    
        line = f.readline()
    f.close()
    
    return triple_map

def read_3d_init(filename,nx,ny,nz,filepath=current_path+"/input/"):
    triple_map = np.zeros((nx,ny,nz))
    f = open(filepath + filename)
    line = f.readline()
    line = f.readline()
    line = f.readline()
    line = f.readline()
    while line:
        each_element = line.split()
        k = (int(each_element[0]) - 1) // (nx * ny)
        j = (int(each_element[0]) - 1 - (nx * ny) * k) // ny
        i = (int(each_element[0]) - 1 - (nx * ny) * k) - ny * j
        triple_map[i,j,k] = int(each_element[1])
        
        line = f.readline()
    f.close()
    
    return triple_map

def get_3d_ic1(nx,ny,nz):
    # Get verification IC (120, 120, 120)
    triple_map = np.zeros((nx,ny,nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if i < nx/2 and j < ny/2: triple_map[i,j,k] = 1
                elif i < nx/2 and j >= ny/2: triple_map[i,j,k] = 2
                elif i >= nx/2 and j < nx/2 - (i - nx/2) * math.sqrt(3): triple_map[i,j,k] = 1
                elif i >= nx/2 and j >= nx/2 + (i - nx/2) * math.sqrt(3) - 1: triple_map[i,j,k] = 2
                else: triple_map[i,j,k] = 3
    
    return triple_map

def get_3d_ic2(nx,ny,nz):
    # Get verification IC (90, 90, 180)
    triple_map = np.zeros((nx,ny,nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if i < nx/2 and j < ny/2: triple_map[i,j,k] = 1
                elif i < nx/2 and j >= ny/2: triple_map[i,j,k] = 2
                else: triple_map[i,j,k] = 3
    
    return triple_map
         
def get_3d_ic3(nx,ny,nz):
    # Get verification IC (105, 120, 135)
    triple_map = np.zeros((nx,ny,nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if i < nx/2 and j < ny/2: triple_map[i,j,k] = 1
                elif i < nx/2 and j >= ny/2: triple_map[i,j,k] = 2
                
                elif i >= nx/2 and j < nx/2 - (i - nx/2) * math.sqrt(3): triple_map[i,j,k] = 1
                elif i >= nx/2 and j >= nx/2 + (i - nx/2)-1: triple_map[i,j,k] = 2
                else: triple_map[i,j,k] = 3
    
    return triple_map 

def get_3d_ic4(nx,ny,nz):
    # Get verification IC (45, 45, 270)
    triple_map = np.zeros((nx,ny,nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if i < nx/2 and j <= ny/2 and j > i: triple_map[i,j,k] = 1
                elif i < nx/2 and j > ny/2 and j <= nx - i: triple_map[i,j,k] = 2
                else: triple_map[i,j,k] = 3
    
    return triple_map       

def calculate_tangent(triple_map,iteration=5):
    nx,ny,nz = triple_map.shape
    num = 0
    issue_num = 0
    triple_grain = []
    triple_coord = []
    triple_normal = []
    triple_angle = []
    for i in range(5,nx-6):
        for j in range(5,ny-6):
            for k in range(5,nz-6):
                nei = np.zeros((2,2,2))
                nei = triple_map[i:i+2,j:j+2,k:k+2]
                nei_flat = nei.flatten()
                # print(nei_flat)
                if len(set(nei_flat)) == 3 and 0 not in nei_flat:
                    
                    each_normal, tri_grains = find_normal_3d(triple_map,i,j,k,nei_flat,iteration) # Get basic normals and grain id sequence
                    triple_normal.append(each_normal) # Save the normals
                    triple_coord.append(np.array([i,j,k])) # Save the coordinate of the triple point
                    triple_grain.append(tri_grains) # Save the grain id sequence
                    triple_angle.append(find_angle_3d(each_normal)) # Save the 3 dihedral angles
                    
                    num += 1 # Get the num of all triple point
                    if abs(sum(find_angle_3d(each_normal))-360) > 5: issue_num += 1
            
    print(f"The number of useful triple junction is {num}")
    print(f"The issue propotion is {issue_num/num*100}%")
    
    return np.array(triple_coord), np.array(triple_angle), np.array(triple_grain)



if __name__ == '__main__':
    filename = "500x500x50_50kgrains.init"
    
    
    iteration=3
    
    # Read the 3D input file
    # nx, ny, nz = 500, 500, 50
    # triple_map = read_3d_init(filename,nx,ny,nz)
    
    # Get verification IC (120, 120, 120)
    # nx, ny, nz = 50, 50, 50
    # triple_map = get_3d_ic1(nx,ny,nz)
    
    # Get verification IC (90, 90, 180)
    # nx, ny, nz = 50, 50, 50
    # triple_map = get_3d_ic2(nx,ny,nz)
    
    # Get verification IC (105, 120, 135)
    nx, ny, nz = 50, 50, 50
    triple_map = get_3d_ic3(nx,ny,nz)
    
    # Get verification IC (45, 45, 270)
    # nx, ny, nz = 50, 50, 50
    # triple_map = get_3d_ic4(nx,ny,nz)
    
    # Calculate the tanget for corresponding triple junction
    triple_coord, triple_angle, triple_grain = calculate_tangent(triple_map, iteration)
    # The triple_coord is the coordination of the triple junction(left-upper voxel), axis 0 is the index of triple, axis 1 is the coordination (i,j,k)
    # The triple_angle saves the three dihedral angle for corresponding triple junction, 
    #   axis 0 is the index of triple, 
    #   axis 1 is the three dihedral angle
    # The triple_grain saves the sequence of three grains for corresponding triplr point
    #   axis 0 is the index of the triple,
    #   axis 1 is the three grains
    
    # np.save('triple_data_105_120_135',triple_map)






