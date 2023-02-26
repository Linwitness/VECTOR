#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 16:51:24 2022

@author: lin.yang
"""

import numpy as np
from numpy import seterr
seterr(all='raise')
import math
import myInput

def find_window(P,i,j,iteration,refer_id):
    # Find the windows around the voxel i,j, the size depend on iteration
    nx,ny=P.shape
    tableL=2*(iteration+1)+1
    fw_len = tableL
    fw_half = int((fw_len-1)/2)
    window = np.zeros((fw_len,fw_len))
    
    for wi in range(fw_len):
        for wj in range(fw_len):
            global_x = (i-fw_half+wi)%nx
            global_y = (j-fw_half+wj)%ny
            if P[global_x,global_y] == refer_id:
                window[wi,wj] = 1
            else:
                window[wi,wj] = 0
    
    return window

def find_normal_structure(P,i,j,iteration,refer_id):
    smoothed_vector_i, smoothed_vector_j = myInput.output_linear_vector_matrix(iteration)
    
    a = (np.sum(find_window(P,i,j,iteration,refer_id) * smoothed_vector_i) + 
        np.sum(find_window(P,i+1,j,iteration,refer_id) * smoothed_vector_i) +  
        np.sum(find_window(P,i,j+1,iteration,refer_id) * smoothed_vector_i) + 
        np.sum(find_window(P,i+1,j+1,iteration,refer_id) * smoothed_vector_i)) / 4
    b = (np.sum(find_window(P,i,j,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window(P,i+1,j,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window(P,i,j+1,iteration,refer_id) * smoothed_vector_j) + 
        np.sum(find_window(P,i+1,j+1,iteration,refer_id) * smoothed_vector_j)) / 4
    return a, b

def find_normal(P,i,j,nei_flat,iteration):
    # Calculate the nomals for all the four voxels in the triple junction
    nx,ny=P.shape
    tri_norm = np.zeros((4,2))
    tri_grains = np.zeros(3)
    
    if nei_flat[0] == nei_flat[1]:

        tri_norm[0,0], tri_norm[0,1] = find_normal_structure(P,i,j,iteration,P[i,j])
        tri_norm[1,0], tri_norm[1,1] = find_normal_structure(P,i,j,iteration,P[i,j+1])
        tri_norm[2,0], tri_norm[2,1] = find_normal_structure(P,i,j,iteration,P[i+1,j+1])
        tri_norm[3,0], tri_norm[3,1] = find_normal_structure(P,i,j,iteration,P[i+1,j])
        tri_grains = np.array([P[i,j+1], P[i+1,j+1], P[i+1,j]])
        
        
    elif nei_flat[0] == nei_flat[2]:
        
        tri_norm[0,0], tri_norm[0,1] = find_normal_structure(P,i,j,iteration,P[i+1,j])
        tri_norm[1,0], tri_norm[1,1] = find_normal_structure(P,i,j,iteration,P[i,j])
        tri_norm[2,0], tri_norm[2,1] = find_normal_structure(P,i,j,iteration,P[i,j+1])
        tri_norm[3,0], tri_norm[3,1] = find_normal_structure(P,i,j,iteration,P[i+1,j+1])
        tri_grains = np.array([P[i,j], P[i,j+1], P[i+1,j+1]])
        
    elif nei_flat[2] == nei_flat[3]:
        
        tri_norm[0,0], tri_norm[0,1] = find_normal_structure(P,i,j,iteration,P[i+1,j+1])
        tri_norm[1,0], tri_norm[1,1] = find_normal_structure(P,i,j,iteration,P[i+1,j])
        tri_norm[2,0], tri_norm[2,1] = find_normal_structure(P,i,j,iteration,P[i,j])
        tri_norm[3,0], tri_norm[3,1] = find_normal_structure(P,i,j,iteration,P[i,j+1])
        tri_grains = np.array([P[i+1,j], P[i,j], P[i,j+1]])
        
    elif nei_flat[1] == nei_flat[3]:
        
        tri_norm[0,0], tri_norm[0,1] = find_normal_structure(P,i,j,iteration,P[i,j+1])
        tri_norm[1,0], tri_norm[1,1] = find_normal_structure(P,i,j,iteration,P[i+1,j+1])
        tri_norm[2,0], tri_norm[2,1] = find_normal_structure(P,i,j,iteration,P[i+1,j])
        tri_norm[3,0], tri_norm[3,1] = find_normal_structure(P,i,j,iteration,P[i,j])
        tri_grains = np.array([P[i+1,j+1], P[i+1,j], P[i,j]])
    
    for ni in range(4):
        tri_norm[ni] = tri_norm[ni]/np.linalg.norm(tri_norm[ni])
        
    
    return tri_norm, tri_grains

def find_angle_tan(each_normal):
    # Find the three tangent depend on the four normals from four voxels
    tri_tang = np.zeros((3,2))
    tri_angle = np.zeros(3)
    clock90 = np.array([[0,-1],[1,0]])
    anti_clock90 = np.array([[0,1],[-1,0]])
    
    tri_tang[0] = each_normal[0]@clock90
    tri_tang[1] = each_normal[1]@anti_clock90
    tri_tang[2] = -(each_normal[2]+each_normal[3])/np.linalg.norm(each_normal[2]+each_normal[3])
    
    tri_angle[0] = 180 / np.pi * math.acos(np.dot(tri_tang[0], tri_tang[2]))
    tri_angle[1] = 180 / np.pi * math.acos(np.dot(tri_tang[1], tri_tang[2]))
    tri_angle[2] = 180 / np.pi * math.acos(round(np.dot(tri_tang[0], tri_tang[1]),5))
    if abs(sum(tri_angle) - 360) > 5:
        tri_angle[2] = 360 - tri_angle[2]
    
    return tri_angle

def find_angle(each_normal):
    tri_angle = np.zeros(3)
    
    third_normal = (each_normal[0]+each_normal[1])/np.linalg.norm(each_normal[0]+each_normal[1])
    tri_angle[0] = 180 / np.pi * (2 * np.pi - 2 * math.acos(np.dot(each_normal[3], third_normal)))
    tri_angle[1] = 180 / np.pi * (2 * np.pi - 2 * math.acos(np.dot(each_normal[2], third_normal)))
    
    tri_angle[2] = 180 / np.pi * (2 * np.pi - 2 * math.acos(np.dot(each_normal[2], each_normal[3])))
    
    
    # A way to ignore the issue of some special triple angle
    # if abs(sum(tri_angle)-360) > 5:
        # print()
        # print(sum(tri_angle))
        # print(tri_angle)
        # print(each_normal)
        # tri_angle[2] = 360 - tri_angle[0] - tri_angle[1]
    
    return tri_angle

def read_2d_input(filename,nx,ny):
    # Keep the 2d input macrostructure
    
    triple_map = np.zeros((nx,ny))
    f = open(filename)
    line = f.readline()
    line = f.readline()
    # print(line)
    while line:
        each_element = line.split()
        i = int(each_element[0])-1
        j = int(each_element[1])-1
        triple_map[i,j] = int(each_element[6])
    
        line = f.readline()
    f.close()
    
    return triple_map

def calculate_tangent(triple_map,iteration=5):
    num = 0
    issue_num = 0
    triple_grain = []
    triple_coord = []
    triple_normal = []
    triple_angle = []
    triple_angle_tang = []
    for i in range(nx-1):
        for j in range(ny-1):
            nei = np.zeros((2,2))
            nei = triple_map[i:i+2,j:j+2]
            nei_flat = nei.flatten()
            if len(set(nei_flat)) == 3 and 0 not in nei_flat:
                
                # print(str(i)+" "+str(j))
                
                each_normal, grain_sequence = find_normal(triple_map,i,j,nei_flat,iteration) # Get basic normals and grain id sequence
                triple_normal.append(each_normal) # Save the normals
                triple_coord.append(np.array([i,j])) # Save the coordinate of the triple point
                triple_grain.append(grain_sequence) # Save the grain id sequence
                triple_angle.append(find_angle(each_normal)) # Save the 3 dihedral angles

                num += 1 # Get the num of all triple point
                if abs(sum(find_angle(each_normal))-360) > 5: issue_num += 1
            
    print(f"The number of useful triple junction is {num}")
    print(f"The issue propotion is {issue_num/num*100}%")
    
    return np.array(triple_coord), np.array(triple_angle), np.array(triple_grain)



if __name__ == '__main__':
    filename = "slice3_complete.txt"
    nx, ny = 500, 500
    
    iteration=5
    
    # Read the 2D input file
    triple_map = read_2d_input(filename,nx,ny)
    
    # Calculate the tanget for corresponding triple junction
    triple_coord, triple_angle, triple_grain = calculate_tangent(triple_map, iteration)
    # The triple_coord is the coordination of the triple junction(left-upper voxel), axis 0 is the index of triple, axis 1 is the coordination (i,j)
    # The triple_angle saves the three dihedral angle for corresponding triple junction, 
    #   axis 0 is the index of triple, 
    #   axis 1 is the three dihedral angle
    # The triple_grain saves the sequence of three grains for corresponding triplr point
    #   axis 0 is the index of the triple,
    #   axis 1 is the three grains
    






