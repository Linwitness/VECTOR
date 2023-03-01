#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 21:00:34 2023

@author: Lin
"""

import os
current_path = os.getcwd()
import sys
sys.path.append(current_path)
sys.path.append(current_path+'/../../')
import numpy as np
from numpy import seterr
seterr(all='raise')
import math

import PACKAGE_MP_Linear as Linear_2D
import PACKAGE_MP_Vertex as Vertex_2D

def get_2d_ic1(nx,ny):
    # Get verification IC (90, 90, 180)
    triple_map = np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            if i < nx/2 and j < ny/2: triple_map[i,j] = 1
            elif i < nx/2 and j >= ny/2: triple_map[i,j] = 2
            else: triple_map[i,j] = 3
    
    return triple_map


def energy_function(normals, delta = 0.9, m = 2):
    refer = np.array([1,0])
    theta_rad = math.acos(round(np.array(normals).dot(refer), 5))
    
    energy = 1 + delta * math.cos(m * theta_rad)
    
    return energy

def calculate_energy(P, i, j):
    site_energy = 0
    
    window = P[i-1:i+2, j-1:j+2]
    center = window[1,1]
    edge = np.array(list(window[0,:])+[window[1,0],window[1,2]]+list(window[2,:]))
    edge_coord = np.array([[i-1,j-1], [i-1,j], [i-1,j+1], [i,j-1], [i,j+1], [i+1,j-1], [i+1,j], [i+1,j+1]])
    
    n1 = get_inclination(P, i, j)
    e1 = energy_function(n1)
    print(f"center: {n1}, {e1}")
    
    for m in range(len(edge_coord)):
        if edge[m] != center:
            n_edge = get_inclination(P , edge_coord[m][0], edge_coord[m][1])
            e_edge = energy_function(n_edge)
            print(f"edge: {n_edge}, {e_edge}")
            
            e_ave = (e1 + e_edge) / 2
            site_energy += e_ave
    
    # for m in range(len(edge_coord)):
    #     if edge[m] != center:
    #         n_edge = get_inclination(P , edge_coord[m][0], edge_coord[m][1])
    #         n_edge = n_edge - n1
    #         n_edge = n_edge / np.linalg.norm(n_edge)
    #         e_edge = energy_function(n_edge)
    #         print(f"edge: {n_edge}, {e_edge}")
            
    #         e_ave = e_edge
    #         site_energy += e_ave
            
    return site_energy

def get_inclination(P, i, j, loop_times = 5, ng = 3):
    nx, ny = P.shape
    cores = 1
    R = np.zeros((nx, ny))
    P0 = np.zeros((nx,ny,ng))
    for m in range(ng):
        P0[:,:,m] = 1.0 * (P==(m+1))
    
    test1 = Linear_2D.linear_class(nx,ny,ng,cores,loop_times,P0,R)
    normal = test1.linear_one_normal_vector_core([i, j])
    normal = np.array([-normal[1], -normal[0]])
    normal = normal / np.linalg.norm(normal)
    
    return normal
    
if __name__ == '__main__':
    
    nx, ny = 50, 50
    ic1 = get_2d_ic1(nx,ny)
    
    ic2 = np.array(ic1)
    ic2[int(nx/2), int(ny/2-1)] = 1
    
    ic3 = np.array(ic2)
    ic3[int(nx/2), int(ny/2)] = 2
    
    eng = calculate_energy(ic3, 25, 25)

