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
import matplotlib.pyplot as plt
import numpy as np
from numpy import seterr
seterr(all='raise')
import math

import PACKAGE_MP_Linear as Linear_2D
import PACKAGE_MP_Vertex as Vertex_2D
import myInput

def get_2d_ic1(nx,ny):
    # Get verification IC (90, 90, 180)
    triple_map = np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            if i < nx/2 and j < ny/2: triple_map[i,j] = 1
            elif i < nx/2 and j >= ny/2: triple_map[i,j] = 2
            else: triple_map[i,j] = 3
    
    return triple_map

def get_2d_ic2(nx,ny):
    # Get verification IC (120, 120, 120)
    triple_map = np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            if i < nx/2 and j < ny/2: triple_map[i,j] = 1
            elif i < nx/2 and j >= ny/2: triple_map[i,j] = 2
            elif i >= nx/2 and j < nx/2 - (i - nx/2) * math.sqrt(3): triple_map[i,j] = 1
            elif i >= nx/2 and j >= nx/2 + (i - nx/2) * math.sqrt(3) - 1: triple_map[i,j] = 2
            else: triple_map[i,j] = 3
    
    return triple_map


def energy_function(normals, delta = 0.6, m = 2):
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
    
    # Old Version
    # for m in range(len(edge_coord)):
    #     if edge[m] != center:
    #         n_edge = get_inclination(P , edge_coord[m][0], edge_coord[m][1])
    #         e_edge = energy_function(n_edge)
    #         print(f"edge: {n_edge}, {e_edge}")
            
    #         e_ave = (e1 + e_edge) / 2
    #         site_energy += e_ave
    
    # Old Version
    # for m in range(len(edge_coord)):
    #     if edge[m] != center:
    #         n_edge = get_inclination(P , edge_coord[m][0], edge_coord[m][1])
    #         n_edge = n_edge - n1
    #         n_edge = n_edge / np.linalg.norm(n_edge)
    #         e_edge = energy_function(n_edge)
    #         print(f"edge: {n_edge}, {e_edge}")
            
    #         e_ave = e_edge
    #         site_energy += e_ave
    
    # Min Energy
    # num_site = 0
    # min_eng = 8
    # for m in range(len(edge_coord)):
    #     if edge[m] != center:
    #         num_site += 1
    #         n_edge = get_inclination(P , edge_coord[m][0], edge_coord[m][1])
    #         e_edge = energy_function(n_edge)
    #         print(f"edge: {n_edge}, {e_edge}")
    #         if e_edge < min_eng: min_eng = e_edge
    # site_energy = min_eng #* num_site
    
    # Max Energy
    # num_site = 0
    # max_eng = 0
    # for m in range(len(edge_coord)):
    #     if edge[m] != center:
    #         num_site += 1
    #         n_edge = get_inclination(P , edge_coord[m][0], edge_coord[m][1])
    #         e_edge = energy_function(n_edge)
    #         print(f"edge: {n_edge}, {e_edge}")
    #         if e_edge > max_eng: max_eng = e_edge
    # site_energy = max_eng #* num_site
    
    # Ave Energy
    # num_site = 0
    # nei_grain_id = set(list(edge)+[center])
    # nei_grain_id.remove(center)
    # nei_grain_id = list(nei_grain_id)
    # engs_grain = np.zeros(len(nei_grain_id))
    # sites_grain = np.zeros(len(nei_grain_id))
    # ave_eng = 0
    # for m in range(len(edge_coord)):
    #     if edge[m] != center:
    #         num_site += 1
    #         n_edge = get_inclination(P , edge_coord[m][0], edge_coord[m][1])
    #         engs_grain[int(nei_grain_id.index(edge[m]))] += energy_function(n_edge)
    #         sites_grain[int(nei_grain_id.index(edge[m]))] += 1
    #         print(f"edge: {n_edge}, {energy_function(n_edge)}")
    # for m in range(len(nei_grain_id)): ave_eng += engs_grain[m] / sites_grain[m]
    # print(f"nei_eng: {engs_grain/sites_grain}")
    # ave_eng += e1
    # site_energy = ave_eng / (len(nei_grain_id) + 1)
    
    # Sum Energy
    num_site = 0
    nei_grain_id = set(list(edge)+[center])
    nei_grain_id.remove(center)
    nei_grain_id = list(nei_grain_id)
    engs_grain = np.zeros(len(nei_grain_id))
    sites_grain = np.zeros(len(nei_grain_id))
    ave_eng = 0
    for m in range(len(edge_coord)):
        if edge[m] != center:
            num_site += 1
            n_edge = get_inclination(P , edge_coord[m][0], edge_coord[m][1])
            engs_grain[int(nei_grain_id.index(edge[m]))] += energy_function(n_edge)
            sites_grain[int(nei_grain_id.index(edge[m]))] += 1
            print(f"edge: {n_edge}, {energy_function(n_edge)}")
    for m in range(len(nei_grain_id)): ave_eng += engs_grain[m] / sites_grain[m]
    print(f"nei_eng: {engs_grain[m]/sites_grain[m]}")
    ave_eng += e1
    site_energy = ave_eng
    
            
    return site_energy

def get_inclination(P, i, j, loop_times = 5, ng = 512):
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
    
    nx, ny = 10, 10
    ic90 = get_2d_ic1(nx,ny)
    eng_matrix = np.zeros((nx,ny))
    # ic120 = get_2d_ic2(nx,ny)
    # ic120_2 = np.array(np.rot90(ic120, 3))
    
    # ic2 = np.array(ic1)
    # ic2[int(nx/2), int(ny/2-1)] = 1
    
    # ic3 = np.array(ic2)
    # ic3[int(nx/2), int(ny/2)] = 2
    
    # filename = "VoronoiIC_512_elong.init"
    # nx, ny, ng = 512, 512, 512
    # ic4, _ = myInput.init2IC(nx, ny, ng, filename, "./")
    # ic4 = ic4[:,:,0]
    
    # x , y = 4, 4
    # eng = calculate_energy(ic90, x, y)
    # print(f"The energy at TJ ({x}, {y}) is {eng}")
    # print()
    
    # print("Vertical:")
    # x, y = 6, 4
    # eng = calculate_energy(ic120, x, y)
    # print(f"The energy at TJ ({x}, {y}) is {eng}")
    # print()
    
    # ic120[x,y] = 1
    # eng = calculate_energy(ic120, x, y)
    # print(f"The energy at TJ ({x}, {y}) is {eng}")
    # print()
    
    # print("Horizontal:")
    # x, y = 4, 3
    # eng = calculate_energy(ic120_2, x, y)
    # print(f"The energy at TJ ({x}, {y}) is {eng}")
    # print()
    
    # ic120_2[x,y] = 1
    # eng = calculate_energy(ic120_2, x, y)
    # print(f"The energy at TJ ({x}, {y}) is {eng}")
    # print()
    
    for i in range(4,6):
        for j in range(4,6):
            eng_matrix[i][j] = calculate_energy(ic90, i ,j)
    plt.imshow(eng_matrix, cmap='gray_r',vmin=0,vmax=3)
    # plt.grid()
    plt.colorbar(orientation='horizontal')
    plt.savefig('colorbar', dpi=400,bbox_inches='tight')

