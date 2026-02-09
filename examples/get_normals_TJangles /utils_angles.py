#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for grain boundary normal vector and triple junction analysis.

This module provides common functions used across:
- get_normals_and_TJangles.py (3D analysis)
- get_TJangles_from_npy.py (2D comparative analysis)
- get_TJanglesE_from_npy.py (2D energy analysis)

Author: Lin Yang
"""

import os
import sys
import numpy as np
from itertools import repeat

# Configure paths for VECTOR framework
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + '/../../')
import myInput


# =============================================================================
# Grain Boundary Site Identification
# =============================================================================

def get_gb_sites_2d(P, grain_num):
    """
    Identify grain boundary sites in 2D microstructure.

    Parameters:
    -----------
    P : ndarray
        3D array representing microstructure (time, x, y)
    grain_num : int
        Total number of grains in the microstructure

    Returns:
    --------
    ggn_gbsites : list of lists
        Grain boundary sites organized by grain ID.
        Each sublist contains [i,j] coordinates of boundary sites for that grain.
    """
    _, nx, ny = np.shape(P)
    timestep = 5  # Buffer zone to avoid boundary effects
    ggn_gbsites = [[] for i in repeat(None, grain_num)]

    for i in range(timestep, nx-timestep):
        for j in range(timestep, ny-timestep):
            ip, im, jp, jm = myInput.periodic_bc(nx, ny, i, j)

            if (((P[0,ip,j]-P[0,i,j])!=0) or ((P[0,im,j]-P[0,i,j])!=0) or
                ((P[0,i,jp]-P[0,i,j])!=0) or ((P[0,i,jm]-P[0,i,j])!=0)) and\
                P[0,i,j] <= grain_num:
                ggn_gbsites[int(P[0,i,j]-1)].append([i,j])

    return ggn_gbsites


def get_gb_sites_3d(P, grain_num):
    """
    Identify 3D grain boundary sites in complex microstructures.

    Parameters:
    -----------
    P : ndarray
        4D array representing 3D microstructure (time, x, y, z)
    grain_num : int
        Total number of grains in the microstructure

    Returns:
    --------
    ggn_gbsites : list of lists
        3D grain boundary sites organized by grain ID.
        Each sublist contains [i,j,k] coordinates of boundary sites for that grain.
    """
    _, nx, ny, nz = np.shape(P)
    timestep = 5  # Buffer zone to avoid boundary effects
    ggn_gbsites = [[] for i in repeat(None, grain_num)]

    for i in range(timestep, nx-timestep):
        for j in range(timestep, ny-timestep):
            for k in range(timestep, nz-timestep):
                ip, im, jp, jm, kp, km = myInput.periodic_bc3d(nx, ny, nz, i, j, k)

                if (((P[0,ip,j,k]-P[0,i,j,k])!=0) or ((P[0,im,j,k]-P[0,i,j,k])!=0) or
                    ((P[0,i,jp,k]-P[0,i,j,k])!=0) or ((P[0,i,jm,k]-P[0,i,j,k])!=0) or
                    ((P[0,i,j,kp]-P[0,i,j,k])!=0) or ((P[0,i,j,km]-P[0,i,j,k])!=0)) and\
                    P[0,i,j,k] <= grain_num:
                    ggn_gbsites[int(P[0,i,j,k]-1)].append([i,j,k])

    return ggn_gbsites


# =============================================================================
# Normal Vector Calculation
# =============================================================================

def norm_list_2d(grain_num, P_matrix):
    """
    Calculate normal vectors for all grain boundary sites (2D).

    Parameters:
    -----------
    grain_num : int
        Total number of grains in the microstructure
    P_matrix : ndarray
        Microstructure array for gradient calculation

    Returns:
    --------
    norm_list : list of ndarrays
        Normal vectors organized by grain number.
        Each array contains [normal_x, normal_y] for boundary sites.
    boundary_site : list of lists
        Corresponding boundary site coordinates.
    """
    boundary_site = get_gb_sites_2d(P_matrix, grain_num)
    norm_list = [np.zeros((len(boundary_site[i]), 2)) for i in range(grain_num)]

    for grain_i in range(grain_num):
        print(f"Processing grain {grain_i} boundary normals...")

        for site in range(len(boundary_site[grain_i])):
            norm = myInput.get_grad(P_matrix, boundary_site[grain_i][site][0],
                                   boundary_site[grain_i][site][1])
            norm_list[grain_i][site,:] = list(norm)

    return norm_list, boundary_site


def norm_list_3d(grain_num, P_matrix):
    """
    Calculate 3D normal vectors for all grain boundary sites.

    Parameters:
    -----------
    grain_num : int
        Total number of grains in the microstructure
    P_matrix : ndarray
        3D microstructure array for gradient calculation

    Returns:
    --------
    norm_list : list of ndarrays
        3D normal vectors organized by grain number.
        Each array contains [normal_x, normal_y, normal_z] for boundary sites.
    boundary_site : list of lists
        Corresponding 3D boundary site coordinates.
    """
    grain_num -= 1  # Adjust grain count for processing
    boundary_site = get_gb_sites_3d(P_matrix, grain_num)
    norm_list = [np.zeros((len(boundary_site[i]), 3)) for i in range(grain_num)]

    for grain_i in range(grain_num):
        print(f"Processing 3D grain {grain_i} boundary normals...")

        for site in range(len(boundary_site[grain_i])):
            norm = myInput.get_grad3d(P_matrix, boundary_site[grain_i][site][0],
                                     boundary_site[grain_i][site][1],
                                     boundary_site[grain_i][site][2])
            norm_list[grain_i][site,:] = list(norm)

    return norm_list, boundary_site


# =============================================================================
# Orientation Reading
# =============================================================================

def get_orientation(grain_num, init_name, ng=None):
    """
    Extract Euler angles from SPPARKS initialization file.

    Parameters:
    -----------
    grain_num : int
        Total number of grains expected
    init_name : str
        Path to SPPARKS initialization file
    ng : int, optional
        Number of grains to return (for subset selection).
        If None, returns grain_num-1 angles.

    Returns:
    --------
    eulerAngle : ndarray
        Array of Euler angles [phi1, Phi, phi2] for each grain.
    """
    eulerAngle = np.ones((grain_num, 3)) * -10

    with open(init_name, 'r', encoding='utf-8') as f:
        for line in f:
            eachline = line.split()

            if len(eachline) == 5 and eachline[0] != '#':
                lineN = int(eachline[1]) - 1
                if eulerAngle[lineN, 0] == -10:
                    eulerAngle[lineN, :] = [float(eachline[2]), float(eachline[3]), float(eachline[4])]

    if ng is None:
        ng = grain_num
    return eulerAngle[:ng-1]


# =============================================================================
# Output Functions
# =============================================================================

def output_inclination_2d(output_name, norm_list, site_list, orientation_list=None):
    """
    Export 2D grain boundary inclination data to file.

    Parameters:
    -----------
    output_name : str
        Output file path for inclination data
    norm_list : list of ndarrays
        Normal vectors organized by grain
    site_list : list of lists
        Boundary site coordinates for each grain
    orientation_list : ndarray, optional
        Euler angles for each grain
    """
    with open(output_name, 'w') as file:
        for i in range(len(norm_list)):
            if orientation_list is not None:
                file.write(f'Grain {i+1} Orientation: {orientation_list[i]} centroid: \n')
            else:
                file.write(f'Grain {i+1} Orientation: empty centroid: \n')

            for j in range(len(norm_list[i])):
                file.write(f'{site_list[i][j][0]}, {site_list[i][j][1]}, '
                          f'{norm_list[i][j][0]}, {norm_list[i][j][1]}\n')

            file.write('\n')


def output_inclination_3d(output_name, norm_list, site_list, orientation_list):
    """
    Export 3D grain boundary inclination data to file.

    Parameters:
    -----------
    output_name : str
        Output file name for 3D inclination data
    norm_list : list of ndarrays
        3D normal vectors organized by grain
    site_list : list of lists
        3D boundary site coordinates for each grain
    orientation_list : ndarray
        Euler angles for each grain
    """
    with open('output/' + output_name, 'w') as file:
        for i in range(len(norm_list)):
            file.write(f'Grain {i+1} Orientation: {orientation_list[i]} centroid: \n')

            for j in range(len(norm_list[i])):
                file.write(f'{site_list[i][j][0]}, {site_list[i][j][1]}, {site_list[i][j][2]}, '
                          f'{norm_list[i][j][0]}, {norm_list[i][j][1]}, {norm_list[i][j][2]}\n')

            file.write('\n')


def output_dihedral_angle_2d(output_name, triple_coord, triple_angle, triple_grain):
    """
    Export 2D triple junction dihedral angle analysis results.

    Parameters:
    -----------
    output_name : str
        Output file path for dihedral angle data
    triple_coord : ndarray
        Coordinates of triple junction points
    triple_angle : ndarray
        Calculated dihedral angles for each triple junction
    triple_grain : ndarray
        Grain IDs associated with each triple junction
    """
    with open(output_name, 'w') as file:
        file.write('triple_index triple_coordination grain_id0:dihedral0 '
                   'grain_id1:dihedral1 grain_id2:dihedral2 angle_sum\n')

        for i in range(len(triple_coord)):
            file.write(f'{i+1}, {triple_coord[i][0]} {triple_coord[i][1]}, '
                       f'{int(triple_grain[i][0])}:{round(triple_angle[i][0],2)} '
                       f'{int(triple_grain[i][1])}:{round(triple_angle[i][1],2)} '
                       f'{int(triple_grain[i][2])}:{round(triple_angle[i][2],2)} '
                       f'{round(np.sum(triple_angle[i]),2)}\n')


def output_dihedral_angle_3d(output_name, triple_coord, triple_angle, triple_grain):
    """
    Export 3D triple junction dihedral angle analysis results.

    Parameters:
    -----------
    output_name : str
        Output file name for dihedral angle data
    triple_coord : ndarray
        3D coordinates of triple junction points
    triple_angle : ndarray
        Calculated dihedral angles for each triple junction
    triple_grain : ndarray
        Grain IDs associated with each triple junction
    """
    with open('output/' + output_name, 'w') as file:
        file.write('triple_index triple_coordination grain_id0:dihedral0 '
                   'grain_id1:dihedral1 grain_id2:dihedral2')

        for i in range(len(triple_coord)):
            file.write(f'{i+1}, {triple_coord[i][0]} {triple_coord[i][1]} {triple_coord[i][2]}, '
                       f'{triple_grain[i][0]}:{triple_angle[i][0]} '
                       f'{triple_grain[i][1]}:{triple_angle[i][1]} '
                       f'{triple_grain[i][2]}:{triple_angle[i][2]}')


# =============================================================================
# Helper Functions
# =============================================================================

def find_window(P, i, j, iteration, refer_id):
    """
    Generate local window around specified voxel for neighbor analysis.

    Parameters:
    -----------
    P : ndarray
        2D microstructure array
    i, j : int
        Center coordinates for window
    iteration : int
        Window half-size parameter (total size = 2*(iteration+1)+1)
    refer_id : int
        Reference grain ID for comparison

    Returns:
    --------
    window : ndarray
        Binary array where 1=same grain, 0=different grain
    """
    nx, ny = P.shape
    tableL = 2 * (iteration + 1) + 1
    fw_len = tableL
    fw_half = int((fw_len - 1) / 2)
    window = np.zeros((fw_len, fw_len))

    for wi in range(fw_len):
        for wj in range(fw_len):
            global_x = (i - fw_half + wi) % nx
            global_y = (j - fw_half + wj) % ny

            if P[global_x, global_y] == refer_id:
                window[wi, wj] = 1
            else:
                window[wi, wj] = 0

    return window


def data_smooth(data_array, smooth_level=2):
    """
    Apply moving average smoothing to time series data.

    Parameters:
    -----------
    data_array : ndarray
        Input time series data for smoothing
    smooth_level : int, optional
        Half-width of smoothing window (default=2)

    Returns:
    --------
    data_array_smoothed : ndarray
        Smoothed time series data
    """
    data_array_smoothed = np.zeros(len(data_array))

    for i in range(len(data_array)):
        if i < smooth_level:
            data_array_smoothed[i] = np.sum(data_array[0:i+smooth_level+1]) / (i+smooth_level+1)
        elif (len(data_array) - 1 - i) < smooth_level:
            data_array_smoothed[i] = np.sum(data_array[i-smooth_level:]) / (len(data_array)-i+smooth_level)
        else:
            data_array_smoothed[i] = np.sum(data_array[i-smooth_level:i+smooth_level+1]) / (smooth_level*2+1)

    return data_array_smoothed
