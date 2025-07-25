#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-processing utilities for VECTOR (VoxEl-based boundary inClination smooThing AlgORithms)
Author: Lin Yang
"""

import numpy as np
import multiprocess as mp
import os
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import PACKAGE_MP_3DLinear as smooth_3d
import PACKAGE_MP_Linear as smooth
import myInput
from itertools import combinations

###########################################
# 1. Basic Matrix Operations
###########################################
"""
Basic utility functions for handling matrix operations and indexing in the grain structure analysis.
These functions provide core functionality used by other modules.
"""

def get_line(i, j):
    """Get the row order of grain i and grain j in MisoEnergy.txt (i < j)
    This function calculates the index in a triangular matrix representation for grain boundary properties 
    between two grains i and j.
    
    Args:
        i (int): First grain ID
        j (int): Second grain ID
    Returns:
        int: Index in triangular matrix representation
    """
    if i < j: return int(i+(j-1)*(j)/2)
    else: return int(j+(i-1)*(i)/2)

def get_index_id_from_matrix(matrix_id, nx, ny, nz):
    """Convert 3D coordinates to linear index
    Args:
        matrix_id (tuple): 3D coordinates (x, y, z)
        nx (int): Size of x dimension
        ny (int): Size of y dimension
        nz (int): Size of z dimension
    Returns:
        int: Linear index
    """
    x_index, y_index, z_index = matrix_id
    return x_index * ny * nz + y_index * nz + z_index

###########################################
# 2. Grain Boundary Analysis
###########################################
"""
Functions for analyzing grain boundaries, including:
- Curvature calculation using inside/outside method
- Normal vector calculations
- Boundary site identification
"""

def IO_curvature(microstructure):
    """Calculate the curvature for each grain boundary in the microstructure using inside/outside method.
    This is an optimized version that processes the entire structure using vectorized operations.
    
    Args:
        microstructure (ndarray): 3D numpy array containing grain IDs
        
    Returns:
        dict: Dictionary mapping grain boundary indices to tuple containing:
            - Number of inner voxels
            - Number of outer voxels  
            - Number of faces
            - Calculated curvature
    """
    voxel_and_face_num_edge = dict() # each GB: voxel_inner, voxel_outer, face_num, curvature
    nx, ny, nz = microstructure.shape
    edge_list = [0,1,3,5,8,12]
    
    # for periodic boundary conditions
    offsets = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    idx = np.indices((nx, ny, nz)).reshape(3, -1).T
    
    all_neighbors = np.zeros((len(idx),len(offsets)))
    for i in range(len(offsets)):
        neighbor_idx = idx + offsets[i]
        neighbor_idx = np.mod(neighbor_idx, [nx, ny, nz])
        all_neighbors[:,i] = microstructure[tuple(neighbor_idx.T)]
    num_faces = np.sum((all_neighbors - np.repeat(microstructure[tuple(idx.T)][:,np.newaxis],7,1)) != 0,axis=1)
    
    for i in (range(len(all_neighbors))):
        neighbors_unique_id = np.unique(all_neighbors[i])
        if len(neighbors_unique_id) == 2:
            pair_id = get_line(neighbors_unique_id[0],neighbors_unique_id[1])
            
            if pair_id not in voxel_and_face_num_edge: example = np.zeros(4)
            else: example = voxel_and_face_num_edge[pair_id]
            # figure out the num of faces for each sites
            example[2] += num_faces[i]/2 # add the true face to dict
            # figure out the number of edges
            if microstructure[tuple(idx[i])] == neighbors_unique_id[0]: # smaller
                example[0] += edge_list[int(num_faces[i]-1)]
            else: # larger
                example[1] += edge_list[int(num_faces[i]-1)]
            voxel_and_face_num_edge[pair_id] = example
        # if len(neighbors_unique_id) >= 2:
        #     for g1, g2 in combinations(neighbors_unique_id, 2):
        #         pair = sorted([g1, g2])
        #         pair_id = get_line(pair[0], pair[1])
        #         if pair_id not in voxel_and_face_num_edge:
        #             example = np.zeros(4)
        #         else:
        #             example = voxel_and_face_num_edge[pair_id]
        #         # Add half of the face count for this voxel to the total faces for this grain boundary pair
        #         different_pairs = len(neighbors_unique_id) * (len(neighbors_unique_id) - 1) / 2
        #         example[2] += num_faces[i] / (2 * max(1, different_pairs))
        #         # Assign edge count based on whether the current voxel belongs to the smaller grain id in the pair
        #         if microstructure[tuple(idx[i])] == pair[0]:
        #             example[0] += edge_list[int(num_faces[i] - 1)]
        #         elif microstructure[tuple(idx[i])] == pair[1]:
        #             example[1] += edge_list[int(num_faces[i] - 1)]
        #         else:
        #             continue
        #         voxel_and_face_num_edge[pair_id] = example
    
    # curvature point to the center of smaller grain id is positive
    for key, value in voxel_and_face_num_edge.items():
        value[3] = np.pi / 4 * (value[0] - value[1]) / value[2]
        voxel_and_face_num_edge[key] = value
    
    return voxel_and_face_num_edge

def IO_curvature_segment_numface(microstructure):
    """Get numbers of faces for each voxel in a localized grain boundary segment.
    Used for analyzing individual boundary segments rather than full microstructure.
    
    Args:
        microstructure (ndarray): 3D numpy array of a single grain boundary segment
        
    Returns:
        tuple: Arrays of neighbor IDs and number of faces per voxel
    """
    voxel_and_face_num_edge = dict() # each GB: voxel_inner, voxel_outer, face_num, curvature
    nx, ny, nz = microstructure.shape
    
    # for periodic boundary conditions
    offsets = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    idx = np.indices((nx, ny, nz)).reshape(3, -1).T
    
    all_neighbors = np.zeros((len(idx),len(offsets)))
    for i in range(len(offsets)):
        neighbor_idx = idx + offsets[i]
        neighbor_idx = np.mod(neighbor_idx, [nx, ny, nz])
        all_neighbors[:,i] = microstructure[tuple(neighbor_idx.T)]
    num_faces = np.sum((all_neighbors - np.repeat(microstructure[tuple(idx.T)][:,np.newaxis],7,1)) != 0,axis=1)
    
    return all_neighbors, num_faces

def IO_curvature_segment_curvature(num_faces, location_smaller, location_larger, matrix_shape):
    """Calculate curvature for a specific grain boundary segment between two grains.
    Uses the number of faces and voxel locations to compute localized curvature.
    
    Args:
        num_faces (ndarray): Array of face counts per voxel
        location_smaller (list): Coordinates of voxels in smaller grain ID 
        location_larger (list): Coordinates of voxels in larger grain ID
        matrix_shape (tuple): Shape of the microstructure matrix
        
    Returns:
        ndarray: Array containing inner voxels, outer voxels, faces and curvature
    """
    nx, ny, nz = matrix_shape
    edge_list = [0,1,3,5,8,12]
    example = np.zeros(4)
    for i in range(len(location_smaller)):
        index = get_index_id_from_matrix(location_smaller[i], nx, ny, nz)
        example[0] += edge_list[int(num_faces[index]-1)]
        example[2] += num_faces[index]/2 # add the true face to dict
        
    for i in range(len(location_larger)):
        index = get_index_id_from_matrix(location_larger[i], nx, ny, nz)
        example[1] += edge_list[int(num_faces[index]-1)]
        example[2] += num_faces[index]/2 # add the true face to dict
    
    # curvature point to the center of smaller grain id is positive
    example[3] = np.pi / 4 * (example[1] - example[0]) / example[2]
    
    return example

def get_normal_vector(grain_structure_figure_one):
    """Calculate normal vectors for grain boundaries in 2D microstructure.
    Uses smooth linear interpolation method.
    
    Args:
        grain_structure_figure_one (ndarray): 2D microstructure array
        
    Returns:
        tuple: (smooth_field, boundary_sites, boundary_sites_by_grain) - Smoothed field and boundary locations
    """
    nx = grain_structure_figure_one.shape[0]
    ny = grain_structure_figure_one.shape[1]
    ng = np.max(grain_structure_figure_one)
    cores = 8
    loop_times = 5
    P0 = grain_structure_figure_one
    R = np.zeros((nx,ny,2))
    smooth_class = smooth.linear_class(nx,ny,ng,cores,loop_times,P0,R)

    smooth_class.linear_main("inclination")
    P = smooth_class.get_P()
    sites = smooth_class.get_all_gb_list()
    sites_together = []
    for id in range(len(sites)): sites_together += sites[id]
    print(f"Total num of GB sites: {len(sites_together)}")

    return P, sites_together, sites

def get_normal_vector_3d(grain_structure_figure_one, verification_system = True):
    """Calculate normal vectors for grain boundaries in 3D microstructure.
    Uses smooth linear interpolation method.
    
    Args:
        grain_structure_figure_one (ndarray): 3D microstructure array
        verification_system (bool): Enable verification outputs
        
    Returns:
        tuple: (smooth_field, boundary_sites, boundary_sites_by_grain) - Smoothed field and boundary locations
    """
    nx = grain_structure_figure_one.shape[0]
    ny = grain_structure_figure_one.shape[1]
    nz = grain_structure_figure_one.shape[2]
    ng = np.max(grain_structure_figure_one)
    cores = 16
    loop_times = 5
    P0 = grain_structure_figure_one[:,:,:,np.newaxis]
    R = np.zeros((nx,ny,nz,3))
    smooth_class = smooth_3d.linear3d_class(nx,ny,nz,ng,cores,loop_times,P0,R,'p',verification_system)

    smooth_class.linear3d_main("inclination")
    P = smooth_class.get_P()
    sites = smooth_class.get_all_gb_list()
    sites_together = []
    for id in range(len(sites)): sites_together += sites[id]
    print(f"Total num of GB sites: {len(sites_together)}")

    return P, sites_together, sites

###########################################
# 3. Orientation Analysis
###########################################
"""
Functions for crystallographic calculations including:
- Euler angle conversions
- Quaternion operations
- Misorientation analysis
- Symmetry operations
"""

def euler2quaternion(yaw, pitch, roll):
    """Convert Euler angles to quaternion representation.
    
    Args:
        yaw (float): Yaw angle in radians
        pitch (float): Pitch angle in radians
        roll (float): Roll angle in radians
        
    Returns:
        list: Quaternion [qx, qy, qz, qw]
    """
    qx = np.cos(pitch/2.)*np.cos((yaw+roll)/2.)
    qy = np.sin(pitch/2.)*np.cos((yaw-roll)/2.)
    qz = np.sin(pitch/2.)*np.sin((yaw-roll)/2.)
    qw = np.cos(pitch/2.)*np.sin((yaw+roll)/2.)

    return [qx, qy, qz, qw]

def symquat(index, Osym = 24):
    """Get quaternion for specific symmetry operation.
    
    Args:
        index (int): Index of symmetry operation
        Osym (int): Number of symmetry operations (24=cubic, 12=hexagonal)
        
    Returns:
        ndarray: Quaternion representing symmetry operation
    """
    q = np.zeros(4)

    if Osym == 24:
        SYM = np.array([[1, 0, 0,  0, 1, 0,  0, 0, 1],
                        [1, 0, 0,  0, -1, 0,  0, 0, -1],
                        [1, 0, 0,  0, 0, -1,  0, 1, 0],
                        [1, 0, 0,  0, 0, 1,  0, -1, 0],
                        [-1, 0, 0,  0, 1, 0,  0, 0, -1],
                        [-1, 0, 0,  0, -1, 0,  0, 0, 1],
                        [-1, 0, 0,  0, 0, -1,  0, -1, 0],
                        [-1, 0, 0,  0, 0, 1,  0, 1, 0],
                        [0, 1, 0, -1, 0, 0,  0, 0, 1],
                        [0, 1, 0,  0, 0, -1, -1, 0, 0],
                        [0, 1, 0,  1, 0, 0,  0, 0, -1],
                        [0, 1, 0,  0, 0, 1,  1, 0, 0],
                        [0, -1, 0,  1, 0, 0,  0, 0, 1],
                        [0, -1, 0,  0, 0, -1,  1, 0, 0],
                        [0, -1, 0, -1, 0, 0,  0, 0, -1],
                        [0, -1, 0,  0, 0, 1, -1, 0, 0],
                        [0, 0, 1,  0, 1, 0, -1, 0, 0],
                        [0, 0, 1,  1, 0, 0,  0, 1, 0],
                        [0, 0, 1,  0, -1, 0,  1, 0, 0],
                        [0, 0, 1, -1, 0, 0,  0, -1, 0],
                        [0, 0, -1,  0, 1, 0,  1, 0, 0],
                        [0, 0, -1, -1, 0, 0,  0, 1, 0],
                        [0, 0, -1,  0, -1, 0, -1, 0, 0],
                        [0, 0, -1,  1, 0, 0,  0, -1, 0]])
    elif Osym == 12:
        a = np.sqrt(3)/2
        SYM = np.array([[1,  0, 0,  0,   1, 0,  0, 0,  1],
                        [-0.5,  a, 0, -a, -0.5, 0,  0, 0,  1],
                        [-0.5, -a, 0,  a, -0.5, 0,  0, 0,  1],
                        [0.5,  a, 0, -a, 0.5, 0,  0, 0,  1],
                        [-1,  0, 0,  0,  -1, 0,  0, 0,  1],
                        [0.5, -a, 0,  a, 0.5, 0,  0, 0,  1],
                        [-0.5, -a, 0, -a, 0.5, 0,  0, 0, -1],
                        [1,  0, 0,  0,  -1, 0,  0, 0, -1],
                        [-0.5,  a, 0,  a, 0.5, 0,  0, 0, -1],
                        [0.5,  a, 0,  a, -0.5, 0,  0, 0, -1],
                        [-1,  0, 0,  0,   1, 0,  0, 0, -1],
                        [0.5, -a, 0, -a, -0.5, 0,  0, 0, -1]])

    if (1+SYM[index, 0]+SYM[index, 4]+SYM[index, 8]) > 0:
        q4 = np.sqrt(1+SYM[index, 0]+SYM[index, 4]+SYM[index, 8])/2
        q[0] = q4
        q[1] = (SYM[index, 7]-SYM[index, 5])/(4*q4)
        q[2] = (SYM[index, 2]-SYM[index, 6])/(4*q4)
        q[3] = (SYM[index, 3]-SYM[index, 1])/(4*q4)
    elif (1+SYM[index, 0]-SYM[index, 4]-SYM[index, 8]) > 0:
        q4 = np.sqrt(1+SYM[index, 0]-SYM[index, 4]-SYM[index, 8])/2
        q[0] = (SYM[index, 7]-SYM[index, 5])/(4*q4)
        q[1] = q4
        q[2] = (SYM[index, 3]+SYM[index, 1])/(4*q4)
        q[3] = (SYM[index, 2]+SYM[index, 6])/(4*q4)
    elif (1-SYM[index, 0]+SYM[index, 4]-SYM[index, 8]) > 0:
        q4 = np.sqrt(1-SYM[index, 0]+SYM[index, 4]-SYM[index, 8])/2
        q[0] = (SYM[index, 2]-SYM[index, 6])/(4*q4)
        q[1] = (SYM[index, 3]+SYM[index, 1])/(4*q4)
        q[2] = q4
        q[3] = (SYM[index, 7]+SYM[index, 5])/(4*q4)
    elif (1-SYM[index, 0]-SYM[index, 4]+SYM[index, 8]) > 0:
        q4 = np.sqrt(1-SYM[index, 0]-SYM[index, 4]+SYM[index, 8])/2
        q[0] = (SYM[index, 3]-SYM[index, 1])/(4*q4)
        q[1] = (SYM[index, 2]+SYM[index, 6])/(4*q4)
        q[2] = (SYM[index, 7]+SYM[index, 5])/(4*q4)
        q[3] = q4

    return q

def quat_Multi(q1, q2):
    """Multiply two quaternions.
    
    Args:
        q1 (ndarray): First quaternion
        q2 (ndarray): Second quaternion
        
    Returns:
        ndarray: Product quaternion
    """
    q = np.zeros(4)
    q[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    q[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    q[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    q[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]

    return q

def in_cubic_fz(m_axis):
    """Check if misorientation axis lies in fundamental zone.
    
    Args:
        m_axis (ndarray): Misorientation axis vector
        
    Returns:
        bool: True if axis is in fundamental zone
    """
    # check if the misoreintation axis in fundamental zone
    # three core axis
    axis_a = np.array([1,0,0])
    axis_b = np.array([1,1,0])/np.sqrt(2)
    axis_c = np.array([1,1,1])/np.sqrt(3)
    # if in fz
    judgement_0 = np.dot(np.cross(axis_a, axis_b), m_axis) >= 0
    judgement_1 = np.dot(np.cross(axis_b, axis_c), m_axis) >= 0
    judgement_2 = np.dot(np.cross(axis_c, axis_a), m_axis) >= 0
    # print(f"{judgement_0}, {judgement_1}, {judgement_2}")
    return judgement_0*judgement_1*judgement_2

def quaternions_fz(q1, q2, symm2quat_matrix, Osym=24):
    """Calculate misorientation between two orientations in fundamental zone.
    
    Args:
        q1 (ndarray): First orientation quaternion
        q2 (ndarray): Second orientation quaternion
        symm2quat_matrix (ndarray): Matrix of symmetry quaternions
        Osym (int): Number of symmetry operations
        
    Returns:
        tuple: (misom, axis) - Misorientation angle and axis
    """
    q = np.zeros(4)
    misom = 2*np.pi
    axis = np.array([1, 0, 0])
    # print(f"q1: {q1}, q2: {q2}")
    for i in range(0, Osym):
        for j in range(0, Osym):
            # get misorientation quaternion q
            q1b = quat_Multi(symm2quat_matrix[i], q1)
            q2b = quat_Multi(symm2quat_matrix[j], q2)
            q2b[1:] = -q2b[1:]
            q = quat_Multi(q1b, q2b)
            # print(q)
            # get the q and inverse of q
            q_and_inverse = np.array([q,q])
            q_and_inverse[1,1:] = -q_and_inverse[1,1:]
            # get m_axis and inverse m_axis
            base = np.sqrt(1-q[0]*q[0])
            if base: axis_tmp = q_and_inverse[:,1:]/base
            else: axis_tmp = np.array([[1, 0, 0],[1, 0, 0]])
            # judge if the m_axis in fundamental zone or not
            in_cubic_fz_result = in_cubic_fz(axis_tmp.T)
            if not np.sum(in_cubic_fz_result): continue
            
            # find the index of m_axis in fundamental zone or not
            true_index = np.squeeze(np.where(in_cubic_fz_result))
            # find the minimal miso angle
            miso0 = 2*math.acos(round(q[0], 5))
            if miso0 > np.pi: miso0 = miso0 - 2*np.pi
            if abs(miso0) < misom:
                misom = abs(miso0)
                qmin = q_and_inverse[true_index]
                axis = axis_tmp[true_index]

    return misom, axis

def multiP_calM(i, quartAngle, symm2quat_matrix, Osym):
    """Calculate misorientation parameters between grain pair.
    
    Args:
        i (list): Two grain IDs
        quartAngle (ndarray): Quaternions for all grains
        symm2quat_matrix (ndarray): Matrix of symmetry quaternions  
        Osym (int): Number of symmetry operations
        
    Returns:
        ndarray: [gamma, axis_x, axis_y, axis_z] where gamma is misorientation angle
    """
    # output the value of MisoEnergy by inout the two grain ID: i[0] and i[1]

    qi = quartAngle[i[0]-1, :]
    qj = quartAngle[i[1]-1, :]

    theta, axis = quaternions_fz(qi, qj, symm2quat_matrix, Osym)
    # theta = theta*(theta<1)+(theta>1)
    # gamma = theta*(1-np.log(theta))
    gamma = theta
    return np.insert(axis, 0, gamma)

def pre_operation_misorientation(grainNum, init_filename, Osym=24):
    """Prepare matrices for misorientation calculations.
    
    Args:
        grainNum (int): Number of grains
        init_filename (str): Path to init file with Euler angles
        Osym (int): Number of symmetry operations
        
    Returns:
        tuple: (symm2quat_matrix, quartAngle) - Symmetry quaternions and grain orientation quaternions
    """
    # create the marix to store euler angle and misorientation
    quartAngle = np.ones((grainNum, 4))*-2

    # Create a quaternion matrix to show symmetry
    symm2quat_matrix = np.zeros((Osym, 4))
    for i in range(0, Osym):
        symm2quat_matrix[i, :] = symquat(i, Osym)

    # read the input euler angle from *.init
    with open(init_filename, 'r', encoding='utf-8') as f:
        for line in f:
            eachline = line.split()

            if len(eachline) == 5 and eachline[0] != '#':
                lineN = int(eachline[1])-1
                if quartAngle[lineN, 0] == -2:
                    quartAngle[lineN, :] = euler2quaternion(float(eachline[2]), float(eachline[3]), float(eachline[4]))

    return symm2quat_matrix, quartAngle

###########################################
# 4. Grain Analysis and Statistics
###########################################
"""
Functions for analyzing grain properties including:
- Size calculations
- Shape analysis
- Statistical measures
"""

def get_grain_size_from_data(npy_data):
    """Calculate grain sizes from microstructure evolution data.
    
    Args:
        npy_data (ndarray): 4D array of microstructure evolution data
        
    Returns:
        ndarray: Array of grain radii over time, calculated as sqrt(area/pi)
    """
    num_steps = npy_data.shape[0]
    num_grains = int(npy_data[0].max())
    grain_size_array = np.zeros((num_steps, num_grains))

    for i in tqdm(range(num_steps)):
        for j in range(num_grains):
            grain_id = j + 1
            grain_size_array[i, j] = np.sum(npy_data[i] == grain_id)
    grain_size_array = np.sqrt(grain_size_array/np.pi)
    return grain_size_array

def get_ave_grain_size_from_cluster(clname, grain_num):
    """Calculate average grain size from SPPARKS cluster file.
    
    Args:
        clname (str): Path to cluster file
        grain_num (int): Total number of grains
        
    Returns:
        tuple: (Time, Size) - Arrays of timesteps and average grain sizes
    """
    Time = []
    Size = []
    # Open filename
    with open(clname, 'r') as file:
        line = file.readline()
        while line:
            eachline = line.split()
            if len(eachline) == 3:
                if eachline[0] == 'Time':
                    if len(Time) != 0:
                        reduced_grain_num = sum(Size_one_step > 0)
                        Size.append(sum((Size_one_step / np.pi) ** 0.5) / reduced_grain_num)
                    Time.append(float(eachline[2]))
                    Size_one_step = np.zeros(grain_num)
            if len(eachline) == 7:
                if eachline[0].isdigit(): Size_one_step[int(eachline[1])-1] += float(eachline[3])
            line = file.readline()
    # Last Calculation of ave grain num
    reduced_grain_num = sum(Size_one_step > 0)
    Size.append(sum((Size_one_step / np.pi) ** 0.5) / reduced_grain_num)
    Time = np.array(Time)
    Size = np.array(Size)
    if len(Time) != len(Size): print("Shouldn't happen! "+ str(len(Time) + " " + str(len(Size))))
    return Time, Size

def get_poly_center(micro_matrix, step):
    """Calculate centers of non-periodic grains in microstructure.
    
    Args:
        micro_matrix (ndarray): 4D array of microstructure evolution
        step (int): Timestep to analyze
        
    Returns:
        tuple: (center_list, ave_radius_list) - Grain centers and average radii
    """
    # Get the center of all non-periodic grains in matrix
    num_grains = int(np.max(micro_matrix[step,:]))
    center_list = np.zeros((num_grains,2))
    sites_num_list = np.zeros(num_grains)
    ave_radius_list = np.zeros(num_grains)
    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i,j] = i
            coord_refer_j[i,j] = j

    table = micro_matrix[step,:,:,0]
    for i in range(num_grains):
        sites_num_list[i] = np.sum(table == i+1)

        if (sites_num_list[i] < 500) or \
           (np.max(coord_refer_i[table == i+1]) - np.min(coord_refer_i[table == i+1]) == micro_matrix.shape[1]) or \
           (np.max(coord_refer_j[table == i+1]) - np.min(coord_refer_j[table == i+1]) == micro_matrix.shape[2]): # grains on bc are ignored
          center_list[i, 0] = 0
          center_list[i, 1] = 0
          sites_num_list[i] == 0
        else:
          center_list[i, 0] = np.sum(coord_refer_i[table == i+1]) / sites_num_list[i]
          center_list[i, 1] = np.sum(coord_refer_j[table == i+1]) / sites_num_list[i]
    ave_radius_list = np.sqrt(sites_num_list / np.pi)

    return center_list, ave_radius_list

def get_poly_statistical_radius(micro_matrix, sites_list, step):
    """Calculate maximum radius deviation for each grain.
    
    Args:
        micro_matrix (ndarray): 4D array of microstructure evolution
        sites_list (list): List of grain boundary sites by grain
        step (int): Timestep to analyze
        
    Returns:
        float: Area-weighted average of maximum radius deviation
    """
    # Get the max offset of average radius and real radius
    center_list, ave_radius_list = get_poly_center(micro_matrix, step)
    num_grains = int(np.max(micro_matrix[step,:]))

    max_radius_offset_list = np.zeros(num_grains)
    for n in range(num_grains):
        center = center_list[n]
        ave_radius = ave_radius_list[n]
        sites = sites_list[n]

        if ave_radius != 0:
            for sitei in sites:
                [i,j] = sitei
                current_radius = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                radius_offset = abs(current_radius - ave_radius)
                if radius_offset > max_radius_offset_list[n]: max_radius_offset_list[n] = radius_offset

            max_radius_offset_list[n] = max_radius_offset_list[n] / ave_radius

    max_radius_offset = np.average(max_radius_offset_list[max_radius_offset_list!=0])
    area_list = np.pi*ave_radius_list*ave_radius_list
    if np.sum(area_list) == 0: max_radius_offset = 0
    else: max_radius_offset = np.sum(max_radius_offset_list * area_list) / np.sum(area_list)

    return max_radius_offset

def get_poly_statistical_ar(micro_matrix, step):
    """Calculate average aspect ratio of grains.
    
    Args:
        micro_matrix (ndarray): 4D array of microstructure evolution
        step (int): Timestep to analyze
        
    Returns:
        float: Area-weighted average aspect ratio
    """
    # Get the average aspect ratio
    num_grains = int(np.max(micro_matrix[step,:]))
    sites_num_list = np.zeros(num_grains)
    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i,j] = i
            coord_refer_j[i,j] = j

    aspect_ratio_i = np.zeros((num_grains,2))
    aspect_ratio_j = np.zeros((num_grains,2))
    aspect_ratio = np.zeros(num_grains)
    table = micro_matrix[step,:,:,0]

    aspect_ratio_i_list = [[] for _ in range(int(num_grains))]
    aspect_ratio_j_list = [[] for _ in range(int(num_grains))]
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            grain_id = int(table[i][j]-1)
            sites_num_list[grain_id] +=1
            aspect_ratio_i_list[grain_id].append(coord_refer_i[i][j])
            aspect_ratio_j_list[grain_id].append(coord_refer_j[i][j])

    for i in range(num_grains):
        aspect_ratio_i[i, 0] = len(list(set(aspect_ratio_i_list[i])))
        aspect_ratio_j[i, 1] = len(list(set(aspect_ratio_j_list[i])))
        if aspect_ratio_j[i, 1] == 0: aspect_ratio[i] = 0
        else: aspect_ratio[i] = aspect_ratio_i[i, 0] / aspect_ratio_j[i, 1]

    aspect_ratio = np.sum(aspect_ratio * sites_num_list) / np.sum(sites_num_list)

    return aspect_ratio

###########################################
# 5. Normal Vector Distribution Analysis
###########################################
"""
Functions for analyzing the distribution of grain boundary normal vectors
and calculating related statistics
"""

def get_normal_vector_slope(P, sites, step, para_name, bias=None):
    """Calculate and plot slope distribution of normal vectors in 2D.
    
    Args:
        P (ndarray): Smoothed field containing normal vector information
        sites (list): List of boundary site coordinates
        step (int): Current timestep
        para_name (str): Parameter name for plot legend
        bias (float, optional): Optional bias to add to distribution
        
    Returns:
        float: Always returns 0 (used for tracking plot generation)
    """
    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)

    freqArray = np.zeros(binNum)
    degree = []
    for sitei in sites:
        [i,j] = sitei
        dx,dy = myInput.get_grad(P,i,j)
        degree.append(math.atan2(-dy, dx) + math.pi)
    for i in range(len(degree)):
        freqArray[int((degree[i]/math.pi*180-xLim[0])/binValue)] += 1
    freqArray = freqArray/sum(freqArray*binValue)

    # bias situation
    if bias is not None:
        freqArray = freqArray + bias
        freqArray = freqArray/sum(freqArray*binValue)
    # Plot
    plt.plot(xCor/180*math.pi, freqArray, linewidth=2, label=para_name)

    return 0

def get_normal_vector_slope_3d(P, sites, step, para_name, angle_index=0, bias=None):
    """Calculate and plot slope distribution of normal vectors in 3D.
    
    Args:
        P (ndarray): Smoothed field containing normal vector information
        sites (list): List of boundary site coordinates 
        step (int): Current timestep
        para_name (str): Parameter name for plot legend
        angle_index (int): Which angle to analyze (0=xy, 1=xz, 2=yz)
        bias (float, optional): Optional bias to add to distribution
        
    Returns:
        ndarray: Frequency array of normal vector angles
    """
    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)

    freqArray = np.zeros(binNum)
    degree = []
    for sitei in sites:
        [i,j,k] = sitei
        dx,dy,dz = myInput.get_grad3d(P,i,j,k)
        if angle_index == 0:
            dx_fake = dx
            dy_fake = dy
        elif angle_index == 1:
            dx_fake = dx
            dy_fake = dz
        elif angle_index == 2:
            dx_fake = dy
            dy_fake = dz

        # Normalize
        if math.sqrt(dy_fake**2+dx_fake**2) < 1e-5: continue
        dy_fake_norm = dy_fake / math.sqrt(dy_fake**2+dx_fake**2)
        dx_fake_norm = dx_fake / math.sqrt(dy_fake**2+dx_fake**2)

        degree.append(math.atan2(-dy_fake_norm, dx_fake_norm) + math.pi)
    for n in range(len(degree)):
        freqArray[int((degree[n]/math.pi*180-xLim[0])/binValue)] += 1
    freqArray = freqArray/sum(freqArray*binValue)

    if bias is not None:
        freqArray = freqArray + bias
        freqArray = freqArray/sum(freqArray*binValue)

    # Plot
    plt.plot(np.append(xCor,xCor[0])/180*math.pi, np.append(freqArray,freqArray[0]),linewidth=2,label=para_name)

    return freqArray

def simple_magnitude(freqArray):
    """Calculate anisotropic magnitude from inclination distribution.
    
    Args:
        freqArray (ndarray): Array of angle frequencies
        
    Returns:
        tuple: (magnitude_ave, magnitude_stan) - Average and standard deviation of anisotropic magnitude
    """
    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    # prefect circle
    freqArray_circle = np.ones(binNum)
    freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)
    # max/average(difference between currect distribuition and perfect distribution) over average of perfect distribution
    magnitude_max = np.max(abs(freqArray - freqArray_circle))/np.average(freqArray_circle)
    magnitude_ave = np.average(abs(freqArray - freqArray_circle))/np.average(freqArray_circle)
    # standard
    magnitude_stan = np.sqrt(np.sum((abs(freqArray - freqArray_circle)/np.average(freqArray_circle) - magnitude_ave)**2)/binNum)

    return magnitude_ave, magnitude_stan

###########################################
# 6. Visualization Functions
###########################################
"""
Functions for creating visualizations of microstructure and energy data:
- Static plots
- Animations
- Data representation
"""

def plot_energy_figure(timestep, energy_figure, figure_path=None):
    """Plot the energy distribution at a specific timestep.
    
    Args:
        timestep (int): Timestep to plot
        energy_figure (ndarray): Energy values over time
        figure_path (str, optional): Path to save the figure
    """
    imgs = []
    fig, ax = plt.subplots()

    cv0 = np.squeeze(energy_figure[timestep])
    cv0 = np.rot90(cv0,1)
    im = ax.imshow(cv0,vmin=0,vmax=6,cmap='Accent')
    cb = fig.colorbar(im)
    tx = ax.set_title(f'time step = {timestep}')
    if figure_path != None:
        plt.savefig('energy_{timestep}step')
    plt.show()

def plot_energy_video(timestep, energy_figure, figure_path, delta = 0):
    """Create an animation of energy evolution over multiple timesteps.
    
    Args:
        timestep (list): List of timesteps to include
        energy_figure (ndarray): Energy values over time
        figure_path (str): Path to save the video
        delta (float, optional): Offset for colormap range
    """
    imgs = []
    fig, ax = plt.subplots()

    cv0 = energy_figure[1,:,:,0] #np.squeeze(energy_figure[0])
    cv0 = np.rot90(cv0,1)
    if delta == 0: colormap_map = np.max(cv0)
    else: colormap_map = (1 + delta) * 8
    im = ax.imshow(cv0,vmin=np.min(cv0),vmax=colormap_map,cmap='Accent',interpolation='none')
    cb = fig.colorbar(im)
    tx = ax.set_title(f'time step = {timestep[0]}')
    # plt.show()

    def animate(i):
        arr=energy_figure[i,:,:,0] #np.squeeze(energy_figure[i])
        arr=np.rot90(arr,1)
        im.set_data(arr)
        tx.set_text(f'time step = {timestep[i]}')

    ani = animation.FuncAnimation(fig, animate, frames=len(timestep))
    FFMpegWriter = animation.writers['ffmpeg']
    writer = animation.FFMpegWriter(fps=math.floor(len(timestep)/5), bitrate=10000)
    ani.save(figure_path+".mp4",writer=writer)

def plot_structure_figure(step, structure_figure, figure_path):
    """Plot microstructure at a specific timestep.
    
    Args:
        step (int): Timestep to plot
        structure_figure (ndarray): Microstructure evolution data
        figure_path (str): Path to save the figure
    """
    plt.close()
    fig, ax = plt.subplots()

    cv_initial = np.squeeze(structure_figure[0])
    cv0 = np.squeeze(structure_figure[step])
    cv0 = np.rot90(cv0,1)
    im = ax.imshow(cv0,vmin=np.min(cv_initial),vmax=np.max(cv_initial),cmap='rainbow',interpolation='none') #jet rainbow plasma
    # cb = fig.colorbar(im)
    # cb.set_ticks([10000,20000])
    # cb.set_ticklabels([ '1e4', '2e4'])
    # cb.ax.tick_params(labelsize=20)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.tick_params(which = 'both', size = 0, labelsize = 0)

    plt.savefig(figure_path + f"_ts{step*30}.png", dpi=400,bbox_inches='tight')

def plot_structure_video(timestep, structure_figure, figure_path, dimension = 2, depth = 0):
    """Create an animation of microstructure evolution.
    
    Args:
        timestep (list): List of timesteps to include
        structure_figure (ndarray): Microstructure evolution data 
        figure_path (str): Path to save video
        dimension (int): Dimension to project (0,1,2 for x,y,z)
        depth (int): Slice position in projected dimension
    """
    imgs = []
    fig, ax = plt.subplots()
    if dimension == 0: structure_2d_project = structure_figure[:,depth,:,:]
    elif dimension == 1: structure_2d_project = structure_figure[:,:,depth,:]
    elif dimension == 2: structure_2d_project = structure_figure[:,:,:,depth]
    else: print("Please input the right dimension!")

    cv0 = structure_2d_project[1,:,:] #np.squeeze(structure_figure[0])
    cv0 = np.rot90(cv0,1)
    im = ax.imshow(cv0,vmin=np.min(cv0),vmax=np.max(cv0),cmap='rainbow',interpolation='none') #jet rainbow plasma
    cb = fig.colorbar(im)
    tx = ax.set_title(f'time step = {timestep[0]}')
    # plt.show()

    def animate(i):
        arr=structure_2d_project[i,:,:] #np.squeeze(structure_figure[i])
        arr=np.rot90(arr,1)
        im.set_data(arr)
        tx.set_text(f'time step = {timestep[i]}')

    ani = animation.FuncAnimation(fig, animate, frames=len(timestep))
    FFMpegWriter = animation.writers['ffmpeg']
    writer = animation.FFMpegWriter(fps=math.floor(len(timestep)/5), bitrate=10000)
    ani.save(figure_path+".mp4",writer=writer,dpi=400)

###########################################
# 7. File Format Conversion
###########################################
"""
Functions for converting between different file formats:
- SPPARKS dump files
- Init files
- Image arrays
- Neighbor list generation
"""

def dump2img(dump_path, num_steps=None, extract_data='type', extract_step=None):
    """Convert SPPARKS dump file format to image/array format.
    Processes either single dump file or series of numbered dump files.
    
    Args:
        dump_path (str): Path to dump file(s)
        num_steps (int, optional): Number of timesteps to process
        extract_data (str): Type of data to extract from dump file
        extract_step (int, optional): Specific timestep to extract
        
    Returns:
        tuple: (timesteps, grain_structure_figure) - Arrays of timesteps and grain structures
    """
    # Create grain structure figure from dump file with site ID.
    # if extarct_step is not None, function will extarct one step figrue (extract_step),
    # only work for dump_type 1 currently

    # dump file (type 0) or dump.* files (type 1)
    if os.path.exists(dump_path+".dump"):
        dump_type = 0
        dump_file_name_0 = dump_path+".dump"
    elif os.path.exists(dump_path+f".dump.{0 if extract_step == None else int(extract_step)}"):
        dump_type = 1
        dump_file_name_0 = dump_path+f".dump.{0 if extract_step == None else int(extract_step)}"
    else: print("There is no correct dump file for "+dump_path)

    with open(dump_file_name_0) as file:
        box_size = np.zeros(3)
        for i, line in enumerate(file):
            if i==3: num_sites = int(line)
            if i==5: box_size[0] = np.array(line.split(), dtype=float)[-1]
            if i==6: box_size[1] = np.array(line.split(), dtype=float)[-1]
            if i==7: box_size[2] = np.array(line.split(), dtype=float)[-1]
            if i==8: name_vars = line.split()[2:]
            if i>8: break
    box_size = np.ceil(box_size).astype(int) #reformat box_size
    entry_length = num_sites+9 #there are 9 header lines in each entry

    # total lines for dump
    if num_steps!=None: total_lines = num_steps*entry_length
    else: total_lines=None

    time_steps=[]
    grain_structure_figure=[]
    if dump_type == 0:
        with open(dump_file_name_0) as file:
            for i, line in tqdm(enumerate(file), "DECODING (%s.dump)"%dump_path[-20:], total=total_lines):
                [entry_num, line_num] = np.divmod(i,entry_length) #what entry number and entry line number does this line number indicate
                if line_num==0: entry = np.zeros(box_size) #set the energy figure matrix
                if line_num==1: time_steps.append(int(float(line.split()[-1]))) #log the time step
                atom_num = line_num-9 #track which atom line we're on
                if atom_num>=0 and atom_num<num_sites:
                    line_split = np.array(line.split(), dtype=float)
                    site_x = int(line_split[name_vars.index('x')])
                    site_y = int(line_split[name_vars.index('y')])
                    site_z = int(line_split[name_vars.index('z')])
                    entry[site_x,site_y,site_z] = line_split[name_vars.index(extract_data)] #record valid atom lines
                if line_num==entry_length-1:
                    grain_structure_figure.append(entry)
    elif dump_type == 1:
        if extract_step == None:
            dump_item = 0
            dump_file_name_item = dump_path+".dump."+str(int(dump_item))
            while os.path.exists(dump_file_name_item):
                with open(dump_file_name_item) as file:
                    for i, line in tqdm(enumerate(file), "EXTRACTING SPPARKS DUMP (%s.dump)"%dump_file_name_item[-20:], total=entry_length):
                        if i==0: entry = np.zeros(box_size) #set the energy figure matrix
                        if i==1: time_steps.append(int(float(line.split()[-1]))) #log the time step
                        atom_num = i-9 #track which atom line we're on
                        if atom_num>=0 and atom_num<num_sites:
                            line_split = np.array(line.split(), dtype=float)
                            site_x = int(line_split[name_vars.index('x')])
                            site_y = int(line_split[name_vars.index('y')])
                            site_z = int(line_split[name_vars.index('z')])
                            entry[site_x,site_y,site_z] = line_split[name_vars.index(extract_data)] #record valid atom lines
                        if i==entry_length-1: grain_structure_figure.append(entry)
                # jump to next dump.*
                dump_item += 1
                dump_file_name_item = dump_path+".dump."+str(int(dump_item))
        else:
            # extarct only specific steps
            dump_item = int(extract_step)
            dump_file_name_item = dump_path+".dump."+str(int(dump_item))
            with open(dump_file_name_item) as file:
                for i, line in tqdm(enumerate(file), "EXTRACTING SPPARKS DUMP (%s.dump)"%dump_file_name_item[-20:], total=entry_length):
                    if i==0: entry = np.zeros(box_size) #set the energy figure matrix
                    if i==1: time_steps.append(int(float(line.split()[-1]))) #log the time step
                    atom_num = i-9 #track which atom line we're on
                    if atom_num>=0 and atom_num<num_sites:
                        line_split = np.array(line.split(), dtype=float)
                        site_x = int(line_split[name_vars.index('x')])
                        site_y = int(line_split[name_vars.index('y')])
                        site_z = int(line_split[name_vars.index('z')])
                        entry[site_x,site_y,site_z] = line_split[name_vars.index(extract_data)] #record valid atom lines
                    if i==entry_length-1: grain_structure_figure.append(entry)
    else: print("wrong dump file input!")
    grain_structure_figure = np.array(grain_structure_figure)
    time_steps = np.array(time_steps)

    return time_steps, grain_structure_figure

def init2img(box_size, init_file_path_input):
    """Convert init file format to image array format.
    
    Args:
        box_size (tuple): (nx, ny, nz) dimensions
        init_file_path_input (str): Init file path
        
    Returns:
        ndarray: Image array with grain IDs
    """
    # make init as a img matrix
    
    size_x,size_y,size_z = box_size
    img_array = np.zeros(size_y*size_x*size_z)

    with open(init_file_path_input, 'r') as file: 
        for i, line in enumerate(file):
            if i >= 3: img_array[int(line.split()[0])-1] = int(line.split()[1])

    if size_z>1: fig = img_array.reshape(size_x,size_y,size_z)
    else: fig = img_array.reshape(size_x,size_y)

    return fig

def image2init(img, EulerAngles, fp):
    """Convert image format to SPPARKS init file format.
    
    Args:
        img (ndarray): Image array with grain IDs
        EulerAngles (ndarray): Array of Euler angles per grain
        fp (str): Output file path
    """
    # put image into init files
    
    # Set local variables
    size = img.shape
    dim = len(size)
    IC = [0]*(np.product(size)+3)
    # Write the information in the SPPARKS format and save the file
    IC[0] = '# This line is ignored\n'
    IC[1] = 'Values\n'
    IC[2] = '\n'
    k=0
    if dim==3:
        for i in tqdm(range(size[0])):
            for j in range(size[1]):
                for h in range(size[2]):
                    SiteID = int(img[i,j,h])
                    IC[k+3] = str(k+1) + ' ' + str(SiteID) + ' ' + str(EulerAngles[SiteID-1,0]) + ' ' + str(EulerAngles[SiteID-1,1]) + ' ' + str(EulerAngles[SiteID-1,2]) + '\n'
                    k = k + 1
    else:
        for i in tqdm(range(size[0])):
            for j in range(size[1]):
                SiteID = int(img[i,j])
                IC[k+3] = str(k+1) + ' ' + str(SiteID) + ' ' + str(EulerAngles[SiteID-1,0]) + ' ' + str(EulerAngles[SiteID-1,1]) + ' ' + str(EulerAngles[SiteID-1,2]) + '\n'
                k = k + 1
    with open(fp, 'w') as file:
        file.writelines(IC)
    # Completion message
    print("NEW IC WRITTEN TO FILE: %s"%fp)
    return

def output_init_from_dump(dump_file_path, euler_angle_array, init_file_path_output):
    """Convert SPPARKS dump file to init file with Euler angles.
    
    Args:
        dump_file_path (str): Path to dump file
        euler_angle_array (ndarray): Array of Euler angles per grain
        init_file_path_output (str): Output init file path
        
    Returns:
        tuple: (box_size, entry_length) - Simulation box size and dump file entry length
    """
    # output the init file with euler_angle_array and one dump file
    # Read necessary information from dump file
    with open(dump_file_path) as file:
        box_size = np.zeros(3)
        for i, line in enumerate(file):
            if i==3: num_sites = int(line)
            if i==5: box_size[0] = np.array(line.split(), dtype=float)[-1]
            if i==6: box_size[1] = np.array(line.split(), dtype=float)[-1]
            if i==7: box_size[2] = np.array(line.split(), dtype=float)[-1]
            if i==8: name_vars = line.split()[2:]
            if i>8: break
    box_size = np.ceil(box_size).astype(int) #reformat box_size
    entry_length = num_sites+9 #there are 9 header lines in each entry


    # write the IC files and read the dump data
    # create IC file
    IC_nei = []
    IC_nei.append("# This line is ignored\n")
    IC_nei.append("Values\n")
    IC_nei.append("\n")
    with open(init_file_path_output, 'w') as output_file:
        output_file.writelines( IC_nei )
    IC_nei = []
    # read and write
    with open(init_file_path_output, 'a') as output_file:
        with open(dump_file_path) as file:
            for i, line in tqdm(enumerate(file), "EXTRACTING SPPARKS DUMP (%s.dump)"%dump_file_path[-20:], total=entry_length):
                if i==1: time_step = int(float(line.split()[-1])) #log the time step
                atom_num = i-9 #track which atom line we're on
                if atom_num>=0 and atom_num<num_sites:
                    line_split = np.array(line.split(), dtype=float)
                    grain_id = int(line_split[1])-1
                    output_file.write(f"{int(line_split[0])} {int(line_split[1])} {euler_angle_array[grain_id, 0]} {euler_angle_array[grain_id, 1]} {euler_angle_array[grain_id, 2]}\n")

    return box_size, entry_length

def output_init_neighbor_from_init(interval, box_size, init_file_path_input, init_file_path_output):
    """Generate neighbor list init file from basic init file.
    Uses direct iteration approach.
    
    Args:
        interval (int): Neighbor search radius
        box_size (tuple): (nx, ny, nz) dimensions
        init_file_path_input (str): Input init file path
        init_file_path_output (str): Output init file with neighbors
        
    Returns:
        bool: True if successful
    """
    # Output the init_nighbor5 with init file

    nei_num = (2*interval+3)**3-1
    size_x,size_y,size_z = box_size
    img = np.zeros((size_y,size_x,size_z)) #Figure of all sites with GrainID

    print(f"> img matrix start.")
    for k in tqdm(range(size_z)): # z-axis
        for i in range(size_y): # y-axis
            for j in range(size_x): # x-axis
                img[i,j,k] = int(k*size_x*size_y + i*size_x + j)
    print(f"> img matrix end")

    # distinguish the 2D and 3D cases
    if size_z == 1:
        IC_nei = []
        IC_nei.append("# This line is ignored\n")
        IC_nei.append("2 dimension\n")
        IC_nei.append(f"{nei_num} max neighbors\n")
        IC_nei.append(f"{size_x*size_y} sites\n")
        IC_nei.append(f"0 {size_x} xlo xhi\n")
        IC_nei.append(f"0 {size_y} ylo yhi\n")
        IC_nei.append("0 1 zlo zhi\n")
        IC_nei.append("\n")
        IC_nei.append("Sites\n")
        IC_nei.append("\n")
        with open(init_file_path_output, 'w') as file:
            file.writelines( IC_nei )
        IC_nei = []
        print("> Sites start writing")
        with open(init_file_path_output, 'a') as file:
            for i in range(size_y): # y-axis
                for j in range(size_x): # x-axis
                    file.write(f"{int(img[i,j,0] + 1)} {float(j)} {float(i)} 0.5\n")
        print("> Sites end writing")
        IC_nei.append("\n")
        IC_nei.append("Neighbors\n")
        IC_nei.append("\n")
        with open(init_file_path_output, 'a') as file:
            file.writelines( IC_nei )
        IC_nei = []

        print("> Neighbors start writing")
        max_length_neighbors = 0
        with open(init_file_path_output, 'a') as file:
            for i in range(size_y): # y-axis
                for j in range(size_x): # x-axis
                    tmp_nei = f"{int(img[i,j,0] + 1)} "
                    for m in range(-(interval+1),interval+2):
                        for n in range(-(interval+1),interval+2):
                            if m==0 and n==0: continue
                            tmp_i = (i+m)%size_y
                            tmp_j = (j+n)%size_x
                            tmp_nei += f" {int(img[tmp_i, tmp_j, 0]+1)}"

                    IC_nei.append(tmp_nei+"\n")
                    if len(tmp_nei) > max_length_neighbors: max_length_neighbors = len(tmp_nei)
                    file.write(tmp_nei+"\n")
            file.write("\n")
        print(f"The max length of neighbor data line is {max_length_neighbors}")
        print("> Neighbors end writing")
        print("> Values start writing")
        with open(init_file_path_input, 'r') as f_read:
            tmp_values = f_read.readlines()
        print("> Values read done")
        with open(init_file_path_output, 'a') as file:
            file.writelines(tmp_values[1:])
        print("> Values end writing")

    else:
        IC_nei = []
        IC_nei.append("# This line is ignored\n")
        IC_nei.append("3 dimension\n")
        IC_nei.append(f"{nei_num} max neighbors\n")
        IC_nei.append(f"{size_x*size_y*size_z} sites\n")
        IC_nei.append(f"0 {size_x} xlo xhi\n")
        IC_nei.append(f"0 {size_y} ylo yhi\n")
        IC_nei.append(f"0 {size_z} zlo zhi\n")
        IC_nei.append("\n")
        IC_nei.append("Sites\n")
        IC_nei.append("\n")
        with open(init_file_path_output, 'w') as file:
            file.writelines( IC_nei )
        IC_nei = []

        print("> Sites start writing")
        with open(init_file_path_output, 'a') as file:
            for k in tqdm(range(size_z)): # z-axis
                for i in range(size_y): # y-axis
                    for j in range(size_x): # x-axis
                        file.write(f"{int(img[i,j,k] + 1)} {float(j)} {float(i)} {float(k)}\n")
        print("> Sites end writing")

        IC_nei.append("\n")
        IC_nei.append("Neighbors\n")
        IC_nei.append("\n")
        with open(init_file_path_output, 'a') as file:
            file.writelines( IC_nei )
        IC_nei = []

        print("> Neighbors start writing")
        max_length_neighbors = 0
        with open(init_file_path_output, 'a') as file:
            for k in tqdm(range(size_z)): # z-axis
                for i in range(size_y): # y-axis
                    for j in range(size_x): # x-axis
                        tmp_nei = f"{int(img[i,j,k] + 1)} "
                        offsets = np.array(np.meshgrid(
                        np.arange(-(interval + 1), interval + 2),
                        np.arange(-(interval + 1), interval + 2),
                        np.arange(-(interval + 1), interval + 2),
                        )).T.reshape(-1, 3)
                        # Filter out the [0, 0, 0] offset since we want to skip it
                        offsets = offsets[np.any(offsets != 0, axis=1)]
                        # Compute the indices with wrapping around boundaries (using np.mod)
                        indices = (np.array([i, j, k]) + offsets) % np.array([size_y, size_x, size_z])
                        # Extract the values from 'img' using advanced indexing
                        neighbour_values = img[indices[:, 0], indices[:, 1], indices[:, 2]].astype('int')
                        # Convert values to 1-based indexing and concatenate into a string
                        tmp_nei += ' '.join(map(str, neighbour_values + 1))
                        if len(tmp_nei) > max_length_neighbors: max_length_neighbors = len(tmp_nei)
                        file.write(tmp_nei+"\n")
            file.write("\n")
        print(f"The max length of neighbor data line is {max_length_neighbors}")
        print("> Neighbors end writing")

        print("> Values start writing")
        with open(init_file_path_input, 'r') as f_read:
            tmp_values = f_read.readlines()
        print("> Values read done")
        with open(init_file_path_output, 'a') as file:
            file.writelines(tmp_values[1:])
        print("> Values end writing")
    return True

def output_init_neighbor_from_init_mp(interval, box_size, init_file_path_input, init_file_path_output):
    """Generate neighbor list init file from basic init file.
    Uses multiprocessing for better performance.
    
    Args:
        interval (int): Neighbor search radius
        box_size (tuple): (nx, ny, nz) dimensions 
        init_file_path_input (str): Input init file path
        init_file_path_output (str): Output init file with neighbors
        
    Returns:
        bool: True if successful
    """
    # Output the init_nighbor5 with init file
    size_x,size_y,size_z = box_size
    dimension = int(2 if size_z==1 else 3)
    nei_num = (2*interval+3)**dimension-1

    print(f"> img matrix start.")
    # Create arrays representing the range of each dimension
    k_range = np.arange(size_z).reshape(1, 1, size_z)
    i_range = np.arange(size_y).reshape(size_y, 1, 1)
    j_range = np.arange(size_x).reshape(1, size_x, 1)
    img = (k_range * size_x * size_y + i_range * size_x + j_range).astype(int)
    print(f"> img matrix end")

    IC_nei = []
    IC_nei.append("# This line is ignored\n")
    IC_nei.append(f"{dimension} dimension\n")
    IC_nei.append(f"{nei_num} max neighbors\n")
    IC_nei.append(f"{size_x*size_y*size_z} sites\n")
    IC_nei.append(f"0 {size_x} xlo xhi\n")
    IC_nei.append(f"0 {size_y} ylo yhi\n")
    IC_nei.append(f"0 {size_z} zlo zhi\n")
    IC_nei.append("\n")
    IC_nei.append("Sites\n")
    IC_nei.append("\n")
    with open(init_file_path_output, 'w') as file:
        file.writelines( IC_nei )
    IC_nei = []

    print("> Sites start writing")
    with open(init_file_path_output, 'a') as file:
        for k in tqdm(range(size_z)): # z-axis
            for i in range(size_y): # y-axis
                for j in range(size_x): # x-axis
                    file.write(f"{int(img[i,j,k] + 1)} {float(j)} {float(i)} {0.5 if dimension==2 else float(k)}\n")
    print("> Sites end writing")
    IC_nei.append("\n")
    IC_nei.append("Neighbors\n")
    IC_nei.append("\n")
    with open(init_file_path_output, 'a') as file:
        file.writelines( IC_nei )
    IC_nei = []

    # distinguish the 2D and 3D cases
    print("> Neighbors start writing")
    if dimension == 2:
        # offset value setting before neighboring start
        offsets = np.array(np.meshgrid(
        np.arange(-(interval + 1), interval + 2),
        np.arange(-(interval + 1), interval + 2),
        )).T.reshape(-1, 2)
        # Filter out the [0, 0, 0] offset since we want to skip it
        offsets = offsets[np.any(offsets != 0, axis=1)]
        # function for multiprocess
        def process_chunk(start_k, end_k, file_name):
            max_length_neighbors = 0
            with open(file_name, 'w') as file:
                for i in tqdm(range(start_k, end_k)): # y-axis
                    for j in range(size_x): # x-axis
                        tmp_nei = f"{int(img[i,j,0] + 1)} "
                        # Compute the indices with wrapping around boundaries (using np.mod)
                        indices = (np.array([i, j]) + offsets) % np.array([size_y, size_x])
                        # Extract the values from 'img' using advanced indexing
                        neighbour_values = img[indices[:, 0], indices[:, 1],0].astype('int')
                        # Convert values to 1-based indexing and concatenate into a string
                        tmp_nei += ' '.join(map(str, neighbour_values + 1))
                        max_length_neighbors = max(max_length_neighbors, len(tmp_nei))
                        file.write(tmp_nei + "\n")
            print(f"The max length of neighbor data line is {max_length_neighbors}")
        # necessary init for multi cores writing
        num_processes = mp.cpu_count() # or choose a number that suits your machine
        chunk_size = size_y / num_processes
        processes = []
        temp_files = []
        # Assign tasks for processors
        for p in range(num_processes):
            start_k = int(p * chunk_size)
            end_k = int((p + 1) * chunk_size) if p != num_processes - 1 else size_y
            temp_file = f'{init_file_path_output}_temp_{p}.txt'
            temp_files.append(temp_file)
            process = mp.Process(target=process_chunk, args=(start_k, end_k, temp_file))
            processes.append(process)
            process.start()
        # Wait for all processes to complete
        for process in processes:
            process.join()
        # Concatenate all temporary files
        with open(init_file_path_output, 'a') as outfile:
            for fname in tqdm(temp_files, "Concatenating "):
                with open(fname) as infile:
                    outfile.write(infile.read())
                os.remove(fname)  # Optional: remove temp file after concatenation
            outfile.write("\n")
    else:
        # offset value setting before neighboring start
        offsets = np.array(np.meshgrid(
        np.arange(-(interval + 1), interval + 2),
        np.arange(-(interval + 1), interval + 2),
        np.arange(-(interval + 1), interval + 2),
        )).T.reshape(-1, 3)
        # Filter out the [0, 0, 0] offset since we want to skip it
        offsets = offsets[np.any(offsets != 0, axis=1)]
        # function for multiprocess
        def process_chunk(start_k, end_k, file_name):
            max_length_neighbors = 0
            with open(file_name, 'w') as file:
                for k in tqdm(range(start_k, end_k)): # z-axis
                    for i in range(size_y): # y-axis
                        for j in range(size_x): # x-axis
                            tmp_nei = f"{int(img[i,j,k] + 1)} "
                            # Compute the indices with wrapping around boundaries (using np.mod)
                            indices = (np.array([i, j, k]) + offsets) % np.array([size_y, size_x, size_z])
                            # Extract the values from 'img' using advanced indexing
                            neighbour_values = img[indices[:, 0], indices[:, 1], indices[:, 2]].astype('int')
                            # Convert values to 1-based indexing and concatenate into a string
                            tmp_nei += ' '.join(map(str, neighbour_values + 1))
                            max_length_neighbors = max(max_length_neighbors, len(tmp_nei))
                            file.write(tmp_nei + "\n")
            print(f"The max length of neighbor data line is {max_length_neighbors}")

        # necessary init for multi cores writing
        num_processes = mp.cpu_count() # or choose a number that suits your machine
        chunk_size = size_z / num_processes
        processes = []
        temp_files = []
        # Assign tasks for processors
        for p in range(num_processes):
            start_k = int(p * chunk_size)
            end_k = int((p + 1) * chunk_size) if p != num_processes - 1 else size_z
            temp_file = f'{init_file_path_output}_temp_{p}.txt'
            temp_files.append(temp_file)
            process = mp.Process(target=process_chunk, args=(start_k, end_k, temp_file))
            processes.append(process)
            process.start()
        # Wait for all processes to complete
        for process in processes:
            process.join()
        # Concatenate all temporary files
        with open(init_file_path_output, 'a') as outfile:
            for fname in tqdm(temp_files, "Concatenating "):
                with open(fname) as infile:
                    outfile.write(infile.read())
                os.remove(fname)  # Optional: remove temp file after concatenation
            outfile.write("\n")
    print("> Neighbors end writing")

    print("> Values start writing")
    with open(init_file_path_input, 'r') as f_read:
        tmp_values = f_read.readlines()
    print("> Values read done")
    with open(init_file_path_output, 'a') as file:
        file.writelines(tmp_values[1:])
    print("> Values end writing")
    return True

###########################################
# 8. Analysis Utilities
###########################################
"""
Additional utility functions for analysis tasks:
- Step calculation
- Expected state identification
"""

def calculate_expected_step(input_npy_data, expected_grain_num=200):
    """Find timestep where grain count matches expected number.
    
    Args:
        input_npy_data (list): List of .npy files with microstructure evolution data
        expected_grain_num (int): Target number of grains
        
    Returns:
        tuple: (special_step_distribution, microstructure_list) - Timesteps and corresponding microstructures
    """
    # calculate the timestep for expected_grain_num in npy dataset

    num_input = len(input_npy_data)
    special_step_distribution = np.zeros(num_input)
    microstructure_list = []

    for input_i in range(num_input):
        npy_data = np.load(input_npy_data[input_i])
        step_num = npy_data.shape[0]
        grain_num_list = np.zeros(step_num)
        for i in tqdm(range(step_num)):
            grain_num_list[i] = len(np.unique(npy_data[i,:].flatten()))
        special_step_distribution[input_i] = int(np.argmin(abs(grain_num_list - expected_grain_num)))
        microstructure_list.append(npy_data[int(special_step_distribution[input_i]),:])
    print("> Step calculation done")

    return special_step_distribution, microstructure_list

if __name__ == '__main__':
    path  = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_poly/"
    file_name = ["p_ori_ave_aveE_100_20k_multiCore64_delta0.8_m2_J1_refer_1_0_0_seed56689_kt066"]
    figure_path = file_name
    num_grain = 20000
    output_name = ["ave_02_distribution"]

    for i in range(len(file_name)):
        if not os.path.exists("results/"+figure_path[i]+".npy"):
            timestep, grain_structure_figure = dump2img(path+file_name[i], 81)
            np.save("results/"+figure_path[i],grain_structure_figure)
        else:
            grain_structure_figure = np.load(path + "results/"+figure_path[i]+".npy")
            timestep = 1.0 * np.array(range(len(grain_structure_figure)))

        if not os.path.exists("results/"+figure_path[i]): os.mkdir("results/"+figure_path[i])
        slope_list = np.zeros(len(timestep))
        for step in range(len(timestep)):
            newplace = np.rot90(grain_structure_figure[step,:,:,:], 1, (0,1))
            P, sites = get_normal_vector_3d(newplace)
            if len(sites) == 0: continue
            slope_list[step] = plot_normal_vector_distribution_3d(P, sites, step, figure_path[i]+"/"+figure_path[i])

        os.system(f'ffmpeg -framerate 30 -i results/{figure_path[i]}/{figure_path[i]}_step.%04d.png \
                    -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p \
                    results/{figure_path[i]}/{output_name[i]}.mp4')
        plt.clf()
        fig = plt.subplots()
        # plt.bar(xCor,freqArray,width=binValue*0.7)
        plt.plot(timestep, slope_list*1e4,'-o',linewidth=2,label='slope of normal distribution')
        plt.xlabel("Timestep")
        plt.ylabel("Slope/1e-4")
        plt.ylim([-20, 20])
        plt.legend()
        plt.savefig(f'results/normal_distribution_slope/{figure_path[i]}_slope.png',dpi=400,bbox_inches='tight')
