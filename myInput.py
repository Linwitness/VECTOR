#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input utilities for VECTOR (VoxEl-based boundary inClination smooThing AlgORithms)
This module provides functions for creating and managing input data for VECTOR simulations.
Author: Lin Yang
"""

import os
current_path = os.getcwd()
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py

###########################################
# 1. Input File Processing
###########################################
"""
Functions for reading and processing input files from various formats:
- SPPARKS .init files
- Dream3D files
- Custom formats
"""

def init2IC(nx, ny, ng, filename, filepath=current_path+"/input/"):
    """Convert 2D SPPARKS .init file to internal format
    
    Args:
        nx, ny: Grid dimensions
        ng: Number of grains
        filename: Name of .init file
        filepath: Path to input directory
        
    Returns:
        tuple: (grain structure array, reference array)
    """
    R = np.zeros((nx, ny, 2))

    with open(filepath+filename, 'r') as file:
        beginNum = 3
        fig = []

        while beginNum >= 0:
            line = file.readline()
            beginNum -= 1

        if line[0] != '1':
            print("Please change beginning line! " + line)

        while line:
            eachline = line.split()
            fig.append([int(eachline[1])])
            line = file.readline()

    fig = np.array(fig)
    fig = fig.reshape(nx, ny)
    fig = np.flipud(fig)
    fig = fig[:, :, None]

    return fig, R

def init2IC3d(nx, ny, nz, ng, filename, dream3d=False, filepath=current_path+"/input/"):
    """Convert 3D SPPARKS/Dream3D .init file to internal format
    
    Args:
        nx, ny, nz: Grid dimensions
        ng: Number of grains
        filename: Name of .init file
        dream3d: Whether file is in Dream3D format
        filepath: Path to input directory
        
    Returns:
        tuple: (3D grain structure array, reference array)
    """
    R = np.zeros((nx, ny, nz, 3))

    with open(filepath+filename, 'r') as file:
        beginNum = 3
        fig = np.zeros((nx*ny*nz))

        while beginNum >= 0:
            line = file.readline()
            beginNum -= 1

        if line[0] != '1':
            print("Please change beginning line! " + line)

        while line:
            eachline = line.split()
            fig[int(eachline[0])-1] = int(eachline[1])
            line = file.readline()

    fig = fig.reshape(nz, nx, ny)
    fig = fig.transpose((1, 2, 0))
    if dream3d:
        pass
    else:
        fig = np.flipud(fig)
    fig = fig[:, :, :, None]
    return fig, R

###########################################
# 2. 2D Initial Condition Generators 
###########################################
"""
Functions for generating various 2D initial microstructures:
- Geometric shapes (circles, etc.)
- Voronoi structures
- Complex patterns
"""

def Circle_IC(nx, ny, r=50):
    """Generate 2D circular grain initial condition
    
    Args:
        nx, ny: Grid dimensions
        r: Circle radius (default=50)
        
    Returns:
        tuple: (grain structure array, reference array)
    """
    ng = 2
    P = np.zeros((nx, ny, ng))
    R = np.zeros((nx, ny, 3))

    for i in range(0, nx):
        for j in range(0, ny):
            radius = math.sqrt((j-ny/2)**2+(i-nx/2)**2)
            if radius < r:
                P[i, j, 0] = 1.
                if radius != 0:
                    R[i, j, 0] = (j-ny/2)/radius
                    R[i, j, 1] = (i-nx/2)/radius
                    R[i, j, 2] = 1/radius
            else:
                P[i, j, 1] = 1.
                if radius != 0:
                    R[i, j, 0] = (j-ny/2)/radius
                    R[i, j, 1] = (i-nx/2)/radius
                    R[i, j, 2] = 1/radius

    return P, R

def QuarterCircle_IC(nx, ny):
    """Generate 2D quarter-circle grain initial condition
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        tuple: (grain structure array, reference array)
    """
    ng = 2
    P = np.zeros((nx, ny, ng))
    R = np.zeros((nx, ny, 2))

    for i in range(0, nx):
        for j in range(0, ny):
            radius = math.sqrt((j-ny/2)**2+(i-nx/2)**2)
            if radius < 40 and i < nx/2 and j < ny/2:
                P[i, j, 0] = 1.
                if radius != 0:
                    R[i, j, 0] = (j-ny/2)/radius
                    R[i, j, 1] = (i-nx/2)/radius
            else:
                P[i, j, 1] = 1.
                if radius != 0:
                    R[i, j, 0] = (j-ny/2)/radius
                    R[i, j, 1] = (i-nx/2)/radius

    return P, R

def Voronoi_IC(nx, ny, ng):
    """Generate 2D Voronoi tessellation initial condition
    
    Args:
        nx, ny: Grid dimensions
        ng: Number of grains
        
    Returns:
        tuple: (grain structure array, reference array)
    """
    P = np.zeros((nx, ny, ng))
    R = np.zeros((nx, ny, 3))

    GCoords = np.array([[ 36., 132.],
                        [116.,  64.],
                        [ 43.,  90.],
                        [128., 175.],
                        [194.,  60.]])

    # (400,400,5)
    # GCoords = np.array([[ 69., 321.],
    #                     [298., 134.],
    #                     [174., 138.],
    #                     [294., 392.],
    #                     [ 69., 324.]])

    # (100,100,5)
    # GCoords = np.array([[20., 95.],
    #                     [27., 61.],
    #                     [37., 93.],
    #                     [65., 18.],
    #                     [25., 17.]])

    # (50,50,5)
    # GCoords = np.array([[ 0., 35.],
    #                     [43., 36.],
    #                     [43.,  9.],
    #                     [38., 37.],
    #                     [28., 36.]])


    for i in range(0, nx):
        for j in range(0, ny):
            MinDist = math.sqrt((GCoords[0,1]-j)**2+(GCoords[0,0]-i)**2)
            GG = 0
            for G in range(1, ng):
                dist = math.sqrt((GCoords[G, 1]-j)**2+(GCoords[G, 0]-i)**2)
                if dist < MinDist:
                    GG = G
                    MinDist = dist
            P[i, j, GG] = 1.

    for i in range(0, nx):
        for j in range(0, ny):
            if i > 0 and i <= 93 and j <= 120 and j > 104:
                R[i, j, 0] = -10.0/math.sqrt(101)
                R[i, j, 1] = 1.0/math.sqrt(101)
            elif i == 0 and j == ny-1:
                R[i, j, 0] = -math.sqrt(0.5)
                R[i, j, 1] = math.sqrt(0.5)
            elif i == 0:
                R[i, j, 0] = 0
                R[i, j, 1] = 1
            elif j == ny-1:
                R[i, j, 0] = 1
                R[i, j, 1] = 0
            elif j >= 123 and j < ny-1 and i <= 96 and i >= 60:
                R[i, j, 0] = -9.0/math.sqrt(442)
                R[i, j, 1] = -19.0/math.sqrt(442)
            elif (i == 94 or i == 95 or i == 96) and (j == 121 or j == 122):
                R[i, j, 0] = -2.0/math.sqrt(5)
                R[i, j, 1] = 1.0/math.sqrt(5)

    # (400,400,5)
    # for i in range(0,nx):
    #     for j in range(0,ny):
    #         if i==0 and j==322:
    #             R[i,j,0] = -math.sqrt(0.5)
    #             R[i,j,1] = math.sqrt(0.5)
    #         elif i==0 and j==160:
    #             R[i,j,0] = -7.0/math.sqrt(74)
    #             R[i,j,1] = -5.0/math.sqrt(74)
    #         elif i==192 and j==322:
    #             R[i,j,0] = math.sqrt(0.5)
    #             R[i,j,1] = math.sqrt(0.5)
    #         elif i==206 and j==278:
    #             R[i,j,0] = 0
    #             R[i,j,1] = 1.0
    #         elif i==0:
    #             R[i,j,0] = 0
    #             R[i,j,1] = 1
    #         elif j==322:
    #             R[i,j,0] = 1
    #             R[i,j,1] = 0
    #         elif i>0 and i<=205 and j<=278 and j>=160:
    #             R[i,j,0] = -103.0/math.sqrt(14090)
    #             R[i,j,1] = 59.0/math.sqrt(14090)
    #         elif j>=279 and j<322 and i<=205 and i>=192:
    #             R[i,j,0] = 13.0/math.sqrt(2105)
    #             R[i,j,1] = 44.0/math.sqrt(2105)


    # (100,100,5)
    # for i in range(0,nx):
    #     for j in range(0,ny):
    #         if i==0 and j==ny-1:
    #             R[i,j,0] = math.sqrt(0.5)
    #             R[i,j,1] = -math.sqrt(0.5)
    #         elif i==0 and j==74:
    #             R[i,j,0] = -118885/math.sqrt(23777079626)
    #             R[i,j,1] = -98201/math.sqrt(23777079626)
    #         elif i==29 and j==ny-1:
    #             R[i,j,0] = 98894/math.sqrt(22966870792)
    #             R[i,j,1] = 114834/math.sqrt(22966870792)
    #         elif i==26 and j==79:
    #             R[i,j,0] = -80009/math.sqrt(13351496770)
    #             R[i,j,1] = 83367/math.sqrt(13351496770)
    #         elif i==0:
    #             R[i,j,0] = 0
    #             R[i,j,1] = 1
    #         elif j==ny-1:
    #             R[i,j,0] = 1
    #             R[i,j,1] = 0
    #         elif i>0 and i<=26 and j<=79 and j>=74:
    #             R[i,j,0] = -27.0/math.sqrt(754)
    #             R[i,j,1] = 5.0/math.sqrt(754)
    #         elif j>=79 and j<ny-1 and i<=29 and i>=26:
    #             R[i,j,0] = -3.0/math.sqrt(634)
    #             R[i,j,1] = 25.0/math.sqrt(634)

    # (50,50,5)
    # for i in range(0,nx):
    #     for j in range(0,ny):
    #         if i==0 and j==0:
    #             R[i,j,0] = -math.sqrt(0.5)
    #             R[i,j,1] = -math.sqrt(0.5)
    #         elif i==0 and j==ny-1:
    #             R[i,j,0] = math.sqrt(0.5)
    #             R[i,j,1] = -math.sqrt(0.5)
    #         elif i==13 and j==ny-1:
    #             R[i,j,0] = 99967/math.sqrt(19487370058)
    #             R[i,j,1] = 97437/math.sqrt(19487370058)
    #         elif (i==8 and j==0) or (i==9 and j==ny-1):
    #             R[i,j,0] = math.sqrt(0.5)
    #             R[i,j,1] = math.sqrt(0.5)
    #         elif i==14 and j==10:
    #             R[i,j,0] = -14218/math.sqrt(3119555693)
    #             R[i,j,1] = 54013/math.sqrt(3119555693)
    #         elif i==0:
    #             R[i,j,0] = 0
    #             R[i,j,1] = 1
    #         elif j==ny-1 or j==0:
    #             R[i,j,0] = 1
    #             R[i,j,1] = 0
    #         elif i>=8 and i<=14 and j<=10 and j>=0:
    #             R[i,j,0] = -3.0/math.sqrt(34)
    #             R[i,j,1] = 5.0/math.sqrt(34)
    #         elif j>=10 and j<=ny-1 and i<=14 and i>=13:
    #             R[i,j,0] = 1.0/math.sqrt(1522)
    #             R[i,j,1] = 39.0/math.sqrt(1522)

    return P, R

def Complex2G_IC(nx, ny, wavelength=20):
    """Generate 2D sinusoidal grain boundary initial condition
    
    Args:
        nx, ny: Grid dimensions
        wavelength: Sinusoidal wavelength (default=20)
        
    Returns:
        tuple: (grain structure array, reference array)
    """
    ng = 2
    P = np.zeros((nx, ny, ng))
    R = np.zeros((nx, ny, 2))
    A = wavelength/2

    for i in range(0, nx):
        slope = -1.0/(10*math.pi/A*math.cos(math.pi/A*(i+A/2)))
        length = math.sqrt(1 + slope**2)

        for j in range(0, ny):
            if j < ny/2 + 10*math.sin((i+A/2)*3.1415926/A):
                P[i, j, 0] = 1.
                R[i, j, 0] = slope/length
                R[i, j, 1] = 1.0/length
            else:
                P[i, j, 1] = 1.
                R[i, j, 0] = slope/length
                R[i, j, 1] = 1.0/length

    for i in range(0, nx):
        for j in range(0, ny):
            if j == 0 or j == nx-1:
                R[i, j, 0] = 1
                R[i, j, 1] = 0

    return P, R

def Abnormal_IC(nx, ny):
    """Generate 2D abnormal grain growth initial condition
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        tuple: (grain structure array, reference array)
    """
    ng = 2
    P = np.zeros((nx, ny, ng))
    R = np.zeros((nx, ny, 2))

    file = open(f"input/AG{nx}x{ny}.txt")
    lines = file.readlines()

    row = 0
    for line in lines:
        line = line.strip().split()
        for i in range(0, len(line)):
            P[row, i, 0] = float(line[i])
            P[row, i, 1] = 1-float(line[i])
        row += 1

    if nx == 200:
        R1 = np.load('npy/ACabnormal20_R.npy')
        R2 = np.load('npy/BLabnormal04_R.npy')
        R3 = np.load('npy/LSabnormal01_R.npy')
        R4 = np.load('npy/VTabnormal03_R.npy')

        m = 0
        for i in range(0, nx):
            for j in range(0, ny):
                if R4[i, j, 1]*R1[i, j, 1]+R4[i, j, 0]*R1[i, j, 0] < -0.7:
                    R4[i, j, 0] = -R4[i, j, 0]
                    R4[i, j, 1] = -R4[i, j, 0]
                    m += 1

        for i in range(0, nx):
            for j in range(0, ny):
                R[i, j, 0] = (R1[i, j, 0] + R2[i, j, 0] + R3[i, j, 0] + R4[i, j, 0])/4
                R[i, j, 1] = (R1[i, j, 1] + R2[i, j, 1] + R3[i, j, 1] + R4[i, j, 1])/4
                length = math.sqrt(R[i, j, 0]**2+R[i, j, 1]**2)
                if length == 0:
                    R[i, j, 0] = 0
                    R[i, j, 1] = 0
                else:
                    R[i, j, 0] = R[i, j, 0]/length
                    R[i, j, 1] = R[i, j, 1]/length

    return P, R

def SmallestGrain_IC(nx, ny):
    """Generate 2D initial condition with small grains
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        tuple: (grain structure array, reference array)
    """
    ng = 2
    P = np.zeros((nx, ny, ng))

    for i in range(0, nx):
        for j in range(0, ny):
            if i == 25 and j == 10:
                P[i, j, 0] = 1
            elif i >= 50 and i <= 90 and j == 10:
                P[i, j, 0] = 1
            elif i >= 24 and i <= 25 and j >= 25 and j <= 26:
                P[i, j, 0] = 1
            elif i >= 50 and i <= 90 and j >= 25 and j <= 26:
                P[i, j, 0] = 1
            elif i >= 23 and i <= 25 and j >= 40 and j <= 42:
                P[i, j, 0] = 1
            elif i >= 50 and i <= 90 and j >= 40 and j <= 42:
                P[i, j, 0] = 1
            elif i >= 22 and i <= 25 and j >= 60 and j <= 63:
                P[i, j, 0] = 1
            elif i >= 50 and i <= 90 and j >= 60 and j <= 63:
                P[i, j, 0] = 1
            elif i >= 21 and i <= 25 and j >= 83 and j <= 87:
                P[i, j, 0] = 1
            elif i >= 50 and i <= 90 and j >= 83 and j <= 87:
                P[i, j, 0] = 1
            else:
                P[i, j, 1] = 1

    return P

###########################################
# 3. 3D Initial Condition Generators
###########################################
"""
Functions for generating various 3D initial microstructures:
- Geometric shapes (spheres, etc.)
- Voronoi structures
- Complex patterns
"""

def Circle_IC3d(nx, ny, nz, r=25):
    """Generate 3D spherical grain initial condition
    
    Args:
        nx, ny, nz: Grid dimensions
        r: Sphere radius (default=25)
        
    Returns:
        tuple: (3D grain structure array, reference array)
    """
    ng = 2
    P = np.zeros((nx, ny, nz, ng))
    R = np.zeros((nx, ny, nz, 4))

    for i in range(0, nx):
        for j in range(0, ny):
            for k in range(0, nz):
                radius = math.sqrt((j-ny/2)**2+(i-nx/2)**2+(k-nz/2)**2)
                if radius < r:
                    P[i, j, k, 0] = 1.
                    if radius != 0:
                        R[i, j, k, 0] = (j-ny/2)/radius
                        R[i, j, k, 1] = (i-nx/2)/radius
                        R[i, j, k, 2] = (k-nz/2)/radius
                        R[i, j, k, 3] = 1/radius
                else:
                    P[i, j, k, 1] = 1.
                    if radius != 0:
                        R[i, j, k, 0] = (j-ny/2)/radius
                        R[i, j, k, 1] = (i-nx/2)/radius
                        R[i, j, k, 2] = (k-nz/2)/radius
                        R[i, j, k, 3] = 1/radius

    return P, R

def Complex2G_IC3d(nx, ny, nz, wavelength=20):
    """Generate 3D sinusoidal grain boundary initial condition
    
    Args:
        nx, ny, nz: Grid dimensions
        wavelength: Sinusoidal wavelength (default=20)
        
    Returns:
        tuple: (3D grain structure array, reference array)
    """
    ng = 2
    P = np.zeros((nx, ny, nz, ng))
    R = np.zeros((nx, ny, nz, 4))
    A = wavelength/2

    for i in range(0, nx):
        for j in range(0, ny):
            dk_di = 5*math.pi/(2*A)*math.cos(math.pi/A*(0.5*i+A/2))
            dk_dj = 5*math.pi/(2*A)*math.cos(math.pi/A*(0.5*j+A/2))
            dk_didi = -5*math.pi/(2*A)*math.pi/(2*A)*math.sin(math.pi/A*(0.5*i+A/2))
            dk_djdj = -5*math.pi/(2*A)*math.pi/(2*A)*math.sin(math.pi/A*(0.5*j+A/2))
            dk_didj = 0
            
            vector_i = np.array([1, 0, dk_di])
            length_i = math.sqrt(1 + (dk_di)**2)
            vector_i = vector_i/length_i
            vector_j = np.array([0, 1, dk_dj])
            length_j = math.sqrt(1 + (dk_dj)**2)
            vector_j = vector_j/length_j

            for k in range(0, nz):
                if k < nz/2 + 5*math.sin((0.5*i+A/2)*3.1415926/A) + 5*math.sin((0.5*j+A/2)*3.1415926/A):
                    P[i, j, k, 0] = 1.
                    R[i, j, k, :3] = np.cross(vector_i, vector_j)
                    tmp_r = R[i, j, k, :3]/np.linalg.norm(R[i, j, k, :3])
                    R[i, j, k, :3] = [tmp_r[1], tmp_r[0], tmp_r[2]]
                    R[i, j, k, 3] = ((1 + (dk_di)**2) * dk_djdj - 
                                     2 * dk_di * dk_dj * dk_didj + 
                                     (1 + (dk_dj)**2) * dk_didi) /\
                                    (2 * (1 + dk_di**2 + dk_dj**2)**(1.5))
                else:
                    P[i, j, k, 1] = 1.
                    R[i, j, k, :3] = -np.cross(vector_i, vector_j)
                    tmp_r = R[i, j, k, :3]/np.linalg.norm(R[i, j, k, :3])
                    R[i, j, k, :3] = [tmp_r[1], tmp_r[0], tmp_r[2]]
                    R[i, j, k, 3] = ((1 + (dk_di)**2) * dk_djdj - 
                                     2 * dk_di * dk_dj * dk_didj + 
                                     (1 + (dk_dj)**2) * dk_didi) /\
                                    (2 * (1 + dk_di**2 + dk_dj**2)**(1.5))

    for k in [0, nz-1]:
        for i in range(0, nx):
            for j in range(0, ny):
                R[i, j, k, 2] = 2.*((k > nz/2)*1-0.5)
                R[i, j, k, 3] = 0

    return P, R

###########################################
# 4. Smoothing Matrix Generation
###########################################
"""
Functions for generating smoothing matrices used in various algorithms:
- Linear smoothing matrices
- Vector matrices
- Output formatting
"""

def output_linear_smoothing_matrix(iteration):
    """Generate 2D linear smoothing matrix for given iteration
    
    Args:
        iteration: Number of smoothing iterations
        
    Returns:
        ndarray: Smoothing matrix
    """
    matrix_length = 2*iteration+3
    matrix = np.zeros((iteration, matrix_length, matrix_length))
    matrix_unit = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
    matrix[iteration-1, iteration:iteration+3, iteration:iteration+3] = matrix_unit

    for i in range(iteration-2, -1, -1):
        for j in range(i+1, matrix_length-i-1):
            for k in range(i+1, matrix_length-i-1):
                matrix[i, j, k] += np.sum(matrix[i+1, j-1:j+2, k-1:k+2] * matrix_unit)

    return matrix[0, 1:-1, 1:-1]

def output_linear_smoothing_matrix3D(iteration):
    """Generate 3D linear smoothing matrix for given iteration
    
    Args:
        iteration: Number of smoothing iterations
        
    Returns:
        ndarray: 3D smoothing matrix
    """
    matrix_length = 2*iteration+3
    sa, sb, sc, sd = 1/8, 1/16, 1/32, 1/64
    matrix = np.zeros((iteration, matrix_length, matrix_length, matrix_length))
    matrix_unit = np.array([[[sd, sc, sd], [sc, sb, sc], [sd, sc, sd]],
                            [[sc, sb, sc], [sb, sa, sb], [sc, sb, sc]],
                            [[sd, sc, sd], [sc, sb, sc], [sd, sc, sd]]])
    matrix[iteration-1, iteration:iteration+3, iteration:iteration+3, iteration:iteration+3] = matrix_unit

    for i in range(iteration-2, -1, -1):
        for j in range(i+1, matrix_length-i-1):
            for k in range(i+1, matrix_length-i-1):
                for p in range(i+1, matrix_length-i-1):
                    matrix[i, j, k, p] += np.sum(matrix[i+1, j-1:j+2, k-1:k+2, p-1:p+2] * matrix_unit)

    return matrix[0, 1:-1, 1:-1, 1:-1]

def output_linear_vector_matrix(iteration, clip=0):
    """Generate 2D vector matrix from smoothing status
    
    Args:
        iteration: Number of iterations
        clip: Number of boundary elements to clip (default=0)
        
    Returns:
        tuple: (i-component matrix, j-component matrix)
    """
    matrix_length = 2*iteration+3
    matrix_j = np.zeros((matrix_length, matrix_length))
    matrix_i = np.zeros((matrix_length, matrix_length))
    smoothing_matrix = output_linear_smoothing_matrix(iteration)
    matrix_j[1:-1, 2:] = smoothing_matrix
    matrix_j[1:-1, 0:-2] += -smoothing_matrix
    matrix_i[2:, 1:-1] = smoothing_matrix
    matrix_i[0:-2, 1:-1] += -smoothing_matrix
    matrix_i = matrix_i[clip:matrix_length-clip, clip:matrix_length-clip]
    matrix_j = matrix_j[clip:matrix_length-clip, clip:matrix_length-clip]

    return matrix_i, matrix_j

def output_linear_vector_matrix3D(iteration, clip=0):
    """Generate 3D vector matrix from smoothing status
    
    Args:
        iteration: Number of iterations
        clip: Number of boundary elements to clip (default=0)
        
    Returns:
        tuple: (i,j,k-component matrices)
    """
    matrix_length = 2*iteration+3
    matrix_j = np.zeros((matrix_length, matrix_length, matrix_length))
    matrix_i = np.zeros((matrix_length, matrix_length, matrix_length))
    matrix_k = np.zeros((matrix_length, matrix_length, matrix_length))
    smoothing_matrix = output_linear_smoothing_matrix3D(iteration)
    matrix_j[1:-1, 2:, 1:-1] = smoothing_matrix
    matrix_j[1:-1, 0:-2, 1:-1] += -smoothing_matrix
    matrix_i[2:, 1:-1, 1:-1] = smoothing_matrix
    matrix_i[0:-2, 1:-1, 1:-1] += -smoothing_matrix
    matrix_k[1:-1, 1:-1, 2:] = smoothing_matrix
    matrix_k[1:-1, 1:-1, 0:-2] += -smoothing_matrix
    matrix_i = matrix_i[clip:matrix_length-clip, clip:matrix_length-clip, clip:matrix_length-clip]
    matrix_j = matrix_j[clip:matrix_length-clip, clip:matrix_length-clip, clip:matrix_length-clip]
    matrix_k = matrix_k[clip:matrix_length-clip, clip:matrix_length-clip, clip:matrix_length-clip]

    return matrix_i, matrix_j, matrix_k

def output_smoothed_matrix(simple_test3, linear_smoothing_matrix):
    """Apply smoothing matrix to test data in 2D
    
    Args:
        simple_test3: Input test data
        linear_smoothing_matrix: Smoothing matrix to apply
        
    Returns:
        ndarray: Smoothed matrix
    """
    edge = int(np.floor(np.shape(linear_smoothing_matrix)[0]/2))
    ilen, jlen = np.shape(simple_test3)
    smoothed_matrix3 = np.zeros((ilen, jlen))
    for i in range(edge, ilen-edge):
        for j in range(edge, jlen-edge):
            smoothed_matrix3[i, j] = np.sum(simple_test3[i-edge:i+edge+1, j-edge:j+edge+1]*linear_smoothing_matrix)

    return smoothed_matrix3

def output_smoothed_matrix3D_old(simple_test3, linear_smoothing_matrix):
    """Apply smoothing matrix to test data in 3D
    
    Args:
        simple_test3: Input test data
        linear_smoothing_matrix: Smoothing matrix to apply
        
    Returns:
        ndarray: Smoothed matrix
    """
    edge = int(np.floor(np.shape(linear_smoothing_matrix)[0]/2))
    ilen, jlen, klen = np.shape(simple_test3)
    smoothed_matrix3 = np.zeros((ilen, jlen, klen))
    for i in range(edge, ilen-edge):
        for j in range(edge, jlen-edge):
            for k in range(edge, klen-edge):
                smoothed_matrix3[i, j, k] = np.sum(simple_test3[i-edge:i+edge+1, j-edge:j+edge+1, k-edge:k+edge+1]*linear_smoothing_matrix)

    return smoothed_matrix3

def output_smoothed_matrix3D(simple_test3, linear_smoothing_matrix):
    """
    Smooth a 3D array by applying a 3D linear smoothing kernel in a vectorized fashion.
    
    The function computes the weighted sum (convolution) over each voxel's neighborhood,
    using the kernel (linear_smoothing_matrix) and places the result in the central region of
    the output array. Voxels along the boundary (which do not have a full neighborhood)
    remain zero.
    
    Args:
        simple_test3 (ndarray): 3D input array.
        linear_smoothing_matrix (ndarray): 3D smoothing kernel (assumed cubic).
        
    Returns:
        ndarray: Smoothed 3D array of the same shape as simple_test3.
    """
    # Determine the half-width (edge) of the smoothing kernel.
    edge = int(np.floor(np.shape(linear_smoothing_matrix)[0] / 2))
    
    # Extract all overlapping windows from simple_test3 with the same shape as the smoothing kernel.
    # The resulting shape is (ilen - 2*edge, jlen - 2*edge, klen - 2*edge, kernel_dim, kernel_dim, kernel_dim)
    windows = np.lib.stride_tricks.sliding_window_view(simple_test3, linear_smoothing_matrix.shape)
    
    # Compute the weighted sum over the last three dimensions for each window.
    # This is equivalent to the elementwise multiplication and sum in the original triple loop.
    smoothed_valid = np.tensordot(windows, linear_smoothing_matrix, axes=([3, 4, 5], [0, 1, 2]))
    
    # Prepare the output array and insert the smoothed values in the valid region.
    smoothed_matrix3 = np.zeros_like(simple_test3)
    smoothed_matrix3[edge:-edge, edge:-edge, edge:-edge] = smoothed_valid
    
    return smoothed_matrix3

def output_dream3d(P0, path):
    """Output data in Dream3D format
    
    Args:
        P0: Input data
        path: Output file path
    """
    if len(P0.shape) == 3:
        matrix = np.zeros(P0.shape[0:len(P0.shape)-1])
        for i in range(0, P0.shape[len(P0.shape)-1]):
            matrix += P0[:, :, i]*(i+1)
        bounds = [matrix.shape[0], matrix.shape[1], 1]
    elif len(P0.shape) == 4:
        matrix = np.zeros(P0.shape[0:len(P0.shape)-1])
        for i in range(0, P0.shape[len(P0.shape)-1]):
            matrix += P0[:, :, :, i]*(i+1)
        bounds = np.array(matrix.shape)
    else:
        print("WTF!!!")
        print(matrix.shape)
    with h5py.File(path, 'w') as f:
        f["bounds"] = bounds
        f["IC"] = matrix

###########################################
# 5. Boundary Condition Utilities
###########################################
"""
Functions for handling various boundary conditions:
- Periodic boundaries
- Repeating boundaries 
- Domain edge cases
"""

def periodic_bc(nx, ny, i, j):
    """Apply 2D periodic boundary conditions
    
    Args:
        nx, ny: Grid dimensions
        i, j: Current indices
        
    Returns:
        tuple: (ip,im,jp,jm) neighboring indices with periodic BCs
    """
    ip = i + 1
    im = i - 1
    jp = j + 1
    jm = j - 1
    if ip > nx - 1:
        ip = 0
    if im < 0:
        im = nx - 1
    if jp > ny - 1:
        jp = 0
    if jm < 0:
        jm = ny - 1
    return ip, im, jp, jm

def periodic_bc3d(nx, ny, nz, i, j, k):
    """Apply 3D periodic boundary conditions
    
    Args:
        nx, ny, nz: Grid dimensions
        i, j, k: Current indices
        
    Returns:
        tuple: (ip,im,jp,jm,kp,km) neighboring indices with periodic BCs
    """
    ip = i + 1
    im = i - 1
    jp = j + 1
    jm = j - 1
    kp = k + 1
    km = k - 1
    if ip > nx - 1:
        ip = 0
    if im < 0:
        im = nx - 1
    if jp > ny - 1:
        jp = 0
    if jm < 0:
        jm = ny - 1
    if kp > nz - 1:
        kp = 0
    if km < 0:
        km = nz - 1
    return ip, im, jp, jm, kp, km

def repeat_bc3d(nx, ny, nz, i, j, k):
    """Apply 3D repeating boundary conditions
    
    Args:
        nx, ny, nz: Grid dimensions
        i, j, k: Current indices
        
    Returns:
        tuple: (ip,im,jp,jm,kp,km) neighboring indices with repeat BCs
    """
    ip = i + 1
    im = i - 1
    jp = j + 1
    jm = j - 1
    kp = k + 1
    km = k - 1
    if ip > nx - 1:
        ip = nx
    if im < 0:
        im = 0
    if jp > ny - 1:
        jp = ny
    if jm < 0:
        jm = 0
    if kp > nz - 1:
        kp = nz
    if km < 0:
        km = 0
    return ip, im, jp, jm, kp, km

def filter_bc3d(nx, ny, nz, i, j, k, length):
    """Remove the surface voxels on boundary conditions
    
    Args:
        nx, ny, nz: Grid dimensions
        i, j, k: Current indices
        length: Length of boundary to filter
        
    Returns:
        bool: Whether the voxel is within the boundary
    """
    if i-length < 0:
        return False
    if i+length > nx-1:
        return False
    if j-length < 0:
        return False
    if j+length > ny-1:
        return False
    if k-length < 0:
        return False
    if k+length > nz-1:
        return False
    return True

def get_grad(P, i, j):
    """Calculate gradient in 2D
    
    Args:
        P: Input data
        i, j: Current indices
        
    Returns:
        tuple: (gradient in x, gradient in y)
    """
    DX = P[2, i, j]
    DY = P[1, i, j]
    H = 1.
    VecX = -H*DX
    VecY = -H*DY
    VecLen = math.sqrt(VecX**2+VecY**2)
    if VecLen == 0:
        VecScale = 1
    else:
        VecScale = H/VecLen
    return VecScale*VecX, -VecScale*VecY

def get_grad3d(P, i, j, k):
    """Calculate gradient in 3D
    
    Args:
        P: Input data
        i, j, k: Current indices
        
    Returns:
        tuple: (gradient in x, gradient in y, gradient in z)
    """
    DX = P[2, i, j, k]
    DY = P[1, i, j, k]
    DZ = P[3, i, j, k]
    H = 1.0
    VecX = -H*DX
    VecY = -H*DY
    VecZ = -H*DZ
    VecLen = math.sqrt(VecX**2+VecY**2+VecZ**2)
    if VecLen == 0:
        VecScale = 1
    else:
        VecScale = H/VecLen
    return VecScale*VecX, -VecScale*VecY, VecScale*VecZ

def split_cores(cores, sc_d=2):
    """Split cores num into two or three closed index values of two
    
    Args:
        cores: Number of cores
        sc_d: Dimension (default=2)
        
    Returns:
        tuple: Split core dimensions
    """
    sc_p = 0
    while cores != 1:
        cores = cores/2
        sc_p += 1

    sc_length = 2**(math.ceil(sc_p/sc_d))
    sc_width = 2**(math.floor(sc_p/sc_d))

    if sc_d == 3:
        sc_height = int(2**sc_p/(sc_length*sc_width))
        return sc_length, sc_width, sc_height

    return sc_length, sc_width

def split_IC(split_V, cores, dimentions=2, sic_nx_order=1, sic_ny_order=2, sic_nz_order=3):
    """Split a large matrix into several small matrix based on cores num
    
    Args:
        split_V: Input matrix
        cores: Number of cores
        dimentions: Number of dimensions (default=2)
        sic_nx_order: Order of x dimension (default=1)
        sic_ny_order: Order of y dimension (default=2)
        sic_nz_order: Order of z dimension (default=3)
        
    Returns:
        list: List of split matrices
    """
    if dimentions == 2:
        sic_lc, sic_wc = split_cores(cores)
    elif dimentions == 3:
        sic_lc, sic_wc, sic_hc = split_cores(cores, dimentions)

    new_arrayin = np.array_split(split_V, sic_wc, axis=sic_nx_order)
    new_arrayout = []
    for arrayi in new_arrayin:
        arrayi = np.array_split(arrayi, sic_lc, axis=sic_ny_order)
        if dimentions == 3:
            new_array3 = []
            for arrayj in arrayi:
                arrayj = np.array_split(arrayj, sic_hc, axis=sic_nz_order)
                new_array3.append(arrayj)
            new_arrayout.append(new_array3)
        else:
            new_arrayout.append(arrayi)

    return new_arrayout
