#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crystallographic Utility Functions for Quaternion and Symmetry Operations

This module provides quaternion-based crystallographic orientation utilities
for misorientation calculations in polycrystalline materials analysis.

Functions:
- euler2quaternion: Convert Euler angles to quaternion representation
- symquat: Generate quaternion from crystallographic symmetry matrix
- quat_Multi: Quaternion multiplication operation
- quaternions: Calculate misorientation between two orientations with symmetry

Author: Lin Yang
"""

import numpy as np
import math


def euler2quaternion(yaw, pitch, roll):
    """
    Convert Euler angles to quaternion representation.

    Parameters:
    -----------
    yaw : float
        First Euler angle (phi1) in radians
    pitch : float
        Second Euler angle (Phi) in radians
    roll : float
        Third Euler angle (phi2) in radians

    Returns:
    --------
    list : Quaternion [qx, qy, qz, qw]
    """
    qx = np.cos(pitch/2.)*np.cos((yaw+roll)/2.)
    qy = np.sin(pitch/2.)*np.cos((yaw-roll)/2.)
    qz = np.sin(pitch/2.)*np.sin((yaw-roll)/2.)
    qw = np.cos(pitch/2.)*np.sin((yaw+roll)/2.)

    return [qx, qy, qz, qw]


def symquat(index, Osym=24):
    """
    Convert crystallographic symmetry matrix to quaternion representation.

    Parameters:
    -----------
    index : int
        Index of symmetry operation (0 to Osym-1)
    Osym : int, optional
        Number of symmetry operations (24 for cubic, 12 for hexagonal)

    Returns:
    --------
    np.ndarray : Quaternion [q0, q1, q2, q3]
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
    """
    Multiply two quaternions.

    Parameters:
    -----------
    q1, q2 : array_like
        Quaternions [q0, q1, q2, q3]

    Returns:
    --------
    np.ndarray : Product quaternion
    """
    q = np.zeros(4)
    q[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    q[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    q[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    q[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]

    return q


def quaternions(q1, q2, symm2quat_matrix, Osym=24):
    """
    Calculate the misorientation between two quaternion orientations.

    Accounts for crystallographic symmetry by checking all symmetry-equivalent
    orientations and returning the minimum misorientation angle.

    Parameters:
    -----------
    q1, q2 : array_like
        Quaternions representing two crystallographic orientations
    symm2quat_matrix : np.ndarray
        Pre-computed symmetry quaternion matrix (Osym x 4)
    Osym : int, optional
        Number of symmetry operations (24 for cubic)

    Returns:
    --------
    tuple : (misorientation_angle, rotation_axis)
        - misorientation_angle: Minimum angle in radians
        - rotation_axis: Unit vector defining rotation axis
    """
    q = np.zeros(4)
    misom = 2*np.pi

    for i in range(0, Osym):
        for j in range(0, Osym):
            q1b = quat_Multi(symm2quat_matrix[i], q1)
            q2b = quat_Multi(symm2quat_matrix[j], q2)

            q2b[1] = -q2b[1]
            q2b[2] = -q2b[2]
            q2b[3] = -q2b[3]

            q = quat_Multi(q1b, q2b)
            miso0 = 2*math.acos(round(q[0], 5))

            if miso0 > np.pi:
                miso0 = miso0 - 2*np.pi
            if abs(miso0) < misom:
                misom = abs(miso0)
                qmin = q.copy()

    miso0 = 2*math.acos(round(qmin[0], 5))
    if miso0 > np.pi:
        miso0 = miso0 - 2*np.pi

    if math.sin(miso0/2):
        axis = qmin[1:]/math.sin(miso0/2)
    else:
        axis = np.array([0, 0, 1])

    return abs(miso0), axis


def pre_operation_misorientation(grainNum, init_filename, Osym=24):
    """
    Initialize quaternion matrices for misorientation calculations.

    Parameters:
    -----------
    grainNum : int
        Number of grains in the microstructure
    init_filename : str
        Path to initialization file with Euler angles
    Osym : int, optional
        Number of symmetry operations

    Returns:
    --------
    tuple : (symm2quat_matrix, quartAngle)
        - symm2quat_matrix: Pre-computed symmetry quaternions (Osym x 4)
        - quartAngle: Grain orientations as quaternions (grainNum x 4)
    """
    # Create the matrix to store quaternion angle
    quartAngle = np.ones((grainNum, 4))*-2

    # Create a quaternion matrix to show symmetry
    symm2quat_matrix = np.zeros((Osym, 4))
    for i in range(0, Osym):
        symm2quat_matrix[i, :] = symquat(i, Osym)

    # Read the input Euler angle from *.init
    with open(init_filename, 'r', encoding='utf-8') as f:
        for line in f:
            eachline = line.split()

            if len(eachline) == 5 and eachline[0] != '#':
                lineN = int(eachline[1])-1
                if quartAngle[lineN, 0] == -2:
                    quartAngle[lineN, :] = euler2quaternion(
                        float(eachline[2]), float(eachline[3]), float(eachline[4]))

    return symm2quat_matrix, quartAngle


def multiP_calM(i, quartAngle, symm2quat_matrix, Osym):
    """
    Calculate misorientation between grain pair for multiprocessing.

    Parameters:
    -----------
    i : list
        Grain ID pair [grain_i, grain_j]
    quartAngle : np.ndarray
        Quaternion orientations for all grains
    symm2quat_matrix : np.ndarray
        Symmetry quaternion matrix
    Osym : int
        Number of symmetry operations

    Returns:
    --------
    np.ndarray : [misorientation_angle, axis_x, axis_y, axis_z]
    """
    qi = quartAngle[i[0], :]
    qj = quartAngle[i[1], :]

    theta, axis = quaternions(qi, qj, symm2quat_matrix, Osym)
    gamma = theta
    return np.insert(axis, 0, gamma)
