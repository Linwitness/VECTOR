#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared test utilities for smoothing algorithm verification.
Contains error calculation functions used across all algorithm test suites.

Author: Lin Yang
"""

import numpy as np
import myInput


def calculate_normal_vector_error_2d(P, R, gb_sites):
    """Calculate error between computed and theoretical normal vectors (2D).

    For each grain boundary site, computes the angle between the calculated
    normal vector and the theoretical value from R.

    Args:
        P: Phase field results containing calculated normal vectors
        R: Reference normal vectors
        gb_sites: List of grain boundary sites as (i, j) tuples

    Returns:
        float: RMS angle error in radians
        float: Maximum angle error in radians
    """
    angles = []
    for i, j in gb_sites:
        dx, dy = myInput.get_grad(P, i, j)
        calc_vec = np.array([dx, dy])
        ref_vec = np.array([R[i,j,0], R[i,j,1]])

        # Normalize vectors
        calc_vec = calc_vec / np.linalg.norm(calc_vec)
        ref_vec = ref_vec / np.linalg.norm(ref_vec)

        # Calculate angle, accounting for vector direction ambiguity
        dot_product = np.clip(np.abs(np.dot(calc_vec, ref_vec)), -1.0, 1.0)
        angle = np.arccos(dot_product)
        angles.append(angle)

    angles = np.array(angles)
    rms_error = np.sqrt(np.mean(angles**2))
    max_error = np.max(angles)

    return rms_error, max_error


def calculate_normal_vector_error_3d(P, R, gb_sites):
    """Calculate error between computed and theoretical normal vectors (3D).

    For each grain boundary site, computes the angle between the calculated
    normal vector and the theoretical value from R.

    Args:
        P: Phase field results containing calculated normal vectors
        R: Reference normal vectors
        gb_sites: List of grain boundary sites as (i, j, k) tuples

    Returns:
        float: RMS angle error in radians
        float: Maximum angle error in radians
    """
    angles = []
    for i, j, k in gb_sites:
        dx, dy, dz = myInput.get_grad3d(P, i, j, k)
        calc_vec = np.array([dx, dy, dz])
        ref_vec = np.array([R[i,j,k,0], R[i,j,k,1], R[i,j,k,2]])

        # Normalize vectors
        calc_vec = calc_vec / np.linalg.norm(calc_vec)
        ref_vec = ref_vec / np.linalg.norm(ref_vec)

        # Calculate angle, accounting for vector direction ambiguity
        dot_product = np.clip(np.abs(np.dot(calc_vec, ref_vec)), -1.0, 1.0)
        angle = np.arccos(dot_product)
        angles.append(angle)

    angles = np.array(angles)
    rms_error = np.sqrt(np.mean(angles**2))
    max_error = np.max(angles)

    return rms_error, max_error


def calculate_curvature_error_2d(C, radius, gb_sites):
    """Calculate error between computed and theoretical curvature (2D).

    For a circle, theoretical curvature is 1/R everywhere on the boundary.

    Args:
        C: Calculated curvature field
        radius: Circle radius
        gb_sites: List of grain boundary sites as (i, j) tuples

    Returns:
        float: RMS curvature error
        float: Maximum curvature error
        float: Average curvature
        float: Standard deviation of curvature
    """
    theoretical = 1.0/radius
    curvatures = []
    for i, j in gb_sites:
        curvatures.append(C[1,i,j])

    curvatures = np.array(curvatures)
    errors = np.abs(curvatures - theoretical)

    rms_error = np.sqrt(np.mean(errors**2))
    max_error = np.max(errors)
    avg_curvature = np.mean(curvatures)
    std_curvature = np.std(curvatures)

    return rms_error, max_error, avg_curvature, std_curvature


def calculate_curvature_error_3d(C, radius, gb_sites):
    """Calculate error between computed and theoretical curvature (3D).

    For a sphere, theoretical mean curvature is 1/R everywhere on the boundary.

    Args:
        C: Calculated curvature field
        radius: Sphere radius
        gb_sites: List of grain boundary sites as (i, j, k) tuples

    Returns:
        float: RMS curvature error
        float: Maximum curvature error
        float: Average curvature
        float: Standard deviation of curvature
    """
    theoretical = 1.0/radius
    curvatures = []
    for i, j, k in gb_sites:
        curvatures.append(C[1,i,j,k])

    curvatures = np.array(curvatures)
    errors = np.abs(curvatures - theoretical)

    rms_error = np.sqrt(np.mean(errors**2))
    max_error = np.max(errors)
    avg_curvature = np.mean(curvatures)
    std_curvature = np.std(curvatures)

    return rms_error, max_error, avg_curvature, std_curvature


def get_curvature_statistics(C, gb_sites, is_3d=False):
    """Get curvature statistics for Voronoi/non-analytical cases.

    Args:
        C: Calculated curvature field
        gb_sites: List of grain boundary sites
        is_3d: Whether this is a 3D case

    Returns:
        dict: Dictionary with mean, std, min, max curvature values
    """
    curvatures = []
    if is_3d:
        for i, j, k in gb_sites:
            curvatures.append(C[1,i,j,k])
    else:
        for i, j in gb_sites:
            curvatures.append(C[1,i,j])

    curvatures = np.array(curvatures)

    return {
        'mean': np.mean(curvatures),
        'std': np.std(curvatures),
        'min': np.min(curvatures),
        'max': np.max(curvatures)
    }
