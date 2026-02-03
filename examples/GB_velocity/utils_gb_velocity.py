#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grain Boundary Velocity Analysis Utilities
==========================================

This module provides shared utility functions for grain boundary velocity
and anti-curvature analysis. These functions are used across multiple
analysis notebooks in the GB_velocity directory.

The functions support both 2D and 3D grain boundary analysis with various
anisotropy types (isotropic, misorientation-only, inclination-only, full).

Key Functions:
- compute_dV: Calculate net volume change for grain boundaries
- compute_dV_split: Calculate volume change with directional tracking
- compute_necessary_info_split: Complete velocity-curvature analysis
- Get_GB_movement_information: Extract GB movement data between timesteps

Usage:
    # In a notebook cell:
    import sys
    sys.path.append('.')  # or appropriate path
    from utils_gb_velocity import compute_dV, compute_dV_split

Note: These functions require numba for JIT compilation. Install with:
    pip install numba

Created for VECTOR project grain boundary analysis.
@author: Lin
"""

import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorators when numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@njit(parallel=True)
def compute_dV(npy_file_aniso_current, npy_file_aniso_next, pair_id_pair):
    """
    Calculate net volume change for a grain boundary between two timesteps.

    This function computes how many sites switched from one grain to another
    across a grain boundary, providing a measure of boundary velocity.

    Parameters:
    -----------
    npy_file_aniso_current : ndarray
        Current timestep microstructure data (2D or 3D grain ID array)
    npy_file_aniso_next : ndarray
        Next timestep microstructure data (same shape as current)
    pair_id_pair : tuple
        Pair of grain IDs (grain_id_1, grain_id_2) defining the boundary

    Returns:
    --------
    int
        Net volume change (positive = grain_id_1 grew, negative = grain_id_2 grew)

    Notes:
    ------
    - Uses numba JIT compilation for performance
    - Parallel execution across array elements
    - Volume change is symmetric: dV(A,B) = -dV(B,A)
    """
    size_x, size_y = npy_file_aniso_current.shape[:2]
    is_3d = len(npy_file_aniso_current.shape) == 3

    grain_id_1, grain_id_2 = pair_id_pair
    dV = 0

    if is_3d:
        size_z = npy_file_aniso_current.shape[2]
        for i in prange(size_x):
            for j in range(size_y):
                for k in range(size_z):
                    current_id = npy_file_aniso_current[i, j, k]
                    next_id = npy_file_aniso_next[i, j, k]

                    # Site switched from grain_id_2 to grain_id_1
                    if current_id == grain_id_2 and next_id == grain_id_1:
                        dV += 1
                    # Site switched from grain_id_1 to grain_id_2
                    elif current_id == grain_id_1 and next_id == grain_id_2:
                        dV -= 1
    else:
        for i in prange(size_x):
            for j in range(size_y):
                current_id = npy_file_aniso_current[i, j]
                next_id = npy_file_aniso_next[i, j]

                # Site switched from grain_id_2 to grain_id_1
                if current_id == grain_id_2 and next_id == grain_id_1:
                    dV += 1
                # Site switched from grain_id_1 to grain_id_2
                elif current_id == grain_id_1 and next_id == grain_id_2:
                    dV -= 1

    return dV


@njit(parallel=True)
def compute_dV_split(npy_file_aniso_current, npy_file_aniso_next, pair_id_pair):
    """
    Calculate volume change with directional tracking for a grain boundary.

    Unlike compute_dV which returns net change, this function separately
    tracks growth in each direction, enabling analysis of boundary motion
    asymmetry.

    Parameters:
    -----------
    npy_file_aniso_current : ndarray
        Current timestep microstructure data
    npy_file_aniso_next : ndarray
        Next timestep microstructure data
    pair_id_pair : tuple
        Pair of grain IDs (grain_id_1, grain_id_2)

    Returns:
    --------
    tuple (int, int)
        (dV_positive, dV_negative) where:
        - dV_positive: sites that switched from grain_id_2 to grain_id_1
        - dV_negative: sites that switched from grain_id_1 to grain_id_2

    Notes:
    ------
    - Net change = dV_positive - dV_negative
    - Useful for detecting asymmetric boundary motion
    """
    size_x, size_y = npy_file_aniso_current.shape[:2]
    is_3d = len(npy_file_aniso_current.shape) == 3

    grain_id_1, grain_id_2 = pair_id_pair
    dV_positive = 0
    dV_negative = 0

    if is_3d:
        size_z = npy_file_aniso_current.shape[2]
        for i in prange(size_x):
            for j in range(size_y):
                for k in range(size_z):
                    current_id = npy_file_aniso_current[i, j, k]
                    next_id = npy_file_aniso_next[i, j, k]

                    if current_id == grain_id_2 and next_id == grain_id_1:
                        dV_positive += 1
                    elif current_id == grain_id_1 and next_id == grain_id_2:
                        dV_negative += 1
    else:
        for i in prange(size_x):
            for j in range(size_y):
                current_id = npy_file_aniso_current[i, j]
                next_id = npy_file_aniso_next[i, j]

                if current_id == grain_id_2 and next_id == grain_id_1:
                    dV_positive += 1
                elif current_id == grain_id_1 and next_id == grain_id_2:
                    dV_negative += 1

    return dV_positive, dV_negative


def compute_necessary_info_split(key, time_interval, GB_info, energy_info,
                                  npy_file_aniso_current, npy_file_aniso_next):
    """
    Compute complete velocity-curvature analysis for a single grain boundary.

    This is the main analysis function that combines volume change calculation
    with grain boundary information to produce velocity and curvature data.

    Parameters:
    -----------
    key : tuple
        Grain boundary identifier (grain_id_1, grain_id_2)
    time_interval : int
        Time step interval for velocity calculation
    GB_info : dict
        Dictionary containing grain boundary information including:
        - 'curvature': signed curvature value
        - 'area': boundary area/length
        - Other boundary properties
    energy_info : dict
        Dictionary containing energy information for the boundary
    npy_file_aniso_current : ndarray
        Current timestep microstructure data
    npy_file_aniso_next : ndarray
        Next timestep microstructure data

    Returns:
    --------
    dict or None
        Dictionary containing:
        - 'velocity': boundary velocity (area change / time / boundary length)
        - 'curvature': signed curvature
        - 'energy': grain boundary energy
        - 'area': boundary area/length
        - 'dV_split': tuple of directional volume changes
        - 'is_anti_curvature': boolean flag
        Returns None if boundary has insufficient data

    Notes:
    ------
    - Anti-curvature is detected when velocity * curvature < 0
    - Boundaries with zero area or curvature may be filtered out
    """
    if key not in GB_info or key not in energy_info:
        return None

    gb_data = GB_info[key]
    energy_data = energy_info[key]

    # Extract curvature and area
    curvature = gb_data.get('curvature', 0)
    area = gb_data.get('area', 0)

    if area == 0:
        return None

    # Compute volume change
    dV_positive, dV_negative = compute_dV_split(
        npy_file_aniso_current, npy_file_aniso_next, key
    )
    dV = dV_positive - dV_negative

    # Calculate velocity (area change per unit time per unit boundary length)
    velocity = dV / time_interval / area

    # Get energy
    energy = energy_data.get('energy', 1.0)

    # Detect anti-curvature behavior
    is_anti_curvature = (velocity * curvature) < 0

    return {
        'velocity': velocity,
        'curvature': curvature,
        'energy': energy,
        'area': area,
        'dV_split': (dV_positive, dV_negative),
        'is_anti_curvature': is_anti_curvature
    }


def Get_GB_movement_information(GB_info_current, GB_info_next,
                                 npy_file_current, npy_file_next,
                                 time_interval=30):
    """
    Extract grain boundary movement information between two timesteps.

    Processes all grain boundaries present in both timesteps and calculates
    their velocity and curvature properties.

    Parameters:
    -----------
    GB_info_current : dict
        Grain boundary information dictionary for current timestep
    GB_info_next : dict
        Grain boundary information dictionary for next timestep
    npy_file_current : ndarray
        Current timestep microstructure data
    npy_file_next : ndarray
        Next timestep microstructure data
    time_interval : int, optional
        Time step interval (default: 30)

    Returns:
    --------
    dict
        Dictionary mapping GB keys to movement information:
        - Each entry contains velocity, curvature, volume change data

    Notes:
    ------
    - Only processes GBs that exist in both timesteps
    - Useful for tracking individual GB evolution
    """
    results = {}

    # Find GBs present in both timesteps
    common_keys = set(GB_info_current.keys()) & set(GB_info_next.keys())

    for key in common_keys:
        current_data = GB_info_current[key]
        next_data = GB_info_next[key]

        # Calculate volume change
        dV = compute_dV(npy_file_current, npy_file_next, key)

        # Get curvatures
        curvature_current = current_data.get('curvature', 0)
        curvature_next = next_data.get('curvature', 0)

        # Get areas
        area_current = current_data.get('area', 0)
        area_next = next_data.get('area', 0)

        if area_current > 0:
            velocity = dV / time_interval / area_current
        else:
            velocity = 0

        results[key] = {
            'dV': dV,
            'velocity': velocity,
            'curvature_current': curvature_current,
            'curvature_next': curvature_next,
            'area_current': area_current,
            'area_next': area_next
        }

    return results


def filter_anti_curvature_events(gb_movement_data, curvature_threshold=0.0182,
                                  area_threshold=100):
    """
    Filter grain boundary data to identify significant anti-curvature events.

    Applies quality thresholds to remove noise and focus on physically
    meaningful anti-curvature behavior.

    Parameters:
    -----------
    gb_movement_data : dict
        Dictionary of GB movement data from Get_GB_movement_information
    curvature_threshold : float, optional
        Minimum absolute curvature to consider (default: 0.0182)
    area_threshold : int, optional
        Minimum boundary area to consider (default: 100)

    Returns:
    --------
    dict
        Filtered dictionary containing only GBs that:
        - Have sufficient area
        - Have sufficient curvature
        - Show anti-curvature behavior (velocity * curvature < 0)
    """
    filtered = {}

    for key, data in gb_movement_data.items():
        curvature = data.get('curvature_current', 0)
        area = data.get('area_current', 0)
        velocity = data.get('velocity', 0)

        # Apply thresholds
        if abs(curvature) < curvature_threshold:
            continue
        if area < area_threshold:
            continue

        # Check for anti-curvature
        if velocity * curvature < 0:
            filtered[key] = data

    return filtered


def calculate_anti_curvature_fraction(gb_data_list, curvature_threshold=0.0182):
    """
    Calculate the fraction of grain boundaries showing anti-curvature behavior.

    Parameters:
    -----------
    gb_data_list : list
        List of GB data dictionaries, each containing 'velocity' and 'curvature'
    curvature_threshold : float, optional
        Minimum absolute curvature to include in calculation

    Returns:
    --------
    float
        Fraction of GBs with anti-curvature behavior (0.0 to 1.0)

    Notes:
    ------
    - Only considers GBs above the curvature threshold
    - Anti-curvature defined as velocity * curvature < 0
    """
    valid_gbs = [gb for gb in gb_data_list
                 if abs(gb.get('curvature', 0)) >= curvature_threshold]

    if not valid_gbs:
        return 0.0

    anti_curvature_count = sum(
        1 for gb in valid_gbs
        if gb.get('velocity', 0) * gb.get('curvature', 0) < 0
    )

    return anti_curvature_count / len(valid_gbs)


# Energy function utilities

def cosine_energy_function(misorientation_angle, inclination_angle, f_param, t_param):
    """
    Calculate grain boundary energy using cosine energy function.

    Parameters:
    -----------
    misorientation_angle : float
        Misorientation angle between grains (radians)
    inclination_angle : float
        Inclination angle of boundary plane (radians)
    f_param : float
        Misorientation energy weight (typically 0.0 to 1.0)
    t_param : float
        Inclination energy weight (typically 0.0 to 1.0)

    Returns:
    --------
    float
        Grain boundary energy value
    """
    m_term = f_param * (1 - np.cos(4 * misorientation_angle))
    i_term = t_param * (1 - np.cos(4 * inclination_angle))
    return 1 + m_term + i_term


def well_energy_function(misorientation_angle, inclination_angle, f_param, t_param):
    """
    Calculate grain boundary energy using well energy function.

    The well energy function creates discrete energy wells at specific
    crystallographic configurations, producing sharper energy minima
    compared to the cosine function.

    Parameters:
    -----------
    misorientation_angle : float
        Misorientation angle between grains (radians)
    inclination_angle : float
        Inclination angle of boundary plane (radians)
    f_param : float
        Misorientation energy weight
    t_param : float
        Inclination energy weight

    Returns:
    --------
    float
        Grain boundary energy value
    """
    # Well function implementation
    # Note: The exact form may vary - this is a common formulation
    m_well = f_param * np.abs(np.sin(2 * misorientation_angle))
    i_well = t_param * np.abs(np.sin(2 * inclination_angle))
    return 1 + m_well + i_well


if __name__ == '__main__':
    # Simple test to verify imports work
    print("GB Velocity utilities loaded successfully")
    print(f"Numba available: {NUMBA_AVAILABLE}")

    # Test basic function with small arrays
    test_current = np.array([[1, 1, 2], [1, 2, 2], [2, 2, 2]])
    test_next = np.array([[1, 1, 1], [1, 1, 2], [2, 2, 2]])

    dV = compute_dV(test_current, test_next, (1, 2))
    print(f"Test compute_dV: {dV} (expected: 2)")

    dV_pos, dV_neg = compute_dV_split(test_current, test_next, (1, 2))
    print(f"Test compute_dV_split: +{dV_pos}, -{dV_neg}")
