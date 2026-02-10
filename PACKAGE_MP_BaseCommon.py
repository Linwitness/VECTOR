#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utilities for 2D and 3D Base Classes

This module provides shared functionality for PACKAGE_MP_Base2D and PACKAGE_MP_Base3D
to reduce code duplication while maintaining dimension-specific behavior.

Author: Lin Yang
"""

import datetime


def update_core_time(instance, core_time):
    """Update the running_coreTime if the new core_time is greater.

    Args:
        instance: Base2D or Base3D instance
        core_time: Processing time from a worker core
    """
    if core_time > instance.running_coreTime:
        instance.running_coreTime = core_time


def process_result_array(instance, fval, dim):
    """Process result array from parallel workers.

    Handles both curvature (single component) and normal vector
    (2D: 2 components, 3D: 3 components) results.

    Args:
        instance: Base2D or Base3D instance with P and C arrays
        fval: Result array from worker
        dim: Dimension (2 or 3)
    """
    if dim == 2:
        if fval.shape[2] == 1:
            instance.C[1, :, :] += fval[:, :, 0]
        elif fval.shape[2] == 2:
            instance.P[1, :, :] += fval[:, :, 0]
            instance.P[2, :, :] += fval[:, :, 1]
    else:  # dim == 3
        if fval.shape[3] == 1:
            instance.C[1, :, :, :] += fval[:, :, :, 0]
        elif fval.shape[3] == 3:
            instance.P[1, :, :, :] += fval[:, :, :, 0]
            instance.P[2, :, :, :] += fval[:, :, :, 1]
            instance.P[3, :, :, :] += fval[:, :, :, 2]


def res_back_common(instance, back_result, dim, verbose=True):
    """Common callback to aggregate parallel processing results.

    Args:
        instance: Base2D or Base3D instance
        back_result: (fval, core_time) tuple from worker
        dim: Dimension (2 or 3)
        verbose: Whether to print timing info
    """
    res_stime = datetime.datetime.now()
    (fval, core_time) = back_result
    update_core_time(instance, core_time)

    if instance.verification_system:
        print("res_back start...")

    process_result_array(instance, fval, dim)

    res_etime = datetime.datetime.now()
    if instance.verification_system:
        print("my res time is " + str((res_etime - res_stime).total_seconds()))


def res_back_with_V_common(instance, back_result, dim):
    """Common callback to aggregate results including evolution state V.

    Args:
        instance: Base2D or Base3D instance
        back_result: (fval, core_time, V) tuple from worker
        dim: Dimension (2 or 3)
    """
    res_stime = datetime.datetime.now()
    (fval, core_time, instance.V) = back_result
    update_core_time(instance, core_time)

    print("res_back start...")
    process_result_array(instance, fval, dim)

    res_etime = datetime.datetime.now()
    print("my res time is " + str((res_etime - res_stime).total_seconds()))


def calculate_errors_per_site(instance, num_sites):
    """Calculate and set errors_per_site from total errors.

    Args:
        instance: Base2D or Base3D instance with errors attribute
        num_sites: Number of grain boundary sites
    """
    if num_sites > 0:
        instance.errors_per_site = instance.errors / num_sites
    else:
        instance.errors_per_site = 0


def format_page_string(fig_page):
    """Format a page number as a zero-padded string.

    Args:
        fig_page: Integer page/iteration number

    Returns:
        str: Zero-padded string (4 digits)
    """
    if fig_page < 10:
        return '000' + str(fig_page)
    elif fig_page < 100:
        return '00' + str(fig_page)
    elif fig_page < 1000:
        return '0' + str(fig_page)
    else:
        return str(fig_page)
