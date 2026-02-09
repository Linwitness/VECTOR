#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Curvature Calculation Validation Across Multiple Radii and Iterations

This module provides validation studies for the VECTOR framework's curvature
calculation algorithms using the unified ValidationStudy framework.

See curvature_validation.py for the core validation framework.

Author: Lin Yang
Created: Thu Sep 22 17:46:02 2022
"""

from curvature_validation import (
    run_2d_linear_validation,
    run_2d_vertex_validation,
    run_3d_linear_validation,
    run_3d_vertex_validation,
    run_3d_complex_validation,
    ValidationStudy,
    ValidationConfig,
    AlgorithmType
)


if __name__ == '__main__':
    """
    Main Execution Block for Comprehensive Curvature Validation Studies

    This section executes systematic validation of curvature calculation
    algorithms across multiple dimensions, geometric scales, and algorithmic
    approaches. The studies provide comprehensive benchmarking for the
    VECTOR framework's curvature analysis capabilities.

    Execution Strategy:
    ------------------
    1. 3D spherical validation (run_3d_linear_validation): Primary large-scale validation
    2. Complex geometry validation (run_3d_complex_validation): Advanced testing
    3. Additional studies available for comprehensive analysis

    Expected Runtime:
    ----------------
    - 3D spherical validation: 2-4 hours for complete analysis
    - Complex geometry validation: 4-8 hours for full parameter sweep

    Resource Requirements:
    ---------------------
    - Memory: 16-32 GB for large-scale 3D calculations
    - Storage: 1-5 GB for comprehensive validation datasets
    - CPU: Multi-core system recommended for parallel processing
    """
    print("=== Comprehensive Curvature Validation Study ===")
    print("Executing systematic algorithm validation across multiple parameters")
    print()

    # Execute primary 3D spherical validation study
    run_3d_linear_validation()

    # Additional validation studies available:
    # run_3d_complex_validation()  # Complex geometry validation
    # run_2d_linear_validation()   # 2D validation studies
    # run_2d_vertex_validation()   # 2D vertex algorithm validation
    # run_3d_vertex_validation()   # 3D vertex algorithm validation

    print("=== Validation Study Complete ===")
    print("Comprehensive validation data generated for algorithm analysis")
