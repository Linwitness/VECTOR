#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Curvature Calculation Validation Across Multiple Radii and Iterations

This module implements systematic validation studies for the VECTOR framework's
curvature calculation algorithms. It provides comprehensive benchmarking across
different geometric scales (radii), algorithmic approaches (linear vs vertex),
and dimensional contexts (2D vs 3D) with detailed convergence analysis.

Key Features:
-------------
1. Multi-Radius Validation: Systematic accuracy assessment across geometric scales
2. Algorithm Comparison: Linear vs Vertex smoothing method benchmarking
3. Convergence Analysis: Iteration-dependent accuracy and performance evaluation
4. Dimensional Studies: 2D and 3D implementation comparison
5. Performance Profiling: Computational efficiency characterization

Scientific Applications:
-----------------------
- Validation of curvature calculation algorithms for materials science
- Benchmarking computational methods for grain boundary analysis
- Performance optimization for large-scale microstructure simulations
- Algorithm development and comparative studies

Mathematical Foundation:
-----------------------
The validation employs analytical solutions for circular/spherical interfaces:
- 2D circular interface: κ = 1/R (exact analytical solution)
- 3D spherical interface: κ = 2/R (exact analytical solution)
- Complex sinusoidal interfaces: Advanced validation geometries

Algorithm Implementations:
-------------------------
1. Linear Smoothing (BL): Bilinear gradient-based curvature calculation
2. Vertex Smoothing (VT): Vertex-based geometric curvature analysis
3. 2D vs 3D Extensions: Dimensional algorithm comparison
4. Complex Geometries: Advanced validation test cases

Performance Metrics:
-------------------
- Absolute error vs analytical solutions
- Convergence rate with iteration count
- Computational time scaling
- Memory usage optimization
- Per-site error distribution

Dependencies:
------------
- PACKAGE_MP_Linear: 2D linear smoothing algorithms
- PACKAGE_MP_Vertex: 2D vertex-based methods
- PACKAGE_MP_3DLinear: 3D linear smoothing algorithms  
- PACKAGE_MP_3DVertex: 3D vertex-based methods
- myInput: Test geometry generation utilities

Author: Lin Yang
Created: Thu Sep 22 17:46:02 2022

Usage:
------
This module provides systematic validation frameworks for ensuring
accuracy and performance of curvature calculation algorithms across
multiple geometric and computational parameter spaces.
"""

import os
current_path = os.getcwd()+'/'
import sys
sys.path.append(current_path)
sys.path.append(current_path+'../../')
import numpy as np
import math

import PACKAGE_MP_Linear as Linear_2D
import PACKAGE_MP_Vertex as Vertex_2D
import PACKAGE_MP_3DLinear as Linear_3D
import PACKAGE_MP_3DVertex as Vertex_3D
import myInput

def test_2d():
    """
    2D Circular Interface Curvature Validation Study
    
    This function implements comprehensive validation of 2D linear smoothing
    algorithms for curvature calculation using circular test interfaces.
    The circular geometry provides exact analytical solutions for rigorous
    algorithm accuracy assessment.
    
    Validation Framework:
    --------------------
    - Test geometry: Circular interface with radius R=20
    - Theoretical curvature: κ = 1/R = 0.05 (constant)
    - Domain size: 200×200 for statistical validation
    - Iteration range: 1-20 for convergence analysis
    
    Algorithm Under Test:
    --------------------
    PACKAGE_MP_Linear (2D bilinear smoothing):
    - Grain-boundary-aware gradient calculation
    - Multiprocessing optimization for performance
    - Iterative convergence for enhanced accuracy
    - Error analysis and performance profiling
    
    Performance Metrics:
    -------------------
    - Per-site error vs theoretical solution
    - Computational time scaling with iterations
    - Convergence rate characterization
    - Memory usage optimization
    
    Scientific Applications:
    -----------------------
    - Validation of 2D grain boundary curvature analysis
    - Benchmarking for materials characterization software
    - Algorithm optimization for production calculations
    - Quality control for computational accuracy
    
    Output Data:
    -----------
    Saves convergence analysis to BL_Curvature_R{radius}_Iteration_1_{max_iteration}.npz
    containing error metrics and performance timing data.
    """
    # Configure 2D validation domain parameters
    nx, ny = 200, 200          # High-resolution domain for statistical validation
    ng = 2                     # Two-grain circular interface geometry
    cores = 8                  # Multiprocessing configuration
    max_iteration = 20         # Comprehensive convergence analysis range
    radius = 20                # Circle radius for moderate curvature (κ=0.05)
    
    # Configure output file for systematic data storage
    filename_save = f"examples/curvature_calculation/BL_Curvature_R{radius}_Iteration_1_{max_iteration}"

    # Initialize performance tracking arrays
    BL_errors = np.zeros(max_iteration)        # Accuracy vs iteration
    BL_runningTime = np.zeros(max_iteration)   # Performance vs iteration

    # Generate 2D circular interface with analytical solution
    P0, R = myInput.Circle_IC(nx, ny, radius)  # Exact circular geometry

    # Systematic convergence analysis across iteration range
    for cores in [cores]:
        for loop_times in range(1, max_iteration):
            print(f"=== 2D Validation: Iteration {loop_times}, Radius {radius} ===")

            # Initialize 2D linear smoothing class
            test1 = Linear_2D.linear_class(nx, ny, ng, cores, loop_times, P0, R)

            # Execute curvature calculation algorithm
            test1.linear_main("curvature")
            C_ln = test1.get_C()

            # Display comprehensive performance metrics
            print('loop_times = ' + str(test1.loop_times))
            print('running_time = %.2f seconds' % test1.running_time)
            print('running_core time = %.2f seconds' % test1.running_coreTime)
            print('total_errors = %.2f' % test1.errors)
            print('per_errors = %.3f' % test1.errors_per_site)
            print()

            # Store convergence analysis data
            BL_errors[loop_times-1] = test1.errors_per_site
            BL_runningTime[loop_times-1] = test1.running_coreTime

    # Save comprehensive validation results
    np.savez(filename_save, BL_errors=BL_errors, BL_runningTime=BL_runningTime)
    print(f"2D validation results saved to: {filename_save}")

def test_vertex_2d():
    """
    2D Vertex-Based Curvature Calculation Validation Study
    
    This function implements comprehensive validation of 2D vertex-based
    smoothing algorithms for curvature calculation. The vertex approach
    provides an alternative mathematical framework for comparison with
    linear smoothing methods.
    
    Vertex Algorithm Framework:
    --------------------------
    - Geometric vertex-based curvature calculation
    - Alternative to gradient-based linear methods
    - Enhanced accuracy for certain geometric configurations
    - Computational comparison with bilinear smoothing
    
    Validation Objectives:
    ---------------------
    - Algorithm accuracy vs analytical circular solutions
    - Performance comparison with linear smoothing methods
    - Convergence behavior characterization
    - Computational efficiency assessment
    
    Test Configuration:
    ------------------
    - Test geometry: Large circular interface (R=80)
    - Enhanced challenge for vertex algorithm validation
    - Domain size: 200×200 for comprehensive statistics
    - Systematic iteration sweep for convergence analysis
    
    Scientific Significance:
    -----------------------
    - Independent validation of curvature calculation approaches
    - Benchmarking for method selection in research applications
    - Algorithm development and optimization studies
    - Quality assurance for computational materials science
    """
    # Configure 2D vertex validation parameters
    nx, ny = 200, 200          # High-resolution validation domain
    ng = 2                     # Two-grain interface geometry
    cores = 8                  # Parallel processing configuration
    max_iteration = 20         # Complete convergence analysis
    radius = 80                # Large radius for vertex algorithm challenge
    
    filename_save = f"./VT_Curvature_R{radius}_Iteration_1_{max_iteration}"

    # Initialize vertex algorithm tracking arrays
    VT_errors = np.zeros(max_iteration)        # Vertex algorithm accuracy
    VT_runningTime = np.zeros(max_iteration)   # Vertex performance metrics

    # Generate circular test geometry for vertex validation
    P0, R = myInput.Circle_IC(nx, ny, radius)

    # Systematic vertex algorithm convergence analysis
    for cores in [cores]:
        for interval in range(1, max_iteration):
            print(f"=== 2D Vertex Validation: Iteration {interval}, Radius {radius} ===")

            # Initialize 2D vertex smoothing class
            test1 = Vertex_2D.vertex_class(nx, ny, ng, cores, interval, P0, R)

            # Execute vertex-based curvature calculation
            test1.vertex_main("curvature")
            C_ln = test1.get_C()

            # Display vertex algorithm performance metrics
            print('interval = ' + str(test1.interval))
            print('running_time = %.2f seconds' % test1.running_time)
            print('running_core time = %.2f seconds' % test1.running_coreTime)
            print('total_errors = %.2f' % test1.errors)
            print('per_errors = %.3f' % test1.errors_per_site)
            print()

            # Store vertex algorithm convergence data
            VT_errors[interval-1] = test1.errors_per_site
            VT_runningTime[interval-1] = test1.running_coreTime

    # Save vertex validation results for comparison
    np.savez(filename_save, VT_errors=VT_errors, VT_runningTime=VT_runningTime)
    print(f"2D vertex validation results saved to: {filename_save}")


def test_3d():
    """
    3D Spherical Interface Curvature Validation Study
    
    This function implements comprehensive validation of 3D linear smoothing
    algorithms for curvature calculation using spherical test interfaces.
    The 3D extension provides critical validation for volumetric grain
    boundary analysis applications.
    
    3D Validation Framework:
    -----------------------
    - Test geometry: Spherical interface with configurable radius
    - Theoretical curvature: κ = 2/R (exact analytical solution)
    - High-resolution domain: 200×200×200 (8M sites)
    - Comprehensive iteration sweep for convergence validation
    
    Algorithm Under Test:
    --------------------
    PACKAGE_MP_3DLinear (3D linear smoothing):
    - Extension of bilinear methods to 3D volumes
    - Advanced multiprocessing for large-scale computation
    - Sparse matrix implementation for memory efficiency
    - Complex 3D grain boundary geometry handling
    
    Computational Challenges:
    ------------------------
    - Large memory requirements (8M sites)
    - Complex 3D gradient calculation algorithms
    - Enhanced multiprocessing coordination
    - Validation against exact 3D analytical solutions
    
    Performance Scaling:
    -------------------
    - Memory usage: O(nx×ny×nz) for dense 3D arrays
    - Computational time: O(nx×ny×nz×iterations)
    - Parallel efficiency: Multi-core 3D domain decomposition
    - Accuracy vs computational cost trade-offs
    
    Scientific Applications:
    -----------------------
    - Validation of 3D grain boundary curvature algorithms
    - Benchmarking for electron tomography data analysis
    - Performance optimization for 3D microstructure simulations
    - Quality control for volumetric materials characterization
    """
    # Configure 3D validation domain parameters
    nx, ny, nz = 200, 200, 200  # High-resolution 3D domain (8M sites)
    ng = 2                      # Two-grain spherical interface
    cores = 16                  # Enhanced multiprocessing for 3D calculation
    max_iteration = 20          # Comprehensive convergence analysis
    radius = [80]               # Spherical radius array for systematic study

    # Systematic analysis across different radii
    for r in radius:
        print(f"=== 3D Spherical Validation: Radius {r} ===")
        filename_save = f"./BL3D_Curvature_R{r}_Iteration_1_{max_iteration}"

        # Initialize 3D performance tracking arrays
        BL_errors = np.zeros(max_iteration)        # 3D accuracy metrics
        BL_runningTime = np.zeros(max_iteration)   # 3D performance data

        # Generate 3D spherical interface with analytical solution
        P0, R = myInput.Circle_IC3d(nx, ny, nz, r)  # Exact 3D spherical geometry

        # Systematic 3D convergence analysis
        for cores in [cores]:
            for loop_times in range(1, max_iteration):
                print(f"Processing 3D iteration {loop_times} for radius {r}")

                # Initialize 3D linear smoothing class
                test1 = Linear_3D.linear3d_class(nx, ny, nz, ng, cores, loop_times, P0, R, 'np')

                # Execute 3D curvature calculation algorithm
                test1.linear3d_main("curvature")
                C_ln = test1.get_C()

                # Display comprehensive 3D performance metrics
                print('loop_times = ' + str(test1.loop_times))
                print('running_time = %.2f seconds' % test1.running_time)
                print('running_core time = %.2f seconds' % test1.running_coreTime)
                print('total_errors = %.2f' % test1.errors)
                print('per_errors = %.3f' % test1.errors_per_site)
                print()

                # Store 3D convergence analysis data
                BL_errors[loop_times-1] = test1.errors_per_site
                BL_runningTime[loop_times-1] = test1.running_coreTime

        # Save 3D validation results
        np.savez(filename_save, BL_errors=BL_errors, BL_runningTime=BL_runningTime)
        print(f"3D validation results saved to: {filename_save}")

def test_vertex_3d():
    """
    3D Vertex-Based Curvature Calculation Validation Study
    
    This function implements comprehensive validation of 3D vertex-based
    smoothing algorithms for curvature calculation. The 3D vertex approach
    extends geometric curvature methods to volumetric microstructures.
    
    3D Vertex Algorithm Framework:
    -----------------------------
    - Extension of 2D vertex methods to 3D volumes
    - Geometric vertex-based curvature calculation in 3D space
    - Alternative mathematical approach to 3D linear smoothing
    - Enhanced complexity for 3D grain boundary networks
    
    Multi-Radius Validation Study:
    ------------------------------
    Systematic testing across radius values [5, 20, 50, 80, 2, 1]:
    - Wide range of curvature magnitudes (κ = 2/R)
    - Challenge algorithm across different geometric scales
    - Statistical validation with varying numerical demands
    - Comprehensive performance characterization
    
    3D Computational Complexity:
    ----------------------------
    - High-resolution 3D domain (200³ = 8M sites)
    - Complex 3D vertex neighborhood calculations
    - Enhanced memory and computational requirements
    - Advanced parallel processing coordination
    
    Algorithm Comparison Objectives:
    -------------------------------
    - Independent validation of 3D curvature methods
    - Performance comparison with 3D linear algorithms
    - Accuracy assessment across multiple geometric scales
    - Computational efficiency characterization
    """
    # Configure 3D vertex validation parameters
    nx, ny, nz = 200, 200, 200  # High-resolution 3D validation domain
    ng = 2                      # Two-grain spherical interface
    cores = 8                   # Multiprocessing for 3D vertex calculations
    max_iteration = 20          # Complete convergence analysis
    radius = [5, 20, 50, 80, 2, 1]  # Comprehensive radius range

    # Systematic multi-radius validation study
    for r in radius:
        print(f"=== 3D Vertex Validation: Radius {r} ===")
        filename_save = f"./VT3D_Curvature_R{r}_Iteration_1_{max_iteration}"

        # Initialize 3D vertex tracking arrays
        VT_errors = np.zeros(max_iteration)        # 3D vertex accuracy
        VT_runningTime = np.zeros(max_iteration)   # 3D vertex performance

        # Generate 3D spherical test geometry
        P0, R = myInput.Circle_IC3d(nx, ny, nz, r)

        # Systematic 3D vertex convergence analysis
        for cores in [cores]:
            for interval in range(1, max_iteration):
                print(f"Processing 3D vertex iteration {interval} for radius {r}")

                # Initialize 3D vertex smoothing class
                test1 = Vertex_3D.vertex3d_class(nx, ny, nz, ng, cores, interval, P0, R)

                # Execute 3D vertex-based curvature calculation
                test1.vertex3d_main("curvature")
                C_ln = test1.get_C()

                # Display 3D vertex performance metrics
                print('loop_times = ' + str(test1.interval))
                print('running_time = %.2f seconds' % test1.running_time)
                print('running_core time = %.2f seconds' % test1.running_coreTime)
                print('total_errors = %.2f' % test1.errors)
                print('per_errors = %.3f' % test1.errors_per_site)
                print()

                # Store 3D vertex convergence data
                VT_errors[interval-1] = test1.errors_per_site
                VT_runningTime[interval-1] = test1.running_coreTime

        # Save 3D vertex validation results
        np.savez(filename_save, VT_errors=VT_errors, VT_runningTime=VT_runningTime)
        print(f"3D vertex validation results saved to: {filename_save}")

def test_vertex_3dComplex():
    """
    3D Complex Geometry Vertex Algorithm Validation Study
    
    This function implements advanced validation of 3D vertex-based algorithms
    using complex sinusoidal interface geometries. The complex geometries provide
    enhanced challenge for algorithm validation beyond simple spherical interfaces.
    
    Complex Geometry Framework:
    --------------------------
    - Sinusoidal 3D interfaces with variable wavelength parameters
    - Enhanced curvature complexity beyond spherical test cases
    - Challenging validation for realistic grain boundary geometries
    - Advanced numerical algorithm stress testing
    
    Multi-Wavelength Validation:
    ----------------------------
    Systematic testing across wavelength values [5, 20, 50, 80, 2, 1]:
    - Variable spatial frequency content for algorithm challenge
    - Complex curvature distributions for comprehensive validation
    - Enhanced numerical demands for algorithm robustness
    - Statistical analysis across different geometric complexities
    
    Advanced Algorithm Testing:
    --------------------------
    - Extension beyond analytical circular/spherical solutions
    - Validation with realistic grain boundary complexity
    - Performance assessment for production-level geometries
    - Quality control for complex microstructure analysis
    
    Scientific Applications:
    -----------------------
    - Validation for realistic grain boundary analysis
    - Benchmarking for complex microstructure characterization
    - Algorithm development for advanced materials analysis
    - Quality assurance for computational materials research
    """
    # Configure 3D complex geometry validation parameters
    nx, ny, nz = 200, 200, 200  # High-resolution domain for complex validation
    ng = 2                      # Two-grain complex interface
    cores = 8                   # Multiprocessing for complex calculations
    max_iteration = 20          # Complete convergence analysis
    wave = [5, 20, 50, 80, 2, 1]  # Wavelength parameter range

    # Systematic multi-wavelength complex validation
    for w in wave:
        print(f"=== 3D Complex Vertex Validation: Wavelength {w} ===")
        filename_save = f"./VT3DComp_Curvature_wave{w}_Iteration_1_{max_iteration}"

        # Initialize complex geometry tracking arrays
        VT_errors = np.zeros(max_iteration)        # Complex geometry accuracy
        VT_runningTime = np.zeros(max_iteration)   # Complex geometry performance

        # Generate 3D complex sinusoidal interface geometry
        P0, R = myInput.Complex2G_IC3d(nx, ny, nz, w)

        # Systematic complex geometry convergence analysis
        for cores in [cores]:
            for interval in range(1, max_iteration):
                print(f"Processing complex iteration {interval} for wavelength {w}")

                # Initialize 3D vertex smoothing for complex geometry
                test1 = Vertex_3D.vertex3d_class(nx, ny, nz, ng, cores, interval, P0, R)

                # Execute complex geometry curvature calculation
                test1.vertex3d_main("curvature")
                C_ln = test1.get_C()

                # Display complex geometry performance metrics
                print('loop_times = ' + str(test1.interval))
                print('running_time = %.2f seconds' % test1.running_time)
                print('running_core time = %.2f seconds' % test1.running_coreTime)
                print('total_errors = %.2f' % test1.errors)
                print('per_errors = %.3f' % test1.errors_per_site)
                print()

                # Store complex geometry convergence data
                VT_errors[interval-1] = test1.errors_per_site
                VT_runningTime[interval-1] = test1.running_coreTime

        # Save complex geometry validation results
        np.savez(filename_save, VT_errors=VT_errors, VT_runningTime=VT_runningTime)
        print(f"Complex geometry validation results saved to: {filename_save}")


if __name__ == '__main__':
    """
    Main Execution Block for Comprehensive Curvature Validation Studies
    
    This section executes systematic validation of curvature calculation
    algorithms across multiple dimensions, geometric scales, and algorithmic
    approaches. The studies provide comprehensive benchmarking for the
    VECTOR framework's curvature analysis capabilities.
    
    Execution Strategy:
    ------------------
    1. 3D spherical validation (test_3d): Primary large-scale validation
    2. Complex geometry validation (test_vertex_3dComplex): Advanced testing
    3. Additional studies available for comprehensive analysis
    
    Expected Runtime:
    ----------------
    - 3D spherical validation: 2-4 hours for complete analysis
    - Complex geometry validation: 4-8 hours for full parameter sweep
    - Total comprehensive study: 6-12 hours depending on system performance
    
    Resource Requirements:
    ---------------------
    - Memory: 16-32 GB for large-scale 3D calculations
    - Storage: 1-5 GB for comprehensive validation datasets
    - CPU: Multi-core system recommended for parallel processing
    
    Scientific Impact:
    -----------------
    Results provide validation data for:
    - Algorithm accuracy assessment and optimization
    - Performance benchmarking for production calculations
    - Quality control for computational materials science
    - Method development and comparative studies
    """
    print("=== Comprehensive Curvature Validation Study ===")
    print("Executing systematic algorithm validation across multiple parameters")
    print("Expected runtime: 2-4 hours for complete validation")
    print()

    # Execute primary 3D spherical validation study
    test_3d()
    
    # Additional validation studies available:
    # test_vertex_3dComplex()  # Complex geometry validation
    # test_2d()                # 2D validation studies  
    # test_vertex_2d()         # 2D vertex algorithm validation
    # test_vertex_3d()         # 3D vertex algorithm validation

    print("=== Validation Study Complete ===")
    print("Comprehensive validation data generated for algorithm analysis")

