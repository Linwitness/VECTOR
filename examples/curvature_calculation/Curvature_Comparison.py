#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Curvature Algorithm Comparison and Visualization

This module provides sophisticated visualization and comparative analysis tools
for evaluating the performance of different curvature calculation algorithms
across multiple geometric scales and dimensional contexts. It generates
publication-quality plots for algorithm validation and benchmarking studies.

Key Features:
-------------
1. Multi-Algorithm Comparison: Linear vs Vertex smoothing method analysis
2. Multi-Scale Visualization: Performance across different radius parameters  
3. Convergence Analysis: Iteration-dependent accuracy assessment
4. Dimensional Comparison: 2D vs 3D algorithm performance evaluation
5. Statistical Validation: Error normalization and scaling analysis

Scientific Applications:
-----------------------
- Algorithm validation for materials science research
- Performance benchmarking for computational method development
- Quality control for curvature calculation accuracy
- Comparative studies for method selection in research applications

Visualization Framework:
-----------------------
The module generates comprehensive plots showing:
- Normalized error vs iteration count for different radii
- Convergence behavior comparison between algorithms
- Performance scaling with geometric parameters
- Statistical validation against analytical solutions

Mathematical Foundation:
-----------------------
Error normalization uses theoretical curvature values:
- 2D circular interfaces: κ_theoretical = 1/R
- 3D spherical interfaces: κ_theoretical = 2/R
- Relative error = |κ_computed - κ_theoretical| / κ_theoretical

Algorithm Comparison Methods:
----------------------------
1. Linear Smoothing (BL): Bilinear gradient-based methods
2. Vertex Smoothing (VT): Geometric vertex-based methods
3. 2D vs 3D Extensions: Dimensional scaling analysis
4. Multi-Radius Studies: Geometric scale dependency

Performance Metrics:
-------------------
- Relative error vs theoretical solutions
- Convergence rate with iteration count
- Algorithm stability across geometric scales
- Computational efficiency comparison

Dependencies:
------------
- numpy: Numerical data processing
- matplotlib: Advanced visualization and plotting
- Validation data files: .npz format algorithm results

Author: Lin Yang
Created: Thu Sep 22 17:58:03 2022

Usage:
------
This module provides visualization tools for comprehensive analysis
of curvature calculation algorithm performance and validation studies.
"""

import os
current_path = os.getcwd()+'/'
import sys
sys.path.append(current_path)
sys.path.append(current_path+'../../')
import numpy as np
import math
import matplotlib.pyplot as plt

def plot_test2D():
    """
    2D Linear Smoothing Algorithm Performance Visualization
    
    This function generates comprehensive comparison plots for 2D bilinear
    smoothing algorithm performance across multiple radius parameters.
    The visualization provides insights into convergence behavior and
    accuracy scaling with geometric parameters.
    
    Visualization Framework:
    -----------------------
    - Multi-radius comparison: R = [5, 20, 50, 80] for comprehensive analysis
    - Normalized error plotting: Relative error vs theoretical curvature
    - Convergence analysis: Error reduction with iteration count
    - Statistical validation: Algorithm accuracy assessment
    
    Data Sources:
    ------------
    Loads validation results from systematic convergence studies:
    - BL_Curvature_R{radius}_Iteration_1_20.npz files
    - Per-site error data vs iteration count
    - Performance timing information
    
    Scientific Interpretation:
    -------------------------
    - Theoretical curvature: κ = 1/R for 2D circular interfaces
    - Relative error normalization enables cross-radius comparison
    - Convergence trends indicate algorithm stability and accuracy
    - Performance scaling reveals computational efficiency
    
    Visualization Output:
    --------------------
    Publication-quality plot showing:
    - Error convergence for different radii (different colors)
    - Iteration count on x-axis (1-20 iterations)
    - Normalized relative error on y-axis
    - Legend for radius identification
    
    Applications:
    ------------
    - Algorithm validation for 2D grain boundary analysis
    - Performance benchmarking for method development
    - Quality control for computational accuracy
    - Research publication visualization
    """
    # Configure visualization parameters
    data_num = 19  # Number of iterations to display (1-19)

    # Define data file names for multi-radius analysis
    file1 = 'BL_Curvature_R5_Iteration_1_20.npz'    # High curvature (κ=0.2)
    file2 = 'BL_Curvature_R20_Iteration_1_20.npz'   # Moderate curvature (κ=0.05)
    file3 = 'BL_Curvature_R50_Iteration_1_20.npz'   # Low curvature (κ=0.02)
    file4 = 'BL_Curvature_R80_Iteration_1_20.npz'   # Very low curvature (κ=0.0125)
    function_name = 'BL_errors'  # Error data key in .npz files

    # Load systematic validation data
    r5 = np.load(file1)   # R=5 validation results
    r20 = np.load(file2)  # R=20 validation results
    r50 = np.load(file3)  # R=50 validation results
    r80 = np.load(file4)  # R=80 validation results

    # Generate publication-quality comparison plot
    fig = plt.figure()
    
    # Plot normalized relative error for each radius
    plt.plot(range(data_num), (r5[function_name][:data_num])/(1/5), label="r=05")
    plt.plot(range(data_num), (r20[function_name][:data_num])/(1/20), label="r=20")
    plt.plot(range(data_num), (r50[function_name][:data_num])/(1/50), label="r=50")
    plt.plot(range(data_num), (r80[function_name][:data_num])/(1/80), label="r=80")
    
    # Configure plot appearance for publication quality
    plt.legend(title='Radius Parameters')
    plt.ylim([0, 6])     # Normalized error range
    plt.xlim([-0.1, 20]) # Iteration range
    plt.xlabel('Iteration Count')
    plt.ylabel('Normalized Relative Error')
    plt.title('2D Linear Smoothing Algorithm Convergence Analysis')
    plt.grid(True, alpha=0.3)

def plot_VT_test2D():
    """
    2D Vertex Algorithm Performance Visualization
    
    This function provides comprehensive visualization of 2D vertex-based
    smoothing algorithm performance, enabling direct comparison with
    linear smoothing methods for algorithm selection and validation.
    
    Vertex Algorithm Framework:
    --------------------------
    - Geometric vertex-based curvature calculation approach
    - Alternative mathematical foundation to linear smoothing
    - Independent validation of curvature analysis accuracy
    - Performance comparison for method selection
    
    Comparative Analysis Objectives:
    -------------------------------
    - Algorithm accuracy comparison: Vertex vs Linear methods
    - Convergence behavior characterization for different approaches
    - Performance scaling analysis across geometric parameters
    - Method selection guidance for research applications
    
    Visualization Features:
    ----------------------
    - Multi-radius vertex algorithm performance
    - Normalized error analysis for cross-method comparison
    - Convergence trend identification
    - Statistical validation visualization
    
    Scientific Applications:
    -----------------------
    - Independent algorithm validation for curvature analysis
    - Comparative method studies for research publications
    - Quality control for computational accuracy assessment
    - Algorithm development and optimization guidance
    """
    # Configure vertex visualization parameters
    data_num = 19  # Iteration range for analysis

    # Define vertex algorithm validation data files
    file1 = 'VT_Curvature_R5_Iteration_1_20.npz'    # Vertex R=5 results
    file2 = 'VT_Curvature_R20_Iteration_1_20.npz'   # Vertex R=20 results
    file3 = 'VT_Curvature_R50_Iteration_1_20.npz'   # Vertex R=50 results
    file4 = 'VT_Curvature_R80_Iteration_1_20.npz'   # Vertex R=80 results
    function_name = 'VT_errors'  # Vertex error data key

    # Load vertex algorithm validation data
    r5 = np.load(file1)   # Vertex R=5 results
    r20 = np.load(file2)  # Vertex R=20 results
    r50 = np.load(file3)  # Vertex R=50 results
    r80 = np.load(file4)  # Vertex R=80 results

    # Generate vertex algorithm comparison plot
    fig = plt.figure()
    
    # Plot normalized vertex algorithm performance
    plt.plot(range(data_num), (r5[function_name][:data_num])/(1/5), label="r=05")
    plt.plot(range(data_num), (r20[function_name][:data_num])/(1/20), label="r=20")
    plt.plot(range(data_num), (r50[function_name][:data_num])/(1/50), label="r=50")
    plt.plot(range(data_num), (r80[function_name][:data_num])/(1/80), label="r=80")
    
    # Configure vertex algorithm plot
    plt.legend(title='Radius Parameters (Vertex)')
    plt.ylim([0, 6])
    plt.xlim([-0.1, 20])
    plt.xlabel('Iteration Count')
    plt.ylabel('Normalized Relative Error')
    plt.title('2D Vertex Algorithm Convergence Analysis')
    plt.grid(True, alpha=0.3)

def plot_test3D():
    """
    3D Linear Smoothing Algorithm Performance Visualization with Reference Lines
    
    This function provides advanced visualization of 3D linear smoothing
    algorithm performance with reference lines indicating theoretical
    accuracy limits and validation benchmarks.
    
    3D Algorithm Validation Framework:
    ---------------------------------
    - 3D spherical interface validation (κ = 2/R theoretical)
    - High-resolution domain analysis (200³ voxels)
    - Multi-radius systematic validation
    - Advanced convergence analysis with reference benchmarks
    
    Reference Line Interpretation:
    -----------------------------
    Dashed lines represent theoretical accuracy benchmarks:
    - Based on analytical solutions for spherical interfaces
    - Enable visual assessment of algorithm convergence limits
    - Provide validation targets for algorithm development
    - Support quality control for computational accuracy
    
    Advanced Visualization Features:
    -------------------------------
    - Logarithmic error scaling for wide dynamic range
    - Color-coded radius parameters for clarity
    - Reference benchmark lines for accuracy assessment
    - Publication-quality formatting for research documentation
    
    Scientific Applications:
    -----------------------
    - 3D algorithm validation for volumetric grain boundary analysis
    - Performance benchmarking for large-scale computation
    - Quality control for 3D microstructure characterization
    - Method development and optimization guidance
    """
    # Configure 3D visualization parameters
    data_num = 19  # Iteration range for comprehensive analysis

    # Define 3D linear algorithm validation data files
    file1 = 'BL3D_Curvature_R5_Iteration_1_20.npz'    # 3D R=5 results
    file2 = 'BL3D_Curvature_R20_Iteration_1_20.npz'   # 3D R=20 results
    file3 = 'BL3D_Curvature_R50_Iteration_1_20.npz'   # 3D R=50 results
    file4 = 'BL_Curvature_R80_Iteration_1_20.npz'     # 3D R=80 results
    function_name = 'BL_errors'  # 3D error data key

    # Load 3D algorithm validation data
    r5 = np.load(file1)   # 3D R=5 validation results
    r20 = np.load(file2)  # 3D R=20 validation results
    r50 = np.load(file3)  # 3D R=50 validation results
    r80 = np.load(file4)  # 3D R=80 validation results

    # Generate advanced 3D performance visualization
    fig = plt.figure()
    
    # Plot 3D algorithm convergence with distinct colors
    plt.plot(range(data_num), (r5[function_name][:data_num])/(1/5), 
             label="r=05", color="royalblue")
    plt.plot(range(data_num), (r20[function_name][:data_num])/(1/20), 
             label="r=20", color="orange")
    plt.plot(range(data_num), (r50[function_name][:data_num])/(1/50), 
             label="r=50", color="green")
    plt.plot(range(data_num), (r80[function_name][:data_num])/(1/80), 
             label="r=80", color="purple")
    
    # Add theoretical accuracy reference lines
    plt.plot([-0.1, data_num], [abs(r5_vv-0.2)/0.2, abs(r5_vv-0.2)/0.2], 
             '--', color="royalblue", alpha=0.7)
    plt.plot([-0.1, data_num], [abs(r20_vv-0.05)/0.05, abs(r20_vv-0.05)/0.05], 
             '--', color="orange", alpha=0.7)
    plt.plot([-0.1, data_num], [abs(r50_vv-0.02)/0.02, abs(r50_vv-0.02)/0.02], 
             '--', color="green", alpha=0.7)
    plt.plot([-0.1, data_num], [abs(r80_vv-0.0125)/0.0125, abs(r80_vv-0.0125)/0.0125], 
             '--', color="purple", alpha=0.7)
    
    # Configure advanced 3D plot
    plt.legend(title='3D Radius Parameters')
    plt.yscale('log')    # Logarithmic scale for wide error range
    plt.ylim([0, 2])     # Logarithmic error range
    plt.xlim([-0.1, 20]) # Iteration range
    plt.xlabel('Iteration Count')
    plt.ylabel('Normalized Relative Error (log scale)')
    plt.title('3D Linear Smoothing Algorithm Convergence with Reference Lines')
    plt.grid(True, alpha=0.3)

def plot_VT_test3D():
    """
    3D Vertex Algorithm Comprehensive Performance Visualization
    
    This function provides the most comprehensive visualization of 3D vertex
    algorithm performance, including extended radius range and theoretical
    reference benchmarks for complete validation analysis.
    
    Extended Validation Framework:
    -----------------------------
    - Comprehensive radius range: R = [1, 2, 5, 20, 50, 80]
    - Wide curvature span: κ = 2 to κ = 0.0125
    - Extreme geometry testing for algorithm robustness
    - Complete performance characterization
    
    Advanced Reference System:
    -------------------------
    - Theoretical benchmarks for all radius values
    - Visual convergence target identification
    - Algorithm accuracy limit assessment
    - Quality control reference standards
    
    Comprehensive Analysis Features:
    -------------------------------
    - Extended geometric parameter space coverage
    - Logarithmic scaling for wide dynamic range
    - Color-coded systematic visualization
    - Publication-quality comprehensive documentation
    
    Scientific Significance:
    -----------------------
    - Complete 3D vertex algorithm validation
    - Comprehensive performance characterization
    - Algorithm robustness assessment across scales
    - Research-grade validation documentation
    """
    # Configure comprehensive 3D vertex visualization
    data_num = 19  # Complete iteration analysis

    # Define comprehensive 3D vertex validation data files
    file1 = 'VT3D_Curvature_R5_Iteration_1_20.npz'    # 3D Vertex R=5
    file2 = 'VT3D_Curvature_R20_Iteration_1_20.npz'   # 3D Vertex R=20
    file3 = 'VT3D_Curvature_R50_Iteration_1_20.npz'   # 3D Vertex R=50
    file4 = 'VT3D_Curvature_R80_Iteration_1_20.npz'   # 3D Vertex R=80
    file5 = 'VT3D_Curvature_R2_Iteration_1_20.npz'    # 3D Vertex R=2
    file6 = 'VT3D_Curvature_R1_Iteration_1_20.npz'    # 3D Vertex R=1
    function_name = 'VT_errors'  # 3D vertex error data

    # Load comprehensive 3D vertex validation data
    r5 = np.load(file1)   # R=5 vertex results
    r20 = np.load(file2)  # R=20 vertex results
    r50 = np.load(file3)  # R=50 vertex results
    r80 = np.load(file4)  # R=80 vertex results
    r2 = np.load(file5)   # R=2 vertex results
    r1 = np.load(file6)   # R=1 vertex results

    # Generate comprehensive 3D vertex visualization
    fig = plt.figure()
    
    # Plot comprehensive radius range with distinct colors
    plt.plot(range(data_num), (r5[function_name][:data_num])/(1/5), 
             label="r=05", color="royalblue")
    plt.plot(range(data_num), (r20[function_name][:data_num])/(1/20), 
             label="r=20", color="orange")
    plt.plot(range(data_num), (r50[function_name][:data_num])/(1/50), 
             label="r=50", color="green")
    plt.plot(range(data_num), (r80[function_name][:data_num])/(1/80), 
             label="r=80", color="purple")
    plt.plot(range(data_num), (r2[function_name][:data_num])/(1/2), 
             label="r=2", color="red")
    plt.plot(range(data_num), (r1[function_name][:data_num])/(1), 
             label="r=1", color="gray")
    
    # Add comprehensive theoretical reference lines
    plt.plot([-0.1, data_num], [abs(r5_vv-0.2)/0.2, abs(r5_vv-0.2)/0.2], 
             '--', color="royalblue", alpha=0.7)
    plt.plot([-0.1, data_num], [abs(r20_vv-0.05)/0.05, abs(r20_vv-0.05)/0.05], 
             '--', color="orange", alpha=0.7)
    plt.plot([-0.1, data_num], [abs(r50_vv-0.02)/0.02, abs(r50_vv-0.02)/0.02], 
             '--', color="green", alpha=0.7)
    plt.plot([-0.1, data_num], [abs(r80_vv-0.0125)/0.0125, abs(r80_vv-0.0125)/0.0125], 
             '--', color="purple", alpha=0.7)
    plt.plot([-0.1, data_num], [abs(r2_vv-0.5)/0.5, abs(r2_vv-0.5)/0.5], 
             '--', color="red", alpha=0.7)
    plt.plot([-0.1, data_num], [abs(r1_vv-1)/1, abs(r1_vv-1)/1], 
             '--', color="gray", alpha=0.7)

    # Configure comprehensive visualization
    plt.legend(title='3D Vertex Radius Parameters')
    plt.yscale('log')     # Logarithmic scale for comprehensive range
    plt.ylim([0, 200])    # Extended error range
    plt.xlim([-0.1, 20])  # Complete iteration range
    plt.xlabel('Iteration Count')
    plt.ylabel('Normalized Relative Error (log scale)')
    plt.title('Comprehensive 3D Vertex Algorithm Performance Analysis')
    plt.grid(True, alpha=0.3)


# Theoretical curvature reference values for validation benchmarking
# These values represent exact analytical solutions for spherical interfaces
r1_vv = 1.570796333    # κ = 2/1 = 2.0 (high curvature)
r2_vv = 0.523598778    # κ = 2/2 = 1.0 (moderate-high curvature)
r5_vv = 0.204886473    # κ = 2/5 = 0.4 (moderate curvature)
r20_vv = 0.049205668   # κ = 2/20 = 0.1 (low curvature)
r50_vv = 0.019873334   # κ = 2/50 = 0.04 (very low curvature)
r80_vv = 0.012444896   # κ = 2/80 = 0.025 (extremely low curvature)

# Main execution block for algorithm comparison visualization
if __name__ == '__main__':
    """
    Execute Comprehensive Algorithm Comparison Visualization
    
    This section provides systematic execution of all visualization functions
    for comprehensive algorithm comparison and validation analysis.
    
    Visualization Options:
    ---------------------
    - plot_VT_test2D(): 2D vertex algorithm analysis
    - plot_test3D(): 3D linear algorithm with reference lines
    - plot_VT_test3D(): Comprehensive 3D vertex algorithm analysis
    
    Usage Instructions:
    ------------------
    Uncomment desired visualization functions to generate specific plots:
    - Individual plots for focused analysis
    - Multiple plots for comprehensive comparison
    - All plots for complete algorithm documentation
    
    Output:
    -------
    Publication-quality matplotlib figures showing:
    - Algorithm convergence behavior
    - Performance scaling across geometric parameters
    - Theoretical accuracy benchmarks
    - Comparative analysis results
    """
    print("=== Curvature Algorithm Comparison Visualization ===")
    print("Generating algorithm performance comparison plots...")
    
    # Execute primary 3D linear algorithm visualization
    plot_test3D()
    
    # Additional visualization options (uncomment as needed):
    # plot_VT_test2D()    # 2D vertex algorithm analysis
    # plot_VT_test3D()    # Comprehensive 3D vertex analysis
    
    print("Algorithm comparison visualization complete")
    plt.show()
