#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for Allen-Cahn smoothing algorithm implementation.
Tests both 2D and 3D cases for normal vector and curvature calculations.

Author: Lin Yang
"""

import os
import sys
current_path = os.getcwd()
sys.path.append(current_path + '/../../')

import numpy as np
import matplotlib.pyplot as plt
import myInput
from PACKAGE_MP_AllenCahn import allenCahn_class
from . import CONFIG_2D, ALGORITHM_PARAMS, PLOT_CONFIG

# Create output directory for test results
OUTPUT_DIR = os.path.join(current_path, 'output')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def calculate_normal_vector_error(P, R, gb_sites):
    """Calculate error between computed and theoretical normal vectors.
    
    For each grain boundary site, computes the angle between the calculated
    normal vector and the theoretical value from R.
    
    Args:
        P: Phase field results containing calculated normal vectors
        R: Reference normal vectors
        gb_sites: List of grain boundary sites
        
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

def calculate_curvature_error(C, center, radius, gb_sites):
    """Calculate error between computed and theoretical curvature.
    
    For a circle, theoretical curvature is 1/R everywhere on the boundary.
    
    Args:
        C: Calculated curvature field
        center: (x,y) coordinates of circle center
        radius: Circle radius
        gb_sites: List of grain boundary sites
        
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

def test_circle():
    """Test normal vector and curvature calculation on a 2D circle.
    
    A circle provides an ideal test case since:
    1. Normal vectors should point radially inward/outward
    2. Curvature should be constant (κ = 1/R)
    """
    print("\nTesting circle configuration...")
    
    # Get parameters from config
    cfg = CONFIG_2D['circle_test']
    alg_params = ALGORITHM_PARAMS['allen_cahn']
    nx, ny = cfg['grid_size']
    ng = cfg['num_grains']
    cores = cfg['num_cores']
    nsteps = cfg['num_steps']
    radius = cfg['circle_radius']
    center = (nx//2, ny//2)
    
    # Generate circle initial condition
    P0, R = myInput.Circle_IC(nx, ny, r=radius)
    
    # Test normal vectors
    print("\nTesting normal vectors...")
    ac = allenCahn_class(nx, ny, ng, cores, nsteps, P0, R,
                        clip=alg_params['clip'],
                        verification_system=alg_params['verification_system'],
                        curvature_sign=alg_params['curvature_sign'])
    ac.allenCahn_main(purpose="inclination")
    P = ac.get_P()
    gb_sites = ac.get_gb_list()
    
    # Calculate normal vector errors
    rms_error, max_error = calculate_normal_vector_error(P, R, gb_sites)
    print(f"Normal vector RMS error: {rms_error:.4f} radians ({np.degrees(rms_error):.2f} degrees)")
    print(f"Normal vector maximum error: {max_error:.4f} radians ({np.degrees(max_error):.2f} degrees)")
    print(f"Normal vector calculation time: {ac.running_time:.2f}s")
    
    # Test curvature 
    print("\nTesting curvature...")
    ac = allenCahn_class(nx, ny, ng, cores, nsteps, P0, R,
                        clip=alg_params['clip'],
                        verification_system=alg_params['verification_system'],
                        curvature_sign=alg_params['curvature_sign'])
    ac.allenCahn_main(purpose="curvature")
    C = ac.get_C()
    
    # Calculate curvature errors
    rms_error, max_error, avg_k, std_k = calculate_curvature_error(C, center, radius, gb_sites)
    theoretical_k = 1.0/radius
    print(f"Theoretical curvature: {theoretical_k:.6f}")
    print(f"Average calculated curvature: {avg_k:.6f} ± {std_k:.6f}")
    print(f"Curvature RMS error: {rms_error:.6f}")
    print(f"Curvature maximum error: {max_error:.6f}")
    print(f"Curvature calculation time: {ac.running_time:.2f}s")
    
    # Visualize results
    plt.figure(figsize=PLOT_CONFIG['figure_size'])
    
    # Plot initial condition
    plt.subplot(131)
    plt.imshow(P0[:,:,0], cmap=PLOT_CONFIG['colormap_2d'])
    plt.title("Initial Condition")
    
    # Plot normal vectors
    plt.subplot(132) 
    plt.imshow(P[0,:,:], cmap=PLOT_CONFIG['colormap_2d'])
    plt.title("Normal Vectors")
    
    # Add vector arrows at grain boundary
    for i,j in gb_sites:
        dx, dy = myInput.get_grad(P, i, j)
        plt.arrow(j, i, 
                 PLOT_CONFIG['vector_scale']*dx,
                 PLOT_CONFIG['vector_scale']*dy,
                 color=PLOT_CONFIG['vector_color'],
                 width=PLOT_CONFIG['vector_width'],
                 alpha=PLOT_CONFIG['vector_alpha'])
    
    # Plot curvature
    plt.subplot(133)
    plt.imshow(C[1,:,:], cmap=PLOT_CONFIG['curvature_colormap'])
    plt.colorbar()
    plt.title(f"Curvature (Theory: {theoretical_k:.4f})")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'allencahn_circle_test.png'))
    plt.close()

def test_voronoi():
    """Test algorithm on more complex Voronoi grain structure."""
    print("\nTesting Voronoi configuration...")
    
    # Setup parameters
    cfg = CONFIG_2D['voronoi_test']
    alg_params = ALGORITHM_PARAMS['allen_cahn']
    nx, ny = cfg['grid_size']
    ng = cfg['num_grains']
    cores = cfg['num_cores']
    nsteps = cfg['num_steps']
    
    # Generate Voronoi initial condition
    P0, R = myInput.Voronoi_IC(nx, ny, ng)
    
    # Test normal vectors
    print("\nTesting normal vectors...")
    ac = allenCahn_class(nx, ny, ng, cores, nsteps, P0, R,
                            clip=alg_params['clip'],
                            verification_system=alg_params['verification_system'],
                            curvature_sign=alg_params['curvature_sign'])
    ac.allenCahn_main(purpose="inclination")
    P = ac.get_P()
    gb_sites = ac.get_gb_list()
    
    # Calculate normal vector errors
    rms_error, max_error = calculate_normal_vector_error(P, R, gb_sites)
    print(f"Normal vector RMS error: {rms_error:.4f} radians ({np.degrees(rms_error):.2f} degrees)")
    print(f"Normal vector maximum error: {max_error:.4f} radians ({np.degrees(max_error):.2f} degrees)")
    print(f"Normal vector calculation time: {ac.running_time:.2f}s")
    
    # Test curvature
    print("\nTesting curvature...")
    ac = allenCahn_class(nx, ny, ng, cores, nsteps, P0, R,
                            clip=alg_params['clip'],
                            verification_system=alg_params['verification_system'],
                            curvature_sign=alg_params['curvature_sign'])
    ac.allenCahn_main(purpose="curvature")
    C = ac.get_C()
    
    # For Voronoi, we can't compute theoretical curvature,
    # but we can report statistics
    curvatures = []
    for i, j in gb_sites:
        curvatures.append(C[1,i,j])
    curvatures = np.array(curvatures)
    
    print(f"Average curvature: {np.mean(curvatures):.6f} ± {np.std(curvatures):.6f}")
    print(f"Curvature range: [{np.min(curvatures):.6f}, {np.max(curvatures):.6f}]")
    print(f"Curvature calculation time: {ac.running_time:.2f}s")
    
    # Visualize results
    plt.figure(figsize=PLOT_CONFIG['figure_size'])
    
    plt.subplot(131)
    plt.imshow(P0[:,:,0], cmap=PLOT_CONFIG['colormap_2d'])
    plt.title("Initial Condition")
    
    plt.subplot(132)
    plt.imshow(P[0,:,:], cmap=PLOT_CONFIG['colormap_2d'])
    plt.title("Normal Vectors")
    
    gb_sites = ac.get_gb_list()
    for i,j in gb_sites:
        dx, dy = myInput.get_grad(P, i, j)
        plt.arrow(j, i,
                 PLOT_CONFIG['vector_scale']*dx,
                 PLOT_CONFIG['vector_scale']*dy,
                 color=PLOT_CONFIG['vector_color'],
                 width=PLOT_CONFIG['vector_width'],
                 alpha=PLOT_CONFIG['vector_alpha'])
    
    plt.subplot(133)
    plt.imshow(C[1,:,:], cmap=PLOT_CONFIG['curvature_colormap'])
    plt.colorbar()
    plt.title("Curvature")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'allencahn_voronoi_test.png'))
    plt.close()

def test_convergence():
    """Test convergence behavior with increasing iterations."""
    print("\nTesting convergence...")
    
    # Setup base parameters
    cfg = CONFIG_2D['convergence_test']
    alg_params = ALGORITHM_PARAMS['allen_cahn']
    nx, ny = cfg['grid_size']
    ng = cfg['num_grains']
    cores = cfg['num_cores']
    radius = cfg['circle_radius']
    center = (nx//2, ny//2)
    step_range = cfg['step_range']
    
    # Generate circle initial condition
    P0, R = myInput.Circle_IC(nx, ny, r=radius)
    
    # Test range of iteration steps
    step_range = range(2, 21, 2)
    normal_errors = []
    curvature_errors = []
    times = []
    
    for nsteps in step_range:
        print(f"\nTesting {nsteps} steps...")
        
        # Test normal vectors
        ac = allenCahn_class(nx, ny, ng, cores, nsteps, P0, R,
                                clip=alg_params['clip'],
                                verification_system=alg_params['verification_system'],
                                curvature_sign=alg_params['curvature_sign'])
        ac.allenCahn_main(purpose="inclination")
        P = ac.get_P()
        gb_sites = ac.get_gb_list()
        rms_error, _ = calculate_normal_vector_error(P, R, gb_sites)
        normal_errors.append(rms_error)
        
        # Test curvature
        ac = allenCahn_class(nx, ny, ng, cores, nsteps, P0, R,
                                clip=alg_params['clip'],
                                verification_system=alg_params['verification_system'],
                                curvature_sign=alg_params['curvature_sign'])
        ac.allenCahn_main(purpose="curvature")
        C = ac.get_C()
        rms_error, _, _, _ = calculate_curvature_error(C, center, radius, gb_sites)
        curvature_errors.append(rms_error)
        
        times.append(ac.running_time)
        
        print(f"Normal vector RMS error: {normal_errors[-1]:.4f} radians")
        print(f"Curvature RMS error: {curvature_errors[-1]:.6f}")
        print(f"Running time: {times[-1]:.2f}s")
    
    # Calculate convergence rates
    normal_reduction = (normal_errors[0] - normal_errors[-1])/normal_errors[0] * 100
    curvature_reduction = (curvature_errors[0] - curvature_errors[-1])/curvature_errors[0] * 100
    
    print(f"\nFinal convergence results:")
    print(f"Normal vector error reduced by {normal_reduction:.1f}%")
    print(f"Curvature error reduced by {curvature_reduction:.1f}%")
    
    # Plot convergence results
    plt.figure(figsize=PLOT_CONFIG['figure_size'])
    
    plt.subplot(131)
    plt.plot(list(step_range), normal_errors, 'bo-')
    plt.xlabel("Number of Steps")
    plt.ylabel("Normal Vector RMS Error (rad)")
    plt.title("Normal Vector Convergence")
    
    plt.subplot(132)
    plt.plot(list(step_range), curvature_errors, 'go-')
    plt.xlabel("Number of Steps")
    plt.ylabel("Curvature RMS Error")
    plt.title("Curvature Convergence")
    
    plt.subplot(133)
    plt.plot(list(step_range), times, 'ro-')
    plt.xlabel("Number of Steps")
    plt.ylabel("Running Time (s)")
    plt.title("Computational Cost")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'allencahn_convergence_test.png'))
    plt.close()

if __name__ == '__main__':
    print("Running Allen-Cahn algorithm tests...")
    
    test_circle()
    test_voronoi()
    test_convergence()
    
    print("\nAll tests complete. Check output directory for result visualizations.")