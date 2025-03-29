#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for 3D Vertex smoothing algorithm implementation.
Tests 3D cases for normal vector and curvature calculations.

Author: Lin Yang
"""

import os
import sys
current_path = os.getcwd()
sys.path.append(current_path + '/../../')

import numpy as np
import matplotlib.pyplot as plt
import myInput
from PACKAGE_MP_3DVertex import vertex3d_class
from . import CONFIG_3D, ALGORITHM_PARAMS, PLOT_CONFIG

# Create output directory for test results
OUTPUT_DIR = os.path.join(current_path, 'output')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def calculate_normal_vector_error(P, R, gb_sites):
    """Calculate error between computed and theoretical normal vectors."""
    angles = []
    for i, j, k in gb_sites:
        dx, dy, dz = myInput.get_grad3d(P, i, j, k)
        calc_vec = np.array([dx, dy, dz])
        ref_vec = np.array([R[i,j,k,0], R[i,j,k,1], R[i,j,k,2]])
        
        # Normalize vectors
        calc_vec = calc_vec / np.linalg.norm(calc_vec)
        ref_vec = ref_vec / np.linalg.norm(ref_vec)
        
        # Calculate angle
        dot_product = np.clip(np.abs(np.dot(calc_vec, ref_vec)), -1.0, 1.0)
        angle = np.arccos(dot_product)
        angles.append(angle)
    
    angles = np.array(angles)
    rms_error = np.sqrt(np.mean(angles**2))
    max_error = np.max(angles)
    
    return rms_error, max_error

def calculate_curvature_error(C, center, radius, gb_sites):
    """Calculate error between computed and theoretical curvature."""
    theoretical = 1.0/radius  # Mean curvature for a sphere
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

def test_sphere():
    """Test normal vector and curvature calculation on a sphere."""
    print("\nTesting sphere configuration...")
    
    # Get parameters from config
    cfg = CONFIG_3D['sphere_test']
    alg_params = ALGORITHM_PARAMS['3dvertex']
    nx, ny, nz = cfg['grid_size']
    ng = cfg['num_grains']
    cores = cfg['num_cores']
    nsteps = cfg['num_steps']
    radius = cfg['sphere_radius']
    center = (nx//2, ny//2, nz//2)
    
    # Generate sphere initial condition
    P0, R = myInput.Circle_IC3d(nx, ny, nz, radius)
    
    # Test normal vectors
    print("\nTesting normal vectors...")
    vertex3d = vertex3d_class(nx, ny, nz, ng, cores, nsteps, P0, R,
                            alg_params['boundary_condition'],
                            clip=alg_params['clip'],
                            verification_system=alg_params['verification_system'],
                            curvature_sign=alg_params['curvature_sign'])
    vertex3d.vertex3d_main(purpose="inclination")
    P = vertex3d.get_P()
    gb_sites = vertex3d.get_gb_list()
    
    # Calculate normal vector errors
    rms_error, max_error = calculate_normal_vector_error(P, R, gb_sites)
    print(f"Normal vector RMS error: {rms_error:.4f} radians ({np.degrees(rms_error):.2f} degrees)")
    print(f"Normal vector maximum error: {max_error:.4f} radians ({np.degrees(max_error):.2f} degrees)")
    print(f"Normal vector calculation time: {vertex3d.running_time:.2f}s")
    
    # Test curvature 
    print("\nTesting curvature...")
    vertex3d = vertex3d_class(nx, ny, nz, ng, cores, nsteps, P0, R,
                            alg_params['boundary_condition'],
                            clip=alg_params['clip'],
                            verification_system=alg_params['verification_system'],
                            curvature_sign=alg_params['curvature_sign'])
    vertex3d.vertex3d_main(purpose="curvature")
    C = vertex3d.get_C()
    
    # Calculate curvature errors
    rms_error, max_error, avg_k, std_k = calculate_curvature_error(C, center, radius, gb_sites)
    theoretical_k = 1.0/radius
    print(f"Theoretical mean curvature: {theoretical_k:.6f}")
    print(f"Average calculated curvature: {avg_k:.6f} ± {std_k:.6f}")
    print(f"Curvature RMS error: {rms_error:.6f}")
    print(f"Curvature maximum error: {max_error:.6f}")
    print(f"Curvature calculation time: {vertex3d.running_time:.2f}s")
    
    # Visualize results (middle slice)
    z_mid = nz//2
    plt.figure(figsize=PLOT_CONFIG['figure_size'])
    
    plt.subplot(131)
    plt.imshow(P0[:,:,z_mid,0], cmap=PLOT_CONFIG['colormap_3d'])
    plt.title(f"Initial Condition (z={z_mid})")
    
    plt.subplot(132)
    plt.imshow(P[0,:,:,z_mid], cmap=PLOT_CONFIG['colormap_3d'])
    plt.title("Normal Vectors")
    
    for i, j, k in gb_sites:
        if k == z_mid:
            dx, dy, dz = myInput.get_grad3d(P, i, j, k)
            plt.arrow(j, i,
                     PLOT_CONFIG['vector_scale']*dx,
                     PLOT_CONFIG['vector_scale']*dy,
                     color=PLOT_CONFIG['vector_color'],
                     width=PLOT_CONFIG['vector_width'],
                     alpha=PLOT_CONFIG['vector_alpha'])
    
    plt.subplot(133)
    plt.imshow(C[1,:,:,z_mid], cmap=PLOT_CONFIG['curvature_colormap'])
    plt.colorbar()
    plt.title(f"Curvature (Theory: {theoretical_k:.4f})")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'vertex3d_sphere_test.png'))
    plt.close()

def test_convergence():
    """Test convergence behavior with increasing iterations."""
    print("\nTesting convergence...")
    
    # Get parameters from config
    cfg = CONFIG_3D['convergence_test']
    alg_params = ALGORITHM_PARAMS['3dvertex']
    nx, ny, nz = cfg['grid_size']
    ng = cfg['num_grains']
    cores = cfg['num_cores']
    radius = cfg['sphere_radius']
    center = (nx//2, ny//2, nz//2)
    step_range = cfg['step_range']
    
    # Generate sphere initial condition
    P0, R = myInput.Circle_IC3d(nx, ny, nz, radius)
    
    # Test range of iteration steps
    normal_errors = []
    curvature_errors = []
    times = []
    
    for nsteps in step_range:
        print(f"\nTesting {nsteps} steps...")
        
        # Test normal vectors
        vertex3d = vertex3d_class(nx, ny, nz, ng, cores, nsteps, P0, R,
                                alg_params['boundary_condition'],
                                clip=alg_params['clip'],
                                verification_system=alg_params['verification_system'],
                                curvature_sign=alg_params['curvature_sign'])
        vertex3d.vertex3d_main(purpose="inclination")
        P = vertex3d.get_P()
        gb_sites = vertex3d.get_gb_list()
        rms_error, _ = calculate_normal_vector_error(P, R, gb_sites)
        normal_errors.append(rms_error)
        
        # Test curvature
        vertex3d = vertex3d_class(nx, ny, nz, ng, cores, nsteps, P0, R,
                                alg_params['boundary_condition'],
                                clip=alg_params['clip'],
                                verification_system=alg_params['verification_system'],
                                curvature_sign=alg_params['curvature_sign'])
        vertex3d.vertex3d_main(purpose="curvature")
        C = vertex3d.get_C()
        rms_error, _, _, _ = calculate_curvature_error(C, center, radius, gb_sites)
        curvature_errors.append(rms_error)
        
        times.append(vertex3d.running_time)
        
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
    plt.savefig(os.path.join(OUTPUT_DIR, 'vertex3d_convergence_test.png'))
    plt.close()

if __name__ == '__main__':
    print("Running 3D Vertex smoothing algorithm tests...")
    
    test_sphere()
    test_convergence()
    
    print("\nAll tests complete. Check output directory for result visualizations.")