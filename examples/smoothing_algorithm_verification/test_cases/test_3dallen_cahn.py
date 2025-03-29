#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test cases for 3D Allen-Cahn smoothing algorithm.
Author: Lin Yang
"""
import os
import sys
current_path = os.getcwd()
sys.path.append(current_path + '/../../')

import numpy as np
import matplotlib.pyplot as plt
import myInput
from PACKAGE_MP_3DAllenCahn import allenCahn3d_class
from test_config import CONFIG_3D, ALGORITHM_PARAMS

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

def test_sphere():
    """Test normal vector and curvature calculation on a sphere."""
    print("\nTesting sphere configuration...")
    
    # Get parameters from config
    cfg = CONFIG_3D['sphere_test']
    alg_params = ALGORITHM_PARAMS['3dallen_cahn']
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
    allencahn3d = allenCahn3d_class(nx, ny, nz, ng, cores, nsteps, P0, R,
                                alg_params['boundary_condition'],
                                clip=alg_params['clip'],
                                verification_system=alg_params['verification_system'],
                                curvature_sign=alg_params['curvature_sign'])
    allencahn3d.allenCahn3d_main(purpose="inclination")
    P = allencahn3d.get_P()
    gb_sites = allencahn3d.get_gb_list()
    
    # Calculate normal vector errors
    rms_error, max_error = calculate_normal_vector_error(P, R, gb_sites)
    print(f"Normal vector RMS error: {rms_error:.4f} radians ({np.degrees(rms_error):.2f} degrees)")
    print(f"Normal vector maximum error: {max_error:.4f} radians ({np.degrees(max_error):.2f} degrees)")
    print(f"Normal vector calculation time: {allencahn3d.running_time:.2f}s")
    
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
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'allencahn3d_sphere_test.png'))
    plt.close()

def test_convergence():
    """Test convergence behavior with increasing iterations."""
    print("\nTesting convergence...")
    
    # Get parameters from config
    cfg = CONFIG_3D['convergence_test']
    alg_params = ALGORITHM_PARAMS['3dallen_cahn']
    nx, ny, nz = cfg['grid_size']
    ng = cfg['num_grains']
    cores = cfg['num_cores']
    radius = cfg['sphere_radius']
    step_range = cfg['step_range']
    
    # Generate sphere initial condition
    P0, R = myInput.Circle_IC3d(nx, ny, nz, radius)
    
    # Test range of iteration steps
    normal_errors = []
    times = []
    
    for nsteps in step_range:
        print(f"\nTesting {nsteps} steps...")
        
        # Test normal vectors
        allencahn3d = allenCahn3d_class(nx, ny, nz, ng, cores, nsteps, P0, R,
                                    alg_params['boundary_condition'],
                                    clip=alg_params['clip'],
                                    verification_system=alg_params['verification_system'],
                                    curvature_sign=alg_params['curvature_sign'])
        allencahn3d.allenCahn3d_main(purpose="inclination")
        P = allencahn3d.get_P()
        gb_sites = allencahn3d.get_gb_list()
        rms_error, _ = calculate_normal_vector_error(P, R, gb_sites)
        normal_errors.append(rms_error)
        
        times.append(allencahn3d.running_time)
        
        print(f"Normal vector RMS error: {normal_errors[-1]:.4f} radians")
        print(f"Running time: {times[-1]:.2f}s")
    
    # Calculate convergence rates
    normal_reduction = (normal_errors[0] - normal_errors[-1])/normal_errors[0] * 100
    
    print(f"\nFinal convergence results:")
    print(f"Normal vector error reduced by {normal_reduction:.1f}%")
    
    # Plot convergence results
    plt.figure(figsize=PLOT_CONFIG['figure_size'])
    
    plt.subplot(131)
    plt.plot(list(step_range), normal_errors, 'bo-')
    plt.xlabel("Number of Steps")
    plt.ylabel("Normal Vector RMS Error (rad)")
    plt.title("Normal Vector Convergence")
    
    plt.subplot(133)
    plt.plot(list(step_range), times, 'ro-')
    plt.xlabel("Number of Steps")
    plt.ylabel("Running Time (s)")
    plt.title("Computational Cost")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'allencahn3d_convergence_test.png'))
    plt.close()

if __name__ == '__main__':
    print("Running 3D Allen-Cahn smoothing algorithm tests...")
    
    test_sphere()
    test_convergence()
    
    print("\nAll tests complete. Check output directory for result visualizations.")