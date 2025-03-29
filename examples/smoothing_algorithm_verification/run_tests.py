#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main test runner for smoothing algorithm verification.
Executes all algorithm tests and generates comparative analysis.

Author: Lin Yang
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import test modules
from test_cases.test_linear import test_circle as test_linear_circle
from test_cases.test_linear import test_voronoi as test_linear_voronoi
from test_cases.test_linear import test_convergence as test_linear_convergence

from test_cases.test_3dlinear import test_sphere as test_3dlinear_sphere
from test_cases.test_3dlinear import test_convergence as test_3dlinear_convergence

from test_cases.test_vertex import test_circle as test_vertex_circle
from test_cases.test_vertex import test_voronoi as test_vertex_voronoi
from test_cases.test_vertex import test_convergence as test_vertex_convergence

from test_cases.test_3dvertex import test_sphere as test_3dvertex_sphere
from test_cases.test_3dvertex import test_convergence as test_3dvertex_convergence

from test_cases.test_levelset import test_circle as test_levelset_circle
from test_cases.test_levelset import test_voronoi as test_levelset_voronoi
from test_cases.test_levelset import test_convergence as test_levelset_convergence

from test_cases.test_3dlevelset import test_sphere as test_3dlevelset_sphere
from test_cases.test_3dlevelset import test_convergence as test_3dlevelset_convergence

from test_cases.test_allen_cahn import test_circle as test_allen_cahn_circle
from test_cases.test_allen_cahn import test_voronoi as test_allen_cahn_voronoi
from test_cases.test_allen_cahn import test_convergence as test_allen_cahn_convergence

from test_cases.test_3dallen_cahn import test_sphere as test_3dallen_cahn_sphere
from test_cases.test_3dallen_cahn import test_convergence as test_3dallen_cahn_convergence

def generate_report(results_2d, results_3d):
    """Generate comparative report of all algorithm results."""
    
    report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join("output", f"comparison_report_{report_time}.txt")
    
    with open(report_path, "w") as f:
        # Write header
        f.write("Smoothing Algorithm Comparison Report\n")
        f.write("==================================\n\n")
        
        # 2D Results
        f.write("2D Algorithm Results\n")
        f.write("-------------------\n")
        f.write("\nCircle Test Results:\n")
        for algo, results in results_2d["circle"].items():
            f.write(f"\n{algo}:\n")
            f.write(f"  Normal Vector RMS Error: {results['normal_error']:.4f} rad ({np.degrees(results['normal_error']):.2f} deg)\n")
            f.write(f"  Curvature RMS Error: {results['curvature_error']:.6f}\n")
            f.write(f"  Computation Time: {results['time']:.2f}s\n")
        
        f.write("\nVoronoi Test Results:\n")
        for algo, results in results_2d["voronoi"].items():
            f.write(f"\n{algo}:\n")
            f.write(f"  Normal Vector RMS Error: {results['normal_error']:.4f} rad ({np.degrees(results['normal_error']):.2f} deg)\n")
            f.write(f"  Average Curvature: {results['avg_curvature']:.6f} ± {results['std_curvature']:.6f}\n")
            f.write(f"  Computation Time: {results['time']:.2f}s\n")
        
        f.write("\nConvergence Results:\n")
        for algo, results in results_2d["convergence"].items():
            f.write(f"\n{algo}:\n")
            f.write(f"  Normal Vector Error Reduction: {results['normal_reduction']:.1f}%\n")
            f.write(f"  Curvature Error Reduction: {results['curvature_reduction']:.1f}%\n")
            f.write(f"  Final Computation Time: {results['final_time']:.2f}s\n")
        
        # 3D Results
        f.write("\n\n3D Algorithm Results\n")
        f.write("-------------------\n")
        f.write("\nSphere Test Results:\n")
        for algo, results in results_3d["sphere"].items():
            f.write(f"\n{algo}:\n")
            f.write(f"  Normal Vector RMS Error: {results['normal_error']:.4f} rad ({np.degrees(results['normal_error']):.2f} deg)\n")
            f.write(f"  Curvature RMS Error: {results['curvature_error']:.6f}\n")
            f.write(f"  Computation Time: {results['time']:.2f}s\n")
        
        f.write("\n3D Convergence Results:\n")
        for algo, results in results_3d["convergence"].items():
            f.write(f"\n{algo}:\n")
            f.write(f"  Normal Vector Error Reduction: {results['normal_reduction']:.1f}%\n")
            f.write(f"  Curvature Error Reduction: {results['curvature_reduction']:.1f}%\n")
            f.write(f"  Final Computation Time: {results['final_time']:.2f}s\n")

    print(f"\nComparison report saved to: {report_path}")

def run_all_tests(test_2D=True, test_3D=True):
    """Execute all algorithm tests and collect results."""
    
    print("Starting comprehensive algorithm testing...\n")
    
    if test_2D:
        # Initialize results dictionaries
        results_2d = {
            "circle": {},
            "voronoi": {},
            "convergence": {}
        }

        # 2D Algorithm Tests
        print("Testing 2D Linear Algorithm...")
        test_linear_circle()
        test_linear_voronoi()
        test_linear_convergence()
        
        print("\nTesting 2D Vertex Algorithm...")
        test_vertex_circle()
        test_vertex_voronoi()
        test_vertex_convergence()
        
        print("\nTesting 2D Level Set Algorithm...")
        test_levelset_circle()
        test_levelset_voronoi()
        test_levelset_convergence()
        
        print("\nTesting 2D Allen-Cahn Algorithm...")
        test_allen_cahn_circle()
        test_allen_cahn_voronoi()
        test_allen_cahn_convergence()
    
    if test_3D:
        results_3d = {
            "sphere": {},
            "convergence": {}
        }
        
        # 3D Algorithm Tests
        # print("\nTesting 3D Linear Algorithm...")
        # test_3dlinear_sphere()
        # test_3dlinear_convergence()
        
        print("\nTesting 3D Vertex Algorithm...")
        test_3dvertex_sphere()
        test_3dvertex_convergence()
        
        # print("\nTesting 3D Level Set Algorithm...")
        # test_3dlevelset_sphere()
        # test_3dlevelset_convergence()
        
        # print("\nTesting 3D Allen-Cahn Algorithm...")
        # test_3dallen_cahn_sphere()
        # test_3dallen_cahn_convergence()
    
    # Generate comparison plots
    # plot_comparison_results()
    
    # Generate detailed report
    # generate_report(results_2d, results_3d)

def plot_comparison_results():
    """Create comparative visualization of algorithm results."""
    # This will be populated with actual plotting code based on test results
    plot_path = os.path.join("output", "algorithm_comparison.png")
    
    plt.figure(figsize=(15, 10))
    
    # Placeholder for actual plotting code
    # Will include:
    # - Comparative bar plots of errors
    # - Convergence rate comparisons
    # - Computation time comparisons
    # - 2D vs 3D performance comparisons
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nComparison plots saved to: {plot_path}")

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Record start time
    start_time = time.time()
    
    # Run all tests
    run_all_tests(test_2D=False)
    
    # Report total execution time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal test execution time: {total_time:.2f} seconds")