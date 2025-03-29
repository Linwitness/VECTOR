#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Results manager for smoothing algorithm verification tests.
Handles collection, storage, and comparison of test results.

Author: Lin Yang
"""

import os
import json
import numpy as np
from datetime import datetime

class ResultsManager:
    """Manages test results and generates comparisons."""
    
    def __init__(self, output_dir):
        """Initialize results manager.
        
        Args:
            output_dir: Directory to store results and reports
        """
        self.output_dir = output_dir
        self.results = {
            '2d': {
                'linear': {},
                'vertex': {},
                'levelset': {},
            },
            '3d': {
                'linear': {},
                'vertex': {},
                'levelset': {},
            }
        }
        
    def add_result(self, dimension, algorithm, test_type, metrics):
        """Add test result to storage.
        
        Args:
            dimension: '2d' or '3d'
            algorithm: 'linear', 'vertex', or 'levelset'
            test_type: 'circle', 'sphere', 'voronoi', or 'convergence'
            metrics: Dictionary of test metrics
        """
        if test_type not in self.results[dimension][algorithm]:
            self.results[dimension][algorithm][test_type] = []
        
        # Add timestamp to metrics
        metrics['timestamp'] = datetime.now().isoformat()
        self.results[dimension][algorithm][test_type].append(metrics)
    
    def save_results(self):
        """Save current results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return filepath
    
    def load_results(self, filepath):
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            self.results = json.load(f)
    
    def generate_comparison_report(self):
        """Generate detailed comparison report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"comparison_report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("Smoothing Algorithm Comparison Report\n")
            f.write("==================================\n\n")
            
            # 2D Results
            f.write("2D Algorithm Results\n")
            f.write("-------------------\n")
            
            # Circle Tests
            f.write("\nCircle Test Results:\n")
            for algo in ['linear', 'vertex', 'levelset']:
                if 'circle' in self.results['2d'][algo]:
                    latest = self.results['2d'][algo]['circle'][-1]
                    f.write(f"\n{algo.capitalize()}:\n")
                    f.write(f"  Normal Vector RMS Error: {latest['normal_error']:.4f} rad ")
                    f.write(f"({np.degrees(latest['normal_error']):.2f} deg)\n")
                    f.write(f"  Curvature RMS Error: {latest['curvature_error']:.6f}\n")
                    f.write(f"  Computation Time: {latest['time']:.2f}s\n")
            
            # Voronoi Tests
            f.write("\nVoronoi Test Results:\n")
            for algo in ['linear', 'vertex', 'levelset']:
                if 'voronoi' in self.results['2d'][algo]:
                    latest = self.results['2d'][algo]['voronoi'][-1]
                    f.write(f"\n{algo.capitalize()}:\n")
                    f.write(f"  Normal Vector RMS Error: {latest['normal_error']:.4f} rad ")
                    f.write(f"({np.degrees(latest['normal_error']):.2f} deg)\n")
                    f.write(f"  Average Curvature: {latest['avg_curvature']:.6f} ± ")
                    f.write(f"{latest['std_curvature']:.6f}\n")
                    f.write(f"  Computation Time: {latest['time']:.2f}s\n")
            
            # 3D Results
            f.write("\n\n3D Algorithm Results\n")
            f.write("-------------------\n")
            
            # Sphere Tests
            f.write("\nSphere Test Results:\n")
            for algo in ['linear', 'vertex', 'levelset']:
                if 'sphere' in self.results['3d'][algo]:
                    latest = self.results['3d'][algo]['sphere'][-1]
                    f.write(f"\n{algo.capitalize()}:\n")
                    f.write(f"  Normal Vector RMS Error: {latest['normal_error']:.4f} rad ")
                    f.write(f"({np.degrees(latest['normal_error']):.2f} deg)\n")
                    f.write(f"  Curvature RMS Error: {latest['curvature_error']:.6f}\n")
                    f.write(f"  Computation Time: {latest['time']:.2f}s\n")
            
            # 3D Voronoi Tests
            f.write("\n3D Voronoi Test Results:\n")
            for algo in ['linear', 'vertex', 'levelset']:
                if 'voronoi' in self.results['3d'][algo]:
                    latest = self.results['3d'][algo]['voronoi'][-1]
                    f.write(f"\n{algo.capitalize()}:\n")
                    f.write(f"  Normal Vector RMS Error: {latest['normal_error']:.4f} rad ")
                    f.write(f"({np.degrees(latest['normal_error']):.2f} deg)\n")
                    f.write(f"  Average Curvature: {latest['avg_curvature']:.6f} ± ")
                    f.write(f"{latest['std_curvature']:.6f}\n")
                    f.write(f"  Computation Time: {latest['time']:.2f}s\n")
            
            # Convergence Analysis
            f.write("\n\nConvergence Analysis\n")
            f.write("-------------------\n")
            
            for dim in ['2d', '3d']:
                f.write(f"\n{dim.upper()} Convergence Results:\n")
                for algo in ['linear', 'vertex', 'levelset']:
                    if 'convergence' in self.results[dim][algo]:
                        latest = self.results[dim][algo]['convergence'][-1]
                        f.write(f"\n{algo.capitalize()}:\n")
                        f.write(f"  Normal Vector Error Reduction: {latest['normal_reduction']:.1f}%\n")
                        f.write(f"  Curvature Error Reduction: {latest['curvature_reduction']:.1f}%\n")
                        f.write(f"  Final Computation Time: {latest['final_time']:.2f}s\n")
        
        return report_path
    
    def get_algorithm_ranking(self):
        """Generate algorithm rankings based on accuracy and performance."""
        rankings = {
            'accuracy': {
                '2d': {},
                '3d': {}
            },
            'performance': {
                '2d': {},
                '3d': {}
            }
        }
        
        # Calculate rankings for each dimension
        for dim in ['2d', '3d']:
            test_type = 'circle' if dim == '2d' else 'sphere'
            
            # Collect metrics for each algorithm
            accuracy_metrics = []
            performance_metrics = []
            
            for algo in ['linear', 'vertex', 'levelset']:
                if test_type in self.results[dim][algo]:
                    latest = self.results[dim][algo][test_type][-1]
                    
                    # Combine normal vector and curvature errors for accuracy
                    accuracy = np.mean([
                        latest['normal_error'],
                        latest['curvature_error']
                    ])
                    accuracy_metrics.append((algo, accuracy))
                    
                    # Use computation time for performance
                    performance_metrics.append((algo, latest['time']))
            
            # Sort by metrics (lower is better)
            accuracy_metrics.sort(key=lambda x: x[1])
            performance_metrics.sort(key=lambda x: x[1])
            
            # Store rankings
            rankings['accuracy'][dim] = [x[0] for x in accuracy_metrics]
            rankings['performance'][dim] = [x[0] for x in performance_metrics]
        
        return rankings