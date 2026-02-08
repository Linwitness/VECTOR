#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Grain Boundary Normal Vector Distribution Analysis

This script consolidates all plot_normal_distribution_over_time_*.py variants
into a single parameterized module. Supports 2D/3D polycrystalline and circular
geometries with local and HiPerGator environments.

Usage:
    # As a module
    from plot_normal_distribution import NormalDistributionAnalyzer
    analyzer = NormalDistributionAnalyzer(geometry='poly2d', environment='local')
    analyzer.run()

    # Command line
    python plot_normal_distribution.py --geometry poly2d --environment local

Created: 2023-07-31
Author: Lin Yang
Refactored: 2026-02-08 (Phase 7 consolidation)
"""

import os
import sys
import argparse
import numpy as np
from numpy import seterr
seterr(all='raise')
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

# Setup paths
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
sys.path.insert(0, os.path.join(current_path, '../..'))
sys.path.insert(0, os.path.join(current_path, '../calculate_tangent'))

import myInput
import post_processing


# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

# Environment-specific base paths
ENV_PATHS = {
    'local': {
        'poly2d': '/Users/lin/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_poly_for_GG/results/',
        'poly2d_512': '/Users/lin.yang/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_poly_multiCoreCompare/results/',
        'circle': '/Users/lin.yang/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_circle_multiCoreCompare/results/',
        'poly3d': '/Users/lin/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/3d_poly_for_GG/results/',
    },
    'hipergator': {
        'poly2d': '/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/',
        'poly2d_20k': '/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_20k_aveE/results/',
        'poly3d': '/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_poly/results/',
        'sphere3d': '/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_sphere/results/',
    }
}

# Energy function types
ENERGY_TYPES = ['ave', 'min', 'max', 'sum', 'consMin', 'consMax', 'iso']

# Anisotropy levels (sigma values)
SIGMA_LEVELS = ['0.0', '0.2', '0.4', '0.6', '0.8', '0.95']


# =============================================================================
# DATA CONFIGURATION BUILDERS
# =============================================================================

def build_poly2d_config(environment='local', num_grains=20000, delta=0.6):
    """Build configuration for 2D polycrystalline analysis."""
    base_path = ENV_PATHS.get(environment, {}).get('poly2d', '')
    multicore = 'multiCore64' if environment == 'hipergator' else 'multiCore32'

    config = {
        'base_path': base_path,
        'num_grains': num_grains,
        'dimension': 2,
        'cases': {}
    }

    for etype in ENERGY_TYPES:
        if etype == 'iso':
            fname = f"p_ori_ave_aveE_{num_grains}_{multicore}_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
        else:
            fname = f"p_ori_ave_{etype}E_{num_grains}_{multicore}_delta{delta}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
        config['cases'][etype] = {
            'file': fname,
            'label': etype.capitalize() if etype != 'iso' else 'Iso',
            'step': 11 if etype not in ['min', 'max', 'iso'] else (30 if etype == 'min' else 10)
        }

    return config


def build_circle_config(environment='local'):
    """Build configuration for 2D circular grain analysis."""
    base_path = ENV_PATHS.get(environment, {}).get('circle', '')

    config = {
        'base_path': base_path,
        'num_grains': 2,
        'dimension': 2,
        'cases': {}
    }

    for sigma in SIGMA_LEVELS:
        sigma_key = sigma.replace('.', '')
        prefix = 'cT' if float(sigma) >= 0.8 else 'c'
        fname = f"{prefix}_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta{sigma}_m2_refer_1_0_0.npy"
        config['cases'][sigma_key] = {
            'file': fname,
            'label': rf"$\sigma={sigma}$",
            'step': 30
        }

    return config


def build_poly3d_config(environment='local', num_grains=20000):
    """Build configuration for 3D polycrystalline analysis."""
    base_path = ENV_PATHS.get(environment, {}).get('poly3d', '')

    config = {
        'base_path': base_path,
        'num_grains': num_grains,
        'dimension': 3,
        'cases': {}
    }

    for etype in ['ave', 'min', 'sum', 'iso']:
        if etype == 'iso':
            fname = f"p_ori_ave_aveE_100_20k_multiCore64_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
        else:
            fname = f"p_ori_ave_{etype}E_100_20k_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
        config['cases'][etype] = {
            'file': fname,
            'label': f"{etype.capitalize()} case",
            'step': 2
        }

    return config


def build_sphere3d_config(environment='hipergator'):
    """Build configuration for 3D spherical grain analysis."""
    base_path = ENV_PATHS.get(environment, {}).get('sphere3d', '')

    config = {
        'base_path': base_path,
        'num_grains': 2,
        'dimension': 3,
        'cases': {}
    }

    for sigma in SIGMA_LEVELS:
        sigma_key = sigma.replace('.', '')
        fname = f"sphere_ori_aveE_000_000_multiCore16_delta{sigma}_seed56689.npy"
        config['cases'][sigma_key] = {
            'file': fname,
            'label': rf"$\sigma={sigma}$",
            'step': 30
        }

    return config


def build_poly2d_sigma_config(environment='local', num_grains=512):
    """Build configuration for 2D poly with varying sigma levels."""
    base_path = ENV_PATHS.get(environment, {}).get('poly2d_512', '')

    config = {
        'base_path': base_path,
        'num_grains': num_grains,
        'dimension': 2,
        'cases': {}
    }

    # Different steps for different sigma values based on grain evolution
    step_map = {'00': 89, '02': 75, '04': 116, '06': 106, '08': 105, '095': 64}

    for sigma in SIGMA_LEVELS:
        sigma_key = sigma.replace('.', '').replace('0.', '0')
        if len(sigma_key) == 1:
            sigma_key = '0' + sigma_key
        multicore = 'multiCore8' if sigma == '0.6' else 'multiCore16'
        fname = f"p_ori_ave_aveE_{num_grains}_{multicore}_delta{sigma}_m2_J1_refer_1_0_0_seed56689_kt066.npy"
        config['cases'][sigma_key] = {
            'file': fname,
            'label': rf"$\sigma={sigma}$",
            'step': step_map.get(sigma_key, 89)
        }

    return config


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class NormalDistributionAnalyzer:
    """Unified analyzer for grain boundary normal vector distributions."""

    def __init__(self, geometry='poly2d', environment='local', config=None, **kwargs):
        """Initialize the analyzer.

        Args:
            geometry: One of 'poly2d', 'poly2d_sigma', 'circle', 'poly3d', 'sphere3d'
            environment: 'local' or 'hipergator'
            config: Optional custom configuration dict
            **kwargs: Additional parameters passed to config builder
        """
        self.geometry = geometry
        self.environment = environment
        self.current_path = current_path

        # Build or use provided configuration
        if config:
            self.config = config
        else:
            self.config = self._build_config(geometry, environment, **kwargs)

        self.cache_dir = os.path.join(current_path, 'normal_distribution_data')
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(current_path, 'figures'), exist_ok=True)

        self.data = {}
        self.results = {}

    def _build_config(self, geometry, environment, **kwargs):
        """Build configuration based on geometry type."""
        builders = {
            'poly2d': build_poly2d_config,
            'poly2d_sigma': build_poly2d_sigma_config,
            'circle': build_circle_config,
            'poly3d': build_poly3d_config,
            'sphere3d': build_sphere3d_config,
        }
        builder = builders.get(geometry, build_poly2d_config)
        return builder(environment=environment, **kwargs)

    def load_data(self, cases=None):
        """Load microstructure data for specified cases.

        Args:
            cases: List of case keys to load, or None for all
        """
        cases = cases or list(self.config['cases'].keys())
        base_path = self.config['base_path']

        for case_key in cases:
            case_cfg = self.config['cases'].get(case_key)
            if not case_cfg:
                continue
            filepath = os.path.join(base_path, case_cfg['file'])
            if os.path.exists(filepath):
                self.data[case_key] = np.load(filepath)
                print(f"Loaded {case_key}: shape {self.data[case_key].shape}")
            else:
                print(f"Warning: File not found: {filepath}")

        print("Data loading complete.")

    def compute_normal_vectors(self, case_key, step=None):
        """Compute or load cached normal vectors for a case.

        Args:
            case_key: Case identifier
            step: Timestep (uses config default if None)

        Returns:
            tuple: (P, sites) - Smoothed field and boundary sites
        """
        case_cfg = self.config['cases'].get(case_key)
        if not case_cfg or case_key not in self.data:
            return None, None

        step = step if step is not None else case_cfg['step']
        data = self.data[case_key]

        # Prepare microstructure
        newplace = np.rot90(data[step, :, :, :], 1, (0, 1))

        # Select appropriate normal vector function
        if self.config['dimension'] == 3:
            compute_func = post_processing.get_normal_vector_3d
            prefix = f"{self.geometry}_{case_key}_3d"
        else:
            compute_func = post_processing.get_normal_vector
            prefix = f"{self.geometry}_{case_key}"

        P, sites = post_processing.load_or_compute_normal_vectors(
            self.cache_dir, prefix, step, compute_func, newplace
        )

        return P, sites

    def plot_polar_distribution(self, cases=None, output_name=None, bias_case=None,
                                 r_max=0.008, r_tick=0.004, theta_tick=45.0,
                                 angle_index=0, show_legend=True):
        """Generate polar distribution plot.

        Args:
            cases: List of case keys to include
            output_name: Output filename (without extension)
            bias_case: Case key to use for bias correction (None = no correction)
            r_max, r_tick, theta_tick: Polar plot parameters
            angle_index: For 3D, projection plane (0=xy, 1=xz, 2=yz)
            show_legend: Whether to show legend

        Returns:
            dict: Anisotropic magnitude results for each case
        """
        cases = cases or list(self.config['cases'].keys())
        fig, ax = post_processing.setup_polar_figure(r_max, r_tick, theta_tick, fontsize=16)

        # Compute bias if specified
        slope_bias = None
        if bias_case and bias_case in cases:
            P, sites = self.compute_normal_vectors(bias_case)
            if P is not None:
                case_cfg = self.config['cases'][bias_case]
                if self.config['dimension'] == 3:
                    freq = post_processing.get_normal_vector_slope_3d(
                        P, sites, case_cfg['step'], case_cfg['label'], angle_index
                    )
                else:
                    freq = post_processing.get_normal_vector_slope(
                        P, sites, case_cfg['step'], case_cfg['label']
                    )
                slope_bias = post_processing.compute_bias(freq)
                plt.cla()  # Clear the bias reference line
                fig, ax = post_processing.setup_polar_figure(r_max, r_tick, theta_tick, fontsize=16)

        results = {}
        for case_key in cases:
            P, sites = self.compute_normal_vectors(case_key)
            if P is None:
                continue

            case_cfg = self.config['cases'][case_key]
            step = case_cfg['step']
            label = case_cfg['label']

            if self.config['dimension'] == 3:
                freq = post_processing.get_normal_vector_slope_3d(
                    P, sites, step, label, angle_index, slope_bias
                )
            else:
                freq = post_processing.get_normal_vector_slope(
                    P, sites, step, label, slope_bias
                )

            mag_ave, mag_std = post_processing.simple_magnitude(freq)
            results[case_key] = {'magnitude': mag_ave, 'std': mag_std, 'freq': freq}

        if show_legend:
            plt.legend(loc='best', fontsize=14)

        # Add axis labels for 3D
        if self.config['dimension'] == 3:
            axis_labels = [('x', 'y'), ('x', 'z'), ('y', 'z')]
            plt.text(0.0, r_max * 1.1, axis_labels[angle_index][0], fontsize=14)
            plt.text(np.pi/2, r_max * 1.1, axis_labels[angle_index][1], fontsize=14)

        if output_name:
            outpath = os.path.join(current_path, 'figures', f'{output_name}.png')
            plt.savefig(outpath, dpi=400, bbox_inches='tight')
            print(f"Saved: {outpath}")

        self.results = results
        return results

    def plot_magnitude_comparison(self, output_name=None):
        """Plot anisotropic magnitude comparison across sigma values.

        Args:
            output_name: Output filename (without extension)
        """
        if not self.results:
            print("No results available. Run plot_polar_distribution first.")
            return

        plt.close()
        fig = plt.figure(figsize=(5, 5))

        # Extract magnitudes in order
        keys = sorted(self.results.keys())
        magnitudes = [self.results[k]['magnitude'] for k in keys]
        stds = [self.results[k]['std'] for k in keys]

        # Try to extract sigma values from keys
        try:
            x_values = [float('0.' + k) if k.isdigit() else float(k) for k in keys]
        except ValueError:
            x_values = list(range(len(keys)))

        plt.errorbar(x_values, magnitudes, yerr=stds,
                    linestyle='None', marker='None', color='black', linewidth=1, capsize=2)
        plt.plot(x_values, magnitudes, '.-', markersize=8, linewidth=2)

        plt.xlabel(r"$\sigma$", fontsize=16)
        plt.ylabel("Anisotropic Magnitude", fontsize=16)
        plt.ylim([-0.05, max(magnitudes) * 1.2 + 0.1])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        if output_name:
            outpath = os.path.join(current_path, 'figures', f'{output_name}.png')
            plt.savefig(outpath, dpi=400, bbox_inches='tight')
            print(f"Saved: {outpath}")

    def run(self, cases=None, bias_case=None, output_prefix=None):
        """Run complete analysis pipeline.

        Args:
            cases: List of cases to analyze (None = all)
            bias_case: Case for bias correction
            output_prefix: Prefix for output filenames
        """
        prefix = output_prefix or f"normal_distribution_{self.geometry}"

        # Load data
        self.load_data(cases)

        # Generate polar plot
        self.plot_polar_distribution(
            cases=cases,
            output_name=prefix,
            bias_case=bias_case
        )

        # For 3D, also generate XZ and YZ projections
        if self.config['dimension'] == 3:
            for idx, plane in enumerate(['xy', 'xz', 'yz']):
                self.plot_polar_distribution(
                    cases=cases,
                    output_name=f"{prefix}_{plane}",
                    bias_case=bias_case,
                    angle_index=idx
                )

        # Generate magnitude comparison if sigma-based analysis
        if self.geometry in ['circle', 'poly2d_sigma', 'sphere3d']:
            self.plot_magnitude_comparison(output_name=f"{prefix}_magnitude")

        print(f"\n{'='*60}")
        print(f"{self.geometry.upper()} ANALYSIS COMPLETED")
        print(f"{'='*60}")


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPER
# =============================================================================

def run_analysis(geometry='poly2d', environment='local', **kwargs):
    """Wrapper function for backward compatibility.

    Args:
        geometry: Analysis geometry type
        environment: 'local' or 'hipergator'
        **kwargs: Additional parameters

    Returns:
        NormalDistributionAnalyzer: Configured analyzer instance
    """
    analyzer = NormalDistributionAnalyzer(geometry=geometry, environment=environment, **kwargs)
    analyzer.run(**kwargs)
    return analyzer


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified grain boundary normal vector distribution analysis'
    )
    parser.add_argument('--geometry', '-g',
                       choices=['poly2d', 'poly2d_sigma', 'circle', 'poly3d', 'sphere3d'],
                       default='poly2d',
                       help='Geometry type for analysis')
    parser.add_argument('--environment', '-e',
                       choices=['local', 'hipergator'],
                       default='local',
                       help='Computing environment')
    parser.add_argument('--cases', '-c', nargs='+',
                       help='Specific cases to analyze')
    parser.add_argument('--bias-case', '-b',
                       help='Case to use for bias correction')
    parser.add_argument('--output', '-o',
                       help='Output filename prefix')

    args = parser.parse_args()

    analyzer = NormalDistributionAnalyzer(
        geometry=args.geometry,
        environment=args.environment
    )
    analyzer.run(
        cases=args.cases,
        bias_case=args.bias_case,
        output_prefix=args.output
    )


if __name__ == '__main__':
    main()
