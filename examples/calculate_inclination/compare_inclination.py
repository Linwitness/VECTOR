#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Grain Boundary Inclination Comparison Module

This module consolidates inclination comparison functionality for validating
PRIMME grain boundary predictions against SPPARKS Monte Carlo simulations.
Generates comparative polar plots for statistical validation.

Usage:
    # As a module
    from compare_inclination import InclinationComparator
    comparator = InclinationComparator(environment='local')
    comparator.compare_datasets(primme_data, spparks_data, [300, 1600])

    # Command line
    python compare_inclination.py --primme data1.npy --spparks data2.npy --steps 300 1600
    python compare_inclination.py --batch --environment hipergator

Created: 2021-09-30
Author: Lin Yang
Refactored: 2026-02-08 (Phase 8 consolidation)
"""

import os
import sys
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt

# Setup paths
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
sys.path.insert(0, os.path.join(current_path, '../..'))

import myInput
import PACKAGE_MP_Linear as linear2d


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

ENV_PATHS = {
    'local': {
        'input_folder': os.path.join(current_path, 'output/'),
        'output_folder': os.path.join(current_path, 'Images/'),
    },
    'hipergator': {
        'input_folder': '/blue/michael.tonks/share/PRIMME_Inclination_npy_files/',
        'output_folder': '/blue/michael.tonks/share/PRIMME_Inclination_npy_files/figures/',
    }
}


# =============================================================================
# CORE ANALYSIS FUNCTIONS
# =============================================================================

def get_all_gb_list(P0):
    """Identify all grain boundary sites in a 2D microstructure.

    Scans the microstructure to locate sites at grain boundaries by checking
    if neighboring sites belong to different grains. Uses periodic boundary
    conditions for edge handling.

    Args:
        P0 (ndarray): 2D microstructure array with grain IDs

    Returns:
        list: List of [i, j] coordinates of grain boundary sites
    """
    nx, ny = P0.shape
    gb_sites = []

    for i in range(nx):
        for j in range(ny):
            ip, im, jp, jm = myInput.periodic_bc(nx, ny, i, j)

            if (((P0[ip, j] - P0[i, j]) != 0) or
                ((P0[im, j] - P0[i, j]) != 0) or
                ((P0[i, jp] - P0[i, j]) != 0) or
                ((P0[i, jm] - P0[i, j]) != 0)):
                gb_sites.append([i, j])

    return gb_sites


def get_normal_vector(grain_structure, grain_num):
    """Calculate grain boundary normal vectors using bilinear smoothing.

    Args:
        grain_structure (ndarray): 2D microstructure with grain IDs
        grain_num (int): Total number of grains

    Returns:
        tuple: (P, sites_flat, sites_by_grain)
            P: Shape (3, nx, ny) - smoothed field with normal vectors
            sites_flat: Flattened list of all GB site coordinates
            sites_by_grain: GB sites organized by grain ID
    """
    nx, ny = grain_structure.shape
    ng = int(np.max(grain_structure))
    cores = 8
    loop_times = 5

    R = np.zeros((nx, ny, 2))
    smooth_class = linear2d.linear_class(nx, ny, ng, cores, loop_times,
                                          grain_structure, R)

    smooth_class.linear_main("inclination")
    P = smooth_class.get_P()

    sites_by_grain = smooth_class.get_all_gb_list()
    sites_flat = []
    for grain_id in range(len(sites_by_grain)):
        sites_flat += sites_by_grain[grain_id]

    print(f"Total GB sites: {len(sites_flat)}")

    return P, sites_flat, sites_by_grain


def get_normal_vector_slope(P, sites, label, bias=None, ax=None):
    """Calculate inclination angle distribution and plot on polar coordinates.

    Args:
        P (ndarray): Smoothed field with normal vectors
        sites (list): List of GB site coordinates
        label (str): Label for plot legend
        bias (ndarray, optional): Bias correction for frequency distribution
        ax (matplotlib.axes, optional): Polar axis for plotting

    Returns:
        ndarray: Normalized frequency array for inclination angles
    """
    # Angle binning parameters
    x_lim = [0, 360]
    bin_width = 10.01
    bin_num = round((abs(x_lim[0]) + abs(x_lim[1])) / bin_width)
    x_coords = np.linspace((x_lim[0] + bin_width/2), (x_lim[1] - bin_width/2), bin_num)

    freq_array = np.zeros(bin_num)
    degrees = []

    # Calculate inclination angle for each GB site
    for site in sites:
        i, j = site
        dx, dy = P[1:, i, j]
        angle = math.atan2(-dy, dx) + math.pi
        degrees.append(angle)

    # Populate histogram bins
    for angle in degrees:
        angle_deg = angle / math.pi * 180
        bin_idx = int((angle_deg - x_lim[0]) / bin_width)
        if 0 <= bin_idx < bin_num:
            freq_array[bin_idx] += 1

    # Normalize to probability density
    total = sum(freq_array * bin_width)
    if total > 0:
        freq_array = freq_array / total

    # Apply optional bias correction
    if bias is not None:
        freq_array = freq_array + bias
        freq_array = freq_array / sum(freq_array * bin_width)

    # Plot on polar coordinates
    angles_rad = np.append(x_coords, x_coords[0]) / 180 * math.pi
    freq_periodic = np.append(freq_array, freq_array[0])

    if ax is not None:
        ax.plot(angles_rad, freq_periodic, linewidth=2, label=label)
    else:
        plt.plot(angles_rad, freq_periodic, linewidth=2, label=label)

    return freq_array


def setup_polar_plot():
    """Create and configure a polar plot for inclination visualization.

    Returns:
        tuple: (fig, ax) matplotlib figure and polar axes
    """
    plt.close('all')
    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, projection='polar')

    # Configure angular axis
    ax.set_thetagrids(np.arange(0.0, 360.0, 45.0), fontsize=16)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)

    # Configure radial axis
    ax.set_rgrids(np.arange(0, 0.008, 0.004))
    ax.set_rlabel_position(0.0)
    ax.set_rlim(0.0, 0.008)
    ax.set_yticklabels(['0', '4e-3'], fontsize=16)

    # Grid appearance
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    return fig, ax


# =============================================================================
# INCLINATION COMPARATOR CLASS
# =============================================================================

class InclinationComparator:
    """Unified inclination comparator for local and HPC environments."""

    def __init__(self, environment='local'):
        """Initialize the comparator.

        Args:
            environment: 'local' or 'hipergator'
        """
        self.environment = environment

        paths = ENV_PATHS.get(environment, ENV_PATHS['local'])
        self.input_folder = paths['input_folder']
        self.output_folder = paths['output_folder']

        os.makedirs(self.output_folder, exist_ok=True)

    def compare_datasets(self, primme_prefix, spparks_prefix, step_list,
                         output_prefix=None, verbose=True):
        """Compare inclination distributions between PRIMME and SPPARKS.

        Args:
            primme_prefix: Filename prefix for PRIMME data
            spparks_prefix: Filename prefix for SPPARKS data
            step_list: List of timesteps to compare
            output_prefix: Prefix for output files (default: primme_prefix)
            verbose: Print progress information

        Returns:
            dict: Comparison results with file paths
        """
        if output_prefix is None:
            output_prefix = primme_prefix

        results = {'plots_created': [], 'errors': []}

        if verbose:
            print(f"\nComparing inclination distributions:")
            print(f"  PRIMME: {primme_prefix}")
            print(f"  SPPARKS: {spparks_prefix}")
            print(f"  Steps: {step_list}")

        for step in step_list:
            if verbose:
                print(f"\n--- Processing step {step} ---")

            try:
                # Load data files
                spparks_file = os.path.join(self.input_folder, f"{spparks_prefix}{step}.npy")
                primme_file = os.path.join(self.input_folder, f"{primme_prefix}{step}.npy")

                spparks_data = np.load(spparks_file)
                primme_data = np.load(primme_file)

                if verbose:
                    print(f"  SPPARKS shape: {spparks_data.shape}")
                    print(f"  PRIMME shape: {primme_data.shape}")

                # Setup polar plot
                fig, ax = setup_polar_plot()

                # Analyze SPPARKS distribution
                spparks_sites = get_all_gb_list(spparks_data[0, :, :])
                get_normal_vector_slope(spparks_data, spparks_sites, "SPPARKS 20k", ax=ax)

                if verbose:
                    print(f"  SPPARKS GB sites: {len(spparks_sites)}")

                # Analyze PRIMME distribution
                primme_sites = get_all_gb_list(primme_data[0, :, :])
                get_normal_vector_slope(primme_data, primme_sites, "PRIMME 20k", ax=ax)

                if verbose:
                    print(f"  PRIMME GB sites: {len(primme_sites)}")

                # Save plot
                plt.legend(loc=(-0.10, -0.3), fontsize=16, ncol=2)
                output_file = os.path.join(self.output_folder,
                                          f"normal_distribution_{output_prefix}{step}.png")
                plt.savefig(output_file, dpi=400, bbox_inches='tight')
                plt.close()

                results['plots_created'].append(output_file)

                if verbose:
                    print(f"  Saved: {output_file}")

            except Exception as e:
                error_msg = f"Step {step} error: {str(e)}"
                results['errors'].append(error_msg)
                if verbose:
                    print(f"  Error: {error_msg}")

        return results

    def compare_batch(self, cases, spparks_baseline, verbose=True):
        """Run batch comparison for multiple PRIMME cases.

        Args:
            cases: List of dicts with 'prefix' and 'steps' keys
            spparks_baseline: SPPARKS data prefix for comparison
            verbose: Print progress information

        Returns:
            dict: Batch comparison results
        """
        batch_results = {'total': len(cases), 'successful': 0, 'failed': 0, 'results': []}

        if verbose:
            print("=" * 80)
            print("PRIMME VALIDATION: BATCH COMPARATIVE ANALYSIS")
            print("=" * 80)
            print(f"Cases: {len(cases)}")
            print(f"SPPARKS baseline: {spparks_baseline}")

        for i, case in enumerate(cases):
            if verbose:
                print(f"\n[{i+1}/{len(cases)}] {case.get('name', case['prefix'])}")

            try:
                result = self.compare_datasets(
                    primme_prefix=case['prefix'],
                    spparks_prefix=spparks_baseline,
                    step_list=case['steps'],
                    verbose=verbose
                )
                result['case'] = case
                batch_results['results'].append(result)

                if result['errors']:
                    batch_results['failed'] += 1
                else:
                    batch_results['successful'] += 1

            except Exception as e:
                batch_results['failed'] += 1
                batch_results['results'].append({'case': case, 'errors': [str(e)]})

        if verbose:
            print("\n" + "=" * 80)
            print(f"BATCH COMPLETE: {batch_results['successful']}/{len(cases)} successful")
            print("=" * 80)

        return batch_results


# =============================================================================
# PREDEFINED CASE CONFIGURATIONS
# =============================================================================

PRIMME_COMPARISON_CASES = [
    {'name': 'Case2AS_T3', 'prefix': 'Case2AS_T3_tstep_300_600_inclination_step', 'steps': [300, 600]},
    {'name': 'Case2BF_T8', 'prefix': 'Case2BF_T8_tstep_300_1600_inclination_step', 'steps': [300, 1600]},
    {'name': 'Case2BS_T1', 'prefix': 'Case2BS_T1_tstep_300_1600_inclination_step', 'steps': [300, 1600]},
    {'name': 'Case2CF_T4', 'prefix': 'Case2CF_T4_tstep_300_400_inclination_step', 'steps': [300, 400]},
    {'name': 'Case2DF_T1', 'prefix': 'Case2DF_T1_tstep_300_1600_inclination_step', 'steps': [300, 1600]},
    {'name': 'Case2DS_T5', 'prefix': 'Case2DS_T5_tstep_300_800_inclination_step', 'steps': [300, 800]},
    {'name': 'Case2DS_T8', 'prefix': 'Case2DS_T8_tstep_300_1600_inclination_step', 'steps': [300, 1600]},
    {'name': 'Case3AF_T9', 'prefix': 'Case3AF_T9_tstep_300_600_inclination_step', 'steps': [300, 600]},
    {'name': 'Case3AS_T6', 'prefix': 'Case3AS_T6_tstep_300_1600_inclination_step', 'steps': [300, 1600]},
    {'name': 'Case3BF_T7', 'prefix': 'Case3BF_T7_tstep_300_600_inclination_step', 'steps': [300, 600]},
    {'name': 'Case3BS_T10', 'prefix': 'Case3BS_T10_step_300_1600_inclination_step', 'steps': [300, 1600]},
    {'name': 'Case3BS_T6', 'prefix': 'Case3BS_T6_tstep_300_1600_inclination_step', 'steps': [300, 1600]},
    {'name': 'Case3CS_T7', 'prefix': 'Case3CS_T7_tstep_300_400_inclination_step', 'steps': [300, 400]},
]

SPPARKS_BASELINE = "spparks_s1_tstep_300_400_600_800_1600_inclination_step"


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPERS
# =============================================================================

def run_local_comparison():
    """Run local environment comparison (backward compatibility)."""
    comparator = InclinationComparator(environment='local')

    # Local 20k grain comparison
    spparks_prefix = "spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0)_inclination_step"
    primme_prefix = "primme_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1)_kt(0.66)_cut(0)_inclination_step"
    pf_prefix = "phasefiled_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1)_kt(0.66)_cut(0)_inclination_step"

    for step in [300, 1600]:
        fig, ax = setup_polar_plot()

        # Load and analyze each method
        for prefix, label in [(spparks_prefix, 'spparks20k'),
                              (primme_prefix, 'primme20k'),
                              (pf_prefix, 'phasefield20k')]:
            try:
                data = np.load(os.path.join(comparator.input_folder, f"{prefix}{step}.npy"))
                sites = get_all_gb_list(data[0, :, :])
                get_normal_vector_slope(data, sites, label, ax=ax)
            except FileNotFoundError:
                print(f"  Warning: {prefix}{step}.npy not found")

        plt.legend(loc=(-0.10, -0.3), fontsize=16, ncol=2)
        output_file = os.path.join(comparator.output_folder,
                                   f"normal_distribution_20k_step{step}.png")
        plt.savefig(output_file, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_file}")


def run_hipergator_comparison():
    """Run HiPerGator batch comparison (backward compatibility)."""
    comparator = InclinationComparator(environment='hipergator')
    comparator.compare_batch(PRIMME_COMPARISON_CASES, SPPARKS_BASELINE)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified grain boundary inclination comparison'
    )
    parser.add_argument('--primme', help='PRIMME data file prefix')
    parser.add_argument('--spparks', help='SPPARKS data file prefix')
    parser.add_argument('--steps', '-s', nargs='+', type=int, help='Timesteps to compare')
    parser.add_argument('--environment', '-e', choices=['local', 'hipergator'],
                       default='local', help='Computing environment')
    parser.add_argument('--batch', action='store_true', help='Run predefined batch comparison')
    parser.add_argument('--output', '-o', help='Output file prefix')

    args = parser.parse_args()

    if args.batch:
        if args.environment == 'hipergator':
            run_hipergator_comparison()
        else:
            run_local_comparison()
    elif args.primme and args.spparks and args.steps:
        comparator = InclinationComparator(environment=args.environment)
        comparator.compare_datasets(args.primme, args.spparks, args.steps,
                                   output_prefix=args.output)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python compare_inclination.py --primme primme_data_ --spparks spparks_data_ --steps 300 1600")
        print("  python compare_inclination.py --batch --environment hipergator")


if __name__ == '__main__':
    main()
