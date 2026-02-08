#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Grain Boundary Inclination Calculation Module

This module consolidates inclination calculation functionality for both local
and HPC (HiPerGator) environments. It processes HDF5 phase field simulation
data to generate grain boundary inclination vector fields using bilinear
smoothing algorithms.

Usage:
    # As a module
    from calculate_inclination import InclinationCalculator
    calc = InclinationCalculator(environment='local')
    calc.process_single('input/file', 'output/prefix', [300, 1600])

    # Command line
    python calculate_inclination.py --input input.h5 --output output_prefix --steps 300 1600
    python calculate_inclination.py --batch --config cases.json

Created: 2021-09-30
Author: Lin Yang
Refactored: 2026-02-08 (Phase 8 consolidation)
"""

import os
import sys
import argparse
import numpy as np
import math
import h5py
from tqdm import tqdm

# Setup paths
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
sys.path.insert(0, os.path.join(current_path, '../..'))

import myInput
import PACKAGE_MP_Linear as smooth


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

ENV_PATHS = {
    'local': {
        'input_folder': os.path.join(current_path, 'input/'),
        'output_folder': os.path.join(current_path, 'output/'),
    },
    'hipergator': {
        'input_folder': '/blue/michael.tonks/share/PRIMME_Inclination/',
        'output_folder': '/blue/michael.tonks/share/PRIMME_Inclination_npy_files/',
    }
}


# =============================================================================
# CORE INCLINATION CALCULATION
# =============================================================================

def load_h5_data(filepath):
    """Load microstructure data from HDF5 file.

    Args:
        filepath (str): Path to HDF5 file (with or without .h5 extension)

    Returns:
        tuple: (ims_id, euler_angles) - Microstructure IDs and grain orientations
    """
    if not filepath.endswith('.h5'):
        filepath = filepath + '.h5'

    ims_id = None
    euler_angles = None

    with h5py.File(filepath, 'r') as f:
        for simu in f.keys():
            container = f.get(simu)
            for dataset in container.keys():
                data = container[dataset]
                name = dataset.replace(' ', '_')
                if 'ims_id' in name.lower() or name == 'ims_id':
                    ims_id = np.array(data)
                elif 'euler' in name.lower() or name == 'euler_angles':
                    euler_angles = np.array(data)

    if ims_id is None:
        raise ValueError(f"Could not find ims_id dataset in {filepath}")

    return ims_id, euler_angles


def calculate_inclination_vectors(microstructure, cores=8, loop_times=5):
    """Calculate inclination vectors using bilinear smoothing.

    Args:
        microstructure (ndarray): 2D grain ID array
        cores (int): Number of CPU cores for parallel processing
        loop_times (int): Smoothing iterations

    Returns:
        ndarray: Shape (3, nx, ny) - [microstructure, x_vector, y_vector]
    """
    nx, ny = microstructure.shape
    ng = int(np.max(microstructure))
    R = np.zeros((nx, ny, 2))

    # Initialize and run smoothing algorithm
    smooth_class = smooth.linear_class(nx, ny, ng, cores, loop_times,
                                        microstructure, R, 0, False)
    smooth_class.linear_main('inclination')
    P = smooth_class.get_P()

    return P, smooth_class


def normalize_inclination(P):
    """Normalize and transform inclination vectors.

    Args:
        P (ndarray): Raw inclination field from smoothing algorithm

    Returns:
        ndarray: Normalized inclination field with proper coordinates
    """
    P_final = np.array(P)

    # Coordinate transformation for standard convention
    P_final[1] = -P[2]  # x-component = -original_y_component
    P_final[2] = P[1]   # y-component = original_x_component

    # Normalize to unit length
    magnitude = np.sqrt(P_final[1]**2 + P_final[2]**2)
    mask = magnitude != 0
    P_final[1][mask] = P_final[1][mask] / magnitude[mask]
    P_final[2][mask] = P_final[2][mask] / magnitude[mask]

    # Handle NaN values
    P_final = np.nan_to_num(P_final)

    return P_final


# =============================================================================
# INCLINATION CALCULATOR CLASS
# =============================================================================

class InclinationCalculator:
    """Unified inclination calculator for local and HPC environments."""

    def __init__(self, environment='local', cores=8, loop_times=5):
        """Initialize the calculator.

        Args:
            environment: 'local' or 'hipergator'
            cores: CPU cores for parallel processing
            loop_times: Smoothing algorithm iterations
        """
        self.environment = environment
        self.cores = cores
        self.loop_times = loop_times

        paths = ENV_PATHS.get(environment, ENV_PATHS['local'])
        self.input_folder = paths['input_folder']
        self.output_folder = paths['output_folder']

        # Ensure output directory exists
        os.makedirs(self.output_folder, exist_ok=True)

    def process_single(self, input_path, output_prefix, step_list, verbose=True):
        """Process a single HDF5 file for inclination calculation.

        Args:
            input_path: Path to HDF5 file (with or without extension)
            output_prefix: Prefix for output files
            step_list: List of timesteps to process
            verbose: Print progress information

        Returns:
            dict: Processing results with timing and file info
        """
        results = {'steps_processed': [], 'files_created': [], 'errors': []}

        if verbose:
            print(f"Processing: {os.path.basename(input_path)}")

        try:
            # Load HDF5 data
            ims_id, euler_angles = load_h5_data(input_path)
            steps, nz, nx, ny = ims_id.shape
            ng = len(euler_angles) if euler_angles is not None else int(np.max(ims_id))

            if verbose:
                print(f"  Domain: {nx}×{ny}, Grains: {ng}, Steps: {steps}")

        except Exception as e:
            results['errors'].append(f"Load error: {str(e)}")
            return results

        # Process each timestep
        for step in tqdm(step_list, desc="Processing timesteps", disable=not verbose):
            try:
                if step >= steps:
                    results['errors'].append(f"Step {step} out of range (max: {steps-1})")
                    continue

                # Extract microstructure
                microstructure = np.squeeze(ims_id[step, :])

                # Calculate inclination
                P, smooth_class = calculate_inclination_vectors(
                    microstructure, self.cores, self.loop_times
                )

                # Normalize and transform
                P_final = normalize_inclination(P)

                # Save output
                output_file = f"{output_prefix}step{step}"
                np.save(output_file, P_final)

                results['steps_processed'].append(step)
                results['files_created'].append(output_file + '.npy')

                if verbose:
                    print(f"    Step {step}: {smooth_class.running_time:.2f}s, saved to {output_file}.npy")

            except Exception as e:
                results['errors'].append(f"Step {step} error: {str(e)}")

        return results

    def process_batch(self, cases, verbose=True):
        """Process multiple simulation cases in batch.

        Args:
            cases: List of dicts with 'input', 'output', 'steps' keys
            verbose: Print progress information

        Returns:
            dict: Batch processing results
        """
        batch_results = {'total_cases': len(cases), 'successful': 0, 'failed': 0, 'case_results': []}

        if verbose:
            print("=" * 70)
            print("BATCH INCLINATION PROCESSING")
            print("=" * 70)
            print(f"Processing {len(cases)} cases...")

        for i, case in enumerate(cases):
            if verbose:
                print(f"\n[{i+1}/{len(cases)}] {case.get('name', case.get('input', 'Unknown'))}")

            input_path = case.get('input', '')
            output_prefix = case.get('output', '')
            steps = case.get('steps', [])

            # Add folder paths if not absolute
            if not os.path.isabs(input_path):
                input_path = os.path.join(self.input_folder, input_path)
            if not os.path.isabs(output_prefix):
                output_prefix = os.path.join(self.output_folder, output_prefix)

            try:
                result = self.process_single(input_path, output_prefix, steps, verbose=verbose)
                result['case'] = case
                batch_results['case_results'].append(result)

                if result['errors']:
                    batch_results['failed'] += 1
                else:
                    batch_results['successful'] += 1

            except Exception as e:
                batch_results['failed'] += 1
                batch_results['case_results'].append({
                    'case': case,
                    'errors': [str(e)]
                })

        if verbose:
            print("\n" + "=" * 70)
            print(f"BATCH COMPLETE: {batch_results['successful']}/{len(cases)} successful")
            print("=" * 70)

        return batch_results


# =============================================================================
# PREDEFINED CASE CONFIGURATIONS
# =============================================================================

PRIMME_CASES = [
    {'name': 'Case2AS_T3', 'input': 'Case2AS_T3_tstep_300_600',
     'output': 'Case2AS_T3_tstep_300_600_inclination_', 'steps': [300, 600]},
    {'name': 'Case2BF_T8', 'input': 'Case2BF_T8_tstep_300_1600',
     'output': 'Case2BF_T8_tstep_300_1600_inclination_', 'steps': [300, 1600]},
    {'name': 'Case2BS_T1', 'input': 'Case2BS_T1_tstep_300_1600',
     'output': 'Case2BS_T1_tstep_300_1600_inclination_', 'steps': [300, 1600]},
    {'name': 'Case2CF_T4', 'input': 'Case2CF_T4_tstep_300_400',
     'output': 'Case2CF_T4_tstep_300_400_inclination_', 'steps': [300, 400]},
    {'name': 'Case2DF_T1', 'input': 'Case2DF_T1_tstep_300_1600',
     'output': 'Case2DF_T1_tstep_300_1600_inclination_', 'steps': [300, 1600]},
    {'name': 'Case2DS_T5', 'input': 'Case2DS_T5_tstep_300_800',
     'output': 'Case2DS_T5_tstep_300_800_inclination_', 'steps': [300, 800]},
    {'name': 'Case2DS_T8', 'input': 'Case2DS_T8_tstep_300_1600',
     'output': 'Case2DS_T8_tstep_300_1600_inclination_', 'steps': [300, 1600]},
    {'name': 'Case3AF_T9', 'input': 'Case3AF_T9_tstep_300_600',
     'output': 'Case3AF_T9_tstep_300_600_inclination_', 'steps': [300, 600]},
    {'name': 'Case3AS_T6', 'input': 'Case3AS_T6_tstep_300_1600',
     'output': 'Case3AS_T6_tstep_300_1600_inclination_', 'steps': [300, 1600]},
    {'name': 'Case3BF_T7', 'input': 'Case3BF_T7_tstep_300_600',
     'output': 'Case3BF_T7_tstep_300_600_inclination_', 'steps': [300, 600]},
    {'name': 'Case3BS_T10', 'input': 'Case3BS_T10_step_300_1600',
     'output': 'Case3BS_T10_step_300_1600_inclination_', 'steps': [300, 1600]},
    {'name': 'Case3BS_T6', 'input': 'Case3BS_T6_tstep_300_1600',
     'output': 'Case3BS_T6_tstep_300_1600_inclination_', 'steps': [300, 1600]},
    {'name': 'Case3CS_T7', 'input': 'Case3CS_T7_tstep_300_400',
     'output': 'Case3CS_T7_tstep_300_400_inclination_', 'steps': [300, 400]},
    {'name': 'spparks_s1', 'input': 'spparks_s1_tstep_300_400_600_800_1600',
     'output': 'spparks_s1_tstep_300_400_600_800_1600_inclination_', 'steps': [300, 400, 600, 800, 1600]},
]


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPERS
# =============================================================================

def calculate_inclination_data(input_path, output_path, step_list):
    """Backward-compatible wrapper for batch processing function.

    Args:
        input_path: Full path to HDF5 file (without extension)
        output_path: Full path prefix for output files
        step_list: List of timesteps to process
    """
    calc = InclinationCalculator(environment='hipergator')
    calc.process_single(input_path, output_path, step_list, verbose=True)


def run_local_analysis():
    """Run local environment analysis (backward compatibility)."""
    calc = InclinationCalculator(environment='local')
    input_path = os.path.join(current_path, 'input',
                              'phasefiled_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1)_kt(0.66)_cut(0)')
    output_path = os.path.join(current_path, 'output',
                               'phasefiled_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1)_kt(0.66)_cut(0)_inclination_')
    calc.process_single(input_path, output_path, [300, 1600])


def run_hipergator_batch():
    """Run HiPerGator batch analysis (backward compatibility)."""
    calc = InclinationCalculator(environment='hipergator')
    calc.process_batch(PRIMME_CASES)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified grain boundary inclination calculation'
    )
    parser.add_argument('--input', '-i', help='Input HDF5 file path')
    parser.add_argument('--output', '-o', help='Output file prefix')
    parser.add_argument('--steps', '-s', nargs='+', type=int, help='Timesteps to process')
    parser.add_argument('--environment', '-e', choices=['local', 'hipergator'],
                       default='local', help='Computing environment')
    parser.add_argument('--batch', action='store_true', help='Run predefined batch processing')
    parser.add_argument('--cores', type=int, default=8, help='CPU cores for parallel processing')

    args = parser.parse_args()

    if args.batch:
        if args.environment == 'hipergator':
            run_hipergator_batch()
        else:
            run_local_analysis()
    elif args.input and args.output and args.steps:
        calc = InclinationCalculator(environment=args.environment, cores=args.cores)
        calc.process_single(args.input, args.output, args.steps)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python calculate_inclination.py --input data.h5 --output result_ --steps 300 1600")
        print("  python calculate_inclination.py --batch --environment hipergator")


if __name__ == '__main__':
    main()
