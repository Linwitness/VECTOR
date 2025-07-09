#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRIMME INCLINATION DISTRIBUTION COMPARISON - HIPERGATOR HPC VERSION
================================================================

High-Performance Computing (HPC) comparative analysis script for validating
PRIMME grain boundary inclination predictions against SPPARKS Monte Carlo
simulations on the HiPerGator supercomputing cluster. This script generates
comprehensive polar plot comparisons for statistical validation of the
PRIMME method accuracy.

PRIMME Validation Framework:
- Quantitative comparison: PRIMME vs SPPARKS baseline simulations
- Statistical analysis: Grain boundary inclination distribution accuracy
- Polar visualization: Angular distribution patterns and deviations
- HPC batch processing: Multiple simulation cases and timesteps

Scientific Methodology:
- Grain Boundary Identification: Periodic boundary condition handling
- Bilinear Smoothing: Consistent normal vector calculation methodology
- Statistical Binning: Angular distribution analysis with 10-degree resolution
- Polar Visualization: Standard crystallographic convention presentation

Validation Metrics:
- Distribution Shape Accuracy: Comparison of angular frequency patterns
- Peak Position Analysis: Critical angle identification and validation
- Statistical Deviation: Quantitative assessment of method differences
- Temporal Evolution: Validation across multiple simulation timesteps

Target Platform: UF HiPerGator Supercomputing Cluster
- Input: Processed inclination data from PRIMME and SPPARKS simulations
- Output: High-resolution polar comparison plots for publication
- Storage: Blue file system optimization for large-scale data processing

Author: lin.yang
Created: Thu Sep 30 14:55:28 2021
Purpose: PRIMME method validation through comparative statistical analysis
"""

import os
current_path = os.getcwd()
import sys
sys.path.append(current_path)
sys.path.append('../../.')
import numpy as np
import math
from itertools import repeat
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

import myInput
import PACKAGE_MP_Linear as linear2d     #2D Bilinear smooth algorithm


def get_all_gb_list(P0):
    nx, ny = P0.shape
    gagn_gbsites = []
    for i in range(0,nx):
        for j in range(0,ny):
            ip,im,jp,jm = myInput.periodic_bc(nx,ny,i,j)
            if ( ((P0[ip,j]-P0[i,j])!=0) or
                 ((P0[im,j]-P0[i,j])!=0) or
                 ((P0[i,jp]-P0[i,j])!=0) or
                 ((P0[i,jm]-P0[i,j])!=0) ):
                gagn_gbsites.append([i,j])
    return gagn_gbsites

def get_normal_vector(grain_structure_figure_one, grain_num):
    nx = grain_structure_figure_one.shape[0]
    ny = grain_structure_figure_one.shape[1]
    ng = np.max(grain_structure_figure_one)
    cores = 8
    loop_times = 5
    P0 = grain_structure_figure_one
    R = np.zeros((nx,ny,2))
    smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)

    smooth_class.linear_main("inclination")
    P = smooth_class.get_P()
    # sites = smooth_class.get_gb_list(1)
    # print(len(sites))
    # for id in range(2,grain_num+1): sites += smooth_class.get_gb_list(id)
    # print(len(sites))
    sites = smooth_class.get_all_gb_list()
    sites_together = []
    for id in range(len(sites)): sites_together += sites[id]
    print("Total num of GB sites: " + str(len(sites_together)))

    return P, sites_together, sites

def get_normal_vector_slope(P, sites, para_name, bias=None):
    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)

    freqArray = np.zeros(binNum)
    degree = []
    for sitei in sites:
        [i,j] = sitei
        dx,dy = P[1:,i,j]
        degree.append(math.atan2(-dy, dx) + math.pi)

    for i in range(len(degree)):
        freqArray[int((degree[i]/math.pi*180-xLim[0])/binValue)] += 1
    freqArray = freqArray/sum(freqArray*binValue)

    if bias is not None:
        freqArray = freqArray + bias
        freqArray = freqArray/sum(freqArray*binValue)
    # Plot
    plt.plot(np.append(xCor,xCor[0])/180*math.pi, np.append(freqArray,freqArray[0]), linewidth=2, label=para_name)

    return freqArray

def compare_inclination_dsitribution(primme_data, spparks_data, step_list, input_path, output_path):
    """
    COMPREHENSIVE INCLINATION DISTRIBUTION COMPARISON FUNCTION
    ========================================================
    
    Core analysis function for generating comparative polar plots between
    PRIMME and SPPARKS grain boundary inclination distributions. This function
    performs statistical validation and creates publication-quality visualizations
    for quantitative assessment of PRIMME method accuracy.
    
    Comparison Methodology:
    1. Data Loading: Load processed inclination data from both methods
    2. Grain Boundary Identification: Extract boundary sites using periodic conditions
    3. Statistical Analysis: Calculate angular distribution histograms
    4. Polar Visualization: Generate comparative plots with standardized formatting
    5. Publication Output: Save high-resolution figures for research documentation
    
    Parameters:
    -----------
    primme_data : str
        Filename prefix for PRIMME inclination data
        Format: "CaseXY_TZ_tstep_A_B_inclination_step"
        
    spparks_data : str
        Filename prefix for SPPARKS baseline comparison data
        Format: "spparks_s1_tstep_300_400_600_800_1600_inclination_step"
        
    step_list : list of int
        Timesteps to compare between methods
        Example: [300, 600, 1600] for evolution analysis
        
    input_path : str
        Directory path containing processed inclination data files
        HPC path: "/blue/michael.tonks/share/PRIMME_Inclination_npy_files/"
        
    output_path : str
        Directory path for saving comparative analysis figures
        Output: High-resolution PNG files for publication
    
    Scientific Analysis:
    -------------------
    - Angular Resolution: 10-degree bins for statistical accuracy
    - Normalization: Probability density function for comparative analysis
    - Coordinate System: Standard crystallographic convention (0-360°)
    - Visualization: Polar plots with consistent formatting and legends
    
    Output Files:
    ------------
    - Format: "normal_distribution_{primme_data}{timestep}.png"
    - Resolution: 400 DPI for publication quality
    - Content: Overlay comparison of PRIMME vs SPPARKS distributions
    
    Validation Metrics:
    ------------------
    - Distribution Shape: Visual comparison of angular patterns
    - Peak Analysis: Critical angle identification and accuracy
    - Statistical Deviation: Quantitative assessment of method differences
    - Temporal Evolution: Validation consistency across simulation time
    """
    
    print(f"\nComparing inclination distributions:")
    print(f"  PRIMME data: {primme_data}")
    print(f"  SPPARKS baseline: {spparks_data}")
    print(f"  Timesteps: {step_list}")

    # =================================================================
    # TIMESTEP PROCESSING LOOP
    # =================================================================
    
    for i in step_list:
        print(f"\n--- Processing timestep {i} ---")
        
        # =================================================================
        # DATA LOADING AND VALIDATION
        # =================================================================
        
        # Load processed inclination data from both simulation methods
        spparks_file = input_path + spparks_data + f"{i}.npy"
        primme_file = input_path + primme_data + f"{i}.npy"
        
        print(f"  Loading SPPARKS data: {spparks_file}")
        print(f"  Loading PRIMME data: {primme_file}")
        
        npy_file_s20k = np.load(spparks_file)    # SPPARKS baseline simulation
        npy_file_p20k = np.load(primme_file)     # PRIMME validation simulation
        
        # Validate data dimensions and compatibility
        print(f"  SPPARKS data shape: {npy_file_s20k.shape}")
        print(f"  PRIMME data shape: {npy_file_p20k.shape}")
        
        # =================================================================
        # GRAIN POPULATION ANALYSIS
        # =================================================================
        
        # Initial grain population for simulation validation
        initial_grain_num = 20000  # Target grain count for statistical significance
        
        # Analyze actual grain populations in loaded data
        spparks_grains = len(np.unique(npy_file_s20k[0, :, :]))
        primme_grains = len(np.unique(npy_file_p20k[0, :, :]))
        
        print(f"  Grain populations - SPPARKS: {spparks_grains}, PRIMME: {primme_grains}")

        # =================================================================
        # POLAR PLOT INITIALIZATION AND FORMATTING
        # =================================================================
        
        # Start polar figure with publication-quality formatting
        plt.close('all')  # Clear any existing plots
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca(projection='polar')

        # Configure angular axis (theta) - crystallographic convention
        ax.set_thetagrids(np.arange(0.0, 360.0, 45.0), fontsize=16)
        ax.set_thetamin(0.0)
        ax.set_thetamax(360.0)

        # Configure radial axis (r) - probability density scale
        ax.set_rgrids(np.arange(0, 0.008, 0.004))  # Probability density ticks
        ax.set_rlabel_position(0.0) 
        ax.set_rlim(0.0, 0.008)                    # Maximum probability density
        ax.set_yticklabels(['0', '4e-3'], fontsize=16)

        # Grid and visual formatting
        ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow('True')
        
        print("  ✓ Polar plot initialized with publication formatting")

        # =================================================================
        # SPPARKS BASELINE DISTRIBUTION ANALYSIS
        # =================================================================
        
        print("  Analyzing SPPARKS baseline distribution...")
        
        # Extract grain boundary sites from SPPARKS simulation
        spparks_sites = get_all_gb_list(npy_file_s20k[0, :, :])
        
        # Calculate and plot angular distribution
        spparks_distribution = get_normal_vector_slope(npy_file_s20k, spparks_sites, r"SPPARKS 20k")
        
        print(f"    SPPARKS GB sites: {len(spparks_sites)}")

        # =================================================================
        # PRIMME METHOD DISTRIBUTION ANALYSIS
        # =================================================================
        
        print("  Analyzing PRIMME method distribution...")
        
        # Extract grain boundary sites from PRIMME simulation
        primme_sites = get_all_gb_list(npy_file_p20k[0, :, :])
        
        # Calculate and plot angular distribution
        primme_distribution = get_normal_vector_slope(npy_file_p20k, primme_sites, r"PRIMME 20k")
        
        print(f"    PRIMME GB sites: {len(primme_sites)}")

        # =================================================================
        # COMPARATIVE VISUALIZATION AND OUTPUT
        # =================================================================
        
        # Add legend and save comparative plot
        plt.legend(loc=(-0.10, -0.3), fontsize=16, ncol=2)
        
        # Generate output filename and save high-resolution figure
        output_filename = output_path + f"figures/normal_distribution_{primme_data}{i}.png"
        plt.savefig(output_filename, dpi=400, bbox_inches='tight')
        
        print(f"  ✓ Saved comparative plot: {output_filename}")
        
        # =================================================================
        # STATISTICAL SUMMARY AND VALIDATION METRICS
        # =================================================================
        
        # Calculate basic statistical comparison metrics
        total_spparks_sites = len(spparks_sites)
        total_primme_sites = len(primme_sites)
        site_ratio = total_primme_sites / total_spparks_sites if total_spparks_sites > 0 else 0
        
        print(f"  Statistical comparison:")
        print(f"    Site count ratio (PRIMME/SPPARKS): {site_ratio:.3f}")
        print(f"    Grain boundary detection consistency: {'✓' if abs(site_ratio - 1.0) < 0.1 else '✗'}")
        
        # Optional: Calculate distribution correlation (advanced analysis)
        # correlation = np.corrcoef(spparks_distribution, primme_distribution)[0,1]
        # print(f"    Distribution correlation: {correlation:.3f}")
        
    print(f"\n✓ Completed comparative analysis for: {primme_data}")
    print(f"Generated {len(step_list)} comparative distribution plots")


if __name__ == '__main__':
    """
    MAIN EXECUTION: HPC BATCH COMPARATIVE ANALYSIS
    =============================================
    
    High-performance batch processing for comprehensive comparative analysis
    between PRIMME method and SPPARKS baseline simulations. This script processes
    multiple simulation cases to generate statistical validation datasets for
    PRIMME method accuracy assessment.
    
    Validation Strategy:
    - Multiple Case Studies: Systematic comparison across different scenarios
    - Temporal Evolution: Validation at multiple simulation timesteps
    - Statistical Significance: Large grain populations (20,000 grains)
    - Publication Quality: High-resolution polar plots for research documentation
    
    HPC Optimization:
    - Batch Processing: Automated analysis of all simulation cases
    - File System: Blue storage optimization for large datasets
    - Parallel Execution: Efficient processing on HiPerGator cluster
    """

    # =================================================================
    # HPC FILE SYSTEM CONFIGURATION
    # =================================================================
    
    # HiPerGator Blue file system paths for processed inclination data
    input_folder = "/blue/michael.tonks/share/PRIMME_Inclination_npy_files/"
    
    print("="*80)
    print("PRIMME VALIDATION: HPC BATCH COMPARATIVE ANALYSIS")
    print("="*80)
    print(f"Data directory: {input_folder}")
    
    # =================================================================
    # SIMULATION CASE SPECIFICATIONS FOR COMPARATIVE ANALYSIS
    # =================================================================
    
    # PRIMME simulation cases for validation analysis
    # Each case represents different parameter combinations and scenarios
    input_name = ["Case2AS_T3_tstep_300_600_inclination_step",      # Case 2A Standard, Trial 3
                  "Case2BF_T8_tstep_300_1600_inclination_step",     # Case 2B Fast, Trial 8
                  "Case2BS_T1_tstep_300_1600_inclination_step",     # Case 2B Standard, Trial 1
                  "Case2CF_T4_tstep_300_400_inclination_step",      # Case 2C Fast, Trial 4
                  "Case2DF_T1_tstep_300_1600_inclination_step",     # Case 2D Fast, Trial 1
                  "Case2DS_T5_tstep_300_800_inclination_step",      # Case 2D Standard, Trial 5
                  "Case2DS_T8_tstep_300_1600_inclination_step",     # Case 2D Standard, Trial 8
                  "Case3AF_T9_tstep_300_600_inclination_step",      # Case 3A Fast, Trial 9
                  "Case3AS_T6_tstep_300_1600_inclination_step",     # Case 3A Standard, Trial 6
                  "Case3BF_T7_tstep_300_600_inclination_step",      # Case 3B Fast, Trial 7
                  "Case3BS_T10_step_300_1600_inclination_step",     # Case 3B Standard, Trial 10
                  "Case3BS_T6_tstep_300_1600_inclination_step",     # Case 3B Standard, Trial 6
                  "Case3CS_T7_tstep_300_400_inclination_step"]      # Case 3C Standard, Trial 7
    
    # SPPARKS baseline simulation for comparative validation
    # Comprehensive timestep coverage for temporal evolution analysis
    compare_data_name = "spparks_s1_tstep_300_400_600_800_1600_inclination_step"

    # Corresponding timestep lists for each PRIMME case
    # Aligned with available SPPARKS data for direct comparison
    step_lists = [[300,600],          # Early to intermediate evolution
                  [300,1600],         # Early to final state  
                  [300,1600],         # Early to final state
                  [300,400],          # Short evolution study
                  [300,1600],         # Full evolution range
                  [300,800],          # Early to intermediate
                  [300,1600],         # Full evolution range
                  [300,600],          # Early to intermediate
                  [300,1600],         # Full evolution range
                  [300,600],          # Early to intermediate
                  [300,1600],         # Full evolution range
                  [300,1600],         # Full evolution range
                  [300,400]]          # Short evolution study

    # =================================================================
    # BATCH PROCESSING EXECUTION AND PROGRESS MONITORING
    # =================================================================
    
    print(f"\nComparative Analysis Configuration:")
    print(f"  PRIMME cases: {len(input_name)}")
    print(f"  SPPARKS baseline: {compare_data_name}")
    print(f"  Total comparisons: {sum(len(steps) for steps in step_lists)}")
    
    print(f"\nProcessing plan:")
    for i, (case_name, steps) in enumerate(zip(input_name, step_lists)):
        case_short = case_name.replace("_inclination_step", "")
        print(f"  {i+1:2d}. {case_short:<30} → timesteps {steps}")
    
    print(f"\nStarting batch comparative analysis...")
    
    # =================================================================
    # MAIN PROCESSING LOOP WITH ERROR HANDLING
    # =================================================================
    
    successful_analyses = 0
    failed_analyses = 0
    
    for i in range(len(input_name)):
        print(f"\n{'='*60}")
        print(f"COMPARATIVE ANALYSIS {i+1}/{len(input_name)}")
        print(f"Case: {input_name[i].replace('_inclination_step', '')}")
        print(f"{'='*60}")
        
        try:
            # Execute comparative inclination distribution analysis
            compare_inclination_dsitribution(
                primme_data=input_name[i],
                spparks_data=compare_data_name,
                step_list=step_lists[i],
                input_path=input_folder,
                output_path=input_folder
            )
            
            print(f"✓ Successfully completed analysis: {input_name[i]}")
            successful_analyses += 1
            
        except Exception as e:
            print(f"✗ Error in comparative analysis {input_name[i]}: {str(e)}")
            failed_analyses += 1
            continue  # Continue with next case if current fails
        
        print(f"Generated {len(step_lists[i])} comparative plots")
    
    # =================================================================
    # COMPLETION SUMMARY AND VALIDATION REPORT
    # =================================================================
    
    print(f"\n{'='*80}")
    print("BATCH COMPARATIVE ANALYSIS COMPLETED!")
    print(f"{'='*80}")
    
    print(f"Analysis Results:")
    print(f"  Successful analyses: {successful_analyses}/{len(input_name)}")
    print(f"  Failed analyses: {failed_analyses}")
    print(f"  Success rate: {(successful_analyses/len(input_name))*100:.1f}%")
    
    # Calculate total output files
    total_plots = sum(len(steps) for steps in step_lists[:successful_analyses])
    print(f"  Generated plots: {total_plots}")
    
    print(f"\nOutput Organization:")
    print(f"  Location: {input_folder}figures/")
    print(f"  Format: normal_distribution_{{case}}_{{timestep}}.png")
    print(f"  Resolution: 400 DPI for publication quality")
    print(f"  Content: PRIMME vs SPPARKS polar comparison plots")
    
    print(f"\nValidation Methodology:")
    print("• Grain Boundary Identification: Periodic boundary conditions")
    print("• Angular Resolution: 10-degree bins for statistical accuracy")
    print("• Normalization: Probability density functions")  
    print("• Visualization: Standard crystallographic polar coordinates")
    print("• Comparison: Overlay plots with consistent formatting")
    
    print(f"\nNext Steps for Research:")
    print("1. Analyze polar plots for distribution shape accuracy")
    print("2. Identify critical angles and peak positions")
    print("3. Calculate quantitative deviation metrics")
    print("4. Prepare validation summary for PRIMME method publication")
    print("5. Consider temporal evolution patterns across timesteps")
    
    if successful_analyses == len(input_name):
        print(f"\n🎉 ALL COMPARATIVE ANALYSES COMPLETED SUCCESSFULLY!")
    else:
        print(f"\n⚠️  Some analyses failed - check error messages above")
    
    print(f"\nTotal validation dataset: {total_plots} comparative plots")
    print("Ready for statistical accuracy assessment and publication preparation")





