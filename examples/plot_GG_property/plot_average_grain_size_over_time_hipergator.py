#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Grain Growth Analysis: HiPerGator Large-Scale Multi-Energy Method Study

This script provides advanced grain growth analysis capabilities optimized for HiPerGator
supercomputing cluster processing. It combines temporal evolution tracking, statistical
distribution analysis, and power-law growth kinetics fitting across all six energy
calculation methods. The analysis includes regression fitting to determine growth exponents
and comparative assessment of grain size evolution characteristics.

Scientific Purpose:
- Comprehensive multi-energy method comparison (6 anisotropic + 1 isotropic)
- Power-law growth kinetics analysis with R² optimization for growth exponent determination
- Large-scale statistical characterization of grain size distributions
- Advanced temporal evolution visualization with proper scaling
- Growth rate coefficient extraction and comparative analysis

Key Features:
- Complete energy method suite: ave, consMin, sum, min, max, consMax, iso
- Advanced regression analysis using scikit-learn for growth law fitting
- Automated grain area calculation with intelligent caching
- Power-law fitting: R^n - R₀^n = kt (with optimized n-value determination)
- Publication-quality visualization with enhanced formatting
- HiPerGator-optimized data processing for 64-core parallel simulations

Energy Methods Analyzed:
- ave: Average triple junction energy approach (baseline)
- consMin: Conservative minimum energy selection (enhanced small grain stability)
- consMax: Conservative maximum energy selection (balanced large grain growth)
- sum: Summation-based energy calculation (cumulative energy effects)
- min: Pure minimum energy criterion (maximum small grain preservation)
- max: Pure maximum energy criterion (enhanced large grain growth)
- iso: Isotropic reference case (delta=0.0, no anisotropy effects)

Advanced Analysis Features:
- Growth exponent fitting: R^n - R₀^n = kt with n ∈ [0.5, 3.5]
- R² optimization across 301 n-values for best-fit determination
- Temporal scaling: Monte Carlo Steps (MCS) with 30-step intervals
- Area-based analysis: <R>² in Monte Carlo Units (MCU²)
- Statistical validation: R² score reporting for growth law validation

Technical Specifications:
- Initial grain count: 20,000 grains
- Domain: 2D polycrystalline systems with crystallographic orientations
- Processing: 64-core parallel (anisotropic), 32-core (isotropic)
- HiPerGator cluster: University of Florida supercomputing infrastructure
- Regression analysis: 200 timesteps for growth law fitting
- Distribution analysis: Optimized special timesteps for ~2000-grain states

Created on Mon Jul 31 14:33:57 2023
@author: Lin

Applications:
- Advanced grain growth kinetics analysis and model validation
- Power-law growth behavior characterization across energy methods
- Large-scale statistical mechanics of polycrystalline evolution
- Energy method benchmarking with growth rate quantification
- HPC-optimized materials science simulation analysis
"""

# Core scientific computing libraries
import os
current_path = os.getcwd()
import numpy as np                    # Numerical array operations and statistical analysis
from numpy import seterr
seterr(all='raise')                  # Enable numpy error checking for numerical stability
import matplotlib.pyplot as plt      # Publication-quality plotting and visualization
import math                          # Mathematical functions for size calculations
from tqdm import tqdm                # Progress bar for computationally intensive loops
import sys

# Add VECTOR framework paths for simulation analysis modules
sys.path.append(current_path)
sys.path.append(current_path+'/../../')
import myInput                       # VECTOR input parameter management
import PACKAGE_MP_Linear as linear2d # 2D linear algebra operations for grain analysis
sys.path.append(current_path+'/../calculate_tangent/')

# Advanced regression and machine learning libraries for growth law analysis
from sklearn.linear_model import LinearRegression    # Linear regression for power-law fitting
from sklearn.preprocessing import PolynomialFeatures # Polynomial feature transformation
from sklearn.metrics import r2_score                 # R² coefficient determination

if __name__ == '__main__':
    # =============================================================================
    # HiPerGator Supercomputing Cluster Data Configuration
    # =============================================================================
    """
    Data source: University of Florida HiPerGator supercomputing cluster
    Simulation scale: Large-scale 20,000-grain polycrystalline systems
    Processing: 64-core parallel anisotropic simulations, 32-core isotropic reference
    Analysis focus: Complete energy method suite with power-law growth kinetics
    """
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/"
    
    # =============================================================================
    # Complete Energy Method Suite for Comprehensive Analysis
    # =============================================================================
    """
    Full set of energy calculation methods for comprehensive grain growth study:
    Each method produces distinct growth kinetics, size distributions, and power-law behavior
    """
    TJ_energy_type_ave = "ave"         # Average energy method (baseline comparison)
    TJ_energy_type_consMin = "consMin" # Conservative minimum (small grain stabilization)
    TJ_energy_type_sum = "sum"         # Summation-based (cumulative energy effects)
    TJ_energy_type_min = "min"         # Pure minimum (maximum stability enhancement)
    TJ_energy_type_max = "max"         # Pure maximum (maximum growth enhancement)
    TJ_energy_type_consMax = "consMax" # Conservative maximum (balanced enhancement)

    # =============================================================================
    # Large-Scale Simulation Data Files (64-core HiPerGator Processing)
    # =============================================================================
    """
    High-performance simulation datasets with enhanced computational resources:
    - 64-core parallel processing for anisotropic simulations
    - 32-core processing for isotropic reference case
    - Consistent seed and parameters across all methods for direct comparison
    """
    npy_file_name_aniso_ave = f"p_ori_ave_{TJ_energy_type_ave}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMin = f"p_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_sum = f"p_ori_ave_{TJ_energy_type_sum}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_min = f"p_ori_ave_{TJ_energy_type_min}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_max = f"p_ori_ave_{TJ_energy_type_max}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMax = f"p_ori_ave_{TJ_energy_type_consMax}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_iso = "p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    
    # =============================================================================
    # Pre-computed Grain Area Data Files for Computational Efficiency
    # =============================================================================
    """
    Cached grain area calculations to avoid repeated computational overhead:
    Note: File names reference 32-core processing for cached area data
    """
    grain_size_data_name_ave = f"grain_size_p_ori_ave_{TJ_energy_type_ave}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_consMin = f"grain_size_p_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_sum = f"grain_size_p_ori_ave_{TJ_energy_type_sum}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_min = f"grain_size_p_ori_ave_{TJ_energy_type_min}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_max = f"grain_size_p_ori_ave_{TJ_energy_type_max}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_consMax = f"grain_size_p_ori_ave_{TJ_energy_type_consMax}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_iso = "grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"

    # =============================================================================
    # Large-Scale Simulation Data Loading and Validation
    # =============================================================================
    """
    Load complete suite of simulation datasets for comprehensive analysis
    Each dataset represents different energy calculation approach with identical initial conditions
    """
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)
    npy_file_aniso_consMin = np.load(npy_file_folder + npy_file_name_aniso_consMin)
    npy_file_aniso_sum = np.load(npy_file_folder + npy_file_name_aniso_sum)
    npy_file_aniso_min = np.load(npy_file_folder + npy_file_name_aniso_min)
    npy_file_aniso_max = np.load(npy_file_folder + npy_file_name_aniso_max)
    npy_file_aniso_consMax = np.load(npy_file_folder + npy_file_name_aniso_consMax)
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)
    
    # Display comprehensive dataset dimensions for verification
    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    print(f"The consMin data size is: {npy_file_aniso_consMin.shape}")
    print(f"The sum data size is: {npy_file_aniso_sum.shape}")
    print(f"The min data size is: {npy_file_aniso_min.shape}")
    print(f"The max data size is: {npy_file_aniso_max.shape}")
    print(f"The consMax data size is: {npy_file_aniso_consMax.shape}")
    print(f"The iso data size is: {npy_file_iso.shape}")
    print("READING DATA DONE")

    # =============================================================================
    # Comprehensive Data Structure Initialization for Multi-Method Analysis
    # =============================================================================
    """
    Initialize complete data arrays for all seven energy methods:
    - Individual grain areas, sizes, and statistical measures
    - Both size (equivalent radius) and area tracking for comprehensive analysis
    - Separate arrays enable direct method-to-method comparison
    """
    initial_grain_num = 20000                          # Initial grain count
    step_num = npy_file_aniso_ave.shape[0]            # Number of timesteps

    # Average energy method arrays
    grain_num_ave = np.zeros(step_num)                 # Grain count evolution
    grain_area_ave = np.zeros((step_num,initial_grain_num))     # Individual grain areas
    grain_size_ave = np.zeros((step_num,initial_grain_num))     # Individual grain sizes (radius)
    grain_ave_size_ave = np.zeros(step_num)            # Average grain size evolution
    grain_ave_area_ave = np.zeros(step_num)            # Average grain area evolution

    # Conservative minimum energy method arrays
    grain_num_consMin = np.zeros(step_num)
    grain_area_consMin = np.zeros((step_num,initial_grain_num))
    grain_size_consMin = np.zeros((step_num,initial_grain_num))
    grain_ave_size_consMin = np.zeros(step_num)
    grain_ave_area_consMin = np.zeros(step_num)

    # Summation energy method arrays
    grain_num_sum = np.zeros(step_num)
    grain_area_sum = np.zeros((step_num,initial_grain_num))
    grain_size_sum = np.zeros((step_num,initial_grain_num))
    grain_ave_size_sum = np.zeros(step_num)
    grain_ave_area_sum = np.zeros(step_num)

    # Minimum energy method arrays
    grain_num_min = np.zeros(step_num)
    grain_area_min = np.zeros((step_num,initial_grain_num))
    grain_size_min = np.zeros((step_num,initial_grain_num))
    grain_ave_size_min = np.zeros(step_num)
    grain_ave_area_min = np.zeros(step_num)

    # Maximum energy method arrays
    grain_num_max = np.zeros(step_num)
    grain_area_max = np.zeros((step_num,initial_grain_num))
    grain_size_max = np.zeros((step_num,initial_grain_num))
    grain_ave_size_max = np.zeros(step_num)
    grain_ave_area_max = np.zeros(step_num)

    # Conservative maximum energy method arrays
    grain_num_consMax = np.zeros(step_num)
    grain_area_consMax = np.zeros((step_num,initial_grain_num))
    grain_size_consMax = np.zeros((step_num,initial_grain_num))
    grain_ave_size_consMax = np.zeros(step_num)
    grain_ave_area_consMax = np.zeros(step_num)

    # Isotropic reference method arrays
    grain_num_iso = np.zeros(step_num)
    grain_area_iso = np.zeros((step_num,initial_grain_num))
    grain_size_iso = np.zeros((step_num,initial_grain_num))
    grain_ave_size_iso = np.zeros(step_num)
    grain_ave_area_iso = np.zeros(step_num)

    # =============================================================================
    # Advanced Statistical Distribution Analysis Configuration
    # =============================================================================
    """
    Enhanced configuration for normalized grain size distribution analysis:
    - Optimized special timesteps for each energy method to achieve ~2000-grain target
    - Method-specific timing accounts for different growth kinetics
    """
    bin_width = 0.16                    # Grain size distribution bin width
    x_limit = [-0.5, 3.5]             # Range for normalized grain size (R/<R>)
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)  # Number of bins
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)  # Bin centers

    # Distribution arrays and optimized timesteps for each energy method
    grain_size_distribution_ave = np.zeros(bin_num)
    special_step_distribution_ave = 11       # Timestep for ave distribution (≈2000 grains)

    grain_size_distribution_consMin = np.zeros(bin_num)
    special_step_distribution_consMin = 11   # Timestep for consMin distribution (≈2000 grains)

    grain_size_distribution_sum = np.zeros(bin_num)
    special_step_distribution_sum = 11       # Timestep for sum distribution (≈2000 grains)

    grain_size_distribution_min = np.zeros(bin_num)
    special_step_distribution_min = 30       # Timestep for min distribution (≈2000 grains)

    grain_size_distribution_max = np.zeros(bin_num)
    special_step_distribution_max = 15       # Timestep for max distribution (≈2000 grains)

    grain_size_distribution_consMax = np.zeros(bin_num)
    special_step_distribution_consMax = 11   # Timestep for consMax distribution (≈2000 grains)

    grain_size_distribution_iso = np.zeros(bin_num)
    special_step_distribution_iso = 10       # Timestep for iso distribution (≈2000 grains)

    # Start grain size initialing
    if os.path.exists(npy_file_folder + grain_size_data_name_iso):
        grain_area_iso = np.load(npy_file_folder + grain_size_data_name_iso)
    else:
        for i in tqdm(range(step_num)):
            for j in range(npy_file_iso.shape[1]):
                for k in range(npy_file_iso.shape[2]):
                    grain_id = int(npy_file_iso[i,j,k,0])
                    grain_area_iso[i,grain_id-1] += 1
        np.save(npy_file_folder + grain_size_data_name_iso, grain_area_iso)
    if os.path.exists(npy_file_folder + grain_size_data_name_ave):
        grain_area_ave = np.load(npy_file_folder + grain_size_data_name_ave)
    else:
        for i in tqdm(range(step_num)):
            for j in range(npy_file_aniso_ave.shape[1]):
                for k in range(npy_file_aniso_ave.shape[2]):
                    grain_id = int(npy_file_aniso_ave[i,j,k,0])
                    grain_area_ave[i,grain_id-1] += 1
        np.save(npy_file_folder + grain_size_data_name_ave, grain_area_ave)
    if os.path.exists(npy_file_folder + grain_size_data_name_consMin):
        grain_area_consMin = np.load(npy_file_folder + grain_size_data_name_consMin)
    else:
        for i in tqdm(range(step_num)):
            for j in range(npy_file_aniso_consMin.shape[1]):
                for k in range(npy_file_aniso_consMin.shape[2]):
                    grain_id = int(npy_file_aniso_consMin[i,j,k,0])
                    grain_area_consMin[i,grain_id-1] += 1
        np.save(npy_file_folder + grain_size_data_name_consMin, grain_area_consMin)
    if os.path.exists(npy_file_folder + grain_size_data_name_min):
        grain_area_min = np.load(npy_file_folder + grain_size_data_name_min)
    else:
        for i in tqdm(range(step_num)):
            for j in range(npy_file_aniso_min.shape[1]):
                for k in range(npy_file_aniso_min.shape[2]):
                    grain_id = int(npy_file_aniso_min[i,j,k,0])
                    grain_area_min[i,grain_id-1] += 1
        np.save(npy_file_folder + grain_size_data_name_min, grain_area_min)
    if os.path.exists(npy_file_folder + grain_size_data_name_sum):
        grain_area_sum = np.load(npy_file_folder + grain_size_data_name_sum)
    else:
        for i in tqdm(range(step_num)):
            for j in range(npy_file_aniso_sum.shape[1]):
                for k in range(npy_file_aniso_sum.shape[2]):
                    grain_id = int(npy_file_aniso_sum[i,j,k,0])
                    grain_area_sum[i,grain_id-1] += 1
        np.save(npy_file_folder + grain_size_data_name_sum, grain_area_sum)
    if os.path.exists(npy_file_folder + grain_size_data_name_consMax):
        grain_area_consMax = np.load(npy_file_folder + grain_size_data_name_consMax)
    else:
        for i in tqdm(range(step_num)):
            for j in range(npy_file_aniso_consMax.shape[1]):
                for k in range(npy_file_aniso_consMax.shape[2]):
                    grain_id = int(npy_file_aniso_consMax[i,j,k,0])
                    grain_area_consMax[i,grain_id-1] += 1
        np.save(npy_file_folder + grain_size_data_name_consMax, grain_area_consMax)
    if os.path.exists(npy_file_folder + grain_size_data_name_max):
        grain_area_max = np.load(npy_file_folder + grain_size_data_name_max)
    else:
        for i in tqdm(range(step_num)):
            for j in range(npy_file_aniso_max.shape[1]):
                for k in range(npy_file_aniso_max.shape[2]):
                    grain_id = int(npy_file_aniso_max[i,j,k,0])
                    grain_area_max[i,grain_id-1] += 1
        np.save(npy_file_folder + grain_size_data_name_max, grain_area_max)
    print("GRAIN SIZE INITIALING DONE")

    # Start grain size analysis
    for i in tqdm(range(step_num)):
        grain_num_ave[i] = np.sum(grain_area_ave[i,:] != 0) # grain num
        grain_num_consMin[i] = np.sum(grain_area_consMin[i,:] != 0) # grain num
        grain_num_sum[i] = np.sum(grain_area_sum[i,:] != 0) # grain num
        grain_num_min[i] = np.sum(grain_area_min[i,:] != 0) # grain num
        grain_num_consMax[i] = np.sum(grain_area_consMax[i,:] != 0) # grain num
        grain_num_max[i] = np.sum(grain_area_max[i,:] != 0) # grain num
        grain_num_iso[i] = np.sum(grain_area_iso[i,:] != 0) # grain num

        grain_size_ave[i] = (grain_area_ave[i] / np.pi)**0.5
        grain_ave_size_ave[i] = np.sum(grain_size_ave[i]) / grain_num_ave[i] # average grain size
        grain_size_consMin[i] = (grain_area_consMin[i] / np.pi)**0.5
        grain_ave_size_consMin[i] = np.sum(grain_size_consMin[i]) / grain_num_consMin[i] # average grain size
        grain_size_sum[i] = (grain_area_sum[i] / np.pi)**0.5
        grain_ave_size_sum[i] = np.sum(grain_size_sum[i]) / grain_num_sum[i] # average grain size
        grain_size_min[i] = (grain_area_min[i] / np.pi)**0.5
        grain_ave_size_min[i] = np.sum(grain_size_min[i]) / grain_num_min[i] # average grain size
        grain_size_consMax[i] = (grain_area_consMax[i] / np.pi)**0.5
        grain_ave_size_consMax[i] = np.sum(grain_size_consMax[i]) / grain_num_consMax[i] # average grain size
        grain_size_max[i] = (grain_area_max[i] / np.pi)**0.5
        grain_ave_size_max[i] = np.sum(grain_size_max[i]) / grain_num_max[i] # average grain size
        grain_size_iso[i] = (grain_area_iso[i] / np.pi)**0.5
        grain_ave_size_iso[i] = np.sum(grain_size_iso[i]) / grain_num_iso[i] # average grain size

        grain_ave_area_ave[i] = np.sum(grain_area_ave[i]) / grain_num_ave[i] # average grain size
        grain_ave_area_consMin[i] = np.sum(grain_area_consMin[i]) / grain_num_consMin[i] # average grain size
        grain_ave_area_sum[i] = np.sum(grain_area_sum[i]) / grain_num_sum[i] # average grain size
        grain_ave_area_min[i] = np.sum(grain_area_min[i]) / grain_num_min[i] # average grain size
        grain_ave_area_consMax[i] = np.sum(grain_area_consMax[i]) / grain_num_consMax[i] # average grain size
        grain_ave_area_max[i] = np.sum(grain_area_max[i]) / grain_num_max[i] # average grain size
        grain_ave_area_iso[i] = np.sum(grain_area_iso[i]) / grain_num_iso[i] # average grain size

        if i == special_step_distribution_ave:
            # Aniso
            special_size_ave = grain_size_ave[i][grain_size_ave[i] != 0] # remove zero grain size
            special_size_ave = special_size_ave/grain_ave_size_ave[i] # normalize grain size
            for j in range(len(special_size_ave)):
                grain_size_distribution_ave[int((special_size_ave[j]-x_limit[0])/bin_width)] += 1 # Get frequency
        if i == special_step_distribution_consMin:
            # Aniso
            special_size_consMin = grain_size_consMin[i][grain_size_consMin[i] != 0] # remove zero grain size
            special_size_consMin = special_size_consMin/grain_ave_size_consMin[i] # normalize grain size
            for j in range(len(special_size_consMin)):
                grain_size_distribution_consMin[int((special_size_consMin[j]-x_limit[0])/bin_width)] += 1 # Get frequency
        if i == special_step_distribution_sum:
            # Aniso
            special_size_sum = grain_size_sum[i][grain_size_sum[i] != 0] # remove zero grain size
            special_size_sum = special_size_sum/grain_ave_size_sum[i] # normalize grain size
            for j in range(len(special_size_sum)):
                grain_size_distribution_sum[int((special_size_sum[j]-x_limit[0])/bin_width)] += 1 # Get frequency
        if i == special_step_distribution_min:
            # Aniso
            special_size_min = grain_size_min[i][grain_size_min[i] != 0] # remove zero grain size
            special_size_min = special_size_min/grain_ave_size_min[i] # normalize grain size
            for j in range(len(special_size_min)):
                grain_size_distribution_min[int((special_size_min[j]-x_limit[0])/bin_width)] += 1 # Get frequency
        if i == special_step_distribution_consMax:
            # Aniso
            special_size_consMax = grain_size_consMax[i][grain_size_consMax[i] != 0] # remove zero grain size
            special_size_consMax = special_size_consMax/grain_ave_size_consMax[i] # normalize grain size
            for j in range(len(special_size_consMax)):
                grain_size_distribution_consMax[int((special_size_consMax[j]-x_limit[0])/bin_width)] += 1 # Get frequency
        if i == special_step_distribution_max:
            # Aniso
            special_size_max = grain_size_max[i][grain_size_max[i] != 0] # remove zero grain size
            special_size_max = special_size_max/grain_ave_size_max[i] # normalize grain size
            for j in range(len(special_size_max)):
                grain_size_distribution_max[int((special_size_max[j]-x_limit[0])/bin_width)] += 1 # Get frequency
        if i == special_step_distribution_iso:
            # Iso
            special_size_iso = grain_size_iso[i][grain_size_iso[i] != 0] # remove zero grain size
            special_size_iso = special_size_iso/grain_ave_size_iso[i] # normalize grain size
            for j in range(len(special_size_iso)):
                grain_size_distribution_iso[int((special_size_iso[j]-x_limit[0])/bin_width)] += 1 # Get frequency
    grain_size_distribution_ave = grain_size_distribution_ave/np.sum(grain_size_distribution_ave*bin_width) # normalize frequency
    grain_size_distribution_consMin = grain_size_distribution_consMin/np.sum(grain_size_distribution_consMin*bin_width) # normalize frequency
    grain_size_distribution_sum = grain_size_distribution_sum/np.sum(grain_size_distribution_sum*bin_width) # normalize frequency
    grain_size_distribution_min = grain_size_distribution_min/np.sum(grain_size_distribution_min*bin_width) # normalize frequency
    grain_size_distribution_consMax = grain_size_distribution_consMax/np.sum(grain_size_distribution_consMax*bin_width) # normalize frequency
    grain_size_distribution_max = grain_size_distribution_max/np.sum(grain_size_distribution_max*bin_width) # normalize frequency
    grain_size_distribution_iso = grain_size_distribution_iso/np.sum(grain_size_distribution_iso*bin_width) # normalize frequency
    print("GRAIN SIZE ANALYSIS DONE")

    # Start plotting
    plt.clf()
    plt.plot(np.array(range(step_num))*30, grain_ave_area_min/np.pi, label="Min", linewidth=2)
    plt.plot(np.array(range(step_num))*30, grain_ave_area_max/np.pi, label="Max", linewidth=2)
    plt.plot(np.array(range(step_num))*30, grain_ave_area_ave/np.pi, label="Ave", linewidth=2)
    plt.plot(np.array(range(step_num))*30, grain_ave_area_sum/np.pi, label="Sum", linewidth=2)
    plt.plot(np.array(range(step_num))*30, grain_ave_area_consMin/np.pi, label="CMin", linewidth=2)
    plt.plot(np.array(range(step_num))*30, grain_ave_area_consMax/np.pi, label="CMax", linewidth=2)
    plt.plot(np.array(range(step_num))*30, grain_ave_area_iso/np.pi, label="Iso", linewidth=2)

    plt.xlabel("Timestep (MCS)", fontsize=16)
    plt.ylabel(r"$\langle$R$\rangle^2$ (MCU$^2$)", fontsize=16)
    plt.xticks([0,5000,10000,15000],fontsize=16)
    plt.ticklabel_format(style='sci',scilimits=(-1,2),axis='y')
    plt.legend(fontsize=16, ncol=2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(current_path + "/figures/ave_grain_area_over_time.png", dpi=400,bbox_inches='tight')

    plt.clf()
    plt.plot(size_coordination, grain_size_distribution_min, label="Min", linewidth=2)
    plt.plot(size_coordination, grain_size_distribution_max, label="Max", linewidth=2)
    plt.plot(size_coordination, grain_size_distribution_ave, label="Ave", linewidth=2)
    plt.plot(size_coordination, grain_size_distribution_sum, label="Sum", linewidth=2)
    plt.plot(size_coordination, grain_size_distribution_consMin, label="CMin", linewidth=2)
    plt.plot(size_coordination, grain_size_distribution_consMax, label="CMax", linewidth=2)
    plt.plot(size_coordination, grain_size_distribution_iso, label="Iso", linewidth=2)
    plt.xlabel(r"R/$\langle$R$\rangle$", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.title(r"Grain number $\approx$ 2000", fontsize=16)
    plt.savefig(current_path + "/figures/normalized_size_distribution.png", dpi=400,bbox_inches='tight')

    # Fitting n for r^n - r0^n = k t
    short_step_num = 200
    n_value_list = np.linspace(0.5,3.5,301)
    r2_score_min_list = np.zeros(len(n_value_list))
    r2_score_max_list = np.zeros(len(n_value_list))
    r2_score_ave_list = np.zeros(len(n_value_list))
    r2_score_sum_list = np.zeros(len(n_value_list))
    r2_score_consMin_list = np.zeros(len(n_value_list))
    r2_score_consMax_list = np.zeros(len(n_value_list))
    r2_score_iso_list = np.zeros(len(n_value_list))

    for i in range(len(n_value_list)):
        n_value = n_value_list[i]

        # ave
        x_list = np.array(list(range(step_num)))
        y_list = grain_ave_size_ave ** n_value - grain_ave_size_ave[0] ** n_value
        model_ave = LinearRegression().fit(x_list[:short_step_num].reshape(-1,1), y_list[:short_step_num].reshape(-1,1))
        r2_score_ave = model_ave.score(x_list[:short_step_num].reshape(-1,1), y_list[:short_step_num].reshape(-1,1))
        r2_score_ave_list[i] = r2_score_ave

        # min
        y_list = grain_ave_size_min ** n_value - grain_ave_size_min[0] ** n_value
        model_min = LinearRegression().fit(x_list[:short_step_num].reshape(-1,1), y_list[:short_step_num].reshape(-1,1))
        r2_score_min = model_min.score(x_list[:short_step_num].reshape(-1,1), y_list[:short_step_num].reshape(-1,1))
        r2_score_min_list[i] = r2_score_min

        # max
        y_list = grain_ave_size_max ** n_value - grain_ave_size_max[0] ** n_value
        model_max = LinearRegression().fit(x_list[:short_step_num].reshape(-1,1), y_list[:short_step_num].reshape(-1,1))
        r2_score_max = model_max.score(x_list[:short_step_num].reshape(-1,1), y_list[:short_step_num].reshape(-1,1))
        r2_score_max_list[i] = r2_score_max

        # sum
        y_list = grain_ave_size_sum ** n_value - grain_ave_size_sum[0] ** n_value
        model_sum = LinearRegression().fit(x_list[:short_step_num].reshape(-1,1), y_list[:short_step_num].reshape(-1,1))
        r2_score_sum = model_sum.score(x_list[:short_step_num].reshape(-1,1), y_list[:short_step_num].reshape(-1,1))
        r2_score_sum_list[i] = r2_score_sum

        # consMin
        y_list = grain_ave_size_consMin ** n_value - grain_ave_size_consMin[0] ** n_value
        model_consMin = LinearRegression().fit(x_list[:short_step_num].reshape(-1,1), y_list[:short_step_num].reshape(-1,1))
        r2_score_consMin = model_consMin.score(x_list[:short_step_num].reshape(-1,1), y_list[:short_step_num].reshape(-1,1))
        r2_score_consMin_list[i] = r2_score_consMin

        # consMax
        y_list = grain_ave_size_consMax ** n_value - grain_ave_size_consMax[0] ** n_value
        model_consMax = LinearRegression().fit(x_list[:short_step_num].reshape(-1,1), y_list[:short_step_num].reshape(-1,1))
        r2_score_consMax = model_consMax.score(x_list[:short_step_num].reshape(-1,1), y_list[:short_step_num].reshape(-1,1))
        r2_score_consMax_list[i] = r2_score_consMax

        # iso
        y_list = grain_ave_size_iso ** n_value - grain_ave_size_iso[0] ** n_value
        model_iso = LinearRegression().fit(x_list[:short_step_num].reshape(-1,1), y_list[:short_step_num].reshape(-1,1))
        r2_score_iso = model_iso.score(x_list[:short_step_num].reshape(-1,1), y_list[:short_step_num].reshape(-1,1))
        r2_score_iso_list[i] = r2_score_iso

    print(f"For iso, the n is {n_value_list[int(np.argmax(r2_score_iso_list))]}, and the r^2 is {np.max(r2_score_iso_list)}.")
    print(f"For min, the n is {n_value_list[int(np.argmax(r2_score_min_list))]}, and the r^2 is {np.max(r2_score_min_list)}.")
    print(f"For max, the n is {n_value_list[int(np.argmax(r2_score_max_list))]}, and the r^2 is {np.max(r2_score_max_list)}.")
    print(f"For ave, the n is {n_value_list[int(np.argmax(r2_score_ave_list))]}, and the r^2 is {np.max(r2_score_ave_list)}.")
    print(f"For sum, the n is {n_value_list[int(np.argmax(r2_score_sum_list))]}, and the r^2 is {np.max(r2_score_sum_list)}.")
    print(f"For consMin, the n is {n_value_list[int(np.argmax(r2_score_consMin_list))]}, and the r^2 is {np.max(r2_score_consMin_list)}.")
    print(f"For consMax, the n is {n_value_list[int(np.argmax(r2_score_consMax_list))]}, and the r^2 is {np.max(r2_score_consMax_list)}.")








