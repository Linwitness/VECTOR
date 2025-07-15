#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grain Size Distribution Analysis: Comparative Study of 2D MCP vs. 3D Phase Field Methods

This script provides comprehensive analysis and visualization of grain size distributions
across different simulation methodologies and dimensionalities. The analysis compares
2D Monte Carlo Potts (MCP) model results with 3D Phase Field (PF) simulations and
theoretical distributions (Hillert distribution).

Scientific Purpose:
- Cross-validates grain size distributions between MCP and Phase Field methods
- Analyzes dimensional effects (2D vs. 3D) on grain size distribution characteristics
- Compares simulation results with theoretical grain growth models (Hillert)
- Validates energy method effects on statistical grain size behavior

Key Features:
- Multi-energy method grain size distribution comparison
- Normalized grain size analysis (R/<R>) for scale-independent comparison
- Integration with experimental reference data (3705-grain state)
- Statistical binning and frequency analysis for distribution characterization
- Publication-quality visualization with proper normalization

Energy Methods Analyzed:
- ave: Average triple junction energy approach
- consMin: Conservative minimum energy selection
- consMax: Conservative maximum energy selection
- sum: Summation-based energy calculation
- min: Pure minimum energy criterion
- max: Pure maximum energy criterion
- iso: Isotropic reference case (delta=0.0)

Distribution Analysis:
- Target grain count: 3705 grains (matching experimental reference)
- Bin width: 0.16 (optimized for grain size distribution analysis)
- Range: R/<R> from -0.5 to 3.5 (comprehensive size spectrum)
- Normalization: Area-under-curve = 1.0 for proper statistical comparison

Technical Specifications:
- Initial grain count: 20,000 grains
- Domain: 2D polycrystalline systems (MCP) vs. 3D (Phase Field)
- Processing: 64-core parallel (anisotropic), 32-core (isotropic)
- Size calculation: Equivalent circular radius from grain area
- Statistical validation: Frequency normalization and bin width correction

Created on Wed Oct 18 15:44:12 2023
@author: Lin

Applications:
- Grain growth model validation studies
- Statistical mechanics of polycrystalline evolution
- Dimensional scaling analysis (2D-3D comparison)
- Theoretical distribution validation (Hillert vs. simulation)
"""

# Core scientific computing libraries
import os
current_path = os.getcwd()
import numpy as np                    # Numerical array operations and statistical analysis
from numpy import seterr
seterr(all='raise')                  # Enable numpy error checking for numerical stability
import matplotlib.pyplot as plt      # Publication-quality plotting and visualization
import math                          # Mathematical functions for size calculations
from tqdm import tqdm                # Progress bar for long-running computations
import sys

# Add VECTOR framework paths for simulation analysis modules
sys.path.append(current_path)
sys.path.append(current_path+'/../../')
import myInput                       # VECTOR input parameter management
import PACKAGE_MP_Linear as linear2d # 2D linear algebra operations for grain analysis
sys.path.append(current_path+'/../calculate_tangent/')
import csv                           # CSV data handling for Phase Field and experimental reference data

if __name__ == '__main__':
    # =============================================================================
    # HiPerGator Data Source Configuration
    # =============================================================================
    """
    Data source: University of Florida HiPerGator supercomputing cluster
    Simulation type: Large-scale polycrystalline grain growth (20,000 initial grains)
    Analysis focus: Grain size distribution comparison across methods and dimensions
    Reference state: 3705-grain configuration for statistical comparison
    """
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/"

    # =============================================================================
    # Energy Method Definitions for Statistical Analysis
    # =============================================================================
    """
    Energy calculation methods for comparative grain size distribution analysis:
    Each method produces different grain growth kinetics and final size distributions
    """
    TJ_energy_type_ave = "ave"           # Average energy method (baseline statistical behavior)
    TJ_energy_type_consMin = "consMin"   # Conservative minimum (enhanced small grain stability)
    TJ_energy_type_sum = "sum"           # Summation-based (cumulative energy effects)
    TJ_energy_type_min = "min"           # Pure minimum (maximum small grain preservation)
    TJ_energy_type_max = "max"           # Pure maximum (enhanced large grain growth)
    TJ_energy_type_consMax = "consMax"   # Conservative maximum (balanced large grain selection)
    
    # =============================================================================
    # Anisotropic Grain Structure Data Files
    # =============================================================================
    """
    Grain structure evolution data for each energy method
    Format: 4D arrays [time, x, y, grain_id] containing spatial grain arrangements
    """
    npy_file_name_aniso_ave = f"p_ori_ave_{TJ_energy_type_ave}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMin = f"p_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_sum = f"p_ori_ave_{TJ_energy_type_sum}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_min = f"p_ori_ave_{TJ_energy_type_min}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_max = f"p_ori_ave_{TJ_energy_type_max}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMax = f"p_ori_ave_{TJ_energy_type_consMax}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    
    # Isotropic reference case
    npy_file_name_iso = "p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    
    # =============================================================================
    # Grain Size Data Files (Pre-computed Area Analysis)
    # =============================================================================
    """
    Pre-computed grain area data for each energy method
    Format: 2D arrays [time, grain_id] containing individual grain areas
    Used for efficient grain size distribution calculations
    """
    grain_size_data_name_ave = f"grain_size_p_ori_ave_{TJ_energy_type_ave}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_consMin = f"grain_size_p_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_sum = f"grain_size_p_ori_ave_{TJ_energy_type_sum}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_min = f"grain_size_p_ori_ave_{TJ_energy_type_min}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_max = f"grain_size_p_ori_ave_{TJ_energy_type_max}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_consMax = f"grain_size_p_ori_ave_{TJ_energy_type_consMax}E_20000_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    grain_size_data_name_iso = "grain_size_p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    
    # =============================================================================
    # Reference Distribution Data (Experimental and Theoretical)
    # =============================================================================
    """
    CSV files containing reference grain size distributions:
    - 3705: Experimental or high-fidelity simulation reference at specific grain count
    - Hillert: Theoretical grain size distribution from Hillert model
    Both normalized as frequency vs. normalized grain size (R/<R>)
    """
    csv_file_name_3705 = "grain_size_distribution_3705.csv"     # Experimental reference
    csv_file_name_hillert = "grain_size_distribution_Hillert.csv"  # Theoretical Hillert distribution
    
    # =============================================================================
    # Grain Structure Data Loading and Validation
    # =============================================================================
    """
    Load all simulation datasets for comparative analysis
    Each dataset contains temporal evolution of grain structures
    """
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)
    npy_file_aniso_consMin = np.load(npy_file_folder + npy_file_name_aniso_consMin)
    npy_file_aniso_sum = np.load(npy_file_folder + npy_file_name_aniso_sum)
    npy_file_aniso_min = np.load(npy_file_folder + npy_file_name_aniso_min)
    npy_file_aniso_max = np.load(npy_file_folder + npy_file_name_aniso_max)
    npy_file_aniso_consMax = np.load(npy_file_folder + npy_file_name_aniso_consMax)
    npy_file_iso = np.load(npy_file_folder + npy_file_name_iso)
    
    # Display dataset dimensions for verification
    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    print(f"The consMin data size is: {npy_file_aniso_consMin.shape}")
    print(f"The sum data size is: {npy_file_aniso_sum.shape}")
    print(f"The min data size is: {npy_file_aniso_min.shape}")
    print(f"The max data size is: {npy_file_aniso_max.shape}")
    print(f"The consMax data size is: {npy_file_aniso_consMax.shape}")
    print(f"The iso data size is: {npy_file_iso.shape}")
    print("READING DATA DONE")
    
    # =============================================================================
    # Reference Distribution Data Processing
    # =============================================================================
    """
    Load experimental/theoretical reference distributions for validation
    Format: CSV files with normalized grain size (R/<R>) and frequency data
    """
    csv_file_3705_r = []           # Normalized grain sizes for 3705-grain reference
    csv_file_3705_frequency = []   # Frequency distribution for 3705-grain reference
    
    # Load 3705-grain reference distribution
    with open(npy_file_folder + csv_file_name_3705, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            csv_file_3705_r.append(float(row[0]))
            csv_file_3705_frequency.append(float(row[1]))
            
    csv_file_hillert_r = []        # Normalized grain sizes for Hillert distribution
    csv_file_hillert_frequency = []  # Frequency distribution for Hillert model
    
    # Load Hillert theoretical distribution
    with open(npy_file_folder + csv_file_name_hillert, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            csv_file_hillert_r.append(float(row[0]))
            csv_file_hillert_frequency.append(float(row[1]))
            
    # Convert to numpy arrays for efficient numerical operations
    csv_file_3705_r = np.array(csv_file_3705_r)
    csv_file_3705_frequency = np.array(csv_file_3705_frequency)
    csv_file_hillert_r = np.array(csv_file_hillert_r)
    csv_file_hillert_frequency = np.array(csv_file_hillert_frequency)
    
    # =============================================================================
    # Commented Distribution Normalization Code
    # =============================================================================
    """
    Optional normalization approach for ensuring proper area-under-curve = 1.0
    Currently commented out - distributions assumed to be pre-normalized
    
    PF_bin_width = csv_file_3705_r - np.hstack((np.array(0),csv_file_3705_r[:-1]))
    PF_area = np.sum(csv_file_3705_frequency*PF_bin_width)
    Hillert_bin_width = csv_file_hillert_r - np.hstack((np.array(0),csv_file_hillert_r[:-1]))
    Hillert_area = np.sum(csv_file_hillert_frequency*Hillert_bin_width)
    """
        
    # =============================================================================
    # Target Timestep Identification for Comparative Analysis
    # =============================================================================
    """
    Identify timesteps where grain counts match reference state (3705 grains)
    This enables direct statistical comparison at equivalent evolution states
    """
    step_num = npy_file_aniso_ave.shape[0]  # Total number of timesteps
    
    # Calculate grain numbers at each timestep for target identification
    grain_num_MCP_iso = np.zeros(step_num)   # Isotropic grain evolution
    grain_num_MCP_ave = np.zeros(step_num)   # Anisotropic grain evolution (ave method)
    
    for i in range(step_num):
        # Count unique grain IDs at each timestep
        grain_num_MCP_iso[i] = len(list(set(npy_file_iso[i].reshape(-1))))
        grain_num_MCP_ave[i] = len(list(set(npy_file_aniso_ave[i].reshape(-1))))
        
    # Find timesteps closest to target grain count (3705)
    special_time_step = np.argmin(abs(grain_num_MCP_ave - 3705))      # Anisotropic target timestep
    special_time_step_grain_num = grain_num_MCP_ave[special_time_step]  # Actual grain count at target
    special_time_step_iso = np.argmin(abs(grain_num_MCP_iso - 3705))   # Isotropic target timestep
    special_time_step_grain_num_iso = grain_num_MCP_iso[special_time_step_iso]  # Actual grain count at target
    
    # =============================================================================
    # Grain Size Data Loading with Existence Verification
    # =============================================================================
    """
    Load pre-computed grain area data if available
    Grain areas used for equivalent circular radius calculations
    """
    if os.path.exists(npy_file_folder + grain_size_data_name_ave):
        grain_area_ave = np.load(npy_file_folder + grain_size_data_name_ave)
    if os.path.exists(npy_file_folder + grain_size_data_name_iso):
        grain_area_iso = np.load(npy_file_folder + grain_size_data_name_iso)
        
    # =============================================================================
    # Grain Size Distribution Analysis
    # =============================================================================
    """
    Convert grain areas to equivalent circular radii and calculate size distributions
    
    Size calculation: R = sqrt(Area/π) (equivalent circular radius)
    Normalization: R/<R> where <R> is the average grain size
    """
    # Calculate equivalent circular radii from grain areas
    grain_size_ave = (grain_area_ave[special_time_step] / np.pi)**0.5
    grain_ave_size_ave = np.sum(grain_size_ave) / special_time_step_grain_num  # Average grain size (anisotropic)
    
    grain_size_iso = (grain_area_iso[special_time_step_iso] / np.pi)**0.5
    grain_ave_size_iso = np.sum(grain_size_iso) / special_time_step_grain_num_iso  # Average grain size (isotropic)
    
    # =============================================================================
    # Statistical Binning and Distribution Calculation
    # =============================================================================
    """
    Create normalized grain size distributions using histogram binning
    
    Parameters:
    - bin_width: 0.16 (optimized for grain size distribution resolution)
    - x_limit: [-0.5, 3.5] (comprehensive range for normalized sizes)
    - Normalization: Frequency per bin width for proper statistical comparison
    """
    bin_width = 0.16  # Grain size distribution bin width (optimized for resolution)
    x_limit = [-0.5, 3.5]  # Range for normalized grain size (R/<R>)
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)  # Number of bins
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)  # Bin centers
    
    # Initialize distribution arrays
    grain_size_distribution_ave = np.zeros(bin_num)  # Anisotropic distribution
    grain_size_distribution_iso = np.zeros(bin_num)  # Isotropic distribution
    
    # Process anisotropic grain size distribution
    special_size_ave = grain_size_ave[grain_size_ave != 0]  # Remove zero-area grains
    special_size_ave = special_size_ave/grain_ave_size_ave  # Normalize by average size
    for j in range(len(special_size_ave)):
        # Bin each grain size into appropriate histogram bin
        grain_size_distribution_ave[int((special_size_ave[j]-x_limit[0])/bin_width)] += 1
    # Normalize frequency distribution (area under curve = 1.0)
    grain_size_distribution_ave = grain_size_distribution_ave/np.sum(grain_size_distribution_ave*bin_width)
    
    # Process isotropic grain size distribution
    special_size_iso = grain_size_iso[grain_size_iso != 0]  # Remove zero-area grains
    special_size_iso = special_size_iso/grain_ave_size_iso  # Normalize by average size
    for j in range(len(special_size_iso)):
        # Bin each grain size into appropriate histogram bin
        grain_size_distribution_iso[int((special_size_iso[j]-x_limit[0])/bin_width)] += 1
    # Normalize frequency distribution (area under curve = 1.0)
    grain_size_distribution_iso = grain_size_distribution_iso/np.sum(grain_size_distribution_iso*bin_width)
        
    # =============================================================================
    # Comparative Visualization: MCP vs. Phase Field vs. Theory
    # =============================================================================
    """
    Generate publication-quality comparison plot showing:
    - 3D anisotropic Phase Field reference
    - 3D Hillert theoretical distribution
    - 2D anisotropic MCP simulation results
    - 2D isotropic MCP simulation results
    
    Purpose: Validate MCP results against established references and theory
    """
    plt.clf()
    # Reference distributions
    plt.plot(csv_file_3705_r, csv_file_3705_frequency, 
             label="3D aniso PF", linewidth=2)  # 3D Phase Field reference
    plt.plot(csv_file_hillert_r, csv_file_hillert_frequency, 
             label="3D Hillert", linewidth=2)   # Theoretical Hillert distribution
    
    # MCP simulation results
    plt.plot(size_coordination, grain_size_distribution_ave, 
             label=f"2D aniso MCP", linewidth=2)  # 2D Anisotropic MCP
    plt.plot(size_coordination, grain_size_distribution_iso, 
             label=f"2D iso MCP", linewidth=2)    # 2D Isotropic MCP
    
    # Formatting for publication quality
    plt.xlabel(r"R/$\langle$R$\rangle$", fontsize=18)  # Normalized grain size
    plt.ylabel("Frequency", fontsize=18)                # Probability density
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.ylim([0,1.5])      # Frequency range for clear visualization
    plt.xlim(x_limit)      # Normalized size range
    
    # Save high-resolution figure for publication
    plt.savefig(npy_file_folder + "/size_figure/grain_size_distribution_MCP_PF.png", 
                dpi=400, bbox_inches='tight')




