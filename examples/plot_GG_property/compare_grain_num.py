#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grain Number Evolution Comparison: MCP vs. Phase Field Methods

This script provides comprehensive analysis and visualization of grain number evolution 
in polycrystalline systems using both Monte Carlo Potts (MCP) model and Phase Field (PF) 
simulations. The analysis focuses on comparing isotropic and anisotropic energy formulations 
across different energy calculation methods.

Scientific Purpose:
- Validates MCP simulation results against Phase Field reference data
- Analyzes grain number evolution kinetics under different energy formulations
- Quantifies differences between isotropic and anisotropic grain growth behavior
- Provides temporal scaling analysis between MCP and PF simulation frameworks

Key Features:
- Multi-energy method comparison (ave, consMin, sum, min, max, consMax)
- Temporal evolution analysis with proper scaling parameters
- HiPerGator data integration for large-scale simulation datasets
- Publication-quality visualization with logarithmic scaling
- CSV data integration for Phase Field reference comparisons

Energy Methods Analyzed:
- ave: Average triple junction energy approach
- consMin: Conservative minimum energy selection
- consMax: Conservative maximum energy selection  
- sum: Summation-based energy calculation
- min: Pure minimum energy criterion
- max: Pure maximum energy criterion
- iso: Isotropic reference case (delta=0.0)

Technical Specifications:
- Initial grain count: 20,000 grains
- Domain: 2D polycrystalline systems
- Processing: 64-core parallel (anisotropic), 32-core (isotropic)
- Temporal resolution: 30 MCS intervals
- Scaling parameter: 0.195 (MCP-PF temporal alignment)

Created on Wed Oct 18 15:44:12 2023
@author: Lin

Applications:
- Grain growth model validation studies
- Energy method benchmarking and comparison
- Temporal evolution analysis for materials science
- Simulation framework cross-validation
"""

# Core scientific computing libraries
import os
current_path = os.getcwd()
import numpy as np                    # Numerical array operations and data processing
from numpy import seterr
seterr(all='raise')                  # Enable numpy error checking for numerical stability
import matplotlib.pyplot as plt      # Publication-quality plotting and visualization
import math                          # Mathematical functions and constants
from tqdm import tqdm                # Progress bar for long-running computations
import sys

# Add VECTOR framework paths for simulation analysis modules
sys.path.append(current_path)
sys.path.append(current_path+'/../../')
import myInput                       # VECTOR input parameter management
import PACKAGE_MP_Linear as linear2d # 2D linear algebra operations for grain analysis
sys.path.append(current_path+'/../calculate_tangent/')
import csv                           # CSV data handling for Phase Field reference data

if __name__ == '__main__':
    # =============================================================================
    # HiPerGator Data Source Configuration
    # =============================================================================
    """
    Data source: University of Florida HiPerGator supercomputing cluster
    Simulation type: Large-scale polycrystalline grain growth (20,000 initial grains)
    Processing: Multi-core parallel simulations (32-64 cores)
    Domain: 2D oriented polycrystalline systems with anisotropic energy
    """
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/"
    
    # =============================================================================
    # Energy Method Definitions for Comparative Analysis
    # =============================================================================
    """
    Energy calculation methods for triple junction energy formulations:
    - ave: Arithmetic average of triple junction energies
    - consMin: Conservative minimum energy selection with stability constraints
    - consMax: Conservative maximum energy selection for enhanced anisotropy
    - sum: Summation-based approach for cumulative energy effects
    - min: Pure minimum energy criterion for stability analysis
    - max: Pure maximum energy criterion for growth enhancement
    """
    TJ_energy_type_ave = "ave"           # Average energy method (baseline)
    TJ_energy_type_consMin = "consMin"   # Conservative minimum energy
    TJ_energy_type_sum = "sum"           # Summation-based energy
    TJ_energy_type_min = "min"           # Pure minimum energy
    TJ_energy_type_max = "max"           # Pure maximum energy
    TJ_energy_type_consMax = "consMax"   # Conservative maximum energy
    
    # =============================================================================
    # Anisotropic System File Names (delta=0.6, m=2, J=1)
    # =============================================================================
    """
    File naming convention: p_ori_ave_{energy_type}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy
    - p: Polycrystalline system
    - ori: Oriented grains with crystallographic texture
    - ave: Average-based initial energy calculation
    - 20000: Initial grain count
    - multiCore64: 64-core parallel processing
    - delta0.6: Anisotropy parameter (0.6 = moderate anisotropy)
    - m2: Mobility parameter
    - J1: Grain boundary energy parameter
    - seed56689: Random seed for reproducibility
    - kt066: Temperature parameter (0.66)
    """
    npy_file_name_aniso_ave = f"p_ori_ave_{TJ_energy_type_ave}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMin = f"p_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_sum = f"p_ori_ave_{TJ_energy_type_sum}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_min = f"p_ori_ave_{TJ_energy_type_min}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_max = f"p_ori_ave_{TJ_energy_type_max}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMax = f"p_ori_ave_{TJ_energy_type_consMax}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    
    # =============================================================================
    # Isotropic Reference System (delta=0.0)
    # =============================================================================
    """
    Isotropic reference case for comparison with anisotropic results
    - delta0.0: No anisotropy (isotropic grain boundary energy)
    - multiCore32: 32-core processing (reduced computational demand)
    - Same initial conditions and parameters for direct comparison
    """
    npy_file_name_iso = "p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    
    # =============================================================================
    # Phase Field Reference Data (CSV format)
    # =============================================================================
    """
    CSV files contain Phase Field simulation results for validation:
    - Column 1: Simulation timestep
    - Column 2: Grain number (scaled by 1e4 for proper units)
    - Used as ground truth for MCP model validation
    """
    csv_file_name_iso = "grain_num_iso.csv"       # Isotropic Phase Field data
    csv_file_name_aniso = "grain_num_aniso.csv"   # Anisotropic Phase Field data
    
    # =============================================================================
    # Data Loading and Validation
    # =============================================================================
    """
    Load all simulation datasets and verify data integrity
    Each dataset contains temporal evolution of grain structures
    Data format: [timestep, x_coord, y_coord] with grain IDs
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
    # Phase Field Reference Data Processing
    # =============================================================================
    """
    Load and process Phase Field simulation results for comparison
    Phase Field data provides ground truth for validation of MCP results
    Data scaling: Grain numbers multiplied by 1e4 for proper units
    """
    csv_file_iso_step = []          # Timesteps for isotropic PF simulation
    csv_file_iso_grain_num = []     # Grain numbers for isotropic PF simulation
    
    # Load isotropic Phase Field data
    with open(npy_file_folder + csv_file_name_iso, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            csv_file_iso_step.append(float(row[0]))
            csv_file_iso_grain_num.append(float(row[1])*1e4)  # Scale to proper grain count units
            
    csv_file_aniso_step = []        # Timesteps for anisotropic PF simulation
    csv_file_aniso_grain_num = []   # Grain numbers for anisotropic PF simulation
    
    # Load anisotropic Phase Field data
    with open(npy_file_folder + csv_file_name_aniso, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            csv_file_aniso_step.append(float(row[0]))
            csv_file_aniso_grain_num.append(float(row[1])*1e4)  # Scale to proper grain count units
            
    # Convert to numpy arrays for efficient numerical operations
    csv_file_iso_step = np.array(csv_file_iso_step)
    csv_file_iso_grain_num = np.array(csv_file_iso_grain_num)
    csv_file_aniso_step = np.array(csv_file_aniso_step)
    csv_file_aniso_grain_num = np.array(csv_file_aniso_grain_num)
        
    # =============================================================================
    # MCP Grain Number Analysis
    # =============================================================================
    """
    Extract grain numbers from MCP simulation data by counting unique grain IDs
    Method: Count unique values in flattened grain ID arrays at each timestep
    Purpose: Direct comparison with Phase Field grain evolution
    """
    initial_grain_num = 20000       # Initial grain count for all simulations
    step_num = npy_file_aniso_ave.shape[0]  # Number of timesteps in simulation
    
    # Initialize arrays for grain number evolution
    grain_num_MCP_iso = np.zeros(step_num)   # Isotropic MCP grain numbers
    grain_num_MCP_ave = np.zeros(step_num)   # Anisotropic MCP grain numbers (ave method)
    
    # Calculate grain numbers at each timestep
    for i in range(step_num):
        # Count unique grain IDs in each timestep
        grain_num_MCP_iso[i] = len(list(set(npy_file_iso[i].reshape(-1))))
        grain_num_MCP_ave[i] = len(list(set(npy_file_aniso_ave[i].reshape(-1))))
    
    # =============================================================================
    # Temporal Evolution Visualization and Comparison
    # =============================================================================
    """
    Generate publication-quality plot comparing MCP and Phase Field results
    
    Key elements:
    - Temporal scaling parameter (0.195) aligns MCP and PF time scales
    - Logarithmic y-axis for grain number evolution visualization
    - Time offset alignment for proper comparison
    - Limited time range (0-500 MCS) focuses on active growth period
    - Grain number range (1000-20000) captures significant evolution
    """
    scaling_parameter = 0.195  # Temporal scaling factor for MCP-PF alignment
    
    plt.clf()
    # Phase Field results (reference data)
    plt.plot((csv_file_iso_step[12:]-csv_file_iso_step[12])*scaling_parameter, 
             csv_file_iso_grain_num[12:], label="Iso - PF", linewidth=2)
    plt.plot((csv_file_aniso_step-csv_file_iso_step[12])*scaling_parameter, 
             csv_file_aniso_grain_num, label="Aniso - PF", linewidth=2)
    
    # MCP simulation results
    plt.plot(np.linspace(0,(step_num-1)*30,step_num), 
             grain_num_MCP_iso, label="Iso - MCP", linewidth=2)
    plt.plot(np.linspace(0,(step_num-1)*30,step_num), 
             grain_num_MCP_ave, label="Aniso - MCP", linewidth=2)
    
    # Formatting for publication quality
    plt.xlabel("Time step (MCS)", fontsize=20)     # Monte Carlo Steps
    plt.ylabel("Grain number (-)", fontsize=20)    # Dimensionless grain count
    plt.legend(fontsize=20,loc="upper right")       # Method identification
    plt.ylim([1000,20000])                          # Focus on evolution range
    plt.xlim([0,500])                               # Active growth period
    plt.yscale('log')                               # Logarithmic scaling for better visualization
    
    # Save high-resolution figure for publication
    plt.savefig(npy_file_folder + "/size_figure/grain_num_MCP_PF.png", 
                dpi=400, bbox_inches='tight')




