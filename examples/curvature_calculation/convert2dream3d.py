#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DREAM.3D Format Conversion Tool for Spherical Microstructures

This module provides automated conversion of 3D spherical grain structures
from VECTOR format to DREAM.3D HDF5 format, enabling advanced microstructural
analysis and visualization workflows.

DREAM.3D Integration Framework:
------------------------------
DREAM.3D is a powerful open-source platform for microstructural analysis and 
processing. This tool enables seamless data exchange between VECTOR curvature
analysis and DREAM.3D's advanced visualization and analysis capabilities.

Scientific Applications:
-----------------------
- Microstructural visualization in DREAM.3D environment
- Advanced analysis workflow integration
- Data exchange for materials science research
- Quality control for 3D grain structure validation

Created on Mon Oct  3 14:21:12 2022
@author: Lin
"""

import os
current_path = os.getcwd()+'/'
import sys
sys.path.append(current_path+'../../')
import numpy as np
import math
import myInput


# 3D Domain Configuration for DREAM.3D Export
# ==========================================
# High-resolution domain parameters for detailed microstructural analysis
nx, ny, nz = 200, 200, 200  # Voxel resolution: 200³ = 8M voxels
r = 2                       # Sphere radius (voxels) - moderate curvature test case

# Generate 3D Spherical Microstructure
# ====================================
# Create spherical grain embedded in matrix for curvature validation
# P0: 3D grain structure array with phase assignments
# R: Geometric parameters for validation analysis
P0,R = myInput.Circle_IC3d(nx,ny,nz,r)

# DREAM.3D Export Configuration
# =============================
# Generate HDF5 file path with descriptive naming convention
# Format: sphere_domain{nx}x{ny}x{nz}_r{radius}.hdf5
# This naming enables systematic organization of validation datasets
path = current_path + f"sphere_domain{nx}x{ny}x{nz}_r{r}.hdf5"

# Execute DREAM.3D Format Conversion
# =================================
# Convert VECTOR format to DREAM.3D compatible HDF5 structure
# Enables advanced visualization and analysis in DREAM.3D environment
# Output: HDF5 file suitable for DREAM.3D import and processing
myInput.output_dream3d(P0, path)

print(f"=== DREAM.3D Conversion Complete ===")
print(f"Input: 3D spherical microstructure (r={r}, domain={nx}x{ny}x{nz})")
print(f"Output: {path}")
print(f"Ready for DREAM.3D analysis and visualization")