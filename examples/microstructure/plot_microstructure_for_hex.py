#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Hexagonal Microstructure Analysis for Triple Junction Energy Studies
======================================================================

Visualizes hexagonal grain structures for systematic triple junction energy
(TJE) validation studies. Focuses on controlled hexagonal grain arrangements
that provide ideal geometries for testing energy calculation algorithms.

Scientific Background:
- Regular hexagonal grain structures for controlled TJE studies
- 48-grain hexagonal arrangements for systematic energy validation
- Angle-resolved energy calculations for triple junction analysis

Technical Specifications:
- Geometry: 48-grain regular hexagonal arrangement
- Triple junctions: Well-defined 120 degree angles
- Processing: 32-core parallel optimization

Created on Mon Jul 31 14:33:57 2023
@author: Lin
"""

import os
import sys
from examples.shared.path_setup import setup_vector_path
setup_vector_path()

from microstructure_plotter import HexPlotter

if __name__ == '__main__':
    # Data location for hexagonal microstructure TJE validation studies
    npy_file_folder = "/Users/lin/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_hex_for_TJE/results/"
    figure_folder = current_path + "/figures/"

    # Create plotter with default hex configurations
    plotter = HexPlotter.create_default(npy_file_folder, figure_folder)

    # Load all datasets
    plotter.load_data()

    # Generate all visualizations
    plotter.plot_all()
