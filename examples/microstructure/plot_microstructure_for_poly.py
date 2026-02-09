#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Polycrystalline Microstructure Analysis for Delta Parameter Studies
=====================================================================

Provides visualization for medium-scale polycrystalline microstructures with
systematic delta parameter variation. Analyzes effects of anisotropy parameters
on grain growth evolution in 512-grain oriented systems.

Scientific Background:
- Medium-scale 2D polycrystalline grain growth (512 initial grains)
- Systematic delta parameter sensitivity analysis (0.0 to 0.95)
- Oriented grain systems with crystallographic reference directions
- Target grain count studies (convergence to ~10 grain final states)

Created on Mon Jul 31 14:33:57 2023
@author: Lin
"""

import os
import sys

current_path = os.getcwd()
sys.path.append(current_path)
sys.path.append(current_path + '/../../')

from microstructure_plotter import PolyPlotter

if __name__ == '__main__':
    # Data location for medium-scale polycrystalline simulation results
    npy_file_folder = "/Users/lin.yang/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_poly_multiCoreCompare/results/"
    figure_folder = current_path + "/figures/"

    # Create plotter with default poly configurations
    plotter = PolyPlotter.create_default(npy_file_folder, figure_folder)

    # Load all datasets
    plotter.load_data()

    # Generate all visualizations
    plotter.plot_all()
