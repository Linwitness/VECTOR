#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Polycrystalline Microstructure Visualization for Large-Scale Grain Growth
============================================================================

Provides visualization for large-scale polycrystalline microstructures from
SPPARKS Monte Carlo simulations. Focuses on comparative analysis of different
energy calculation methods and their effects on grain growth evolution in
systems with 20,000 initial grains and random initial orientations.

Scientific Background:
- Large-scale 2D grain growth simulation visualization
- Comparative analysis of energy calculation methodologies
- Random initial orientations (randomtheta0)

Technical Specifications:
- Initial grain count: 20,000 grains
- Energy methods: 6 different calculation approaches
- Output format: High-resolution PNG images

Created on Mon Jul 31 14:33:57 2023
@author: Lin
"""

import os
import sys

current_path = os.getcwd()
sys.path.append(current_path)
sys.path.append(current_path + '/../../')

from microstructure_plotter import Poly20kRandomPlotter

if __name__ == '__main__':
    # HiPerGator cluster data directory configuration
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/"
    figure_folder = current_path + "/figures/"

    # Create plotter with default random theta configurations
    plotter = Poly20kRandomPlotter.create_default(npy_file_folder, figure_folder)

    # Load all datasets
    plotter.load_data()

    # Generate all visualizations
    plotter.plot_all()
