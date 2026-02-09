#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Polycrystalline Microstructure Visualization for Oriented Grain Systems
=========================================================================

Provides visualization for large-scale polycrystalline microstructures with
oriented grains from SPPARKS Monte Carlo simulations. Focuses on comparative
analysis of different energy calculation methods in systems with realistic
crystallographic orientation distributions.

Scientific Background:
- Large-scale 2D grain growth with crystallographic orientations
- Comparative analysis of energy calculation methodologies
- Multi-core parallel processing optimization for HiPerGator cluster

Technical Specifications:
- Initial grain count: 20,000 grains with crystallographic orientations
- Parallel processing: 64-core optimization
- Energy methods: 6 different calculation approaches

Created on Mon Jul 31 14:33:57 2023
@author: Lin
"""

import os
import sys

current_path = os.getcwd()
sys.path.append(current_path)
sys.path.append(current_path + '/../../')

from microstructure_plotter import Poly20kPlotter

if __name__ == '__main__':
    # Data file location on HiPerGator cluster storage
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/"
    figure_folder = current_path + "/figures/"

    # Create plotter with default poly20k configurations
    plotter = Poly20kPlotter.create_default(npy_file_folder, figure_folder)

    # Load all datasets
    plotter.load_data()

    # Generate all visualizations
    plotter.plot_all()
