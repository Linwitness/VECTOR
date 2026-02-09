#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Circular Microstructure Analysis for Oriented Grain Systems
==============================================================

Visualizes circular microstructure evolution in oriented grain systems using
SPPARKS Monte Carlo simulations. Focuses on parameter sensitivity studies for
inclination energy methods with systematic variation of delta values,
crystallographic orientations, and mobility parameters.

Scientific Background:
- Simplified circular grain geometry for controlled parameter studies
- Systematic delta value variation (0.0 to 0.95) for sensitivity analysis
- Crystallographic orientation effects in two-grain systems
- Mobility parameter studies (m=2,4,6) for kinetic analysis

Created on Mon Jul 31 14:33:57 2023
@author: Lin
"""

import os
import sys
from examples.shared.path_setup import setup_vector_path
setup_vector_path()

from microstructure_plotter import CirclePlotter

if __name__ == '__main__':
    # Data location for circular microstructure simulation results
    npy_file_folder = "/Users/lin.yang/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/2d_circle_multiCoreCompare/results/"
    figure_folder = current_path + "/figures/"

    # Create plotter with default circle configurations
    plotter = CirclePlotter.create_default(npy_file_folder, figure_folder)

    # Load all datasets
    plotter.load_data()

    # Generate all visualizations
    plotter.plot_all()
