#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for smoothing algorithm verification tests.
Defines common parameters and settings for all test cases.

Author: Lin Yang
"""

# 2D test configurations
CONFIG_2D = {
    'circle_test': {
        'grid_size': (100, 100),
        'num_grains': 2,
        'num_cores': 4,
        'num_steps': 10,
        'circle_radius': 30,
    },
    'voronoi_test': {
        'grid_size': (200, 200),
        'num_grains': 5,
        'num_cores': 4,
        'num_steps': 10,
    },
    'convergence_test': {
        'grid_size': (100, 100),
        'num_grains': 2,
        'num_cores': 4,
        'step_range': range(2, 21, 2),
        'circle_radius': 30,
    }
}

# 3D test configurations
CONFIG_3D = {
    'sphere_test': {
        'grid_size': (50, 50, 50),
        'num_grains': 2,
        'num_cores': 8,
        'num_steps': 10,
        'sphere_radius': 15,
    },
    'convergence_test': {
        'grid_size': (50, 50, 50),
        'num_grains': 2,
        'num_cores': 8,
        'step_range': range(2, 11, 2),
        'sphere_radius': 15,
    }
}

# Algorithm-specific parameters
ALGORITHM_PARAMS = {
    'linear': {
        'verification_system': True,
        'curvature_sign': False,
        'clip': 0,
    },
    'vertex': {
        'verification_system': True,
        'curvature_sign': False,
        'clip': 0,
    },
    'levelset': {
        'verification_system': True,
        'curvature_sign': False,
        'clip': 0,
    },
    'allen_cahn': {
        'verification_system': True,
        'curvature_sign': False,
        'clip': 0,
        'mobility': 1.0,
        'gradient_coeff': 1.0,
    },
    '3dlinear': {
        'verification_system': True,
        'curvature_sign': False,
        'clip': 0,
        'boundary_condition': 'p',  # periodic
    },
    '3dvertex': {
        'verification_system': True,
        'curvature_sign': False,
        'clip': 0,
        'boundary_condition': 'p',
    },
    '3dlevelset': {
        'verification_system': True,
        'curvature_sign': False,
        'clip': 0,
        'boundary_condition': 'p',
    },
    '3dallen_cahn': {
        'verification_system': True,
        'curvature_sign': False,
        'clip': 0,
        'boundary_condition': 'p',
        'mobility': 1.0,
        'gradient_coeff': 1.0,
    }
}

# Output configuration
OUTPUT_CONFIG = {
    'save_plots': True,
    'plot_format': 'png',
    'plot_dpi': 300,
    'generate_report': True,
    'report_format': 'txt',
    'verbose_output': True,
}

# Visualization settings
PLOT_CONFIG = {
    'figure_size': (15, 5),
    'colormap_2d': 'gray',
    'colormap_3d': 'gray',
    'vector_scale': 10,
    'vector_width': 0.1,
    'vector_alpha': 0.5,
    'vector_color': 'red',
    'curvature_colormap': 'coolwarm',
}