#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward compatibility wrapper for example/tutorial analysis.
Delegates to unified plot_normal_distribution.py module.
"""
from plot_normal_distribution import run_analysis

if __name__ == '__main__':
    # Simple example using default poly2d configuration
    run_analysis(geometry='poly2d', environment='hipergator',
                 cases=['ave'], output_prefix='normal_distribution_example')
