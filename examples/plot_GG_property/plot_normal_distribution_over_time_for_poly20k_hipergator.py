#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward compatibility wrapper for 20K poly HiPerGator energy comparison analysis.
Delegates to unified plot_normal_distribution.py module.
"""
from plot_normal_distribution import run_analysis

if __name__ == '__main__':
    run_analysis(geometry='poly2d', environment='hipergator', num_grains=20000)
