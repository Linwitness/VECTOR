#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward compatibility wrapper for 3D sphere HiPerGator analysis.
Delegates to unified plot_normal_distribution.py module.
"""
from plot_normal_distribution import run_analysis

if __name__ == '__main__':
    run_analysis(geometry='sphere3d', environment='hipergator', bias_case='iso')
