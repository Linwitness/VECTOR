#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward compatibility wrapper for local PRIMME inclination calculation.
Delegates to unified calculate_inclination.py module.
"""
from calculate_inclination import run_local_analysis

if __name__ == '__main__':
    run_local_analysis()
