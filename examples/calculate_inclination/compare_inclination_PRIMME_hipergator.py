#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward compatibility wrapper for HiPerGator PRIMME inclination comparison.
Delegates to unified compare_inclination.py module.
"""
from compare_inclination import run_hipergator_comparison

if __name__ == '__main__':
    run_hipergator_comparison()
