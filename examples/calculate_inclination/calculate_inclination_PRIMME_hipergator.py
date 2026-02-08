#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward compatibility wrapper for HiPerGator batch inclination calculation.
Delegates to unified calculate_inclination.py module.
"""
from calculate_inclination import run_hipergator_batch

if __name__ == '__main__':
    run_hipergator_batch()
