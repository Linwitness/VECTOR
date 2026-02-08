#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward compatibility wrapper for well energy analysis on HiPerGator.
Delegates to unified plot_normal_distribution.py module with custom config.
"""
from plot_normal_distribution import NormalDistributionAnalyzer

if __name__ == '__main__':
    # Custom configuration for well energy analysis
    config = {
        'base_path': '/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_wellEnergy/results/',
        'num_grains': 20000,
        'dimension': 2,
        'cases': {
            'iso': {'file': 'p_aveE_20000_Cos_delta0.0_J1_refer_1_0_0_seed56689_kt0.66.npy',
                    'label': 'Iso', 'step': 'auto'},
            '070': {'file': 'p_aveE_20000_Cos_delta0.7_J1_refer_1_0_0_seed56689_kt0.66.npy',
                    'label': r'$\sigma$=0.7', 'step': 'auto'},
            '080': {'file': 'p_aveE_20000_Cos_delta0.8_J1_refer_1_0_0_seed56689_kt0.66.npy',
                    'label': r'$\sigma$=0.8', 'step': 'auto'},
            '090': {'file': 'p_aveE_20000_Cos_delta0.9_J1_refer_1_0_0_seed56689_kt0.66.npy',
                    'label': r'$\sigma$=0.9', 'step': 'auto'},
        }
    }
    analyzer = NormalDistributionAnalyzer(geometry='poly2d', environment='hipergator', config=config)
    analyzer.run(output_prefix='normal_distribution_wellEnergy', bias_case='iso')
