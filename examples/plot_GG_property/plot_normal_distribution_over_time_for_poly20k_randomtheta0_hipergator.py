#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward compatibility wrapper for random theta0 analysis on HiPerGator.
Delegates to unified plot_normal_distribution.py module with custom config.
"""
from plot_normal_distribution import NormalDistributionAnalyzer

if __name__ == '__main__':
    # Custom configuration for random theta0 analysis
    config = {
        'base_path': '/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/',
        'num_grains': 20000,
        'dimension': 2,
        'cases': {
            'iso': {'file': 'p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy',
                    'label': 'Iso', 'step': 10},
            'ave': {'file': 'p_randomtheta0_aveE_20000_delta0.6_J1_refer_1_0_0_seed56689_kt0.66.npy',
                    'label': 'Ave', 'step': 12},
            'min': {'file': 'p_randomtheta0_minE_20000_delta0.6_J1_refer_1_0_0_seed56689_kt0.66.npy',
                    'label': 'Min', 'step': 22},
            'max': {'file': 'p_randomtheta0_maxE_20000_delta0.6_J1_refer_1_0_0_seed56689_kt0.66.npy',
                    'label': 'Max', 'step': 12},
        }
    }
    analyzer = NormalDistributionAnalyzer(geometry='poly2d', environment='hipergator', config=config)
    analyzer.run(output_prefix='normal_distribution_poly_randomtheta0_20k')
