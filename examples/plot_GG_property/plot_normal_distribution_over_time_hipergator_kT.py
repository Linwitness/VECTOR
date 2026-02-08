#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward compatibility wrapper for thermal kT variation analysis.
Delegates to unified plot_normal_distribution.py module with custom config.
"""
from plot_normal_distribution import NormalDistributionAnalyzer

if __name__ == '__main__':
    # Custom configuration for kT analysis
    config = {
        'base_path': '/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/',
        'num_grains': 20000,
        'dimension': 2,
        'cases': {
            'T000': {'file': 'p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt000.npy',
                     'label': r'$kT=0.00$', 'step': 10},
            'T025': {'file': 'p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt025.npy',
                     'label': r'$kT=0.25$', 'step': 10},
            'T050': {'file': 'p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt050.npy',
                     'label': r'$kT=0.50$', 'step': 10},
            'T066': {'file': 'p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy',
                     'label': r'$kT=0.66$', 'step': 10},
            'T095': {'file': 'p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt095.npy',
                     'label': r'$kT=0.95$', 'step': 11},
        }
    }
    analyzer = NormalDistributionAnalyzer(geometry='poly2d', environment='hipergator', config=config)
    analyzer.run(output_prefix='normal_distribution_kT')
