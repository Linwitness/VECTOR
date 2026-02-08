#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward compatibility wrapper for isotropic GBs analysis on HiPerGator.
Delegates to unified plot_normal_distribution.py module with custom config.
"""
from plot_normal_distribution import NormalDistributionAnalyzer

if __name__ == '__main__':
    # Custom configuration for isoGBs analysis
    config = {
        'base_path': '/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/',
        'num_grains': 20000,
        'dimension': 2,
        'cases': {
            'iso': {'file': 'p_ori_ave_aveE_20000_multiCore32_delta0.0_m2_J1_refer_1_0_0_seed56689_kt066.npy',
                    'label': 'Iso', 'step': 10},
            'ave': {'file': 'p_ori_ave_aveE_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_isoGBs.npy',
                    'label': 'Ave', 'step': 11},
            'min': {'file': 'p_ori_ave_minE_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_isoGBs.npy',
                    'label': 'Min', 'step': 30},
            'max': {'file': 'p_ori_ave_maxE_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_isoGBs.npy',
                    'label': 'Max', 'step': 15},
            'sum': {'file': 'p_ori_ave_sumE_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_isoGBs.npy',
                    'label': 'Sum', 'step': 11},
            'consMin': {'file': 'p_ori_ave_consMinE_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_isoGBs.npy',
                        'label': 'CMin', 'step': 11},
            'consMax': {'file': 'p_ori_ave_consMaxE_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_isoGBs.npy',
                        'label': 'CMax', 'step': 11},
        }
    }
    analyzer = NormalDistributionAnalyzer(geometry='poly2d', environment='hipergator', config=config)
    analyzer.run(output_prefix='normal_distribution_poly_20k_isoGBs')
