#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:33:57 2023

@author: Lin
"""

import os
current_path = os.getcwd()
import numpy as np
from numpy import seterr
seterr(all='raise')
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import sys
sys.path.append(current_path+'/../../')
import myInput
import post_processing
import PACKAGE_MP_3DLinear as linear3d

if __name__ == '__main__':


    # File name
    # dump file of last time step
    last_step = 5
    dump_file_foler = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_poly_fully/"
    # dump_file_foler = "/Users/lin/projects/SPPARKS-AGG/examples/Test_SimplifyIncE/3d_poly_for_GG/"
    dump_file_name = f"p_ori_fully5d_fz_aveE_f1.0_t1.0_450_5k_multiCore64_J1_refer_1_0_0_seed56689_kt1.95"
    init_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_poly/IC/"
    init_file_output_folder = dump_file_foler+"IC/"
    init_file_name_original = f"poly_IC450_5k.init"
    init_file_name = f"poly_IC450_5k_s{last_step}.init"
    init_file_name_final = f"poly_IC450_5k_s{last_step}_neighbor5.init"

    # Read init file
    grain_num = 5000
    euler_angle_array = post_processing.init2EAarray(init_file_folder+init_file_name_original, grain_num)
    print("> Read init file done")


    # Read necessary information from dump file
    dump_file_name_0 = dump_file_foler+dump_file_name+f".dump.{int(last_step)}"
    print(f"> Read dump {int(last_step)} file done")
    box_size, entry_length = post_processing.output_init_from_dump(dump_file_name_0, euler_angle_array, init_file_output_folder+init_file_name)
    size_x,size_y,size_z = box_size

    # necessary data for neighbor init file
    interval = 5
    output_neighbr_init = post_processing.output_init_neighbor_from_init_mp(interval, box_size, init_file_output_folder+init_file_name, init_file_output_folder+init_file_name_final)
    print(f"The converting is {output_neighbr_init} success.")
    # output_neighbr_init = output_init_neighbor_from_init(interval, box_size, init_file_folder+init_file_name, init_file_folder+init_file_name_final+"test")
    
    # pigz -p 128 -k poly_IC450_5k_s5_neighbor5.init

