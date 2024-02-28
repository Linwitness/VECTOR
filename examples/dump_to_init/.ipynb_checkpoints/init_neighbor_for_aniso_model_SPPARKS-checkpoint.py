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
import multiprocess as mp
import sys
sys.path.append(current_path+'/../../')
import myInput
import post_processing
import PACKAGE_MP_3DLinear as linear3d

def tmp_mp(interval, box_size, init_file_path_input, init_file_path_output):
    # Output the init_nighbor5 with init file
    size_x,size_y,size_z = box_size
    dimension = int(2 if size_z==1 else 3)
    nei_num = (2*interval+3)**dimension-1
    num_processes = mp.cpu_count() # or choose a number that suits your machine
    
    temp_files = []
    for p in range(num_processes):
        temp_file = f'{init_file_path_output}_temp_{p}.txt'
        temp_files.append(temp_file)
    
    with open(init_file_path_output, 'a') as outfile:
        for fname in tqdm(temp_files, "Concatenating "):
            with open(fname) as infile:
                outfile.write(infile.read())
            os.remove(fname)  # Optional: remove temp file after concatenation
        outfile.write("\n")
    print("> Neighbors end writing")

    print("> Values start writing")
    with open(init_file_path_input, 'r') as f_read:
        tmp_values = f_read.readlines()
    print("> Values read done")
    with open(init_file_path_output, 'a') as file:
        file.writelines(tmp_values[1:])
    print("> Values end writing")
    return True


if __name__ == '__main__':

    # File name
    dump_file_foler = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_poly/"
    init_file_folder = dump_file_foler + "IC/"
    init_file_name = f"poly_IC1000_20k.init"
    init_file_folder_final = "/orange/michael.tonks/lin.yang/IC/"
    init_file_name_final = f"poly_IC1000_20k_neighbor5.init"


    # Read necessary information from dump file
    # dump_file_name_0 = dump_file_foler+dump_file_name+f".dump.{int(last_step)}"
    # box_size, entry_length = post_processing.output_init_from_dump(dump_file_name_0, euler_angle_array, init_file_folder+init_file_name)
    # size_x,size_y,size_z = box_size
    box_size = np.array([1000,1000,1000]).astype('int')

    # necessary data for neighbor init file
    interval = 5
    # output_neighbr_init = post_processing.output_init_neighbor_from_init_mp(interval, box_size, init_file_folder+init_file_name, init_file_folder_final+init_file_name_final)
    output_neighbr_init = tmp_mp(interval, box_size, init_file_folder+init_file_name, init_file_folder_final+init_file_name_final)

