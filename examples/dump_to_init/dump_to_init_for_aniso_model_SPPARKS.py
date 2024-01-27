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


def get_line(i, j):
    """Get the row order of grain i and grain j in MisoEnergy.txt (i < j)"""
    if i < j: return i+(j-1)*(j)/2
    else: return j+(i-1)*(i)/2

if __name__ == '__main__':
    # File name
    # dump file of last time step
    last_steps = 30
    dump_file_foler = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_poly/"
    dump_file_name = ""
    init_file_folder = dump_file_foler + "IC/"
    init_file_name = f"poly_IC150_1k.{last_steps}.init"


