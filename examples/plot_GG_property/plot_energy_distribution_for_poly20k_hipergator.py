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
sys.path.append(current_path)
sys.path.append(current_path+'/../../')
import myInput
import PACKAGE_MP_Linear as linear2d
sys.path.append(current_path+'/../calculate_tangent/')

def get_poly_center(micro_matrix, step):
    # Get the center of all non-periodic grains in matrix
    num_grains = int(np.max(micro_matrix[step,:]))
    center_list = np.zeros((num_grains,2))
    sites_num_list = np.zeros(num_grains)
    ave_radius_list = np.zeros(num_grains)
    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i,j] = i
            coord_refer_j[i,j] = j

    table = micro_matrix[step,:,:,0]
    for i in range(num_grains):
        sites_num_list[i] = np.sum(table == i+1)

        if (sites_num_list[i] < 500) or \
           (np.max(coord_refer_i[table == i+1]) - np.min(coord_refer_i[table == i+1]) == micro_matrix.shape[1]) or \
           (np.max(coord_refer_j[table == i+1]) - np.min(coord_refer_j[table == i+1]) == micro_matrix.shape[2]): # grains on bc are ignored
          center_list[i, 0] = 0
          center_list[i, 1] = 0
          sites_num_list[i] == 0
        else:
          center_list[i, 0] = np.sum(coord_refer_i[table == i+1]) / sites_num_list[i]
          center_list[i, 1] = np.sum(coord_refer_j[table == i+1]) / sites_num_list[i]
    ave_radius_list = np.sqrt(sites_num_list / np.pi)

    return center_list, ave_radius_list


def get_normal_vector(grain_structure_figure_one, grain_num):
    nx = grain_structure_figure_one.shape[0]
    ny = grain_structure_figure_one.shape[1]
    ng = np.max(grain_structure_figure_one)
    cores = 32
    loop_times = 5
    P0 = grain_structure_figure_one
    R = np.zeros((nx,ny,2))
    smooth_class = linear2d.linear_class(nx,ny,ng,cores,loop_times,P0,R)

    smooth_class.linear_main("inclination")
    P = smooth_class.get_P()
    # sites = smooth_class.get_gb_list(1)
    # print(len(sites))
    # for id in range(2,grain_num+1): sites += smooth_class.get_gb_list(id)
    # print(len(sites))
    sites = smooth_class.get_all_gb_list()
    sites_together = []
    for id in range(len(sites)): sites_together += sites[id]
    print("Total num of GB sites: " + str(len(sites_together)))

    return P, sites_together, sites

def get_normal_vector_slope(P, sites, step, para_name, bias=None):
    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)

    freqArray = np.zeros(binNum)
    degree = []
    for sitei in sites:
        [i,j] = sitei
        dx,dy = myInput.get_grad(P,i,j)
        degree.append(math.atan2(-dy, dx) + math.pi)
        # if dx == 0:
        #     degree.append(math.pi/2)
        # elif dy >= 0:
        #     degree.append(abs(math.atan(-dy/dx)))
        # elif dy < 0:
        #     degree.append(abs(math.atan(dy/dx)))
    for i in range(len(degree)):
        freqArray[int((degree[i]/math.pi*180-xLim[0])/binValue)] += 1
    freqArray = freqArray/sum(freqArray*binValue)

    if bias is not None:
        freqArray = freqArray + bias
        freqArray = freqArray/sum(freqArray*binValue)
    # Plot
    # plt.close()
    # fig = plt.figure(figsize=(5, 5))
    # ax = plt.gca(projection='polar')

    # ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    # ax.set_thetamin(0.0)
    # ax.set_thetamax(360.0)

    # ax.set_rgrids(np.arange(0, 0.008, 0.004))
    # ax.set_rlabel_position(0.0)  # 标签显示在0°
    # ax.set_rlim(0.0, 0.008)  # 标签范围为[0, 5000)
    # ax.set_yticklabels(['0', '0.004'],fontsize=14)

    # ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    # ax.set_axisbelow('True')
    plt.plot(np.append(xCor,xCor[0])/180*math.pi, np.append(freqArray,freqArray[0]), linewidth=2, label=para_name)

    # fitting
    fit_coeff = np.polyfit(xCor, freqArray, 1)
    return freqArray

def plot_TJ_energy_distribution(path, file_name, x_lim = [[0, 4]], bin_value=0.01001, energy_per_site = 1):

  for i in range(len(file_name)):
    # Save energy data
    eng_file_name = file_name[i] + "_energy"
    if energy_per_site == 0: save_postfix = ''
    else: save_postfix = '_per_site'
    if not os.path.exists(path + "results/"+eng_file_name+".npy"):
      timestep, energy_figure = normal_distribution.dump2energy(path+file_name[i], 161)
      np.save("results/"+eng_file_name, energy_figure)
    else:
      energy_figure = np.load(path + "results/"+eng_file_name+".npy")
      timestep = 30 * np.array(range(len(energy_figure)))

    # Save structure data
    if not os.path.exists("results/"+file_name[i]+".npy"):
      timestep, grain_structure_figure = normal_distribution.dump2img(path+file_name[i], 161)
      np.save("results/"+file_name[i],grain_structure_figure)
    else:
      grain_structure_figure = np.load(path + "results/"+file_name[i]+".npy")
      timestep = 30 * np.array(range(len(grain_structure_figure)))

    if not os.path.exists("results/"+eng_file_name+save_postfix): os.mkdir("results/"+eng_file_name+save_postfix)
    for step in range(len(timestep)):
      current_structure = grain_structure_figure[step,:]
      current_energy = energy_figure[step,:]

      nx, ny, nz = current_structure.shape
      # xLim = [0, np.ceil(np.max(energy_figure))]
      xLim = x_lim[i]
      binValue = bin_value
      binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
      xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)
      freqArray = np.zeros(binNum)
      for m in range(nx):
        for n in range(ny):
          # find window
          window = find_window(current_structure[:,:,0], m ,n)
          center = window[1,1]
          edge = np.array(list(window[0,:])+[window[1,0],window[1,2]]+list(window[2,:]))
          edge_coord = np.array([[m-1,n-1], [m-1,n], [m-1,n+1], [m,n-1], [m,n+1], [m+1,n-1], [m+1,n], [m+1,n+1]])
          # Jump over the GB
          edge_set = set(edge)
          edge_set.discard(center)
          if len(edge_set) < 2: continue
          # Get number of neighbors
          num_nei = 0
          for nei in edge:
            if center != nei: num_nei += 1
          # Get site energy
          site_energy = current_energy[m, n, 0]
          if energy_per_site == 0:
          # If we use site energy
            freqArray[int((site_energy - xCor[0]) / binValue)] += 1
          else:
          # If we use energy per site
            freqArray[int((site_energy/num_nei - xCor[0]) / binValue)] += 1
      freqArray = freqArray/sum(freqArray)
      ave_energy = freqArray.dot(xCor)

      plt.clf()
      fig = plt.subplots()
      # plt.bar(xCor,freqArray,width=binValue*0.7)
      plt.plot(xCor, freqArray,'-o',linewidth=2,markersize=2,label='TJ energy distribution')
      plt.xlabel("Triple Energy")
      plt.ylabel("Frequence")
      plt.ylim([0, 1.1])
      plt.plot(ave_energy * np.ones(2), plt.ylim(), '--', linewidth=2,label='average TJ energy',color='gray')
      # fitting
      # fit_coeff = np.polyfit(xCor, freqArray, 1)
      # plt.plot(xCor, xCor*fit_coeff[0]+fit_coeff[1],'--',color='k',linewidth=2,label='fitting')
      plt.legend()
      plt.title(f"Time step {timestep[step]}, ave_TJ_eng = {round(ave_energy,3)}")

      if step < 10:
          step_str = '000' + str(step)
      elif step < 100:
          step_str = '00' + str(step)
      elif step < 1000:
          step_str = '0' + str(step)

      plt.savefig(f'results/{eng_file_name+save_postfix}/{eng_file_name+save_postfix}_step.{step_str}.png',dpi=400,bbox_inches='tight')
      plt.close()

    os.system(f'ffmpeg -framerate 30 -i results/{eng_file_name+save_postfix}/{eng_file_name+save_postfix}_step.%04d.png \
                    -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p \
                    results/{eng_file_name+save_postfix}/{eng_file_name+save_postfix}.mp4')

  return


if __name__ == '__main__':
    # File name
    npy_file_folder = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/results/"
    TJ_energy_type_ave = "ave"
    TJ_energy_type_consMin = "consMin"
    TJ_energy_type_sum = "sum"
    TJ_energy_type_min = "min"
    TJ_energy_type_max = "max"
    TJ_energy_type_consMax = "consMax"



    npy_file_name_aniso_ave = f"p_ori_ave_{TJ_energy_type_ave}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMin = f"p_ori_ave_{TJ_energy_type_consMin}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_sum = f"p_ori_ave_{TJ_energy_type_sum}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_min = f"p_ori_ave_{TJ_energy_type_min}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_max = f"p_ori_ave_{TJ_energy_type_max}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"
    npy_file_name_aniso_consMax = f"p_ori_ave_{TJ_energy_type_consMax}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy"

    # Initial data
    npy_file_aniso_ave = np.load(npy_file_folder + npy_file_name_aniso_ave)
    npy_file_aniso_consMin = np.load(npy_file_folder + npy_file_name_aniso_consMin)
    npy_file_aniso_sum = np.load(npy_file_folder + npy_file_name_aniso_sum)
    npy_file_aniso_min = np.load(npy_file_folder + npy_file_name_aniso_min)
    npy_file_aniso_max = np.load(npy_file_folder + npy_file_name_aniso_max)
    npy_file_aniso_consMax = np.load(npy_file_folder + npy_file_name_aniso_consMax)
    print(f"The ave data size is: {npy_file_aniso_ave.shape}")
    print(f"The consMin data size is: {npy_file_aniso_consMin.shape}")
    print(f"The sum data size is: {npy_file_aniso_sum.shape}")
    print(f"The min data size is: {npy_file_aniso_min.shape}")
    print(f"The max data size is: {npy_file_aniso_max.shape}")
    print(f"The consMax data size is: {npy_file_aniso_consMax.shape}")
    print("READING DATA DONE")

    # Initial container
    initial_grain_num = 20000
    step_num = npy_file_aniso_ave.shape[0]

    bin_width = 0.16 # Grain size distribution
    x_limit = [-0.5, 3.5]
    bin_num = round((abs(x_limit[0])+abs(x_limit[1]))/bin_width)
    size_coordination = np.linspace((x_limit[0]+bin_width/2),(x_limit[1]-bin_width/2),bin_num)
    grain_size_distribution_ave = np.zeros(bin_num)
    special_step_distribution_ave = 11 #to get 2000 grains
    grain_size_distribution_consMin = np.zeros(bin_num)
    special_step_distribution_consMin = 11#to get 2000 grains
    grain_size_distribution_sum = np.zeros(bin_num)
    special_step_distribution_sum = 11#to get 2000 grains
    grain_size_distribution_iso = np.zeros(bin_num)
    grain_size_distribution_min = np.zeros(bin_num)
    special_step_distribution_min = 30#to get 2000 grains
    grain_size_distribution_max = np.zeros(bin_num)
    special_step_distribution_max = 15#to get 2000 grains
    grain_size_distribution_consMax = np.zeros(bin_num)
    special_step_distribution_consMax = 11#to get 2000 grains


    # Start polar figure
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='polar')

    ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)

    ax.set_rgrids(np.arange(0, 0.01, 0.004))
    ax.set_rlabel_position(0.0)  # 标签显示在0°
    ax.set_rlim(0.0, 0.01)  # 标签范围为[0, 5000)
    ax.set_yticklabels(['0', '0.004', '0.008'],fontsize=14)

    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')

    # Get bias from kT test
    special_step_distribution_T066_bias = 10
    data_file_name_bias = f'/normal_distribution_data/normal_distribution_T066_bias_sites_step{special_step_distribution_T066_bias}.npy'
    slope_list_bias = np.load(current_path + data_file_name_bias)
    
    aniso_mag = np.zeros(6)
    aniso_mag_stand = np.zeros(6)
    # Aniso - min
    data_file_name_P = f'/normal_distribution_data/normal_distribution_min_P_step{special_step_distribution_min}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_min_sites_step{special_step_distribution_min}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_min[special_step_distribution_min,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_min, "Min case", slope_list_bias)
    aniso_mag[0], aniso_mag_stand[0] = simple_magnitude(slope_list)

    # Aniso - max
    data_file_name_P = f'/normal_distribution_data/normal_distribution_max_P_step{special_step_distribution_max}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_max_sites_step{special_step_distribution_max}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_max[special_step_distribution_max,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_max, "Max case", slope_list_bias)
    aniso_mag[1], aniso_mag_stand[1] = simple_magnitude(slope_list)

    # Aniso - ave
    data_file_name_P = f'/normal_distribution_data/normal_distribution_ave_P_step{special_step_distribution_ave}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_ave_sites_step{special_step_distribution_ave}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_ave[special_step_distribution_ave,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_ave, "Ave case", slope_list_bias)
    aniso_mag[2], aniso_mag_stand[2] = simple_magnitude(slope_list)

    # Aniso - sum
    data_file_name_P = f'/normal_distribution_data/normal_distribution_sum_P_step{special_step_distribution_sum}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_sum_sites_step{special_step_distribution_sum}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_sum[special_step_distribution_sum,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_sum, "Sum case", slope_list_bias)
    aniso_mag[3], aniso_mag_stand[3] = simple_magnitude(slope_list)

    # Aniso - consMin
    data_file_name_P = f'/normal_distribution_data/normal_distribution_consMin_P_step{special_step_distribution_consMin}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_consMin_P_sites_step{special_step_distribution_consMin}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_consMin[special_step_distribution_consMin,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_consMin, "ConsMin case", slope_list_bias)
    aniso_mag[4], aniso_mag_stand[4] = simple_magnitude(slope_list)

    # Aniso - consMax
    data_file_name_P = f'/normal_distribution_data/normal_distribution_consMax_P_step{special_step_distribution_consMax}.npy'
    data_file_name_sites = f'/normal_distribution_data/normal_distribution_consMax_sites_step{special_step_distribution_consMax}.npy'
    if os.path.exists(current_path + data_file_name_P):
        P = np.load(current_path + data_file_name_P)
        sites = np.load(current_path + data_file_name_sites)
    else:
        newplace = np.rot90(npy_file_aniso_consMax[special_step_distribution_consMax,:,:,:], 1, (0,1))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        np.save(current_path + data_file_name_P, P)
        np.save(current_path + data_file_name_sites, sites)

    slope_list = get_normal_vector_slope(P, sites, special_step_distribution_consMax, "ConsMax case", slope_list_bias)
    aniso_mag[5], aniso_mag_stand[5] = simple_magnitude(slope_list)

    plt.legend(loc=(-0.25,-0.3),fontsize=14,ncol=3)
    plt.savefig(current_path + "/figures/normal_distribution_poly_20k_after_removing_bias.png", dpi=400,bbox_inches='tight')

    plt.close()
    fig = plt.figure(figsize=(5, 5))
    label_list = ["Min", "Max", "Ave", "Sum", "ConsMin", "ConsMax"]
    plt.errorbar(np.linspace(0,len(label_list)-1,len(label_list)), aniso_mag, yerr=aniso_mag_stand, linestyle='None', marker='None',color='black',linewidth=1, capsize=2)
    plt.plot(np.linspace(0,len(label_list)-1,len(label_list)), aniso_mag, '.-', markersize=8, label='around 2000 grains', linewidth=2)
    plt.xlabel("Energy type", fontsize=14)
    plt.ylabel("Magnitude", fontsize=14)
    plt.xticks([0,1,2,3,4,5],label_list)
    plt.legend(fontsize=14)
    plt.ylim([-0.05,1.8])
    plt.savefig(current_path + "/figures/anisotropic_poly_20k_magnitude_polar_ave.png", dpi=400,bbox_inches='tight')











