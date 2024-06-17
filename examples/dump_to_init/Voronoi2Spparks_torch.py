#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 17:37:17 2023

@author: Lin
"""

from tqdm import tqdm
import torch
import numpy as np
import os

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def image2init(img, EulerAngles, fp=None):
    '''
    Takes an image of grain IDs (and euler angles assigned to each ID) and writes it to an init file for a SPPARKS simulation
    The initial condition file is written to the 2D or 3D file based on the dimension of 'img'

    Inputs:
        img (numpy, integers): pixels indicate the grain ID of the grain it belongs to
        EulerAngles (numpy): number of grains by three Euler angles
    '''
    # Set local variables
    size = img.shape
    dim = len(img.shape)
    if fp==None: fp = r"./spparks_simulations/spparks.init"
    IC = [0]*(np.product(size)+3)

    # Write the information in the SPPARKS format and save the file
    IC[0] = '# This line is ignored\n'
    IC[1] = 'Values\n'
    IC[2] = '\n'
    k=0

    if dim==3:
        for i in range(0,size[2]):
            for j in range(0,size[1]):
                for h in range(0,size[0]):
                    SiteID = int(img[h,j,i])
                    IC[k+3] = str(k+1) + ' ' + str(int(SiteID+1)) + ' ' + str(EulerAngles[SiteID,0]) + ' ' + str(EulerAngles[SiteID,1]) + ' ' + str(EulerAngles[SiteID,2]) + '\n'
                    k = k + 1

    else:
        for i in range(0,size[1]):
            for j in range(0,size[0]):
                SiteID = int(img[j,i])
                IC[k+3] = str(k+1) + ' ' + str(int(SiteID+1)) + ' ' + str(EulerAngles[SiteID,0]) + ' ' + str(EulerAngles[SiteID,1]) + ' ' + str(EulerAngles[SiteID,2]) + '\n'
                k = k + 1

    with open(fp, 'w') as file:
        file.writelines(IC)

    # Completion message
    print("NEW IC WRITTEN TO FILE: %s"%fp)

def generate_random_grain_centers(size=[128, 64, 32], ngrain=512):
    grain_centers = torch.rand(ngrain, len(size))*torch.Tensor(size)
    return grain_centers

def voronoi2image(size=[128, 64, 32], ngrain=512, memory_limit=1e9, p=2, center_coords0=None, device=device):

    #SETUP AND EDIT LOCAL VARIABLES
    dim = len(size)

    #GENERATE RENDOM GRAIN CENTERS
    # center_coords = torch.cat([torch.randint(0, size[i], (ngrain,1)) for i in range(dim)], dim=1).float().to(device)
    # center_coords0 = torch.cat([torch.randint(0, size[i], (ngrain,1)) for i in range(dim)], dim=1).float()
    if center_coords0 is None: center_coords0 = generate_random_grain_centers(size, ngrain)
    else:
        center_coords0 = torch.Tensor(center_coords0)
        ngrain = center_coords0.shape[0]
    center_coords = torch.Tensor([])
    for i in range(3): #include all combinations of dimension shifts to calculate periodic distances
        for j in range(3):
            if len(size)==2:
                center_coords = torch.cat([center_coords, center_coords0 + torch.Tensor(size)*(torch.Tensor([i,j])-1)])
            else:
                for k in range(3):
                    center_coords = torch.cat([center_coords, center_coords0 + torch.Tensor(size)*(torch.Tensor([i,j,k])-1)])
    center_coords = center_coords.float().to(device)

    #CALCULATE THE MEMORY NEEDED TO THE LARGE VARIABLES
    mem_center_coords = float(64*dim*center_coords.shape[0])
    mem_cords = 64*torch.prod(torch.Tensor(size))*dim
    mem_dist = 64*torch.prod(torch.Tensor(size))*center_coords.shape[0]
    mem_ids = 64*torch.prod(torch.Tensor(size))
    available_memory = memory_limit - mem_center_coords - mem_ids
    batch_memory = mem_cords + mem_dist

    #CALCULATE THE NUMBER OF BATCHES NEEDED TO STAY UNDER THE "memory_limit"
    num_batches = torch.ceil(batch_memory/available_memory).int()
    num_dim_batch = torch.ceil(num_batches**(1/dim)).int() #how many batches per dimension
    dim_batch_size = torch.ceil(torch.Tensor(size)/num_dim_batch).int() #what's the size of each of the batches (per dimension)
    num_dim_batch = torch.ceil(torch.Tensor(size)/dim_batch_size).int() #the actual number of batches per dimension (needed because of rouning error)

    if available_memory>0: #if there is avaiable memory
        #CALCULATE THE ID IMAGE
        all_ids = torch.zeros(size).type(torch.int16)
        ref = [torch.arange(size[i]).int() for i in range(dim)] #aranges of each dimension length
        tmp = tuple([torch.arange(i).int() for i in num_dim_batch]) #make a tuple to iterate with number of batches for dimension
        for itr in tqdm(torch.cartesian_prod(*tmp), 'Finding voronoi: '): #asterisk allows variable number of inputs as a tuple

            start = itr*dim_batch_size #sample start for each dimension
            stop = (itr+1)*dim_batch_size #sample end for each dimension
            stop[stop>=torch.Tensor(size)] = torch.Tensor(size)[stop>=torch.Tensor(size)].int() #reset the stop value to the end of the dimension if it went over
            indicies = [ref[i][start[i]:stop[i]] for i in range(dim)] #sample indicies for each dimension

            coords = torch.cartesian_prod(*indicies).float().to(device) #coordinates for each pixel
            dist = torch.cdist(center_coords, coords, p=p) #distance between each pixel and the "center_coords" (grain centers)
            ids = (torch.argmin(dist, dim=0).reshape(tuple(stop-start))%ngrain).int() #a batch of the final ID image (use modulo/remainder quotient to account for periodic grain centers)

            if dim==2: all_ids[start[0]:stop[0], start[1]:stop[1]] = ids
            else: all_ids[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = ids

        total_memory = batch_memory + mem_center_coords + mem_ids #total memory used by this function
        print("Total Memory: %3.3f GB, Batches: %d"%(total_memory/1e9, num_batches))

        #GENERATE RANDOM EULER ANGLES FOR EACH ID
        euler_angles = torch.stack([2*np.pi*torch.rand((ngrain)), \
                              0.5*np.pi*torch.rand((ngrain)), \
                              2*np.pi*torch.rand((ngrain))], 1)

        return all_ids.cpu().numpy(), euler_angles.cpu().numpy(), center_coords0.numpy()

    else:
        print("Available Memory: %d - Increase memory limit"%available_memory)
        return None, None, None

# savename = '/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/2d_poly_multiCoreCompare/IC/VoronoiIC_1024_5k.init'#The output filename
# size_x, size_y = 1024, 1024  #Number of sites in the x and y directions, so the size of the domain
# grains = 5000  #Number of grains to create

savename = '/orange/michael.tonks/lin.yang/IC/VoronoiIC_450_20k.init'
size_x, size_y, size_z = 450,450,450
grains = 20000

ic, ea, _ = voronoi2image([size_x,size_y,size_z], grains, 100e9)
image2init(ic, ea, savename) #write initial condition


