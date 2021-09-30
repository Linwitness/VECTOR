#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 10:32:12 2021

@author: lin.yang
"""

import os
current_path = os.getcwd()
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# nx,ny = 200,200
# ng = 5
# PI = 3.1415926
# import image from SPPARKS .init file
def init2IC(nx,ny,ng,filename,filepath=current_path+"/input/"):
    R = np.zeros((nx,ny,2))
    
    with open(filepath+filename, 'r') as file:
        beginNum = 3
        fig = []
        
        while beginNum >= 0:
            line = file.readline()
            beginNum -= 1
        
        if line[0] != '1':
            print("Please change beginning line! " + line)
            
        while line:
            eachline = line.split()
            fig.append([int(eachline[1])])
            line = file.readline()
    
    fig = np.array(fig)
    fig = fig.reshape(nx,ny)
    fig = np.flipud(fig)
    fig = fig[:,:,None]
    
    return fig,R

def init2IC3d(nx,ny,nz,ng,filename,dream3d=False,filepath=current_path+"/input/"):
    R = np.zeros((nx,ny,nz,3))
    
    with open(filepath+filename, 'r') as file:
        beginNum = 3
        fig = np.zeros((nx*ny*nz))
        
        while beginNum >= 0:
            line = file.readline()
            beginNum -= 1
        
        if line[0] != '1':
            print("Please change beginning line! " + line)
            
        while line:
            eachline = line.split()
            fig[int(eachline[0])-1]=int(eachline[1])
            # fig.append([int(eachline[1])])
            
            line = file.readline()
    
    # fig = np.array(fig)
    fig = fig.reshape(nz,nx,ny)
    fig = fig.transpose((1,2,0))
    if dream3d:
        pass
    else:
        fig = np.flipud(fig)
    fig = fig[:,:,:,None]
    return fig,R

# Circle IC
def Circle_IC(nx,ny):
# =============================================================================
#     output the circle initial condition, 
#     nx is the sites in x coordination, 
#     ny is the site in y coordination, 
#     ng is the grain number
# =============================================================================
    ng = 2
    P = np.zeros((nx,ny,ng))
    R = np.zeros((nx,ny,2))
    
    for i in range(0,nx):
        for j in range(0,ny):
            radius = math.sqrt((j-ny/2)**2+(i-nx/2)**2)
            if radius < nx/4:
                P[i,j,0] = 1.
                if radius != 0:
                    R[i,j,0] = (j-ny/2)/radius
                    R[i,j,1] = (i-nx/2)/radius
            else:
                P[i,j,1] = 1.
                if radius != 0:
                    R[i,j,0] = (j-ny/2)/radius
                    R[i,j,1] = (i-nx/2)/radius

    return P,R

# 3D Circle IC
def Circle_IC3d(nx,ny,nz):
# =============================================================================
#     output the circle initial condition, 
#     nx is the sites in x coordination, 
#     ny is the site in y coordination, 
#     nz is the site in z coordination,
#     ng is the grain number
# =============================================================================
    ng = 2
    P = np.zeros((nx,ny,nz,ng))
    R = np.zeros((nx,ny,nz,3))
    
    for i in range(0,nx):
        for j in range(0,ny):
            for k in range(0,nz):
                radius = math.sqrt((j-ny/2)**2+(i-nx/2)**2+(k-nz/2)**2)
                if radius < nx/4:
                    P[i,j,k,0] = 1.
                    if radius != 0:
                        R[i,j,k,0] = (j-ny/2)/radius
                        R[i,j,k,1] = (i-nx/2)/radius
                        R[i,j,k,2] = (k-nz/2)/radius
                else:
                    P[i,j,k,1] = 1.
                    if radius != 0:
                        R[i,j,k,0] = (j-ny/2)/radius
                        R[i,j,k,1] = (i-nx/2)/radius
                        R[i,j,k,2] = (k-nz/2)/radius

    return P,R

# Voronoi IC
def Voronoi_IC(nx,ny,ng):
# =============================================================================
#     output the Voronoi initial condition, 
#     nx is the sites in x coordination, 
#     ny is the site in y coordination, 
#     ng is the grain number
# =============================================================================
    P = np.zeros((nx,ny,ng))
    R = np.zeros((nx,ny,2))
    
    # # generate points randomly
    # GCoords = np.zeros((ng,2))
    # for i in range(0,ng):
    #     GCoords[i,0],GCoords[i,1]= np.random.randint(nx),np.random.randint(ny)
    
    # Paint each domain site according to which grain center is closest(200,200,5)
    GCoords = np.array([[ 36., 132.],
                        [116.,  64.],
                        [ 43.,  90.],
                        [128., 175.],
                        [194.,  60.]])
    
    # (400,400,5)
    # GCoords = np.array([[ 69., 321.],
    #                     [298., 134.],
    #                     [174., 138.],
    #                     [294., 392.],
    #                     [ 69., 324.]])
    
    # (100,100,5)
    # GCoords = np.array([[20., 95.],
    #                     [27., 61.],
    #                     [37., 93.],
    #                     [65., 18.],
    #                     [25., 17.]])
    
    # (50,50,5)
    # GCoords = np.array([[ 0., 35.],
    #                     [43., 36.],
    #                     [43.,  9.],
    #                     [38., 37.],
    #                     [28., 36.]])
    
    
    for i in range(0,nx):
        for j in range(0,ny):
            MinDist = math.sqrt((GCoords[0,1]-j)**2+(GCoords[0,0]-i)**2)
            GG = 0
            for G in range(1,ng):
                dist = math.sqrt((GCoords[G,1]-j)**2+(GCoords[G,0]-i)**2) 
                if dist < MinDist:
                    GG = G
                    MinDist = dist
            P[i,j,GG] = 1.
            
    # (200,200,5)
    for i in range(0,nx):
        for j in range(0,ny):
            if i>0 and i<=93 and j<=120 and j>104:
                R[i,j,0] = -10.0/math.sqrt(101)
                R[i,j,1] = 1.0/math.sqrt(101)
            elif i==0 and j==ny-1:
                R[i,j,0] = -math.sqrt(0.5)
                R[i,j,1] = math.sqrt(0.5)
            elif i==0:
                R[i,j,0] = 0
                R[i,j,1] = 1
            elif j==ny-1:
                R[i,j,0] = 1
                R[i,j,1] = 0
            elif j>=123 and j<ny-1 and i<=96 and i>=60:
                R[i,j,0] = -9.0/math.sqrt(442)
                R[i,j,1] = -19.0/math.sqrt(442)
            elif (i==94 or i==95 or i==96) and (j==121 or j==122):
                R[i,j,0] = -2.0/math.sqrt(5)
                R[i,j,1] = 1.0/math.sqrt(5)
    
    # (400,400,5)
    # for i in range(0,nx):
    #     for j in range(0,ny):
    #         if i==0 and j==322:
    #             R[i,j,0] = -math.sqrt(0.5)
    #             R[i,j,1] = math.sqrt(0.5)
    #         elif i==0 and j==160:
    #             R[i,j,0] = -7.0/math.sqrt(74)
    #             R[i,j,1] = -5.0/math.sqrt(74)
    #         elif i==192 and j==322:
    #             R[i,j,0] = math.sqrt(0.5)
    #             R[i,j,1] = math.sqrt(0.5)
    #         elif i==206 and j==278:
    #             R[i,j,0] = 0
    #             R[i,j,1] = 1.0
    #         elif i==0:
    #             R[i,j,0] = 0
    #             R[i,j,1] = 1
    #         elif j==322:
    #             R[i,j,0] = 1
    #             R[i,j,1] = 0
    #         elif i>0 and i<=205 and j<=278 and j>=160:
    #             R[i,j,0] = -103.0/math.sqrt(14090)
    #             R[i,j,1] = 59.0/math.sqrt(14090)
    #         elif j>=279 and j<322 and i<=205 and i>=192:
    #             R[i,j,0] = 13.0/math.sqrt(2105)
    #             R[i,j,1] = 44.0/math.sqrt(2105)
            
                
    # (100,100,5)
    # for i in range(0,nx):
    #     for j in range(0,ny):
    #         if i==0 and j==ny-1:
    #             R[i,j,0] = math.sqrt(0.5)
    #             R[i,j,1] = -math.sqrt(0.5)
    #         elif i==0 and j==74:
    #             R[i,j,0] = -118885/math.sqrt(23777079626)
    #             R[i,j,1] = -98201/math.sqrt(23777079626)
    #         elif i==29 and j==ny-1:
    #             R[i,j,0] = 98894/math.sqrt(22966870792)
    #             R[i,j,1] = 114834/math.sqrt(22966870792)
    #         elif i==26 and j==79:
    #             R[i,j,0] = -80009/math.sqrt(13351496770)
    #             R[i,j,1] = 83367/math.sqrt(13351496770)
    #         elif i==0:
    #             R[i,j,0] = 0
    #             R[i,j,1] = 1
    #         elif j==ny-1:
    #             R[i,j,0] = 1
    #             R[i,j,1] = 0         
    #         elif i>0 and i<=26 and j<=79 and j>=74:
    #             R[i,j,0] = -27.0/math.sqrt(754)
    #             R[i,j,1] = 5.0/math.sqrt(754)
    #         elif j>=79 and j<ny-1 and i<=29 and i>=26:
    #             R[i,j,0] = -3.0/math.sqrt(634)
    #             R[i,j,1] = 25.0/math.sqrt(634)
            
    # (50,50,5)
    # for i in range(0,nx):
    #     for j in range(0,ny):
    #         if i==0 and j==0:
    #             R[i,j,0] = -math.sqrt(0.5)
    #             R[i,j,1] = -math.sqrt(0.5)
    #         elif i==0 and j==ny-1:
    #             R[i,j,0] = math.sqrt(0.5)
    #             R[i,j,1] = -math.sqrt(0.5)
    #         elif i==13 and j==ny-1:
    #             R[i,j,0] = 99967/math.sqrt(19487370058)
    #             R[i,j,1] = 97437/math.sqrt(19487370058)
    #         elif (i==8 and j==0) or (i==9 and j==ny-1):
    #             R[i,j,0] = math.sqrt(0.5)
    #             R[i,j,1] = math.sqrt(0.5)
    #         elif i==14 and j==10:
    #             R[i,j,0] = -14218/math.sqrt(3119555693)
    #             R[i,j,1] = 54013/math.sqrt(3119555693)
    #         elif i==0:
    #             R[i,j,0] = 0
    #             R[i,j,1] = 1
    #         elif j==ny-1 or j==0:
    #             R[i,j,0] = 1
    #             R[i,j,1] = 0         
    #         elif i>=8 and i<=14 and j<=10 and j>=0:
    #             R[i,j,0] = -3.0/math.sqrt(34)
    #             R[i,j,1] = 5.0/math.sqrt(34)
    #         elif j>=10 and j<=ny-1 and i<=14 and i>=13:
    #             R[i,j,0] = 1.0/math.sqrt(1522)
    #             R[i,j,1] = 39.0/math.sqrt(1522)
    
    # R = np.load('npy/voronoi_R.npy')
    
    return P,R



# Sin(x) IC
def Complex2G_IC(nx,ny):
# =============================================================================
#     output the sin(x) initial condition, 
#     nx is the sites in x coordination, 
#     ny is the site in y coordination, 
#     ng is the grain number
# =============================================================================
    
    ng = 2
    P = np.zeros((nx,ny,ng))
    R = np.zeros((nx,ny,2))
     
    for i in range(0,nx):
        # slope
        slope = -1.0/(math.pi*math.cos(math.pi/10*(i+5)))
        length = math.sqrt(1 + slope**2)
        
        for j in range(0,ny):
            if j < ny/2 + 10*math.sin((i+5)*0.31415926):
                P[i,j,0] = 1.
                R[i,j,0] = slope/length
                R[i,j,1] = 1.0/length
            else:
                P[i,j,1] = 1.
                R[i,j,0] = slope/length
                R[i,j,1] = 1.0/length
    
    for i in range(0,nx):
        for j in range(0,ny):
            if (i==0 and j==0) or (i==nx-1 and j==ny-1):
                R[i,j,0]=math.sqrt(0.5)
                R[i,j,1]=math.sqrt(0.5)
            elif (i==0 and j==ny-1) or (i==nx-1 and j==0):
                R[i,j,0]=-math.sqrt(0.5)
                R[i,j,1]=math.sqrt(0.5)
            elif j==0 or j==nx-1:
                R[i,j,0]=1
                R[i,j,1]=0

    return P,R

def Complex2G_IC3d(nx,ny,nz):
# =============================================================================
#     output the circle initial condition, 
#     nx is the sites in x coordination, 
#     ny is the site in y coordination, 
#     nz is the site in z coordination,
#     ng is the grain number
# =============================================================================
    ng = 2
    P = np.zeros((nx,ny,nz,ng))
    R = np.zeros((nx,ny,nz,3))
    
    for i in range(0,nx):
        for j in range(0,ny):
            vector_i = np.array([1, 0, math.pi/4*math.cos(math.pi/10*(0.5*i+5))])
            length_i = math.sqrt(1 + (math.pi/4*math.cos(math.pi/10*(0.5*i+5)))**2)
            vector_i = vector_i/length_i
            vector_j = np.array([0, 1, math.pi/4*math.cos(math.pi/10*(0.5*j+5))])
            length_j = math.sqrt(1 + (math.pi/4*math.cos(math.pi/10*(0.5*j+5)))**2)
            vector_j = vector_j/length_j
            
            
            for k in range(0,nz):
                if k < nz/2 + 5*math.sin((0.5*i+5)*0.31415926) + 5*math.sin((0.5*j+5)*0.31415926):
                    P[i,j,k,0] = 1.
                    R[i,j,k,:] = np.cross(vector_i,vector_j)
                    tmp_r = R[i,j,k,:]/np.linalg.norm(R[i,j,k,:])
                    R[i,j,k,:]=[tmp_r[1],tmp_r[0],tmp_r[2]]
                    # print(f"i={i} j={j} k={k}")
                    # print(R[i,j,k,:])
                else:
                    P[i,j,k,1] = 1. 
                    R[i,j,k,:] = -np.cross(vector_i,vector_j)
                    tmp_r = R[i,j,k,:]/np.linalg.norm(R[i,j,k,:])
                    R[i,j,k,:]=[tmp_r[1],tmp_r[0],tmp_r[2]]
                    
                    
            
    for k in [0,nz-1]:
        for i in range(0,nx):
            for j in range(0,ny):
                R[i,j,k,2] = 2.*((k>nz/2)*1-0.5)
    
    return P, R
        

# A real abnormal grain growth
def Abnormal_IC(nx,ny):
    
    ng = 2
    P = np.zeros((nx,ny,ng))
    R = np.zeros((nx,ny,2))

    file=open(f"input/AG{nx}x{ny}.txt")
    lines=file.readlines()
     
    row=0
    for line in lines:
        # line = " ".join(line)
        line=line.strip().split() 
        for i in range(0,len(line)):
            P[row,i,0]=float(line[i])
            P[row,i,1]=1-float(line[i])
        row+=1

    # abnormal possible true value
    if nx==200:
        R1 = np.load('npy/ACabnormal20_R.npy')
        R2 = np.load('npy/BLabnormal04_R.npy')
        R3 = np.load('npy/LSabnormal01_R.npy')
        R4 = np.load('npy/VTabnormal03_R.npy')
        
        m=0
        for i in range(0,nx):
            for j in range(0,ny):
                if R4[i,j,1]*R1[i,j,1]+R4[i,j,0]*R1[i,j,0] < -0.7:
                    R4[i,j,0] = -R4[i,j,0]
                    R4[i,j,1] = -R4[i,j,0]
                    m+=1
                    # print("i = " + str(i) + " j = " + str(j) + " m = " + str(m))
        
        for i in range(0,nx):
            for j in range(0,ny):
                R[i,j,0] = (R1[i,j,0] + R2[i,j,0] + R3[i,j,0] + R4[i,j,0])/4
                R[i,j,1] = (R1[i,j,1] + R2[i,j,1] + R3[i,j,1] + R4[i,j,1])/4
                length = math.sqrt(R[i,j,0]**2+R[i,j,1]**2)
                if length ==0:
                    R[i,j,0] = 0
                    R[i,j,1] = 0
                else:
                    R[i,j,0] = R[i,j,0]/length
                    R[i,j,1] = R[i,j,1]/length
            
    
    return P,R
    
    
def SmallestGrain_IC(nx,ny):
     ng = 2
     P = np.zeros((nx,ny,ng))
     
     for i in range(0,nx):
         
         for j in range(0,ny):
             
             if i==25 and j==10:
                 P[i,j,0] = 1
             elif i>=50 and i<=90 and j==10:
                 P[i,j,0] = 1
                 
             elif i>=24 and i<=25 and j>=25 and j<=26:
                 P[i,j,0] = 1
             elif i>=50 and i<=90 and j>=25 and j<=26:
                 P[i,j,0] = 1
                 
             elif i>=23 and i<=25 and j>=40 and j<=42:
                 P[i,j,0] = 1
             elif i>=50 and i<=90 and j>=40 and j<=42:
                 P[i,j,0] = 1
                 
             elif i>=22 and i<=25 and j>=60 and j<=63:
                 P[i,j,0] = 1
             elif i>=50 and i<=90 and j>=60 and j<=63:
                 P[i,j,0] = 1
                 
             elif i>=21 and i<=25 and j>=83 and j<=87:
                 P[i,j,0] = 1
             elif i>=50 and i<=90 and j>=83 and j<=87:
                 P[i,j,0] = 1
                 
             else:
                 P[i,j,1] = 1
                 
     return P


#%% Basic function in Smooth Algorithm
 
def periodic_bc(nx,ny,i,j):
    ip = i + 1
    im = i - 1
    jp = j + 1
    jm = j - 1
    #Periodic BC
    if ip > nx - 1:
        ip = 0
    if im < 0:
        im = nx - 1
    if jp > ny - 1:
        jp = 0
    if jm < 0:
        jm = ny - 1
    return ip,im,jp,jm

def periodic_bc3d(nx,ny,nz,i,j,k):
    ip = i + 1
    im = i - 1
    jp = j + 1
    jm = j - 1
    kp = k + 1
    km = k - 1
    #Periodic BC
    if ip > nx - 1:
        ip = 0
    if im < 0:
        im = nx - 1
    if jp > ny - 1:
        jp = 0
    if jm < 0:
        jm = ny - 1
    if kp > nz - 1:
        kp = 0
    if km < 0:
        km = nz - 1
    return ip,im,jp,jm,kp,km

# Using the repeat voxels on boundary conditions
def repeat_bc3d(nx,ny,nz,i,j,k):
    ip = i + 1
    im = i - 1
    jp = j + 1
    jm = j - 1
    kp = k + 1
    km = k - 1
    #Periodic BC
    if ip > nx - 1:
        ip = nx
    if im < 0:
        im = 0
    if jp > ny - 1:
        jp = ny
    if jm < 0:
        jm = 0
    if kp > nz - 1:
        kp = nz
    if km < 0:
        km = 0
    return ip,im,jp,jm,kp,km

# Remove the surface voxels on boundary conditions
def filter_bc3d(nx,ny,nz,i,j,k,length):
    if i-length < 0:
        return False
    if i+length > nx-1:
        return False
    if j-length < 0:
        return False
    if j+length > ny-1:
        return False
    if k-length < 0:
        return False
    if k+length > nz-1:
        return False
    return True

def get_grad(P,i,j):
    DX = P[2,i,j]
    DY = P[1,i,j]
    H = 1.
    VecX = -H*DX
    VecY = -H*DY
    VecLen = math.sqrt(VecX**2+VecY**2)
    if VecLen == 0:
        VecScale = 1
    else:
        VecScale = H/VecLen
    return VecScale*VecX,-VecScale*VecY

def get_grad3d(P,i,j,k):
    DX = P[2,i,j,k]
    DY = P[1,i,j,k]
    DZ = P[3,i,j,k]
    H = 1.0
    VecX = -H*DX
    VecY = -H*DY
    VecZ = -H*DZ
    VecLen = math.sqrt(VecX**2+VecY**2+VecZ**2)
    if VecLen == 0:
        VecScale = 1
    else:
        VecScale = H/VecLen
    return VecScale*VecX,-VecScale*VecY,VecScale*VecZ

def split_cores(cores, sc_d = 2):
    """ Split cores num into two or three closed index values of two """
    sc_p = 0
    while cores != 1:
        cores = cores/2
        sc_p += 1
        
    sc_length  = 2**(math.ceil(sc_p/sc_d))
    sc_width = 2**(math.floor(sc_p/sc_d))
    
    if sc_d == 3:
        sc_height = int(2**sc_p/(sc_length*sc_width))
        return sc_length, sc_width, sc_height
    
    return sc_length, sc_width


def split_IC(split_V,cores,dimentions=2, sic_nx_order = 1, sic_ny_order = 2, sic_nz_order = 3):
    """ Split a large matrix into several small matrix based on cores num"""
    if dimentions==2:
        sic_lc, sic_wc = split_cores(cores)
    elif dimentions==3:
        sic_lc,sic_wc,sic_hc = split_cores(cores,dimentions)
    # sic_width  = nx/sic_wc
    # sic_length  = ny /sic_lc
    
    new_arrayin = np.array_split(split_V, sic_wc, axis = sic_nx_order)
    new_arrayout = []
    for arrayi in new_arrayin:
        arrayi = np.array_split(arrayi, sic_lc, axis = sic_ny_order)
        if dimentions==3:
            new_array3 = []
            for arrayj in arrayi:
                arrayj = np.array_split(arrayj, sic_hc, axis = sic_nz_order)
                new_array3.append(arrayj)
            new_arrayout.append(new_array3)
        else:
            new_arrayout.append(arrayi)
    
    return new_arrayout
