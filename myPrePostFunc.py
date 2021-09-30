#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 17:31:44 2021

@author: lin.yang
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import myInput
import string

#%% Basic function in Smooth Algorithm

def get_gb_num(P,grainID,bc='np'):
    ggn_gbsites = []  
    if bc == 'np':
        edge_l = halfL
    else:
        edge_l = 0
    for i in range(0+edge_l,nx-edge_l):
        for j in range(0+edge_l,ny-edge_l):
            for k in range(0+edge_l,nz-edge_l):
                ip,im,jp,jm,kp,km = myInput.periodic_bc3d(nx,ny,nz,i,j,k)
                if ( ((P[0,ip,j,k]-P[0,i,j,k])!=0) or ((P[0,im,j,k]-P[0,i,j,k])!=0) or\
                     ((P[0,i,jp,k]-P[0,i,j,k])!=0) or ((P[0,i,jm,k]-P[0,i,j,k])!=0) or\
                     ((P[0,i,j,kp]-P[0,i,j,k])!=0) or ((P[0,i,j,km]-P[0,i,j,k])!=0) )\
                       and P[0,i,j,k]==grainID:
                    ggn_gbsites.append([i,j,k])
    return ggn_gbsites


def get_2d_plot(P,init,algo,loop_times,z_surface = 0):
    z_surface = int(nz/2)
    plt.close()
    fig1 = plt.figure(1)
    fig_page = loop_times
    plt.title(f'{algo}-{init} \n loop = '+str(fig_page))
    if fig_page < 10:
        String = '000'+str(fig_page)
    elif fig_page < 100:
        String = '00'+str(fig_page)
    elif fig_page < 1000:
        String = '0'+str(fig_page)
    elif fig_page < 10000:
        String = str(fig_page)
    plt.imshow(P[0,:,:,z_surface], cmap='nipy_spectral', interpolation='nearest')
    
    g2p_gbsites = get_gb_num()
    for gbSite in g2p_gbsites:
        [g2pi,g2pj,g2pk] = gbSite
        if g2pk==z_surface:
            
            g2p_dx,g2p_dy,g2p_dz = myInput.get_grad3d(P,g2pi,g2pj,g2pk)
            plt.arrow(g2pj,g2pi,10*g2p_dx,10*g2p_dy,width=0.1,lw=0.1,alpha=0.8,color='navy')



#%% Post Func

# plot the arrow and figure for Core
      
def plotRvsE(AC_errors,AC_runningTime,BL_errors,BL_runningTime,VT_errors,VT_runningTime,LS_errors,LS_runningTime,cores,edge):
    plt.close()
    lines = plt.plot(AC_errors,AC_runningTime,BL_errors,BL_runningTime,VT_errors,VT_runningTime,LS_errors,LS_runningTime,marker='o',markerfacecolor='none',markersize=4)

    plt.setp(lines[0], linewidth=2, linestyle='-', color = '#0071bd', label = 'Allen-Cahn') 
    plt.setp(lines[1], linewidth=2, linestyle='-', color = '#d95319', label = 'Bilinear') 
    plt.setp(lines[2], linewidth=2, linestyle='-', color = '#edb121', label = 'Vertex') 
    plt.setp(lines[3], linewidth=2, linestyle='-', color = 'black', label = 'Level-Set') 
    
    plt.title("Running time VS Circle Unit error")
    plt.xlabel("Unit error")
    plt.ylabel("Running time")
    plt.legend()
    plt.yscale('log')
    plt.xlim(0,0.150)
    plt.ylim(8,1500)
    
    plt.savefig(f'NT_Circle_4algs_{cores}c_{edge}by{edge}.png',dpi=400,bbox_inches='tight')

def p2and3d(AC_errors,AC_runningTime,BL_errors,BL_runningTime,VT_errors,VT_runningTime,LS_errors,LS_runningTime,xlim,ylim):
    lines = plt.plot(AC_errors,AC_runningTime,BL_errors,BL_runningTime,VT_errors,VT_runningTime,LS_errors,LS_runningTime,marker='o',markerfacecolor='none',markersize=5)

    plt.setp(lines[0], linewidth=4, linestyle='-', color = '#0071bd', label = 'Allen-Cahn') 
    plt.setp(lines[1], linewidth=4, linestyle='-', color = '#d95319', label = 'Bilinear') 
    plt.setp(lines[2], linewidth=4, linestyle='-', color = '#edb121', label = 'Vertex') 
    plt.setp(lines[3], linewidth=4, linestyle='-', color = 'black', label = 'Level-Set') 
    
    # plt.xlabel("Running time (s)",fontsize=35)
    plt.ylabel("Unit error (rad)",fontsize=35)
    plt.legend(fontsize=21,loc="upper right")
    plt.xscale('log')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tick_params(labelsize=35)
    
def p2and3d_gm(AC_errors,AC_runningTime,BL_errors,BL_runningTime,VT_errors,VT_runningTime,LS_errors,LS_runningTime,xlim,ylim):
    lines = plt.plot(AC_errors,AC_runningTime,BL_errors,BL_runningTime,VT_errors,VT_runningTime,LS_errors,LS_runningTime,marker='o',markerfacecolor='none',markersize=5)

    plt.setp(lines[0], linewidth=4, linestyle='-', color = '#0071bd', label = 'Allen-Cahn') 
    plt.setp(lines[1], linewidth=4, linestyle='-', color = '#d95319', label = 'Bilinear') 
    plt.setp(lines[2], linewidth=4, linestyle='-', color = '#edb121', label = 'Vertex') 
    plt.setp(lines[3], linewidth=4, linestyle='-', color = 'black', label = 'Level-Set') 
    
    plt.xlabel("Running time (s)",fontsize=15)
    plt.ylabel("Unit error (rad)",fontsize=15)
    plt.legend(fontsize=14,loc="upper right")
    plt.xscale('log')
    plt.xlim(xlim)
    plt.ylim(ylim)
    y_major_locator=MultipleLocator(0.05)
    plt.gca().yaxis.set_major_locator(y_major_locator)
    plt.tick_params(labelsize=14)
    
def get_gb_num(P,grainID):
    _,nx,ny,nz=np.shape(P)
    ggn_gbsites = []  
    for i in range(0,nx,2):
        for j in range(0,ny,2):
            for k in range(0,nz):
                ip,im,jp,jm,kp,km = myInput.periodic_bc3d(nx,ny,nz,i,j,k)
                if ( ((P[0,ip,j,k]-P[0,i,j,k])!=0) or ((P[0,im,j,k]-P[0,i,j,k])!=0) or\
                     ((P[0,i,jp,k]-P[0,i,j,k])!=0) or ((P[0,i,jm,k]-P[0,i,j,k])!=0) or\
                     ((P[0,i,j,kp]-P[0,i,j,k])!=0) or ((P[0,i,j,km]-P[0,i,j,k])!=0) )\
                       and P[0,i,j,k]==grainID and k!= 0 and i!=0 and j!=0:
                    ggn_gbsites.append([i,j,k])
                    
    return ggn_gbsites

def plot2dIC(IC):
    nx,ny=200,200
    ng=2
    
    if IC == "circle":
        P0,R=myInput.Circle_IC(nx,ny)
    if IC == "voronoi":
        ng=5
        P0,R=myInput.Voronoi_IC(nx,ny,ng)
    if IC == "complex":
        P0,R=myInput.Complex2G_IC(nx,ny)


    P = np.zeros((3,nx,ny))
    for i in range(0,np.shape(P0)[2]):
        P[0,:,:] += P0[:,:,i]*(i+1)

    plt.imshow(P[0,:,:], cmap='gray', interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')

def plot3dIC(IC):

    ng=2
    nx,ny,nz = 100,100,100
    
    if IC == "complex":
        P0,R=myInput.Complex2G_IC3d(nx,ny,nz)
    if IC == "circle":
        P0,R=myInput.Circle_IC3d(nx,ny,nz)
    
    P = np.zeros((4,nx,ny,nz))
    for i in range(0,np.shape(P0)[3]):
        P[0,:,:,:] += P0[:,:,:,i]*(i+1)

    IC_boundary = get_gb_num(P,1)
# fig1 = plt.figure(1)
# ax = fig.add_subplot(111, projection='3d')
    pos_all = np.array(IC_boundary)
    vali1 = ax.scatter(pos_all[:,1],pos_all[:,0],pos_all[:,2],marker='.',c='black')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # plt.axis('off')    

#%% save data of algorithm accuracy and efficiency

npzfile = np.load("RVE_3dalgs_3ABL.npz")
AC3d_runningTime,AC3d_errors,BL3d2_runningTime,BL3d2_errors,LS3d_runningTime,LS3d_errors = npzfile['ACR3d'],npzfile['ACE3d'],npzfile['BLR3d'],npzfile['BLE3d'],npzfile['LSR3d'],npzfile['LSE3d']

np.savez("RVE_3dalgs_Circle",ACR3d=AC3d_runningTime,ACE3d=AC3d_errors,
                                      BLR3d=BL3d2_runningTime,BLE3d=BL3d2_errors,
                                      LSR3d=LS3d_runningTime,LSE3d=LS3d_errors,
                                      VTR3d=VT3d_runningTime,VTE3d=VT3d_errors)

np.savez("RVE_2dalgs_Voronoi",ACR=AC_runningTime,ACE=AC_errors,
                                      BLR=BL_runningTime,BLE=BL_errors,
                                      LSR=LS_runningTime,LSE=LS_errors,
                                      VTR=VT_runningTime,VTE=VT_errors)

  
#%% plot Unit Error VS Running Time


# npzfile = np.load("RVE_3dalgs_Complex.npz")

# lines = plt.plot(npzfile['ACR3d'],npzfile['ACE3d'],npzfile['BLR3d'],npzfile['BLE3d'],npzfile['VTR3d'],npzfile['VTE3d'],npzfile['LSR3d'],npzfile['LSE3d'],marker='o',markerfacecolor='none',markersize=4)
xlim_list = [[0.2,50], [0.2,100], [0.2,100], [0.8,10000], [2,30000]]
ylim_list = [[0,0.175], [0,0.42], [0,0.15], [0,0.42], [0,0.56]]
file_name = ["RVE_2dalgs_Circle.npz", "RVE_2dalgs_Complex.npz","RVE_2dalgs_Voronoi.npz","RVE_3dalgs_Circle.npz","RVE_3dalgs_Complex.npz"]
IC_name = ["circle", "complex", "voronoi", "circle", "complex"]
IC_title = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)"]
# fig = plt.figure(1,figsize=(30,8))
# plt.show()

fig, ax_big = plt.subplots( figsize=(18, 30))   # 这是figure上有一张名叫 ax_big 的空图，上面是有坐标系的

axis = plt.gca()                                # gca 'get current axes' 获取图像的坐标轴对象
axis.spines['right'].set_color('none')                  # 隐藏右和上坐标轴
axis.spines['top'].set_color('none')
axis.spines['bottom'].set_position(('outward', 30))     # 偏移左和下坐标轴
axis.spines['left'].set_position(('outward', 30))
# ax_big.set_xticks([])                                   # 隐藏坐标轴刻度
# ax_big.set_yticks([])
ax_big.axis("off")
# ax_big.set_xlabel('time (s)', fontsize=13, fontstyle='italic')      # 设置字号，斜体
# ax_big.set_ylabel('longitudinal sensor output', fontsize=13, fontstyle='italic')


for i in range(1,11):
    # ax = fig.add_subplot(2,5,i)
    
    if i < 4:
        ax = fig.add_subplot(5,2,(i-1)*2+1)
        l, b, w, h = ax.get_position().bounds
        ax.set_position([l + 0.1*w - 0.15*w, b + 0.1*h, w*0.8, h*0.8])
        plot2dIC(IC_name[i-1])
    elif i < 6:
        ax = fig.add_subplot(5,2,(i-1)*2+1, projection='3d')
        l, b, w, h = ax.get_position().bounds
        ax.set_position([l - 0.05*w - 0.2*w, b - 0.05*h, w*1.1, h*1.1])
        plot3dIC(IC_name[i-1]) 
        
    
    elif i < 9:
        ax = fig.add_subplot(5,2,(i-6)*2+2)
        l, b, w, h = ax.get_position().bounds
        ax.set_position([l - 0.20*w, b + 0.05*h, w*1.1, h*1.0])
        npzfile = np.load(file_name[i-6])
        p2and3d(npzfile['ACR'],npzfile['ACE'],npzfile['BLR'],npzfile['BLE'],npzfile['VTR'],npzfile['VTE'],npzfile['LSR'],npzfile['LSE'],xlim_list[i-6],ylim_list[i-6])
        # if i == 6:
        #     plt.legend(fontsize=30,loc="upper right")
    else:
        ax = fig.add_subplot(5,2,(i-6)*2+2)
        l, b, w, h = ax.get_position().bounds
        ax.set_position([l - 0.20*w, b + 0.05*h, w*1.1, h*1.0])
        npzfile = np.load(file_name[i-6])
        p2and3d(npzfile['ACR3d'],npzfile['ACE3d'],npzfile['BLR3d'],npzfile['BLE3d'],npzfile['VTR3d'],npzfile['VTE3d'],npzfile['LSR3d'],npzfile['LSE3d'],xlim_list[i-6],ylim_list[i-6])
        if i == 10:
            plt.xlabel("Running time (s)",fontsize=35)
    
    if i <6:
        ax.set_title(IC_title[i-1],x=0.124,y=0.85,fontsize=28,weight='bold',backgroundcolor='white')
        # break
    else:
        ax.set_title(IC_title[i-1],x=0.062,y=0.865,fontsize=28,weight='bold',backgroundcolor='white')
        # break
    # if i==2:
    #     break
# ax_big.subplots_adjust(left=0.05, right=1, top=1, bottom=0)
# plt.savefig('paper_ICandAVE_low_colume3.png',dpi=400,bbox_inches='tight',pad_inches = 0)
# plt.show()




# plt.setp(lines[0], linewidth=2, linestyle='-', color = '#0071bd', label = 'Allen-Cahn') 
# plt.setp(lines[1], linewidth=2, linestyle='-', color = '#d95319', label = 'Bilinear') 
# plt.setp(lines[2], linewidth=2, linestyle='-', color = '#edb121', label = 'Vertex') 
# plt.setp(lines[3], linewidth=2, linestyle='-', color = 'black', label = 'Level-Set') 

# # plt.title("Unit error VS Running time",fontsize=18)
# plt.xlabel("Running time (s)",fontsize=15)
# plt.ylabel("Unit error (rad)",fontsize=15)
# plt.legend(fontsize=13)
# plt.xscale('log')
# # plt.xlim(0.1,50)
# # plt.ylim(0,0.175)
# # plt.xlim(0.2,100)
# # plt.ylim(0,0.4)
# # plt.xlim(0.2,100)
# # plt.ylim(0,0.15)
# # plt.xlim(1,10000)
# # plt.ylim(0,0.4)
# plt.xlim([1,20000])
# plt.ylim([0,0.6])

# plt.subplot(122)
# # plt.set_xlabel(fontsize=10)
# # plt.set_ylabel(fontsize=10)

# plt.show()
# plt.savefig(f'Voronoi_ErrorVSRunning.png',dpi=1000,bbox_inches='tight')









