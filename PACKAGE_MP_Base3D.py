#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Class for 3D Grain Boundary Analysis Algorithms

This module provides the shared base class for all 3D algorithm implementations
(3DLinear, 3DAllenCahn, 3DLevelSet, 3DVertex). It contains common functionality for:
- Grid and result array initialization
- Error tracking and computation
- Grain boundary site identification
- Result aggregation from parallel workers

Author: Lin Yang
"""

import os
current_path = os.getcwd()+'/'
import sys
sys.path.append(current_path)
import numpy as np
import math
import matplotlib.pyplot as plt
import myInput
import datetime
import multiprocessing as mp


class Base3D(object):
    """Base class for 3D grain boundary analysis algorithms.

    Provides shared infrastructure for grid management, error calculation,
    grain boundary identification, and result visualization for 3D systems.

    Attributes:
        P (ndarray): Phase field array (4, nx, ny, nz) — grain IDs and normal vectors
        C (ndarray): Curvature array (2, nx, ny, nz) — grain IDs and curvature values
        nx, ny, nz (int): Grid dimensions
        ng (int): Number of grains
        cores (int): CPU cores for parallel processing
        clip (int): Boundary layers to exclude
        errors (float): Accumulated angle errors
        errors_per_site (float): Average error per boundary site
        running_time (float): Total computation time
        running_coreTime (float): Maximum core processing time
    """

    def __init__(self, nx, ny, nz, ng, cores, P0, R, bc='p', clip=0,
                 verification_system=True, curvature_sign=False):
        """Initialize common 3D algorithm parameters.

        Args:
            nx, ny, nz (int): Grid dimensions
            ng (int): Number of grains
            cores (int): Number of CPU cores for parallel processing
            P0 (ndarray): Initial microstructure configuration (3D or 4D array)
            R (ndarray): Reference/analytical solution for validation
            bc (str): Boundary condition type ('p' for periodic, 'np' for non-periodic)
            clip (int): Number of boundary layers to ignore
            verification_system (bool): Enable validation output
            curvature_sign (bool): Calculate signed curvature values
        """
        # Runtime tracking
        self.running_time = 0
        self.running_coreTime = 0
        self.errors = 0
        self.errors_per_site = 0
        self.clip = clip

        # Grid parameters
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.ng = ng
        self.R = R
        self.bc = bc

        # Initialize result arrays
        self.P = np.zeros((4, nx, ny, nz))  # Stores IC and normal vectors (3 components)
        self.C = np.zeros((2, nx, ny, nz))  # Stores curvature

        # Convert individual grain maps to single map
        if len(P0.shape) == 3:
            self.P[0, :, :, :] = np.array(P0)
            self.C[0, :, :, :] = np.array(P0)
        else:
            for i in range(0, np.shape(P0)[3]):
                self.P[0, :, :, :] += P0[:, :, :, i] * (i + 1)
                self.C[0, :, :, :] += P0[:, :, :, i] * (i + 1)

        # Parallel processing
        self.cores = cores

        # Configuration flags
        self.verification_system = verification_system
        self.curvature_sign = curvature_sign

    def get_P(self):
        """Get the phase field and normal vector results.

        Returns:
            ndarray: Shape (4, nx, ny, nz) — grain IDs and 3 normal vector components.
        """
        return self.P

    def get_C(self):
        """Get the curvature calculation results.

        Returns:
            ndarray: Shape (2, nx, ny, nz) — grain IDs and curvature values.
        """
        return self.C

    def get_errors(self):
        """Calculate error between calculated and reference normal vectors.

        Computes angular difference between calculated 3D normal vectors and
        reference values at each grain boundary site.
        """
        ge_gbsites = self.get_gb_list()
        for gbSite in ge_gbsites:
            [gei, gej, gek] = gbSite
            ge_dx, ge_dy, ge_dz = myInput.get_grad3d(self.P, gei, gej, gek)
            self.errors += math.acos(round(abs(ge_dx * self.R[gei, gej, gek, 0] + ge_dy * self.R[gei, gej, gek, 1] + ge_dz * self.R[gei, gej, gek, 2]), 5))

        if len(ge_gbsites) > 0:
            self.errors_per_site = self.errors / len(ge_gbsites)
        else:
            self.errors_per_site = 0

    def get_curvature_errors(self):
        """Calculate error between calculated and reference curvature values.

        For each grain boundary site, computes difference between calculated
        curvature and reference value (stored at index 3 in R for 3D).
        """
        gce_gbsites = self.get_gb_list()
        for gceSite in gce_gbsites:
            [gcei, gcej, gcek] = gceSite
            self.errors += abs(self.R[gcei, gcej, gcek, 3] - self.C[1, gcei, gcej, gcek])

        if len(gce_gbsites) != 0:
            self.errors_per_site = self.errors / len(gce_gbsites)
        else:
            self.errors_per_site = 0

    def get_gb_list(self, grainID=1):
        """Get list of grain boundary sites in 3D.

        Handles both periodic and non-periodic boundary conditions.
        For non-periodic BC, excludes sites within halfL of boundaries.

        Args:
            grainID (int): ID of grain to find boundaries for

        Returns:
            list: List of [i, j, k] coordinates of boundary sites
        """
        ggn_gbsites = []
        edge_l = self._get_edge_length()
        for i in range(0 + edge_l, self.nx - edge_l):
            for j in range(0 + edge_l, self.ny - edge_l):
                for k in range(0 + edge_l, self.nz - edge_l):
                    if myInput.is_grain_boundary_3d(self.P, i, j, k, self.nx, self.ny, self.nz) and self.P[0, i, j, k] == grainID:
                        ggn_gbsites.append([i, j, k])
        return ggn_gbsites

    def _get_edge_length(self):
        """Calculate edge exclusion length based on boundary conditions.

        Returns:
            int: Number of sites to exclude from each edge.
        """
        if hasattr(self, 'bc') and self.bc == 'np' and hasattr(self, 'halfL'):
            return self.halfL
        elif hasattr(self, 'halfL'):
            return 0
        else:
            return 1

    def res_back(self, back_result):
        """Callback to aggregate parallel processing results for 3D.

        Args:
            back_result (tuple): (fval, core_time) where:
                - fval: ndarray of shape (nx, ny, nz, 1) for curvature or
                       (nx, ny, nz, 3) for normal vectors
                - core_time: float, computation time in seconds
        """
        res_stime = datetime.datetime.now()
        (fval, core_time) = back_result
        if core_time > self.running_coreTime:
            self.running_coreTime = core_time

        if self.verification_system == True:
            print("res_back start...")
        if fval.shape[3] == 1:
            self.C[1, :, :, :] += fval[:, :, :, 0]
        elif fval.shape[3] == 3:
            self.P[1, :, :, :] += fval[:, :, :, 0]
            self.P[2, :, :, :] += fval[:, :, :, 1]
            self.P[3, :, :, :] += fval[:, :, :, 2]
        res_etime = datetime.datetime.now()
        if self.verification_system == True:
            print("my res time is " + str((res_etime - res_stime).total_seconds()))

    def res_back_with_V(self, back_result):
        """Callback to aggregate results including evolution state V for 3D.

        Args:
            back_result (tuple): (fval, core_time, V) where V is the
                updated evolution state matrix.
        """
        res_stime = datetime.datetime.now()
        (fval, core_time, self.V) = back_result
        if core_time > self.running_coreTime:
            self.running_coreTime = core_time

        print("res_back start...")
        self.P[1, :, :, :] += fval[:, :, :, 0]
        self.P[2, :, :, :] += fval[:, :, :, 1]
        self.P[3, :, :, :] += fval[:, :, :, 2]
        res_etime = datetime.datetime.now()
        print("my res time is " + str((res_etime - res_stime).total_seconds()))

    def get_2d_plot(self, init, algo, fig_page, z_surface=None, arrow_scale=10,
                    cmap='nipy_spectral', save_prefix='Plot'):
        """Generate 2D slice visualization of 3D microstructure with normal vectors.

        Args:
            init (str): Name of initial condition
            algo (str): Name of algorithm used
            fig_page (int): Parameter value for title
            z_surface (int or None): Z-slice to visualize (None = mid-plane)
            arrow_scale (float): Scale factor for arrow length
            cmap (str): Colormap for grain structure
            save_prefix (str): Prefix for saved file name
        """
        if z_surface is None:
            z_surface = int(self.nz / 2)
        plt.subplots_adjust(wspace=0.2, right=1.8)
        plt.close()
        fig1 = plt.figure(1)
        plt.title(f'{algo}-{init} \n loop = ' + str(fig_page))
        if fig_page < 10:
            String = '000' + str(fig_page)
        elif fig_page < 100:
            String = '00' + str(fig_page)
        elif fig_page < 1000:
            String = '0' + str(fig_page)
        elif fig_page < 10000:
            String = str(fig_page)
        plt.imshow(self.P[0, :, :, z_surface], cmap=cmap, interpolation='nearest')

        g2p_gbsites = self.get_gb_list()
        for gbSite in g2p_gbsites:
            [g2pi, g2pj, g2pk] = gbSite
            if g2pk == z_surface:
                g2p_dx, g2p_dy, g2p_dz = myInput.get_grad3d(self.P, g2pi, g2pj, g2pk)
                plt.arrow(g2pj, g2pi, arrow_scale * g2p_dx, arrow_scale * g2p_dy,
                          width=0.1, lw=0.1, alpha=0.8, color='navy')

        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'{init}-{algo}.{String}.png', dpi=1000, bbox_inches='tight')

    def run_main(self, core_func_inclination, core_func_curvature,
                 purpose="inclination", callback=None, verbose=True,
                 core_extra_args=None):
        """Generic main execution for 3D parallel algorithm dispatch.

        Args:
            core_func_inclination: Function for inclination computation
            core_func_curvature: Function for curvature computation
            purpose (str): "inclination" or "curvature"
            callback: Callback for apply_async (defaults to self.res_back)
            verbose (bool): Print progress messages
            core_extra_args (tuple or None): Extra args to pass to core functions
        """
        if callback is None:
            callback = self.res_back

        starttime = datetime.datetime.now()

        pool = mp.Pool(processes=self.cores)
        main_lc, main_wc, main_hc = myInput.split_cores(self.cores, 3)

        all_sites = np.array([[x, y, z] for x in range(self.nx) for y in range(self.ny) for z in range(self.nz)]).reshape(self.nx, self.ny, self.nz, 3)
        multi_input = myInput.split_IC(all_sites, self.cores, 3, 0, 1, 2)

        res_list = []
        core_func = core_func_inclination if purpose == "inclination" else core_func_curvature
        for mpi in range(main_wc):
            for mpj in range(main_lc):
                for mpk in range(main_hc):
                    if core_extra_args is not None:
                        args = (multi_input[mpi][mpj][mpk],) + core_extra_args
                    else:
                        args = (multi_input[mpi][mpj][mpk],)
                    res_one = pool.apply_async(
                        func=core_func,
                        args=args,
                        callback=callback)
                    res_list.append(res_one)

        pool.close()
        pool.join()

        if verbose:
            print("core done!")

        endtime = datetime.datetime.now()
        self.running_time = (endtime - starttime).total_seconds()
