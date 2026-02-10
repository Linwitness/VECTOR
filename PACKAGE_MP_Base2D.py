#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Class for 2D Grain Boundary Analysis Algorithms

This module provides the shared base class for all 2D algorithm implementations
(Linear, Allen-Cahn, Level Set, Vertex). It contains common functionality for:
- Grid and result array initialization
- Error tracking and computation
- Grain boundary site identification
- Parallel processing domain decomposition
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
import PACKAGE_MP_BaseCommon as BaseCommon


class Base2D(object):
    """Base class for 2D grain boundary analysis algorithms.

    Provides shared infrastructure for grid management, error calculation,
    grain boundary identification, parallel domain decomposition, and
    result visualization.

    Attributes:
        P (ndarray): Phase field array (3, nx, ny) — grain IDs and normal vectors
        C (ndarray): Curvature array (2, nx, ny) — grain IDs and curvature values
        nx (int): Grid points in x direction
        ny (int): Grid points in y direction
        ng (int): Number of grains
        cores (int): CPU cores for parallel processing
        clip (int): Boundary layers to exclude
        errors (float): Accumulated angle errors
        errors_per_site (float): Average error per boundary site
        running_time (float): Total computation time
        running_coreTime (float): Maximum core processing time
    """

    def __init__(self, nx, ny, ng, cores, P0, R, clip=0,
                 verification_system=True, curvature_sign=False):
        """Initialize common algorithm parameters.

        Args:
            nx (int): Number of grid points in x direction
            ny (int): Number of grid points in y direction
            ng (int): Number of grains in the system
            cores (int): Number of CPU cores for parallel processing
            P0 (ndarray): Initial microstructure configuration (2D or 3D array)
            R (ndarray): Reference/analytical solution for validation
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
        self.ng = ng
        self.R = R

        # Initialize result arrays
        self.P = np.zeros((3, nx, ny))  # Stores IC and normal vectors
        self.C = np.zeros((2, nx, ny))  # Stores curvature

        # Convert individual grain maps to single map
        if len(P0.shape) == 2:
            self.P[0, :, :] = np.array(P0)
            self.C[0, :, :] = np.array(P0)
        else:
            for i in range(0, np.shape(P0)[2]):
                self.P[0, :, :] += P0[:, :, i] * (i + 1)
                self.C[0, :, :] += P0[:, :, i] * (i + 1)

        # Parallel processing
        self.cores = cores

        # Configuration flags
        self.verification_system = verification_system
        self.curvature_sign = curvature_sign

    def get_P(self):
        """Get the phase field and normal vector results.

        Returns:
            ndarray: Shape (3, nx, ny) — grain IDs and normal vector components.
        """
        return self.P

    def get_C(self):
        """Get the curvature calculation results.

        Returns:
            ndarray: Shape (2, nx, ny) — grain IDs and curvature values.
        """
        return self.C

    def get_errors(self, negate_dx=False):
        """Calculate error between calculated and reference normal vectors.

        Computes angular difference between calculated normal vectors and
        reference values at each grain boundary site.

        Args:
            negate_dx (bool): If True, negate the dx component before
                comparison (used by Vertex algorithm sign convention).
        """
        ge_gbsites = self.get_gb_list()
        for gbSite in ge_gbsites:
            [gei, gej] = gbSite
            ge_dx, ge_dy = myInput.get_grad(self.P, gei, gej)
            if negate_dx:
                ge_dx = -ge_dx
            self.errors += math.acos(round(abs(ge_dx * self.R[gei, gej, 0] + ge_dy * self.R[gei, gej, 1]), 5))

        if len(ge_gbsites) > 0:
            self.errors_per_site = self.errors / len(ge_gbsites)
        else:
            self.errors_per_site = 0

    def get_curvature_errors(self):
        """Calculate error between calculated and reference curvature values.

        For each grain boundary site, computes difference between calculated
        curvature and reference value.
        """
        gce_gbsites = self.get_gb_list()
        for gbSite in gce_gbsites:
            [gcei, gcej] = gbSite
            self.errors += abs(self.R[gcei, gcej, 2] - self.C[1, gcei, gcej])

        if len(gce_gbsites) != 0:
            self.errors_per_site = self.errors / len(gce_gbsites)
        else:
            self.errors_per_site = 0

    def get_gb_list(self, grainID=1):
        """Get list of grain boundary sites.

        Args:
            grainID (int): ID of grain to find boundaries for

        Returns:
            list: List of [i, j] coordinates of boundary sites
        """
        ggn_gbsites = []
        for i in range(0, self.nx):
            for j in range(0, self.ny):
                if myInput.is_grain_boundary(self.P, i, j, self.nx, self.ny) and self.P[0, i, j] == grainID:
                    ggn_gbsites.append([i, j])
        return ggn_gbsites

    def get_all_gb_list(self):
        """Get grain boundary sites grouped by grain ID.

        Returns:
            list: List of lists, where index i contains boundary sites for grain i+1.
        """
        gagn_gbsites = [[] for _ in range(int(self.ng))]
        for i in range(0, self.nx):
            for j in range(0, self.ny):
                if myInput.is_grain_boundary(self.P, i, j, self.nx, self.ny):
                    gagn_gbsites[int(self.P[0, i, j] - 1)].append([i, j])
        return gagn_gbsites

    def check_subdomain_and_nei(self, A):
        """Determine subdomain ID and neighbor subdomains for parallel processing.

        Maps a global coordinate to its subdomain index and identifies all
        8 neighboring subdomains (with periodic wrapping).

        Args:
            A (list): [i, j] global coordinates of a site

        Returns:
            tuple: (center_subdomain, neighbor_subdomains) where:
                - center_subdomain: [width_id, length_id] subdomain indices
                - neighbor_subdomains: list of 8 [width_id, length_id] neighbors
        """
        ca_length, ca_width = myInput.split_cores(self.cores)
        ca_area_cen = [int(A[0] / self.nx * ca_width), int(A[1] / self.ny * ca_length)]
        ca_area_nei = []
        ca_area_nei.append([int((ca_area_cen[0] - 1) % ca_width), int((ca_area_cen[1] - 1) % ca_length)])
        ca_area_nei.append([int((ca_area_cen[0] - 1) % ca_width), int(ca_area_cen[1])])
        ca_area_nei.append([int((ca_area_cen[0] - 1) % ca_width), int((ca_area_cen[1] + 1) % ca_length)])
        ca_area_nei.append([int(ca_area_cen[0]), int((ca_area_cen[1] + 1) % ca_length)])
        ca_area_nei.append([int((ca_area_cen[0] + 1) % ca_width), int((ca_area_cen[1] + 1) % ca_length)])
        ca_area_nei.append([int((ca_area_cen[0] + 1) % ca_width), int(ca_area_cen[1])])
        ca_area_nei.append([int((ca_area_cen[0] + 1) % ca_width), int((ca_area_cen[1] - 1) % ca_length)])
        ca_area_nei.append([int(ca_area_cen[0]), int((ca_area_cen[1] - 1) % ca_length)])

        return ca_area_cen, ca_area_nei

    def res_back(self, back_result):
        """Callback to aggregate parallel processing results.

        Receives results from worker processes and accumulates them into
        the main P (normal vectors) or C (curvature) arrays.

        Args:
            back_result (tuple): (fval, core_time) where:
                - fval: ndarray of shape (nx, ny, 1) for curvature or
                       (nx, ny, 2) for normal vectors
                - core_time: float, computation time in seconds
        """
        BaseCommon.res_back_common(self, back_result, dim=2)

    def res_back_with_V(self, back_result):
        """Callback to aggregate results including evolution state V.

        Same as res_back but also captures the updated V matrix from
        Allen-Cahn and Level Set algorithms.

        Args:
            back_result (tuple): (fval, core_time, V) where V is the
                updated evolution state matrix.
        """
        BaseCommon.res_back_with_V_common(self, back_result, dim=2)

    def get_2d_plot(self, init, algo, fig_page, arrow_scale=30, cmap='gray',
                    save_prefix='Plot', filter_range=None):
        """Generate 2D visualization of microstructure with normal vectors.

        Args:
            init (str): Name of initial condition
            algo (str): Name of algorithm used
            fig_page (int): Parameter value for title (loop_times, nsteps, etc.)
            arrow_scale (float): Scale factor for arrow length
            cmap (str): Colormap for grain structure
            save_prefix (str): Prefix for saved file name
            filter_range (tuple or None): (min_i, max_i) range to filter arrows,
                or None to show all arrows
        """
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
        plt.imshow(self.P[0, :, :], cmap=cmap, interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'{save_prefix}_PolyGray_noArrows.png', dpi=1000, bbox_inches='tight')

        g2p_gbsites = self.get_gb_list()
        for gbSite in g2p_gbsites:
            [g2pi, g2pj] = gbSite
            g2p_dx, g2p_dy = myInput.get_grad(self.P, g2pi, g2pj)
            if filter_range is None or (g2pi > filter_range[0] and g2pi < filter_range[1]):
                plt.arrow(g2pj, g2pi, arrow_scale * g2p_dx, arrow_scale * g2p_dy,
                          width=0.1, lw=0.1, alpha=0.8, color='navy')

    def run_main(self, core_func_inclination, core_func_curvature,
                 purpose="inclination", callback=None, verbose=True):
        """Generic main execution for parallel algorithm dispatch.

        Args:
            core_func_inclination: Function for inclination computation
            core_func_curvature: Function for curvature computation
            purpose (str): "inclination" or "curvature"
            callback: Callback for apply_async (defaults to self.res_back)
            verbose (bool): Print progress messages
        """
        if callback is None:
            callback = self.res_back

        starttime = datetime.datetime.now()

        pool = mp.Pool(processes=self.cores)
        main_lc, main_wc = myInput.split_cores(self.cores)

        all_sites = np.array([[x, y] for x in range(self.nx) for y in range(self.ny)]).reshape(self.nx, self.ny, 2)
        multi_input = myInput.split_IC(all_sites, self.cores, 2, 0, 1)

        res_list = []
        core_func = core_func_inclination if purpose == "inclination" else core_func_curvature
        for ki in range(main_wc):
            for kj in range(main_lc):
                if verbose:
                    print(f'processor [{ki},{kj}] starting...')
                res_one = pool.apply_async(
                    func=core_func,
                    args=(multi_input[ki][kj],),
                    callback=callback)
                res_list.append(res_one)

        pool.close()
        pool.join()

        if verbose or self.verification_system:
            print("core done!")

        endtime = datetime.datetime.now()
        self.running_time = (endtime - starttime).total_seconds()
