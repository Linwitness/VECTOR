#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Curvature Validation Framework for VECTOR Algorithms

This module provides a unified framework for validating curvature calculation
algorithms across different dimensions (2D/3D), methods (linear/vertex),
and geometric complexities (circular/spherical/complex).

Classes:
    ValidationStudy: Configurable validation study for curvature algorithms

Author: Lin Yang
Created: Thu Sep 22 17:46:02 2022
"""

import os
import sys
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable

# Add VECTOR framework to path
current_path = os.getcwd() + '/'
sys.path.append(current_path)
sys.path.append(current_path + '../../')

import PACKAGE_MP_Linear as Linear_2D
import PACKAGE_MP_Vertex as Vertex_2D
import PACKAGE_MP_3DLinear as Linear_3D
import PACKAGE_MP_3DVertex as Vertex_3D
import myInput


class AlgorithmType(Enum):
    """Curvature calculation algorithm types."""
    LINEAR_2D = "BL"
    VERTEX_2D = "VT"
    LINEAR_3D = "BL3D"
    VERTEX_3D = "VT3D"
    VERTEX_3D_COMPLEX = "VT3DComp"


@dataclass
class ValidationConfig:
    """Configuration for a validation study."""
    algorithm: AlgorithmType
    dimensions: Tuple[int, ...]  # (nx, ny) or (nx, ny, nz)
    cores: int = 8
    max_iteration: int = 20
    radii: Optional[List[int]] = None  # For multi-radius studies
    waves: Optional[List[int]] = None  # For complex geometry studies


class ValidationStudy:
    """
    Unified validation framework for curvature calculation algorithms.

    This class consolidates the validation logic for different algorithm
    types (linear/vertex, 2D/3D) and geometric configurations into a
    single configurable interface.

    Parameters:
    -----------
    config : ValidationConfig
        Configuration specifying algorithm, dimensions, and parameters

    Attributes:
    -----------
    errors : np.ndarray
        Per-iteration error values
    running_times : np.ndarray
        Per-iteration computation times
    """

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.errors = np.zeros(config.max_iteration)
        self.running_times = np.zeros(config.max_iteration)

    def _get_algorithm_class(self):
        """Return the appropriate algorithm class based on config."""
        mapping = {
            AlgorithmType.LINEAR_2D: Linear_2D.linear_class,
            AlgorithmType.VERTEX_2D: Vertex_2D.vertex_class,
            AlgorithmType.LINEAR_3D: Linear_3D.linear3d_class,
            AlgorithmType.VERTEX_3D: Vertex_3D.vertex3d_class,
            AlgorithmType.VERTEX_3D_COMPLEX: Vertex_3D.vertex3d_class,
        }
        return mapping[self.config.algorithm]

    def _get_geometry_generator(self) -> Callable:
        """Return geometry generator based on algorithm type."""
        if self.config.algorithm in [AlgorithmType.LINEAR_2D, AlgorithmType.VERTEX_2D]:
            return myInput.Circle_IC
        elif self.config.algorithm == AlgorithmType.VERTEX_3D_COMPLEX:
            return myInput.Complex2G_IC3d
        else:
            return myInput.Circle_IC3d

    def _get_main_method_name(self) -> str:
        """Return main method name based on algorithm type."""
        if self.config.algorithm in [AlgorithmType.LINEAR_2D, AlgorithmType.VERTEX_2D]:
            if self.config.algorithm == AlgorithmType.LINEAR_2D:
                return "linear_main"
            return "vertex_main"
        else:
            if self.config.algorithm == AlgorithmType.LINEAR_3D:
                return "linear3d_main"
            return "vertex3d_main"

    def _get_iteration_param_name(self) -> str:
        """Return iteration parameter name for the algorithm."""
        if self.config.algorithm in [AlgorithmType.VERTEX_2D, AlgorithmType.VERTEX_3D,
                                      AlgorithmType.VERTEX_3D_COMPLEX]:
            return "interval"
        return "loop_times"

    def run(self, param_value: int, output_prefix: str) -> None:
        """
        Run validation study for a given geometry parameter.

        Parameters:
        -----------
        param_value : int
            Radius for circular/spherical, wavelength for complex geometry
        output_prefix : str
            Prefix for output file path
        """
        dims = self.config.dimensions
        ng = 2
        cores = self.config.cores
        max_iter = self.config.max_iteration

        # Generate filename
        if self.config.algorithm == AlgorithmType.VERTEX_3D_COMPLEX:
            param_name = f"wave{param_value}"
        else:
            param_name = f"R{param_value}"
        filename = f"{output_prefix}{self.config.algorithm.value}_Curvature_{param_name}_Iteration_1_{max_iter}"

        print(f"=== {self.config.algorithm.value} Validation: {param_name} ===")

        # Generate test geometry
        geometry_gen = self._get_geometry_generator()
        if len(dims) == 2:
            P0, R = geometry_gen(dims[0], dims[1], param_value)
        else:
            P0, R = geometry_gen(dims[0], dims[1], dims[2], param_value)

        # Get algorithm class and method
        AlgClass = self._get_algorithm_class()
        main_method = self._get_main_method_name()
        iter_param = self._get_iteration_param_name()

        # Run convergence analysis
        for iteration in range(1, max_iter):
            print(f"Processing iteration {iteration} for {param_name}")

            # Initialize algorithm instance
            if len(dims) == 2:
                instance = AlgClass(dims[0], dims[1], ng, cores, iteration, P0, R)
            else:
                if self.config.algorithm == AlgorithmType.LINEAR_3D:
                    instance = AlgClass(dims[0], dims[1], dims[2], ng, cores, iteration, P0, R, 'np')
                else:
                    instance = AlgClass(dims[0], dims[1], dims[2], ng, cores, iteration, P0, R)

            # Execute curvature calculation
            getattr(instance, main_method)("curvature")
            C = instance.get_C()

            # Display metrics
            iter_val = getattr(instance, iter_param) if hasattr(instance, iter_param) else iteration
            print(f'{iter_param} = {iter_val}')
            print(f'running_time = {instance.running_time:.2f} seconds')
            print(f'running_core time = {instance.running_coreTime:.2f} seconds')
            print(f'total_errors = {instance.errors:.2f}')
            print(f'per_errors = {instance.errors_per_site:.3f}')
            print()

            # Store results
            self.errors[iteration-1] = instance.errors_per_site
            self.running_times[iteration-1] = instance.running_coreTime

        # Save results
        np.savez(filename,
                 errors=self.errors,
                 running_time=self.running_times)
        print(f"Results saved to: {filename}")

    def run_multi_param(self, output_prefix: str = "./") -> None:
        """
        Run validation across multiple parameter values.

        Uses radii for circular/spherical or waves for complex geometries.
        """
        params = self.config.waves if self.config.algorithm == AlgorithmType.VERTEX_3D_COMPLEX else self.config.radii

        if params is None:
            raise ValueError("No parameter values specified in config")

        for param in params:
            self.run(param, output_prefix)


# Convenience functions for common validation configurations

def run_2d_linear_validation(radius: int = 20, max_iteration: int = 20, cores: int = 8):
    """Run 2D linear algorithm validation with circular geometry."""
    config = ValidationConfig(
        algorithm=AlgorithmType.LINEAR_2D,
        dimensions=(200, 200),
        cores=cores,
        max_iteration=max_iteration,
        radii=[radius]
    )
    study = ValidationStudy(config)
    study.run(radius, "examples/curvature_calculation/")


def run_2d_vertex_validation(radius: int = 80, max_iteration: int = 20, cores: int = 8):
    """Run 2D vertex algorithm validation with circular geometry."""
    config = ValidationConfig(
        algorithm=AlgorithmType.VERTEX_2D,
        dimensions=(200, 200),
        cores=cores,
        max_iteration=max_iteration,
        radii=[radius]
    )
    study = ValidationStudy(config)
    study.run(radius, "./")


def run_3d_linear_validation(radii: List[int] = None, max_iteration: int = 20, cores: int = 16):
    """Run 3D linear algorithm validation with spherical geometry."""
    if radii is None:
        radii = [80]
    config = ValidationConfig(
        algorithm=AlgorithmType.LINEAR_3D,
        dimensions=(200, 200, 200),
        cores=cores,
        max_iteration=max_iteration,
        radii=radii
    )
    study = ValidationStudy(config)
    study.run_multi_param("./")


def run_3d_vertex_validation(radii: List[int] = None, max_iteration: int = 20, cores: int = 8):
    """Run 3D vertex algorithm validation with spherical geometry."""
    if radii is None:
        radii = [5, 20, 50, 80, 2, 1]
    config = ValidationConfig(
        algorithm=AlgorithmType.VERTEX_3D,
        dimensions=(200, 200, 200),
        cores=cores,
        max_iteration=max_iteration,
        radii=radii
    )
    study = ValidationStudy(config)
    study.run_multi_param("./")


def run_3d_complex_validation(waves: List[int] = None, max_iteration: int = 20, cores: int = 8):
    """Run 3D vertex algorithm validation with complex sinusoidal geometry."""
    if waves is None:
        waves = [5, 20, 50, 80, 2, 1]
    config = ValidationConfig(
        algorithm=AlgorithmType.VERTEX_3D_COMPLEX,
        dimensions=(200, 200, 200),
        cores=cores,
        max_iteration=max_iteration,
        waves=waves
    )
    study = ValidationStudy(config)
    study.run_multi_param("./")


if __name__ == '__main__':
    """
    Main Execution Block for Curvature Validation Studies

    This section provides examples of running validation studies using
    the unified framework. Uncomment desired studies to execute.
    """
    print("=== Curvature Validation Framework ===")
    print("Using unified ValidationStudy class for all algorithm types")
    print()

    # Example: Run 3D linear validation (default)
    run_3d_linear_validation()

    # Additional validation studies (uncomment as needed):
    # run_2d_linear_validation()
    # run_2d_vertex_validation()
    # run_3d_vertex_validation()
    # run_3d_complex_validation()

    print("=== Validation Complete ===")
