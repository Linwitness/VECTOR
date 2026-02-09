#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microstructure Plotting Base Class
==================================

Provides a base class and configuration system for microstructure visualization.
This module consolidates common functionality from multiple plotting scripts
to reduce code duplication and improve maintainability.

Usage:
    from microstructure_plotter import MicrostructurePlotter, DatasetConfig

    config = MicrostructurePlotter.Config(
        name="circle",
        data_folder="/path/to/data/",
        figure_folder="./figures/",
        initial_grain_num=2,
        colormap='gray_r'
    )

    plotter = MicrostructurePlotter(config)
    plotter.add_dataset("000", "filename.npy", timestep=30)
    plotter.load_data()
    plotter.plot_all()

Created for VECTOR project microstructure analysis.
@author: Lin
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np


@dataclass
class DatasetConfig:
    """Configuration for a single dataset to be plotted."""
    filename: str
    timestep: int
    suffix: str = ""  # Added to figure filename

    def __post_init__(self):
        if not self.suffix:
            # Extract suffix from filename if not provided
            self.suffix = self.filename.split('.')[0][-3:]


@dataclass
class PlotterConfig:
    """Configuration for the MicrostructurePlotter."""
    name: str
    data_folder: str
    figure_folder: str
    initial_grain_num: int
    colormap: str = 'rainbow'
    figure_prefix: str = ""

    def __post_init__(self):
        if not self.figure_prefix:
            self.figure_prefix = f"microstructure_{self.name}"


class MicrostructurePlotter:
    """
    Base class for microstructure visualization.

    Consolidates common patterns from multiple plotting scripts:
    - Data loading from numpy files
    - Timestep selection for visualization
    - Figure generation using post_processing module

    Attributes:
        config: PlotterConfig with plotting parameters
        datasets: Dict mapping dataset names to DatasetConfig
        data: Dict mapping dataset names to loaded numpy arrays
    """

    Config = PlotterConfig  # Alias for convenience

    def __init__(self, config: PlotterConfig):
        """
        Initialize the plotter with configuration.

        Parameters:
            config: PlotterConfig instance with plotting parameters
        """
        self.config = config
        self.datasets: Dict[str, DatasetConfig] = {}
        self.data: Dict[str, np.ndarray] = {}

        # Ensure post_processing is available
        self._setup_imports()

    def _setup_imports(self):
        """Set up necessary imports for post_processing module."""
        current_path = os.getcwd()
        if current_path not in sys.path:
            sys.path.append(current_path)
        parent_path = os.path.join(current_path, '..', '..')
        if parent_path not in sys.path:
            sys.path.append(parent_path)

        global post_processing
        try:
            import post_processing as pp
            post_processing = pp
        except ImportError:
            # Will be imported later when needed
            pass

    def add_dataset(self, name: str, filename: str, timestep: int,
                    suffix: Optional[str] = None):
        """
        Add a dataset configuration.

        Parameters:
            name: Identifier for the dataset
            filename: Name of the numpy file
            timestep: Timestep to visualize
            suffix: Optional suffix for figure filename
        """
        self.datasets[name] = DatasetConfig(
            filename=filename,
            timestep=timestep,
            suffix=suffix if suffix else name
        )

    def add_datasets(self, datasets: Dict[str, Dict]):
        """
        Add multiple datasets at once.

        Parameters:
            datasets: Dict mapping names to dicts with 'filename', 'timestep',
                     and optional 'suffix' keys
        """
        for name, params in datasets.items():
            self.add_dataset(
                name=name,
                filename=params['filename'],
                timestep=params['timestep'],
                suffix=params.get('suffix', name)
            )

    def load_data(self, verbose: bool = True):
        """
        Load all configured datasets.

        Parameters:
            verbose: If True, print loading progress
        """
        if verbose:
            print(f"Loading {self.config.name} microstructure data...")

        for name, dataset in self.datasets.items():
            filepath = os.path.join(self.config.data_folder, dataset.filename)
            self.data[name] = np.load(filepath)
            if verbose:
                print(f"  {name}: {self.data[name].shape}")

        if verbose:
            print("READING DATA DONE")

    def load_single(self, name: str) -> np.ndarray:
        """
        Load a single dataset.

        Parameters:
            name: Dataset identifier

        Returns:
            Loaded numpy array
        """
        if name not in self.data:
            dataset = self.datasets[name]
            filepath = os.path.join(self.config.data_folder, dataset.filename)
            self.data[name] = np.load(filepath)
        return self.data[name]

    def plot_dataset(self, name: str, cmap: Optional[str] = None):
        """
        Plot a single dataset.

        Parameters:
            name: Dataset identifier
            cmap: Colormap override (uses config.colormap if None)
        """
        import post_processing

        if name not in self.data:
            self.load_single(name)

        dataset = self.datasets[name]
        data = self.data[name]

        # Build figure path
        figure_path = os.path.join(
            self.config.figure_folder,
            f"{self.config.figure_prefix}_{dataset.suffix}"
        )

        # Use provided colormap or config default
        colormap = cmap if cmap else self.config.colormap

        # Extract grain ID data (first feature dimension)
        grain_data = data[:, :, :, 0] if data.ndim == 4 else data

        post_processing.plot_structure_figure(
            dataset.timestep,
            grain_data,
            figure_path,
            cmap=colormap
        )

    def plot_all(self, verbose: bool = True):
        """
        Plot all configured datasets.

        Parameters:
            verbose: If True, print progress
        """
        if verbose:
            print(f"Generating {self.config.name} microstructure visualizations...")

        for name in self.datasets:
            self.plot_dataset(name)

        if verbose:
            print(f"{self.config.name.capitalize()} microstructure analysis complete!")


class CirclePlotter(MicrostructurePlotter):
    """Plotter for circular two-grain microstructure studies."""

    @classmethod
    def create_default(cls, data_folder: str, figure_folder: str = "./figures/"):
        """Create plotter with default circle dataset configurations."""
        config = PlotterConfig(
            name="circle",
            data_folder=data_folder,
            figure_folder=figure_folder,
            initial_grain_num=2,
            colormap='gray_r'
        )
        plotter = cls(config)

        # Delta sensitivity series
        delta_values = ["0.0", "0.2", "0.4", "0.6", "0.8", "0.95"]
        for delta in delta_values:
            d_str = delta.replace(".", "")
            prefix = "cT_ori_aveE" if float(delta) >= 0.8 else "c_ori_aveE"
            plotter.add_dataset(
                name=f"delta_{d_str}",
                filename=f"{prefix}_000_000_multiCore16_kt066_seed56689_scale1_delta{delta}_m2_refer_1_0_0.npy",
                timestep=30,
                suffix=d_str
            )

        # Orientation series (delta=0.95)
        orientations = [
            ("087", "_0.87_0.5_0"),
            ("071", "_0.71_0.71_0"),
            ("050", "_0.5_0.87_0"),
            ("100", "_0_1_0"),
        ]
        for suffix, refer in orientations:
            plotter.add_dataset(
                name=f"orient_{suffix}",
                filename=f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta0.95_m2_refer{refer}.npy",
                timestep=28,
                suffix=f"095_{suffix}"
            )

        # Mobility series (delta=0.95)
        for m in [4, 6]:
            plotter.add_dataset(
                name=f"mobility_m{m}",
                filename=f"cT_ori_aveE_000_000_multiCore16_kt066_seed56689_scale1_delta0.95_m{m}_refer_1_0_0.npy",
                timestep=14,
                suffix=f"095_m{m}"
            )

        return plotter


class PolyPlotter(MicrostructurePlotter):
    """Plotter for medium-scale polycrystalline delta parameter studies."""

    @classmethod
    def create_default(cls, data_folder: str, figure_folder: str = "./figures/"):
        """Create plotter with default poly dataset configurations."""
        config = PlotterConfig(
            name="poly",
            data_folder=data_folder,
            figure_folder=figure_folder,
            initial_grain_num=512,
            colormap='rainbow'
        )
        plotter = cls(config)

        # Delta parameter datasets with optimized timesteps
        datasets = {
            "000": {"delta": "0.0", "cores": 16, "timestep": 89},
            "020": {"delta": "0.2", "cores": 16, "timestep": 75},
            "040": {"delta": "0.4", "cores": 16, "timestep": 116},
            "060": {"delta": "0.6", "cores": 8, "timestep": 106},  # 8-core variant
            "080": {"delta": "0.8", "cores": 16, "timestep": 105},
            "095": {"delta": "0.95", "cores": 16, "timestep": 64},
        }

        for suffix, params in datasets.items():
            plotter.add_dataset(
                name=f"delta_{suffix}",
                filename=f"p_ori_ave_aveE_512_multiCore{params['cores']}_delta{params['delta']}_m2_J1_refer_1_0_0_seed56689_kt066.npy",
                timestep=params['timestep'],
                suffix=suffix
            )

        return plotter


class Poly20kPlotter(MicrostructurePlotter):
    """Plotter for large-scale oriented polycrystalline energy method studies."""

    @classmethod
    def create_default(cls, data_folder: str, figure_folder: str = "./figures/"):
        """Create plotter with default poly20k dataset configurations."""
        config = PlotterConfig(
            name="poly20k",
            data_folder=data_folder,
            figure_folder=figure_folder,
            initial_grain_num=20000,
            colormap='rainbow'
        )
        plotter = cls(config)

        # Energy method datasets with optimized timesteps
        energy_methods = {
            "ave": {"prefix": "p", "timestep": 11},
            "consMin": {"prefix": "pm", "timestep": 11},
            "sum": {"prefix": "pm", "timestep": 11},
            "min": {"prefix": "pm", "timestep": 30},
            "max": {"prefix": "pm", "timestep": 15},
            "consMax": {"prefix": "pm", "timestep": 11},
        }

        for method, params in energy_methods.items():
            plotter.add_dataset(
                name=method,
                filename=f"{params['prefix']}_ori_ave_{method}E_20000_multiCore64_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066.npy",
                timestep=params['timestep'],
                suffix=method.lower()
            )

        return plotter


class Poly20kRandomPlotter(MicrostructurePlotter):
    """Plotter for large-scale random orientation polycrystalline studies."""

    @classmethod
    def create_default(cls, data_folder: str, figure_folder: str = "./figures/"):
        """Create plotter with default poly20k random theta configurations."""
        config = PlotterConfig(
            name="randomtheta0_poly20k",
            data_folder=data_folder,
            figure_folder=figure_folder,
            initial_grain_num=20000,
            colormap='rainbow'
        )
        plotter = cls(config)

        # Energy method datasets with optimized timesteps
        energy_methods = {
            "ave": 12,
            "consMin": 12,
            "sum": 12,
            "min": 22,
            "max": 12,
            "consMax": 13,
        }

        for method, timestep in energy_methods.items():
            plotter.add_dataset(
                name=method,
                filename=f"p_randomtheta0_{method}E_20000_delta0.6_J1_refer_1_0_0_seed56689_kt0.66.npy",
                timestep=timestep,
                suffix=method.lower()
            )

        return plotter


class HexPlotter(MicrostructurePlotter):
    """Plotter for hexagonal triple junction energy validation studies."""

    @classmethod
    def create_default(cls, data_folder: str, figure_folder: str = "./figures/"):
        """Create plotter with default hex dataset configurations."""
        config = PlotterConfig(
            name="hex",
            data_folder=data_folder,
            figure_folder=figure_folder,
            initial_grain_num=48,
            colormap='gray_r'
        )
        plotter = cls(config)

        # Single dataset for TJE validation
        plotter.add_dataset(
            name="initial",
            filename="h_ori_ave_aveE_hex_multiCore32_delta0.6_m2_J1_refer_1_0_0_seed56689_kt066_angle.npy",
            timestep=0,
            suffix="initial"
        )

        return plotter


if __name__ == '__main__':
    # Test module loading
    print("Microstructure plotter module loaded successfully")
    print("Available classes: MicrostructurePlotter, CirclePlotter, PolyPlotter,")
    print("                   Poly20kPlotter, Poly20kRandomPlotter, HexPlotter")
