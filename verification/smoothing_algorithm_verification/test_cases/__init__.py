"""
Test cases package for smoothing algorithm verification.
Imports configuration and provides common utilities.

Author: Lin Yang
"""

import os
import sys

# Add parent directory to path for importing test_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_config import (
    CONFIG_2D,
    CONFIG_3D,
    ALGORITHM_PARAMS,
    OUTPUT_CONFIG,
    PLOT_CONFIG
)

from .test_utils import (
    calculate_normal_vector_error_2d,
    calculate_normal_vector_error_3d,
    calculate_curvature_error_2d,
    calculate_curvature_error_3d,
    get_curvature_statistics
)