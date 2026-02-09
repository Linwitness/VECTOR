"""
Path Configuration Utilities for VECTOR Framework Examples
===========================================================

Provides standardized path setup for all example scripts to access
the VECTOR framework modules consistently.

Usage:
------
    from examples.shared.path_setup import setup_vector_path
    setup_vector_path()

    # Now you can import VECTOR modules
    import myInput
    import post_processing
"""

import sys
from pathlib import Path


def setup_vector_path():
    """
    Add VECTOR framework root to sys.path.

    This function ensures the VECTOR framework root directory is in
    sys.path, allowing example scripts to import framework modules
    regardless of their location in the examples directory tree.

    The function is idempotent - calling it multiple times has no
    additional effect.
    """
    vector_root = Path(__file__).parent.parent.parent
    vector_root_str = str(vector_root)
    if vector_root_str not in sys.path:
        sys.path.insert(0, vector_root_str)


def get_vector_root():
    """
    Get the absolute path to the VECTOR framework root directory.

    Returns:
    --------
    Path : pathlib.Path object pointing to VECTOR root
    """
    return Path(__file__).parent.parent.parent
