"""
Sensitive attribute and proxy detection.
"""

from .correlation import find_column_correlation, find_sensitive_correlations
from .detection import detect_names_df, load_config

__all__ = ["find_column_correlation", "find_sensitive_correlations", "detect_names_df", "load_config"]
