"""
PED Lyngby Model - Utilities Module

This module contains utility functions used across different scenario implementations.
"""

import os
import yaml
import pandas as pd
import numpy as np

def load_config(config_file, params_file):
    """
    Load configuration and parameters from YAML files.

    Args:
        config_file (str): Path to the main config file
        params_file (str): Path to the component parameters file

    Returns:
        tuple: (config, params) dictionaries containing the configuration and parameters
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    return config, params

def load_or_generate_profile(filename, default_peak_mw, data_dir, index):
    """
    Attempt to load a time series profile from a file, or generate a placeholder if the file is not found.

    Args:
        filename (str): Name of the file to load
        default_peak_mw (float): Default peak value in MW for the generated profile
        data_dir (str): Path to the data directory
        index (pd.DatetimeIndex): Index for the time series

    Returns:
        pd.Series: The loaded or generated profile
    """
    filepath = os.path.join(data_dir, 'timeseries', filename)
    try:
        # !!! ADJUST CSV READING AS NEEDED (e.g., delimiter, header, index_col) !!!
        profile_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        # Select the relevant column and time slice
        profile = profile_df.iloc[:, 0].reindex(index).fillna(0) # Take first column, reindex, fill gaps
        print(f"Loaded profile from: {filepath}")
        return profile
    except FileNotFoundError:
        print(f"Warning: File not found '{filepath}'. Generating placeholder profile.")
        np.random.seed(sum(ord(c) for c in filename)) # Seed based on filename
        base = np.sin(np.linspace(0, 2 * np.pi, 24)) * 0.4 + 0.6
        profile_array = np.tile(base, len(index) // 24 + 1)[:len(index)] * default_peak_mw
        return pd.Series(profile_array, index=index)
    except Exception as e:
         print(f"Error loading '{filepath}': {e}. Using placeholder.")
         # Fallback to placeholder on other errors too
         np.random.seed(sum(ord(c) for c in filename))
         base = np.sin(np.linspace(0, 2 * np.pi, 24)) * 0.4 + 0.6
         profile_array = np.tile(base, len(index) // 24 + 1)[:len(index)] * default_peak_mw
         return pd.Series(profile_array, index=index)


