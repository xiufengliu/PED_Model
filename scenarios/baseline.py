"""
PED Lyngby Model - Baseline Scenario

This module implements the baseline scenario for the PED Lyngby Model.
The baseline scenario represents the current situation or a minimal setup with 
existing loads (stadium, pool, general) and minimal local assets.
"""

import pypsa
import pandas as pd
import numpy as np
import os

from .utils import load_config, load_or_generate_profile, setup_basic_network

def create_network(config_file, params_file, data_path):
    """
    Builds the PyPSA network for the baseline scenario.
    
    Args:
        config_file (str): Path to the main config file (e.g., config.yml).
        params_file (str): Path to the component parameters file (e.g., component_params.yml).
        data_path (str): Path to the input data directory.
        
    Returns:
        pypsa.Network: The configured PyPSA network object.
    """
    print("Building baseline network...")
    
    # Load configuration
    config, params = load_config(config_file, params_file)
    
    # Set up basic network with common elements
    network = setup_basic_network(config, params, data_path)
    
    # Add baseline PV
    pv_params = params.get('baseline_pv', {})
    pv_capacity_mw = pv_params.get('capacity_kw', 50) / 1000  # Convert kW to MW
    pv_avail_series = load_or_generate_profile('solar_pv_generation.csv', 1.0, data_path, network.snapshots)
    
    if pv_capacity_mw > 0:
        network.add("Generator", "Existing PV",
                    bus="District LV Bus",
                    p_nom=pv_capacity_mw,
                    p_max_pu=pv_avail_series,  # Time-varying availability
                    marginal_cost=pv_params.get('marginal_cost', 0))
        print(f"Added Existing PV: {pv_params.get('capacity_kw', 50)} kWp")
    else:
        print("No baseline PV capacity specified.")
    
    # Add placeholder storage components (zero capacity for baseline)
    network.add("StorageUnit", "Placeholder Battery", bus="District LV Bus", p_nom=0, max_hours=0)
    network.add("Store", "Placeholder Thermal Storage", bus="District Heat Source", e_nom=0)
    print("Added placeholder storage components (zero capacity for baseline)")
    
    print("Baseline network build complete.")
    return network
