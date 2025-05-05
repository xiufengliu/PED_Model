"""
PED Lyngby Model - High PV Scenario

This module implements the high PV scenario for the PED Lyngby Model.
The high PV scenario explores maximizing solar PV deployment on available surfaces 
within the district while keeping storage and heating systems at baseline levels.
"""

import pypsa
import pandas as pd
import numpy as np
import os

from .utils import load_config, load_or_generate_profile, setup_basic_network

def create_network(config_file, params_file, data_path):
    """
    Builds the PyPSA network for the high PV scenario.
    This scenario maximizes solar PV deployment on available surfaces within the district.
    
    Args:
        config_file (str): Path to the main config file (e.g., config.yml).
        params_file (str): Path to the component parameters file (e.g., component_params.yml).
        data_path (str): Path to the input data directory.
        
    Returns:
        pypsa.Network: The configured PyPSA network object.
    """
    print("Building high PV network...")
    
    # Load configuration
    config, params = load_config(config_file, params_file)
    
    # Set up basic network with common elements
    network = setup_basic_network(config, params, data_path)
    
    # Get high PV parameters
    high_pv_params = params.get('high_pv', {})
    
    # Load PV generation profile (per unit)
    pv_avail_series = load_or_generate_profile('solar_pv_generation.csv', 1.0, data_path, network.snapshots)
    
    # 1. Stadium Rooftop PV
    stadium_pv_capacity_mw = high_pv_params.get('stadium_pv_capacity_kw', 200) / 1000  # Convert kW to MW
    if stadium_pv_capacity_mw > 0:
        network.add("Generator", "Stadium PV",
                    bus="Stadium Elec",  # Connect directly to stadium electrical bus
                    p_nom=stadium_pv_capacity_mw,
                    p_max_pu=pv_avail_series,  # Time-varying availability
                    marginal_cost=high_pv_params.get('marginal_cost', 0))
        print(f"Added Stadium PV: {high_pv_params.get('stadium_pv_capacity_kw', 200)} kWp")
    
    # 2. Swimming Pool Rooftop PV
    pool_pv_capacity_mw = high_pv_params.get('pool_pv_capacity_kw', 150) / 1000  # Convert kW to MW
    if pool_pv_capacity_mw > 0:
        network.add("Generator", "Pool PV",
                    bus="Pool Elec",  # Connect directly to pool electrical bus
                    p_nom=pool_pv_capacity_mw,
                    p_max_pu=pv_avail_series,  # Time-varying availability
                    marginal_cost=high_pv_params.get('marginal_cost', 0))
        print(f"Added Pool PV: {high_pv_params.get('pool_pv_capacity_kw', 150)} kWp")
    
    # 3. General District PV (other buildings, carports, etc.)
    general_pv_capacity_mw = high_pv_params.get('general_pv_capacity_kw', 300) / 1000  # Convert kW to MW
    if general_pv_capacity_mw > 0:
        network.add("Generator", "General PV",
                    bus="General Elec",  # Connect directly to general electrical bus
                    p_nom=general_pv_capacity_mw,
                    p_max_pu=pv_avail_series,  # Time-varying availability
                    marginal_cost=high_pv_params.get('marginal_cost', 0))
        print(f"Added General District PV: {high_pv_params.get('general_pv_capacity_kw', 300)} kWp")
    
    # Calculate and print total PV capacity
    total_pv_capacity_kw = (high_pv_params.get('stadium_pv_capacity_kw', 200) + 
                           high_pv_params.get('pool_pv_capacity_kw', 150) + 
                           high_pv_params.get('general_pv_capacity_kw', 300))
    print(f"Total PV capacity in high_pv scenario: {total_pv_capacity_kw} kWp")
    
    # Add placeholder storage components (zero capacity for high_pv scenario)
    network.add("StorageUnit", "Placeholder Battery", bus="District LV Bus", p_nom=0, max_hours=0)
    network.add("Store", "Placeholder Thermal Storage", bus="District Heat Source", e_nom=0)
    print("Added placeholder storage components (zero capacity for high_pv scenario)")
    
    print("High PV network build complete.")
    return network
