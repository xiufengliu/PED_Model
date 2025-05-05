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

def setup_basic_network(config, params, data_path):
    """
    Set up a basic PyPSA network with common elements used across scenarios.
    
    Args:
        config (dict): Configuration dictionary
        params (dict): Parameters dictionary
        data_path (str): Path to the data directory
        
    Returns:
        pypsa.Network: A basic network with common elements
    """
    import pypsa
    
    # Get simulation settings
    sim_settings = config.get('simulation_settings', {})
    
    # Create network
    network = pypsa.Network(multi_invest=False)
    
    # Define time steps
    n_hours = sim_settings.get('num_hours', 72)
    start_date = sim_settings.get('start_date', "2025-01-01")
    timestamps = pd.date_range(start=start_date, periods=n_hours, freq='h')
    network.set_snapshots(timestamps)
    print(f"Set network snapshots: {len(network.snapshots)} hours, starting {start_date}")
    
    # Define Energy Carriers
    network.add("Carrier", "electricity")
    network.add("Carrier", "heat")
    print("Added carriers: electricity, heat")
    
    # Define Buses
    network.add("Bus", "Grid Connection", carrier="electricity")
    network.add("Bus", "District LV Bus", carrier="electricity") # Main electrical bus
    network.add("Bus", "Stadium Elec", carrier="electricity")
    network.add("Bus", "Pool Elec", carrier="electricity")
    network.add("Bus", "General Elec", carrier="electricity")
    network.add("Bus", "Stadium Heat", carrier="heat")
    network.add("Bus", "Pool Heat", carrier="heat")
    network.add("Bus", "General Heat", carrier="heat")
    network.add("Bus", "District Heat Source", carrier="heat") # e.g., connection to DH or local boiler
    print("Added electrical and heat buses")
    
    # Set up grid connection
    grid_params = params.get('grid', {})
    grid_cost = grid_params.get('import_cost_eur_per_mwh', 50)
    grid_capacity = grid_params.get('capacity_mw', 10)
    
    network.add("Generator", "Grid",
                bus="Grid Connection",
                carrier="electricity",
                marginal_cost=grid_cost,
                p_nom=grid_capacity,
                p_min_pu=-1, # Allow export up to p_nom
                p_max_pu=1)  # Allow import up to p_nom
    print(f"Added Grid connection: Capacity={grid_capacity} MW, Cost={grid_cost} EUR/MWh")
    
    # Link Grid Connection Point to District Bus (e.g., transformer)
    network.add("Link", "Substation",
                bus0="Grid Connection",
                bus1="District LV Bus",
                p_nom=grid_capacity,
                efficiency=grid_params.get('transformer_efficiency', 0.98))
    
    # Add loads
    load_params = params.get('loads', {})
    
    # Stadium Loads
    stadium_elec_load = load_or_generate_profile('stadium_load_profile.csv', 
                                                load_params.get('stadium_elec_peak_mw', 0.5), 
                                                data_path, timestamps)
    stadium_heat_load = load_or_generate_profile('stadium_load_profile.csv', 
                                                load_params.get('stadium_heat_peak_mw', 0.8), 
                                                data_path, timestamps)
    network.add("Load", "Stadium Elec Load", bus="Stadium Elec", p_set=stadium_elec_load)
    network.add("Load", "Stadium Heat Load", bus="Stadium Heat", p_set=stadium_heat_load)
    network.add("Link", "Stadium Elec Link", bus0="District LV Bus", bus1="Stadium Elec", p_nom=10)
    network.add("Link", "Stadium Heat Link", bus0="District Heat Source", bus1="Stadium Heat", p_nom=10)
    print("Added Stadium Loads")
    
    # Swimming Pool Loads
    pool_elec_load = load_or_generate_profile('swimming_pool_load_profile.csv', 
                                             load_params.get('pool_elec_peak_mw', 0.3), 
                                             data_path, timestamps)
    pool_heat_load = load_or_generate_profile('swimming_pool_load_profile.csv', 
                                             load_params.get('pool_heat_peak_mw', 1.0), 
                                             data_path, timestamps)
    network.add("Load", "Pool Elec Load", bus="Pool Elec", p_set=pool_elec_load)
    network.add("Load", "Pool Heat Load", bus="Pool Heat", p_set=pool_heat_load)
    network.add("Link", "Pool Elec Link", bus0="District LV Bus", bus1="Pool Elec", p_nom=10)
    network.add("Link", "Pool Heat Link", bus0="District Heat Source", bus1="Pool Heat", p_nom=10)
    print("Added Swimming Pool Loads")
    
    # General District Loads
    general_elec_load = load_or_generate_profile('electricity_demand_general.csv', 
                                                load_params.get('general_elec_peak_mw', 1.5), 
                                                data_path, timestamps)
    general_heat_load = load_or_generate_profile('heat_demand_general.csv', 
                                                load_params.get('general_heat_peak_mw', 2.0), 
                                                data_path, timestamps)
    network.add("Load", "General Elec Load", bus="General Elec", p_set=general_elec_load)
    network.add("Load", "General Heat Load", bus="General Heat", p_set=general_heat_load)
    network.add("Link", "General Elec Link", bus0="District LV Bus", bus1="General Elec", p_nom=10)
    network.add("Link", "General Heat Link", bus0="District Heat Source", bus1="General Heat", p_nom=10)
    print("Added General District Loads")
    
    # Add heat source
    heat_source_params = params.get('baseline_heat_source', {})
    heat_source_capacity_mw = heat_source_params.get('capacity_mw_th', 5)
    heat_source_cost = heat_source_params.get('cost_eur_per_mwh_th', 40)
    
    if heat_source_capacity_mw > 0:
        network.add("Generator", "Heat Source",
                    bus="District Heat Source",
                    carrier="heat",
                    p_nom=heat_source_capacity_mw,
                    marginal_cost=heat_source_cost)
        print(f"Added Heat Source: Capacity={heat_source_capacity_mw} MWth, Cost={heat_source_cost} EUR/MWh")
    else:
         print("No Heat Source capacity specified.")
    
    return network
