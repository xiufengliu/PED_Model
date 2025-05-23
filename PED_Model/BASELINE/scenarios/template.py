"""
PED Lyngby Model - Template for New Scenarios

This is a template file for creating new scenarios. Copy this file and modify it
to implement a new scenario.

Steps to create a new scenario:
1. Copy this file to a new file named after your scenario (e.g., pv_battery.py)
2. Implement the create_network function with your scenario-specific logic
3. Add your scenario to the SCENARIO_FUNCTIONS dictionary in __init__.py
4. Add your scenario parameters to component_params.yml
5. Add your scenario configuration to config.yml

Important notes:
- All scenarios use variable electricity prices from grid_prices.csv by default
- This is configured in component_params.yml with import_cost_eur_per_mwh: variable
- If you need to use a fixed price instead, modify the grid parameters in your scenario
"""

import pypsa
import pandas as pd
import numpy as np
import os

from .utils import load_config, load_or_generate_profile, setup_basic_network

def create_network(config_file, params_file, data_path):
    """
    Builds the PyPSA network for the new scenario.

    Args:
        config_file (str): Path to the main config file (e.g., config.yml).
        params_file (str): Path to the component parameters file (e.g., component_params.yml).
        data_path (str): Path to the input data directory.

    Returns:
        pypsa.Network: The configured PyPSA network object.
    """
    print("Building new scenario network...")

    # Load configuration
    config, params = load_config(config_file, params_file)

    # Set up basic network with common elements
    # This includes buses, grid connection with variable electricity prices,
    # and baseline heat source
    #
    # Note on variable electricity prices:
    # - The grid connection uses variable electricity prices from grid_prices.csv
    # - This is configured in component_params.yml with import_cost_eur_per_mwh: variable
    # - The prices are loaded by load_electricity_price_profile() in utils.py
    # - They are used as the marginal cost for the grid generator
    network = setup_basic_network(config, params, data_path)

    # Get scenario-specific parameters
    # scenario_params = params.get('your_scenario_name', {})

    # Add scenario-specific components
    # ...

    print("New scenario network build complete.")
    return network
