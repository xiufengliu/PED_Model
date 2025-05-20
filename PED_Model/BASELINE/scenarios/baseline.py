"""
PED Social Building Model - Baseline Scenario

This module implements the baseline scenario for the PED Social Building Model.
The baseline scenario represents the current situation or a minimal setup with
an existing social building with 165 apartments and minimal local assets.
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
    print("Building baseline network for social building...")

    # Load configuration
    config, params = load_config(config_file, params_file)

    # Set up basic network with common elements
    network = setup_basic_network(config, params, data_path)

    # Add social building
    social_building_params = params.get('social_building', {})
    num_apartments = social_building_params.get('num_apartments', 165)

    # Load electricity demand profile
    elec_profile_name = social_building_params.get('electricity_load_profile', 'electricity_demand.csv')
    elec_demand = load_or_generate_profile(elec_profile_name, 0.2, data_path, network.snapshots)

    # Load heat demand profile
    heat_profile_name = social_building_params.get('heat_demand_profile', 'heat_demand.csv')
    heat_demand = load_or_generate_profile(heat_profile_name, 0.3, data_path, network.snapshots)

    # Add social building buses
    network.add("Bus", "Social Building Elec", carrier="electricity")
    network.add("Bus", "Social Building Heat", carrier="heat")

    # Add social building loads
    network.add("Load", "Social Building Elec Load", bus="Social Building Elec", p_set=elec_demand)
    network.add("Load", "Social Building Heat Load", bus="Social Building Heat", p_set=heat_demand)

    # Connect social building to district buses with extendable capacity
    network.add("Link", "Social Building Elec Link",
                bus0="District LV Bus",
                bus1="Social Building Elec",
                p_nom=10,
                p_nom_extendable=True,  # Allow capacity to be extended if needed
                p_nom_max=100)  # Maximum capacity

    network.add("Link", "Social Building Heat Link",
                bus0="District Heat Source",
                bus1="Social Building Heat",
                p_nom=10,
                p_nom_extendable=True,  # Allow capacity to be extended if needed
                p_nom_max=100)  # Maximum capacity

    print(f"Added Social Building with {num_apartments} apartments")

    # Add baseline PV
    pv_params = params.get('baseline_pv', {})
    pv_capacity_mw = pv_params.get('capacity_kw', 15) / 1000  # Convert kW to MW
    pv_profile_name = social_building_params.get('pv_generation_profile', 'Solar_PV_Generation.csv')
    pv_avail_series = load_or_generate_profile(pv_profile_name, 1.0, data_path, network.snapshots)

    if pv_capacity_mw > 0:
        network.add("Generator", "Existing PV",
                    bus="Social Building Elec",  # Connect directly to the social building
                    p_nom=pv_capacity_mw,
                    p_max_pu=pv_avail_series,  # Time-varying availability
                    marginal_cost=pv_params.get('marginal_cost', 0))
        print(f"Added Existing PV: {pv_params.get('capacity_kw', 15)} kWp")
    else:
        print("No baseline PV capacity specified.")

    # Add placeholder storage components (zero capacity for baseline)
    network.add("StorageUnit", "Placeholder Battery", bus="District LV Bus", p_nom=0, max_hours=0)
    network.add("Store", "Placeholder Thermal Storage", bus="District Heat Source", e_nom=0)
    print("Added placeholder storage components (zero capacity for baseline)")

    print("Baseline network build complete.")
    return network
