"""
PED Social Building Model - Utilities Module

This module contains utility functions used across different scenario implementations.
"""

import os
import yaml
import pandas as pd
import numpy as np
import importlib
from pathlib import Path

def load_config(config_file, params_file):
    """
    Load configuration and parameters from YAML files.

    Args:
        config_file (str): Path to the main config file
        params_file (str): Path to the component parameters file

    Returns:
        tuple: (config, params) dictionaries containing the configuration and parameters
    """
    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(params_file, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    return config, params

# scripts/utils.py



def get_scenario_function(scenario_name: str):

    scen_mod_path = Path(__file__).parent.parent / 'scenarios' / f"{scenario_name}.py"
    if not scen_mod_path.exists():
        raise ImportError(f"No scenario module found for '{scenario_name}'")
    # Import via the scenarios package
    module = importlib.import_module(f"scenarios.{scenario_name}")
    if not hasattr(module, 'create_network'):
        raise AttributeError(f"'create_network' not found in scenarios.{scenario_name}")
    return module.create_network


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

def load_electricity_price_profile(data_dir, index):
    """
    Load electricity price profile from CSV file.
    """
    filepath = os.path.join(data_dir, 'timeseries', 'grid_prices.csv')
    try:
        # 1) Leggi il CSV in price_df
        price_df = pd.read_csv(filepath, index_col=0, parse_dates=True)

        # 2) Estrai la colonna dei prezzi e riallinea sugli snapshot
        price_profile = price_df.iloc[:, 0].reindex(index).ffill()

        # 3) Debug: mostra la descrizione prima di convertire
        print("Prezzi originali (prima di *1000):\n", price_profile.describe())

        # 4) Se davvero servisse la conversione da EUR/kWh a EUR/MWh
        print("Converting electricity prices from EUR/kWh to EUR/MWh")
        price_profile = price_profile * 1000

        # 5) Debug: controlla la serie dopo la conversione
        print("Prezzi convertiti (€/MWh):\n", price_profile.describe())

        print(f"Loaded electricity price profile from: {filepath}")
        return price_profile

    except FileNotFoundError:
        print(f"Warning: Electricity price file not found '{filepath}'. Using default constant price.")
        return pd.Series(50.0, index=index)  # Default price: 50 EUR/MWh

    except Exception as e:
        print(f"Error loading electricity price profile from '{filepath}': {e}. Using default constant price.")
        return pd.Series(50.0, index=index)


def load_thermal_price_profile(data_dir, index):
    """
    Load thermal energy price profile from CSV file.

    Args:
        data_dir (str): Path to the data directory
        index (pd.DatetimeIndex): Index for the time series

    Returns:
        pd.Series: Thermal energy price profile indexed by snapshots
    """
    filepath = os.path.join(data_dir, 'timeseries', 'thermal_energy_prices_denmark.csv')
    try:
        # Load the price profile from the CSV file
        price_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        # Select the relevant column and time slice
        # Note: Thermal prices are daily, so we need to resample to match the hourly index
        daily_prices = price_df['Price_(EUR/kWh)']
        # Create a Series with the daily prices
        daily_price_series = pd.Series(daily_prices.values, index=price_df.index)
        # Resample to hourly frequency using forward fill (each hour in a day gets the same price)
        hourly_price_series = daily_price_series.resample('h').ffill()
        # Reindex to match the simulation timeframe
        price_profile = hourly_price_series.reindex(index).fillna(45.0) # Fill gaps with default price
        # SUBITO dopo la creazione di price_profile
        print("Converting thermal prices from EUR/kWh to EUR/MWh")
        price_profile *= 1000

        print(f"Loaded thermal energy price profile from: {filepath}")
        print(f"Average thermal energy price: {price_profile.mean():.2f} EUR/MWh")
        return price_profile
    except FileNotFoundError:
        print(f"Warning: Thermal energy price file not found '{filepath}'. Using default constant price.")
        return pd.Series(45.0, index=index)  # Default price: 45 EUR/MWh
    except Exception as e:
        print(f"Error loading thermal energy price profile from '{filepath}': {e}. Using default constant price.")
        return pd.Series(45.0, index=index)  # Default price: 45 EUR/MWh

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
    n_hours = sim_settings.get('num_hours', 8760)  # Default to full year
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
    network.add("Bus", "District Heat Source", carrier="heat") # e.g., connection to DH or local boiler
    print("Added electrical and heat buses")

    # Set up grid connection
    grid_params = params.get('grid', {})
    grid_capacity = grid_params.get('capacity_mw', 10)

    # Load variable electricity price profile
    price_profile = load_electricity_price_profile(data_path, network.snapshots)

    # Calculate average price for display purposes
    avg_price = price_profile.mean()

    network.add("Generator", "Grid",
                bus="Grid Connection",
                carrier="electricity",
                marginal_cost=price_profile,  # Use variable price profile
                p_nom=grid_capacity,
                p_nom_extendable=True,  # Allow capacity to be extended if needed
                p_nom_max=20.0,  # Maximum capacity (very high to ensure feasibility)
                p_min_pu=-1, # Allow export up to p_nom
                p_max_pu=1)  # Allow import up to p_nom
    print(f"Added Grid connection: Capacity={grid_capacity} MW, Variable Cost (avg={avg_price:.2f} EUR/MWh)")

    # Link Grid Connection Point to District Bus (e.g., transformer)
    network.add("Link", "Substation",
                bus0="Grid Connection",
                bus1="District LV Bus",
                p_nom=grid_capacity,
                p_nom_extendable=True,  # Allow capacity to be extended if needed
                p_nom_max=20.0,  # Maximum capacity (very high to ensure feasibility)
                efficiency=grid_params.get('transformer_efficiency', 0.98))

    # Add heat source
    heat_source_params = params.get('baseline_heat_source', {})
    heat_source_capacity_mw = heat_source_params.get('capacity_mw_th', 1.0)
    heat_source_cost_type = heat_source_params.get('cost_eur_per_mwh_th', 40)
    heat_source_type = heat_source_params.get('type', 'gas_boiler')
    heat_source_efficiency = heat_source_params.get('efficiency_if_boiler', 0.9)

    # Check if we're using variable thermal energy prices
    if heat_source_cost_type == 'variable':
        # Load variable thermal energy price profile
        thermal_price_profile = load_thermal_price_profile(data_path, network.snapshots)
        # Calculate average price for display purposes
        avg_thermal_price = thermal_price_profile.mean()
        heat_source_cost = thermal_price_profile
        cost_display = f"Variable (avg={avg_thermal_price:.2f})"
    else:
        # Use fixed cost
        heat_source_cost = float(heat_source_cost_type)
        cost_display = f"{heat_source_cost}"

    if heat_source_capacity_mw > 0:
        if heat_source_type == 'gas_boiler':
            # For gas boiler, we add a Generator with higher capacity to ensure feasibility
            if isinstance(heat_source_cost, pd.Series):
                # For variable costs, divide each hourly price by efficiency
                marginal_cost = heat_source_cost / heat_source_efficiency
            else:
                # For fixed cost, divide the single value by efficiency
                marginal_cost = heat_source_cost / heat_source_efficiency

            network.add("Generator", "Heat Source",
                        bus="District Heat Source",
                        carrier="heat",
                        p_nom=heat_source_capacity_mw,
                        p_nom_extendable=True,  # Allow capacity to be extended if needed
                        p_nom_max=10.0,  # Maximum capacity (very high to ensure feasibility)
                        marginal_cost=marginal_cost)
            print(f"Added Gas Boiler: Capacity={heat_source_capacity_mw} MWth (extendable), Cost={cost_display} EUR/MWh, Efficiency={heat_source_efficiency}")
        else:
            # For district heating import, we just add a generator
            network.add("Generator", "Heat Source",
                        bus="District Heat Source",
                        carrier="heat",
                        p_nom=heat_source_capacity_mw,
                        p_nom_extendable=True,  # Allow capacity to be extended if needed
                        p_nom_max=10.0,  # Maximum capacity (very high to ensure feasibility)
                        marginal_cost=heat_source_cost)
            print(f"Added District Heating Import: Capacity={heat_source_capacity_mw} MWth (extendable), Cost={cost_display} EUR/MWh")
    else:
         print("No Heat Source capacity specified.")

    return network
