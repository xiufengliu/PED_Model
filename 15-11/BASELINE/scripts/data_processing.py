#!/usr/bin/env python3
"""
Data processing utilities for the PED Lyngby Model.

This module contains functions for loading, processing, and preparing data
for the PED Lyngby Model scenarios.
"""

import os
import pandas as pd
import numpy as np


def load_timeseries(filename, data_path, default_peak_mw=1.0, index=None):
    """
    Load a time series from a CSV file.

    Args:
        filename (str): Name of the file to load
        data_path (str): Path to the data directory
        default_peak_mw (float): Default peak value in MW for the generated profile if file not found
        index (pd.DatetimeIndex): Index for the time series if file not found

    Returns:
        pd.Series: The loaded time series
    """
    filepath = os.path.join(data_path, 'timeseries', filename)
    try:
        # Adjust CSV reading as needed (e.g., delimiter, header, index_col)
        profile_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        # Select the relevant column and time slice
        if 'solar_radiation' in profile_df.columns:
            profile = profile_df['solar_radiation']
        else:
            # Fallback: Se la colonna non è trovata (anche se hai confermato che c'è, è buona pratica avere un fallback)
            print(f"ATTENZIONE: Colonna 'solar_radiation' non trovata in {filepath}. Usando la prima colonna come fallback.")
            profile = profile_df.iloc[:, 0] # Fallback alla prima colonna
        print(f"Loaded profile from: {filepath}")
        return profile
    except FileNotFoundError:
        print(f"Warning: File not found '{filepath}'. Generating placeholder profile.")
        if index is None:
            raise ValueError("Index must be provided if file not found")
        np.random.seed(sum(ord(c) for c in filename))  # Seed based on filename
        base = np.sin(np.linspace(0, 2 * np.pi, 24)) * 0.4 + 0.6
        profile_array = np.tile(base, len(index) // 24 + 1)[:len(index)] * default_peak_mw
        return pd.Series(profile_array, index=index)
    except Exception as e:
        print(f"Error loading '{filepath}': {e}. Using placeholder.")
        if index is None:
            raise ValueError("Index must be provided if file not found")
        # Fallback to placeholder on other errors too
        np.random.seed(sum(ord(c) for c in filename))
        base = np.sin(np.linspace(0, 2 * np.pi, 24)) * 0.4 + 0.6
        profile_array = np.tile(base, len(index) // 24 + 1)[:len(index)] * default_peak_mw
        return pd.Series(profile_array, index=index)


def load_parameters(filename, data_path):
    """
    Load parameters from a CSV file.

    Args:
        filename (str): Name of the file to load
        data_path (str): Path to the data directory

    Returns:
        pd.DataFrame: The loaded parameters
    """
    filepath = os.path.join(data_path, 'parameters', filename)
    try:
        params_df = pd.read_csv(filepath, index_col=0)
        print(f"Loaded parameters from: {filepath}")
        return params_df
    except FileNotFoundError:
        print(f"Warning: Parameters file not found '{filepath}'.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading parameters '{filepath}': {e}.")
        return pd.DataFrame()


def prepare_weather_data(weather_file, data_path, start_date, end_date):
    """
    Prepare weather data for the simulation.

    Args:
        weather_file (str): Name of the weather file
        data_path (str): Path to the data directory
        start_date (str): Start date for the simulation
        end_date (str): End date for the simulation

    Returns:
        pd.DataFrame: Processed weather data
    """
    # Load weather data
    weather_path = os.path.join(data_path, 'timeseries', weather_file)
    try:
        weather_df = pd.read_csv(weather_path, index_col=0, parse_dates=True)

        # Filter to the desired date range
        weather_df = weather_df.loc[start_date:end_date]

        # Check if we have all the required data
        if weather_df.empty:
            print(f"Warning: No weather data found for the specified date range: {start_date} to {end_date}")
            return pd.DataFrame()

        # Process the data as needed (e.g., unit conversions, calculations)
        # ...

        return weather_df

    except FileNotFoundError:
        print(f"Warning: Weather file not found '{weather_path}'.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing weather data '{weather_path}': {e}.")
        return pd.DataFrame()


def calculate_pv_generation(weather_df, pv_surface_m2, pv_efficiency_pv, pv_inverter_efficiency):
    """
    Calcola la generazione PV in kW basandosi su GHI (irradianza), area superficiale ed efficienze.
    Assume che l'irradianza in weather_df sia nella colonna 'solar_radiation' e sia in W/m².
    """

    weather_df.columns = ['solar_radiation'] # Rinomina per coerenza interna

    # Verifica la presenza dei dati meteo e della colonna 'solar_radiation'
    if weather_df.empty or 'solar_radiation' not in weather_df.columns:
        print("Warning: Dati meteo o colonna 'solar_radiation' mancanti per il calcolo della generazione PV. Restituzione generazione PV zero.")
        # Restituisce una serie di zeri con l'indice corretto o una serie vuota
        return pd.Series(0, index=weather_df.index) if not weather_df.empty else pd.Series()

    irradiance_w_per_m2 = weather_df['solar_radiation'] # Estrae l'irradianza in W/m² dalla colonna corretta

    # Calcola la potenza AC istantanea in Watt
    # Formula: Irradianza (W/m²) * Superficie (m²) * Efficienza_PV (adimensionale) * Efficienza_Inverter (adimensionale)
    pv_generation_watts = irradiance_w_per_m2 * pv_surface_m2 * pv_efficiency_pv * pv_inverter_efficiency

    # Converti i Watt in kilowatt (kW) prima di restituire
    return pv_generation_watts / 1000 # Restituisce la potenza PV in kW
if __name__ == "__main__":
    # Example usage
    print("This module is not meant to be run directly.")
    print("Import it and use its functions in your scripts.")