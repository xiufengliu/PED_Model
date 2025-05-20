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
        profile = profile_df.iloc[:, 0]
        if index is not None:
            profile = profile.reindex(index).fillna(0)  # Reindex and fill gaps
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


def calculate_pv_generation(weather_df, pv_capacity_kw, tilt=30, azimuth=180):
    """
    Calculate PV generation based on weather data.

    Args:
        weather_df (pd.DataFrame): Weather data with GHI, DNI, DHI, temp_air columns
        pv_capacity_kw (float): PV capacity in kW
        tilt (float): Panel tilt angle in degrees
        azimuth (float): Panel azimuth angle in degrees (180=south)

    Returns:
        pd.Series: PV generation in kW
    """
    # This is a simplified placeholder function
    # In a real implementation, you would use a library like pvlib
    # to calculate PV generation based on weather data

    if weather_df.empty:
        print("Warning: No weather data provided for PV generation calculation.")
        return pd.Series()

    # Simplified calculation based only on GHI
    if 'ghi' in weather_df.columns:
        # Very simple model: PV output is proportional to GHI
        # with efficiency factors for tilt, azimuth, temperature, etc.
        ghi = weather_df['ghi']

        # Simplified efficiency factors
        tilt_factor = 1.0  # Placeholder
        azimuth_factor = 1.0  # Placeholder
        temp_factor = 1.0  # Placeholder

        # Calculate PV generation
        pv_generation = ghi * pv_capacity_kw / 1000 * tilt_factor * azimuth_factor * temp_factor

        return pv_generation
    else:
        print("Warning: GHI data not found in weather data.")
        return pd.Series()


if __name__ == "__main__":
    # Example usage
    print("This module is not meant to be run directly.")
    print("Import it and use its functions in your scripts.")