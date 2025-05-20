#!/usr/bin/env python3
"""
Generate synthetic data for the PED Lyngby Model.

This script creates synthetic time series data for:
- Load profiles (electricity and heat)
- PV generation profile
- Weather data
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def ensure_dir(directory):
    """Ensure that a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_load_profile(start_date, periods, peak_mw, profile_type, random_seed=42):
    """
    Generate a synthetic load profile.
    
    Args:
        start_date (str): Start date in format 'YYYY-MM-DD'
        periods (int): Number of hours to generate
        peak_mw (float): Peak load in MW
        profile_type (str): Type of profile ('stadium', 'pool', 'general')
        random_seed (int): Random seed for reproducibility
        
    Returns:
        pd.Series: Synthetic load profile
    """
    np.random.seed(random_seed)
    timestamps = pd.date_range(start=start_date, periods=periods, freq='h')
    
    # Base load as percentage of peak
    if profile_type == 'stadium':
        base_load_pct = 0.1  # Stadium has low base load
    elif profile_type == 'pool':
        base_load_pct = 0.3  # Pool has medium base load
    else:  # general
        base_load_pct = 0.2  # General district has medium-low base load
    
    # Initialize with base load
    load = np.ones(periods) * base_load_pct * peak_mw
    
    # Add daily pattern
    for i, timestamp in enumerate(timestamps):
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek  # 0=Monday, 6=Sunday
        is_weekend = day_of_week >= 5
        
        if profile_type == 'stadium':
            # Stadium has events in evenings and weekends
            if is_weekend:
                if 14 <= hour <= 22:  # Weekend events
                    load[i] += (0.7 + 0.3 * np.random.random()) * peak_mw
            else:
                if 18 <= hour <= 22:  # Weekday evening events
                    load[i] += (0.5 + 0.3 * np.random.random()) * peak_mw
        
        elif profile_type == 'pool':
            # Pool has morning and evening peaks
            if 6 <= hour <= 9:  # Morning peak
                load[i] += (0.4 + 0.2 * np.random.random()) * peak_mw
            elif 16 <= hour <= 21:  # Evening peak
                load[i] += (0.5 + 0.3 * np.random.random()) * peak_mw
            
            # Higher usage on weekends
            if is_weekend and 10 <= hour <= 18:
                load[i] += (0.3 + 0.2 * np.random.random()) * peak_mw
        
        else:  # general
            # General district has typical residential/commercial pattern
            if 7 <= hour <= 9:  # Morning peak
                load[i] += (0.4 + 0.2 * np.random.random()) * peak_mw
            elif 17 <= hour <= 21:  # Evening peak
                load[i] += (0.6 + 0.2 * np.random.random()) * peak_mw
            elif 0 <= hour <= 5:  # Night valley
                load[i] -= (0.1 * np.random.random()) * peak_mw  # Reduce load at night
    
    # Add some random noise
    load += np.random.normal(0, 0.05 * peak_mw, periods)
    
    # Ensure load is positive and doesn't exceed peak
    load = np.clip(load, 0.05 * peak_mw, peak_mw)
    
    # Add seasonal variation
    days = periods // 24
    seasonal_factor = 1 + 0.2 * np.sin(np.linspace(0, 2 * np.pi, days))
    seasonal_factor = np.repeat(seasonal_factor, 24)[:periods]
    
    # Apply seasonal factor (more pronounced for heat than electricity)
    if 'heat' in profile_type:
        load *= seasonal_factor
    else:
        load *= (1 + 0.2 * (seasonal_factor - 1))  # Less seasonal variation for electricity
    
    return pd.Series(load, index=timestamps)

def generate_pv_profile(start_date, periods, capacity_factor=0.15, random_seed=42):
    """
    Generate a synthetic PV generation profile.
    
    Args:
        start_date (str): Start date in format 'YYYY-MM-DD'
        periods (int): Number of hours to generate
        capacity_factor (float): Annual capacity factor
        random_seed (int): Random seed for reproducibility
        
    Returns:
        pd.Series: Synthetic PV generation profile (per unit of capacity)
    """
    np.random.seed(random_seed)
    timestamps = pd.date_range(start=start_date, periods=periods, freq='h')
    
    # Initialize with zeros
    pv_profile = np.zeros(periods)
    
    # Add daily pattern
    for i, timestamp in enumerate(timestamps):
        hour = timestamp.hour
        day_of_year = timestamp.dayofyear
        
        # Daylight hours (simplified)
        if 6 <= hour <= 18:
            # Solar intensity based on hour (peak at noon)
            hour_factor = np.sin(np.pi * (hour - 6) / 12)
            
            # Seasonal variation (peak in summer)
            seasonal_factor = 0.5 + 0.5 * np.sin(np.pi * (day_of_year - 172) / 182.5)
            
            # Combine factors
            pv_profile[i] = hour_factor * seasonal_factor
            
            # Add random variation for cloud cover
            cloud_factor = 0.7 + 0.3 * np.random.random()
            pv_profile[i] *= cloud_factor
    
    # Scale to achieve desired capacity factor
    avg_output = np.mean(pv_profile)
    scaling_factor = capacity_factor / avg_output
    pv_profile *= scaling_factor
    
    # Ensure values are between 0 and 1
    pv_profile = np.clip(pv_profile, 0, 1)
    
    return pd.Series(pv_profile, index=timestamps)

def generate_weather_data(start_date, periods, random_seed=42):
    """
    Generate synthetic weather data.
    
    Args:
        start_date (str): Start date in format 'YYYY-MM-DD'
        periods (int): Number of hours to generate
        random_seed (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Synthetic weather data
    """
    np.random.seed(random_seed)
    timestamps = pd.date_range(start=start_date, periods=periods, freq='h')
    
    # Initialize DataFrame
    weather_data = pd.DataFrame(index=timestamps)
    
    # Generate temperature data
    days = periods // 24
    # Seasonal temperature pattern (°C)
    seasonal_temp = 10 + 15 * np.sin(np.pi * (np.arange(days) - 172) / 182.5)
    seasonal_temp = np.repeat(seasonal_temp, 24)[:periods]
    
    # Daily temperature pattern
    daily_pattern = np.tile(np.sin(np.linspace(0, 2 * np.pi, 24)) * 5, days + 1)[:periods]
    
    # Combine patterns and add random noise
    temperature = seasonal_temp + daily_pattern + np.random.normal(0, 2, periods)
    weather_data['temperature'] = temperature
    
    # Generate wind speed data (m/s)
    wind_speed = 5 + 3 * np.sin(np.linspace(0, 20 * np.pi, periods)) + np.random.normal(0, 1.5, periods)
    wind_speed = np.clip(wind_speed, 0, 25)  # Clip to reasonable range
    weather_data['wind_speed'] = wind_speed
    
    # Generate solar irradiation data (W/m²)
    ghi = np.zeros(periods)
    for i, timestamp in enumerate(timestamps):
        hour = timestamp.hour
        day_of_year = timestamp.dayofyear
        
        # Daylight hours
        if 6 <= hour <= 18:
            # Solar intensity based on hour (peak at noon)
            hour_factor = np.sin(np.pi * (hour - 6) / 12)
            
            # Seasonal variation (peak in summer)
            seasonal_factor = 300 + 700 * np.sin(np.pi * (day_of_year - 172) / 182.5)
            
            # Combine factors
            ghi[i] = hour_factor * seasonal_factor
            
            # Add random variation for cloud cover
            cloud_factor = 0.3 + 0.7 * np.random.random()
            ghi[i] *= cloud_factor
    
    weather_data['ghi'] = ghi
    
    return weather_data

def main():
    """Generate all synthetic datasets."""
    print("Generating synthetic data for PED Lyngby Model...")
    
    # Parameters
    start_date = '2025-01-01'
    periods = 8760  # One year of hourly data
    
    # Create directories
    data_dir = 'data/input'
    timeseries_dir = os.path.join(data_dir, 'timeseries')
    ensure_dir(timeseries_dir)
    
    # Generate load profiles
    print("Generating load profiles...")
    
    # Electricity loads
    stadium_elec = generate_load_profile(start_date, periods, 0.5, 'stadium_elec', 42)
    stadium_elec.to_csv(os.path.join(timeseries_dir, 'stadium_elec_load.csv'))
    
    pool_elec = generate_load_profile(start_date, periods, 0.3, 'pool_elec', 43)
    pool_elec.to_csv(os.path.join(timeseries_dir, 'pool_elec_load.csv'))
    
    general_elec = generate_load_profile(start_date, periods, 1.5, 'general_elec', 44)
    general_elec.to_csv(os.path.join(timeseries_dir, 'general_elec_load.csv'))
    
    # Heat loads
    stadium_heat = generate_load_profile(start_date, periods, 0.8, 'stadium_heat', 45)
    stadium_heat.to_csv(os.path.join(timeseries_dir, 'stadium_heat_load.csv'))
    
    pool_heat = generate_load_profile(start_date, periods, 1.0, 'pool_heat', 46)
    pool_heat.to_csv(os.path.join(timeseries_dir, 'pool_heat_load.csv'))
    
    general_heat = generate_load_profile(start_date, periods, 2.0, 'general_heat', 47)
    general_heat.to_csv(os.path.join(timeseries_dir, 'general_heat_load.csv'))
    
    # Generate PV profile
    print("Generating PV profile...")
    pv_profile = generate_pv_profile(start_date, periods, 0.15, 48)
    pv_profile.to_csv(os.path.join(timeseries_dir, 'solar_pv_generation.csv'))
    
    # Generate weather data
    print("Generating weather data...")
    weather_data = generate_weather_data(start_date, periods, 49)
    weather_data.to_csv(os.path.join(timeseries_dir, 'weather_data.csv'))
    
    print("Data generation complete. Files saved to:", timeseries_dir)
    
    # Create plots for verification
    print("Creating verification plots...")
    plots_dir = os.path.join(data_dir, 'plots')
    ensure_dir(plots_dir)
    
    # Plot electricity loads
    plt.figure(figsize=(12, 6))
    plt.plot(stadium_elec.iloc[:168], label='Stadium')
    plt.plot(pool_elec.iloc[:168], label='Pool')
    plt.plot(general_elec.iloc[:168], label='General')
    plt.title('Electricity Load Profiles (First Week)')
    plt.xlabel('Time')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'electricity_loads.png'))
    
    # Plot heat loads
    plt.figure(figsize=(12, 6))
    plt.plot(stadium_heat.iloc[:168], label='Stadium')
    plt.plot(pool_heat.iloc[:168], label='Pool')
    plt.plot(general_heat.iloc[:168], label='General')
    plt.title('Heat Load Profiles (First Week)')
    plt.xlabel('Time')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'heat_loads.png'))
    
    # Plot PV profile
    plt.figure(figsize=(12, 6))
    plt.plot(pv_profile.iloc[:168])
    plt.title('PV Generation Profile (First Week)')
    plt.xlabel('Time')
    plt.ylabel('Generation (per unit)')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'pv_profile.png'))
    
    print("Verification plots saved to:", plots_dir)

if __name__ == "__main__":
    main()
