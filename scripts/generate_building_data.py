#!/usr/bin/env python3
"""
Generate synthetic data for the social housing building.

This script creates synthetic time series data for:
- Electricity demand for the social housing building
- Heat demand for the social housing building
- PV generation profile for the rooftop installation
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

def generate_building_electricity_demand(start_date, periods, num_apartments=165, avg_apartment_kw=0.5, random_seed=42):
    """
    Generate a synthetic electricity demand profile for a residential building.

    Args:
        start_date (str): Start date in format 'YYYY-MM-DD'
        periods (int): Number of hours to generate
        num_apartments (int): Number of apartments in the building
        avg_apartment_kw (float): Average electricity demand per apartment in kW
        random_seed (int): Random seed for reproducibility

    Returns:
        pd.Series: Synthetic electricity demand profile in MW
    """
    np.random.seed(random_seed)
    timestamps = pd.date_range(start=start_date, periods=periods, freq='h')

    # Base load as percentage of peak (always on)
    base_load_pct = 0.2

    # Initialize with base load
    total_peak_mw = num_apartments * avg_apartment_kw / 1000  # Convert to MW
    load = np.ones(periods) * base_load_pct * total_peak_mw

    # Add daily pattern
    for i, timestamp in enumerate(timestamps):
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek  # 0=Monday, 6=Sunday
        is_weekend = day_of_week >= 5

        # Morning peak (people getting ready for work/school)
        if 6 <= hour <= 9:
            if is_weekend:
                # Later and lower peak on weekends
                if 8 <= hour <= 10:
                    load[i] += (0.3 + 0.2 * np.random.random()) * total_peak_mw
            else:
                # Higher peak on weekdays
                load[i] += (0.5 + 0.3 * np.random.random()) * total_peak_mw

        # Evening peak (people returning home, cooking, etc.)
        elif 17 <= hour <= 22:
            # Higher and longer evening peak
            peak_factor = 0.7 + 0.3 * np.random.random()
            if 19 <= hour <= 21:  # Prime time
                peak_factor += 0.2
            load[i] += peak_factor * total_peak_mw

        # Night valley
        elif 0 <= hour <= 5:
            # Only base load at night
            pass

        # Midday
        else:
            if is_weekend:
                # Higher midday usage on weekends
                load[i] += (0.3 + 0.2 * np.random.random()) * total_peak_mw
            else:
                # Lower midday usage on weekdays (most people out)
                load[i] += (0.1 + 0.1 * np.random.random()) * total_peak_mw

    # Add some random noise
    load += np.random.normal(0, 0.05 * total_peak_mw, periods)

    # Ensure load is positive and doesn't exceed a reasonable maximum
    load = np.clip(load, 0.05 * total_peak_mw, 1.2 * total_peak_mw)

    # Add seasonal variation (less pronounced for electricity than for heat)
    days = periods // 24
    seasonal_factor = 1 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, days))
    seasonal_factor = np.repeat(seasonal_factor, 24)[:periods]
    load *= seasonal_factor

    return pd.Series(load, index=timestamps)

def generate_building_heat_demand(start_date, periods, num_apartments=165, avg_heating_kw=1.0, random_seed=43):
    """
    Generate a synthetic heat demand profile for a residential building.

    Args:
        start_date (str): Start date in format 'YYYY-MM-DD'
        periods (int): Number of hours to generate
        num_apartments (int): Number of apartments in the building
        avg_heating_kw (float): Average heat demand per apartment in kW
        random_seed (int): Random seed for reproducibility

    Returns:
        pd.Series: Synthetic heat demand profile in MW
    """
    np.random.seed(random_seed)
    timestamps = pd.date_range(start=start_date, periods=periods, freq='h')

    # Base load as percentage of peak (always on for hot water)
    base_load_pct = 0.1

    # Initialize with base load
    total_peak_mw = num_apartments * avg_heating_kw / 1000  # Convert to MW
    load = np.ones(periods) * base_load_pct * total_peak_mw

    # Add daily pattern
    for i, timestamp in enumerate(timestamps):
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek  # 0=Monday, 6=Sunday
        is_weekend = day_of_week >= 5

        # Morning peak (heating up after night setback)
        if 5 <= hour <= 9:
            if is_weekend:
                # Later and lower peak on weekends
                if 7 <= hour <= 10:
                    load[i] += (0.4 + 0.2 * np.random.random()) * total_peak_mw
            else:
                # Higher peak on weekdays
                load[i] += (0.6 + 0.2 * np.random.random()) * total_peak_mw

        # Evening peak (people returning home)
        elif 16 <= hour <= 22:
            # Higher evening peak
            load[i] += (0.5 + 0.3 * np.random.random()) * total_peak_mw

        # Night setback
        elif 23 <= hour or hour <= 4:
            # Reduced heating at night
            load[i] += (0.1 + 0.1 * np.random.random()) * total_peak_mw

        # Midday
        else:
            if is_weekend:
                # Higher midday usage on weekends
                load[i] += (0.4 + 0.2 * np.random.random()) * total_peak_mw
            else:
                # Lower midday usage on weekdays (most people out)
                load[i] += (0.2 + 0.1 * np.random.random()) * total_peak_mw

    # Add some random noise
    load += np.random.normal(0, 0.05 * total_peak_mw, periods)

    # Ensure load is positive and doesn't exceed a reasonable maximum
    load = np.clip(load, 0.05 * total_peak_mw, 1.2 * total_peak_mw)

    # Add strong seasonal variation for heating
    days = periods // 24
    # Seasonal factor with winter peak (assuming start_date is January 1)
    seasonal_factor = 0.2 + 0.8 * (1 + np.cos(np.linspace(0, 2 * np.pi, days))) / 2
    seasonal_factor = np.repeat(seasonal_factor, 24)[:periods]
    load *= seasonal_factor

    return pd.Series(load, index=timestamps)

def generate_pv_profile(start_date, periods, capacity_factor=0.15, random_seed=44):
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
    scaling_factor = capacity_factor / avg_output if avg_output > 0 else 0
    pv_profile *= scaling_factor

    # Ensure values are between 0 and 1
    pv_profile = np.clip(pv_profile, 0, 1)

    return pd.Series(pv_profile, index=timestamps)

def main():
    """Generate all synthetic datasets for the social housing building."""
    print("Generating synthetic data for social housing building...")

    # Parameters
    start_date = '2025-01-01'
    periods = 8760  # One year of hourly data
    num_apartments = 165
    num_floors = 13
    floor_area_m2 = 9500
    roof_area_m2 = 700

    # Create directories
    data_dir = 'data/input'
    timeseries_dir = os.path.join(data_dir, 'timeseries')
    ensure_dir(timeseries_dir)

    # Generate electricity demand
    print("Generating electricity demand profile...")
    # Calculate average apartment electricity demand
    # Target peak of 0.23 MW for the building
    avg_apartment_kw = 0.23 * 1000 / num_apartments  # ~1.4 kW per apartment

    electricity_demand = generate_building_electricity_demand(
        start_date, periods, num_apartments, avg_apartment_kw=avg_apartment_kw
    )
    electricity_demand.to_csv(os.path.join(timeseries_dir, 'electricity_demand.csv'))

    # Generate heat demand
    print("Generating heat demand profile...")
    # Calculate average apartment heat demand
    # Target peak of 0.32 MW for the building
    avg_heating_kw = 0.32 * 1000 / num_apartments  # ~1.94 kW per apartment

    heat_demand = generate_building_heat_demand(
        start_date, periods, num_apartments, avg_heating_kw=avg_heating_kw
    )
    heat_demand.to_csv(os.path.join(timeseries_dir, 'heat_demand.csv'))

    # Generate PV profile
    print("Generating PV generation profile...")
    # Danish solar conditions, capacity factor around 0.12-0.15
    pv_profile = generate_pv_profile(start_date, periods, capacity_factor=0.14)
    pv_profile.to_csv(os.path.join(timeseries_dir, 'solar_pv_generation.csv'))

    print("Data generation complete. Files saved to:", timeseries_dir)

    # Create plots for verification
    print("Creating verification plots...")
    plots_dir = os.path.join(data_dir, 'plots')
    ensure_dir(plots_dir)

    # Plot electricity demand
    plt.figure(figsize=(12, 6))
    plt.plot(electricity_demand.iloc[:168])
    plt.title('Electricity Demand Profile (First Week)')
    plt.xlabel('Time')
    plt.ylabel('Demand (MW)')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'electricity_demand.png'))

    # Plot heat demand
    plt.figure(figsize=(12, 6))
    plt.plot(heat_demand.iloc[:168])
    plt.title('Heat Demand Profile (First Week)')
    plt.xlabel('Time')
    plt.ylabel('Demand (MW)')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'heat_demand.png'))

    # Plot PV profile
    plt.figure(figsize=(12, 6))
    plt.plot(pv_profile.iloc[:168])
    plt.title('PV Generation Profile (First Week)')
    plt.xlabel('Time')
    plt.ylabel('Generation (per unit)')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'pv_profile.png'))

    # Plot monthly averages
    electricity_monthly = electricity_demand.resample('M').mean()
    heat_monthly = heat_demand.resample('M').mean()

    plt.figure(figsize=(12, 6))
    plt.plot(electricity_monthly.index.strftime('%b'), electricity_monthly.values, 'b-o', label='Electricity')
    plt.plot(heat_monthly.index.strftime('%b'), heat_monthly.values, 'r-o', label='Heat')
    plt.title('Monthly Average Demand')
    plt.xlabel('Month')
    plt.ylabel('Average Demand (MW)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'monthly_demand.png'))

    print("Verification plots saved to:", plots_dir)

if __name__ == "__main__":
    main()
