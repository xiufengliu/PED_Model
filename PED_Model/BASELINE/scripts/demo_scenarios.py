#!/usr/bin/env python3
"""
Simplified demonstration script for PED Social Building Model.
This script simulates the baseline scenario for the social building with 165 apartments
for a full year (8760 hours) without requiring PyPSA.
"""

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_configs():
    """Load configuration files"""
    # Use os.path.join to create platform-independent paths
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

    with open(os.path.join(config_dir, "config.yml"), 'r') as f:
        config = yaml.safe_load(f)

    with open(os.path.join(config_dir, "component_params.yml"), 'r') as f:
        params = yaml.safe_load(f)

    return config, params

def load_profiles(data_path, n_hours=8760, start_date="2025-01-01"):
    """Load or generate profiles for the social building"""
    timestamps = pd.date_range(start=start_date, periods=n_hours, freq='h')

    # Create a dictionary to store all profiles
    profiles = {}

    # Load electricity demand profile
    elec_demand_file = os.path.join(data_path, 'timeseries', 'electricity_demand.csv')
    try:
        elec_demand_df = pd.read_csv(elec_demand_file, index_col=0, parse_dates=True)
        elec_demand = elec_demand_df.iloc[:, 0].reindex(timestamps).fillna(0)
        print(f"Loaded electricity demand from: {elec_demand_file}")
    except Exception as e:
        print(f"Error loading electricity demand: {e}. Generating synthetic profile.")
        # Generate synthetic profile with seasonal and daily patterns
        day_hours = np.arange(n_hours) % 24
        day_of_year = np.arange(n_hours) // 24 % 365

        # Base load with seasonal variation
        seasonal_factor = 0.2 * np.sin(2 * np.pi * day_of_year / 365) + 1  # Higher in winter

        # Daily pattern
        daily_pattern = np.ones(n_hours)
        for i, hour in enumerate(day_hours):
            if 7 <= hour <= 9:  # Morning peak
                daily_pattern[i] = 1.5
            elif 17 <= hour <= 21:  # Evening peak
                daily_pattern[i] = 1.8
            elif 0 <= hour <= 5:  # Night valley
                daily_pattern[i] = 0.6

        # Combine patterns
        elec_demand = 100 * seasonal_factor * daily_pattern  # Base of 100 kW
        elec_demand = pd.Series(elec_demand, index=timestamps)

    profiles['elec_demand'] = elec_demand

    # Load heat demand profile
    heat_demand_file = os.path.join(data_path, 'timeseries', 'heat_demand.csv')
    try:
        heat_demand_df = pd.read_csv(heat_demand_file, index_col=0, parse_dates=True)
        heat_demand = heat_demand_df.iloc[:, 0].reindex(timestamps).fillna(0)
        print(f"Loaded heat demand from: {heat_demand_file}")
    except Exception as e:
        print(f"Error loading heat demand: {e}. Generating synthetic profile.")
        # Generate synthetic profile with strong seasonal pattern
        day_of_year = np.arange(n_hours) // 24 % 365

        # Strong seasonal variation for heat
        seasonal_factor = 0.8 * np.cos(2 * np.pi * (day_of_year - 15) / 365) + 0.2  # Higher in winter
        seasonal_factor = np.maximum(seasonal_factor, 0.1)  # Ensure minimum heat demand

        # Daily pattern
        day_hours = np.arange(n_hours) % 24
        daily_pattern = np.ones(n_hours)
        for i, hour in enumerate(day_hours):
            if 6 <= hour <= 9:  # Morning peak
                daily_pattern[i] = 1.3
            elif 17 <= hour <= 22:  # Evening peak
                daily_pattern[i] = 1.4
            elif 0 <= hour <= 5:  # Night valley
                daily_pattern[i] = 0.8

        # Combine patterns
        heat_demand = 300 * seasonal_factor * daily_pattern  # Base of 300 kW
        heat_demand = pd.Series(heat_demand, index=timestamps)

    profiles['heat_demand'] = heat_demand

    # Load PV generation profile
    pv_profile_file = os.path.join(data_path, 'timeseries', 'solar_pv_generation.csv')
    try:
        pv_profile_df = pd.read_csv(pv_profile_file, index_col=0, parse_dates=True)
        pv_profile = pv_profile_df.iloc[:, 0].reindex(timestamps).fillna(0)
        print(f"Loaded PV generation profile from: {pv_profile_file}")
    except Exception as e:
        print(f"Error loading PV profile: {e}. Generating synthetic profile.")
        # Generate synthetic profile with seasonal and daily patterns
        day_hours = np.arange(n_hours) % 24
        day_of_year = np.arange(n_hours) // 24 % 365

        # Seasonal variation
        seasonal_factor = 0.7 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + 0.3  # Higher in summer
        seasonal_factor = np.maximum(seasonal_factor, 0)  # No negative generation

        # Daily pattern (only during daylight)
        pv_profile = np.zeros(n_hours)
        for i, hour in enumerate(day_hours):
            if 6 <= hour <= 18:  # Daylight hours
                pv_profile[i] = np.sin(np.pi * (hour - 6) / 12)

        # Combine patterns
        pv_profile = pv_profile * seasonal_factor

        # Add some randomness for clouds
        np.random.seed(42)  # For reproducibility
        cloud_factor = 0.7 + 0.3 * np.random.random(n_hours)
        pv_profile = pv_profile * cloud_factor

        pv_profile = pd.Series(pv_profile, index=timestamps)

    profiles['pv_profile'] = pv_profile

    # Load electricity price profile
    price_file = os.path.join(data_path, 'timeseries', 'grid_prices.csv')
    try:
        # Read the CSV file with the correct format
        price_df = pd.read_csv(price_file)

        # Check if the file has the expected columns
        if 'Datetime (Local)' in price_df.columns and 'Price (EUR/MWhe)' in price_df.columns:
            # Convert the datetime column to pandas datetime
            price_df['Datetime (Local)'] = pd.to_datetime(price_df['Datetime (Local)'])

            # Create a new dataframe with the correct structure for our simulation
            # This approach avoids the duplicate index issue
            hourly_prices = []

            # Create a list of all hours in our simulation
            sim_hours = pd.date_range(start=timestamps[0], periods=len(timestamps), freq='h')

            # For each hour in our simulation, find the closest price in the dataset
            for hour in sim_hours:
                # If we need to adjust the year (data is for 2026 but we need 2025)
                if price_df['Datetime (Local)'].dt.year.iloc[0] == 2026 and hour.year == 2025:
                    # Look for the same month, day, and hour but in 2026
                    search_hour = hour.replace(year=2026)
                else:
                    search_hour = hour

                # Find the closest timestamp in the price data
                time_diff = abs(price_df['Datetime (Local)'] - search_hour)
                closest_idx = time_diff.idxmin()

                # Get the price for that timestamp
                price = price_df.loc[closest_idx, 'Price (EUR/MWhe)']
                hourly_prices.append(price)

            # Create a Series with the prices
            price_profile = pd.Series(hourly_prices, index=timestamps)

            print(f"Loaded electricity price profile from: {price_file}")
            print(f"Average electricity price: {price_profile.mean():.2f} EUR/MWh")
        else:
            raise ValueError("CSV file does not have the expected columns")
    except Exception as e:
        print(f"Error loading price profile: {e}. Using default price.")
        # Default constant price
        price_profile = pd.Series(50.0, index=timestamps)

    profiles['price_profile'] = price_profile

    return profiles

def simulate_baseline(params, profiles, n_hours=8760):
    """Simulate the baseline scenario for the social building"""
    results = {}

    # Get parameters
    num_apartments = params['social_building'].get('num_apartments', 165)
    baseline_pv_capacity_kw = 62.0  # Hardcoded to achieve 7% self-sufficiency
    grid_export_price = params['grid'].get('export_price_eur_per_mwh', 20)
    heat_cost = params['baseline_heat_source'].get('cost_eur_per_mwh_th', 40)
    heat_efficiency = params['baseline_heat_source'].get('efficiency_if_boiler', 0.9)

    print(f"Simulating baseline scenario for social building with {num_apartments} apartments...")

    # Get profiles
    elec_demand = profiles['elec_demand']
    heat_demand = profiles['heat_demand']
    pv_profile = profiles['pv_profile']
    price_profile = profiles['price_profile']

    # Calculate PV generation
    # The PV profile represents generation from a system with nominal capacity of 33.6 kW
    # Scale it to match our desired capacity
    nominal_capacity_kw = 33.6  # Maximum value in the dataset
    scaling_factor = baseline_pv_capacity_kw / nominal_capacity_kw
    print(f"Baseline PV capacity: {baseline_pv_capacity_kw} kW")
    pv_generation = pv_profile * scaling_factor

    print(f"PV nominal capacity in dataset: {nominal_capacity_kw:.1f} kW")
    print(f"PV scaling factor: {scaling_factor:.4f}")
    print(f"Expected annual PV production: {pv_profile.sum() * scaling_factor / 1000:.2f} MWh/year")
    print(f"Actual annual PV production: {pv_generation.sum() / 1000:.2f} MWh/year")

    # Calculate grid interaction
    net_load = elec_demand - pv_generation
    grid_import = net_load.clip(lower=0)
    grid_export = (-net_load).clip(lower=0)

    # Calculate energy values (kWh)
    total_elec_demand_kwh = elec_demand.sum()
    total_heat_demand_kwh = heat_demand.sum()
    total_pv_generation_kwh = pv_generation.sum()
    total_grid_import_kwh = grid_import.sum()
    total_grid_export_kwh = grid_export.sum()

    # Calculate costs with variable electricity prices
    import_cost = (grid_import * price_profile / 1000).sum()  # EUR (using variable prices)
    export_revenue = (grid_export * grid_export_price / 1000).sum()  # EUR
    heat_cost_total = (heat_demand * heat_cost / heat_efficiency / 1000).sum()  # EUR
    total_cost = import_cost + heat_cost_total - export_revenue

    # Self-sufficiency and self-consumption
    self_sufficiency = (total_pv_generation_kwh / total_elec_demand_kwh) * 100 if total_elec_demand_kwh > 0 else 0
    self_consumption = ((total_pv_generation_kwh - total_grid_export_kwh) / total_pv_generation_kwh) * 100 if total_pv_generation_kwh > 0 else 0

    # Calculate CO2 emissions
    # Emission factors (kg CO2 per MWh)
    grid_emission_factor = 275  # Danish grid emission factor for 2025 (projected)
    heat_emission_factor = 220  # Natural gas boiler emission factor

    # Calculate emissions
    # Convert kWh to MWh first (divide by 1000), then apply emission factor (kg CO2/MWh)
    # Then convert kg CO2 to tonnes CO2 (divide by 1000)
    grid_import_emissions = (total_grid_import_kwh / 1000) * grid_emission_factor / 1000  # tonnes CO2
    heat_emissions = (total_heat_demand_kwh / 1000) * heat_emission_factor / heat_efficiency / 1000  # tonnes CO2
    total_emissions = grid_import_emissions + heat_emissions  # tonnes CO2

    # Calculate emissions per apartment
    emissions_per_apartment = total_emissions / num_apartments  # tonnes CO2 per apartment

    # Store results
    results['num_apartments'] = num_apartments
    results['total_elec_demand_kwh'] = total_elec_demand_kwh
    results['total_heat_demand_kwh'] = total_heat_demand_kwh
    results['total_pv_generation_kwh'] = total_pv_generation_kwh
    results['total_grid_import_kwh'] = total_grid_import_kwh
    results['total_grid_export_kwh'] = total_grid_export_kwh
    results['import_cost'] = import_cost
    results['export_revenue'] = export_revenue
    results['heat_cost'] = heat_cost_total
    results['total_cost'] = total_cost
    results['self_sufficiency'] = self_sufficiency
    results['self_consumption'] = self_consumption
    results['avg_elec_price'] = price_profile.mean()
    results['grid_emission_factor'] = grid_emission_factor
    results['heat_emission_factor'] = heat_emission_factor
    results['grid_import_emissions'] = grid_import_emissions
    results['heat_emissions'] = heat_emissions
    results['total_emissions'] = total_emissions
    results['emissions_per_apartment'] = emissions_per_apartment

    # Store time series for plotting
    results['time_series'] = {
        'timestamps': profiles['elec_demand'].index,
        'elec_demand': elec_demand,
        'heat_demand': heat_demand,
        'pv_generation': pv_generation,
        'grid_import': grid_import,
        'grid_export': grid_export,
        'price_profile': price_profile
    }

    return results

def create_visualizations(results, output_dir="data/output/demo_social_building"):
    """Create visualizations of the simulation results"""
    print("\nCreating visualizations...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create figures directory inside output_dir
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Visualization path: {figures_dir}")

    # Create self-consumption visualization
    create_self_consumption_viz(results, figures_dir, output_dir)

    # Create summary dataframe
    # Extract key metrics
    total_elec_demand = results['total_elec_demand_kwh'] / 1000  # MWh
    total_heat_demand = results['total_heat_demand_kwh'] / 1000  # MWh
    total_pv_generation = results['total_pv_generation_kwh'] / 1000  # MWh
    total_grid_import = results['total_grid_import_kwh'] / 1000  # MWh
    total_grid_export = results['total_grid_export_kwh'] / 1000  # MWh
    # Heat production is equal to heat demand in the baseline scenario
    total_heat_production = total_heat_demand  # MWh

    # Calculate costs
    # Get the average electricity price from the time series data
    avg_elec_price = results['time_series']['price_profile'].mean()
    total_elec_cost = results['import_cost']
    total_heat_cost = results['heat_cost']
    total_energy_cost = total_elec_cost + total_heat_cost

    # Calculate self-sufficiency metrics
    self_consumption = results['self_consumption']
    self_sufficiency = results['self_sufficiency']

    # Get CO2 emissions
    grid_import_emissions = results.get('grid_import_emissions', 0)
    heat_emissions = results.get('heat_emissions', 0)
    total_emissions = results.get('total_emissions', 0)
    emissions_per_apartment = results.get('emissions_per_apartment', 0)

    # Create summary dataframe
    summary = pd.DataFrame({
        'Metric': [
            'Total Electricity Demand', 'Total Heat Demand',
            'Total PV Generation', 'Total Grid Import', 'Total Grid Export',
            'Total Heat Production', 'Average Electricity Price',
            'Total Electricity Cost', 'Total Heat Cost', 'Total Energy Cost',
            'PV Self-Consumption', 'Electricity Self-Sufficiency',
            'Grid Import CO2 Emissions', 'Heat CO2 Emissions', 'Total CO2 Emissions',
            'CO2 Emissions per Apartment'
        ],
        'Value': [
            f"{total_elec_demand:.2f} MWh/year", f"{total_heat_demand:.2f} MWh/year",
            f"{total_pv_generation:.2f} MWh/year", f"{total_grid_import:.2f} MWh/year",
            f"{total_grid_export:.2f} MWh/year", f"{total_heat_production:.2f} MWh/year",
            f"{avg_elec_price:.2f} EUR/MWh",
            f"{total_elec_cost:.2f} EUR/year", f"{total_heat_cost:.2f} EUR/year",
            f"{total_energy_cost:.2f} EUR/year",
            f"{self_consumption:.2f}%", f"{self_sufficiency:.2f}%",
            f"{grid_import_emissions:.2f} tonnes CO2/year", f"{heat_emissions:.2f} tonnes CO2/year",
            f"{total_emissions:.2f} tonnes CO2/year", f"{emissions_per_apartment:.2f} tonnes CO2/apartment/year"
        ]
    })

    # Save summary to CSV
    summary_file = os.path.join(output_dir, 'baseline_summary.csv')
    summary.to_csv(summary_file, index=False)
    print(f"Summary results saved to {summary_file}")

    return summary

def create_self_consumption_viz(results, figures_dir, output_dir=None):
    """Create visualization for self-consumption and self-sufficiency"""
    # Extract time series data
    ts = results['time_series']
    timestamps = ts['timestamps']

    # Calculate hourly self-consumption and self-sufficiency
    pv_gen = ts['pv_generation']
    elec_demand = ts['elec_demand']
    grid_export = ts['grid_export']

    # Calculate hourly self-consumed PV
    self_consumed_pv = pv_gen - grid_export

    # Calculate hourly self-consumption rate (as percentage)
    hourly_self_consumption = (self_consumed_pv / pv_gen) * 100
    hourly_self_consumption = hourly_self_consumption.fillna(0)  # Handle division by zero

    # Calculate hourly self-sufficiency rate (as percentage)
    hourly_self_sufficiency = (self_consumed_pv / elec_demand) * 100
    hourly_self_sufficiency = hourly_self_sufficiency.fillna(0)  # Handle division by zero

    # 1. Monthly average self-consumption and self-sufficiency
    monthly_self_consumption = hourly_self_consumption.resample('ME').mean()
    monthly_self_sufficiency = hourly_self_sufficiency.resample('ME').mean()

    plt.figure(figsize=(12, 6))

    # Convert index to month names for better visualization
    months = [d.strftime('%b') for d in monthly_self_consumption.index]

    plt.bar(months, monthly_self_consumption, color='#2ca02c', alpha=0.7, label='Self-Consumption (%)')
    plt.bar(months, monthly_self_sufficiency, color='#1f77b4', alpha=0.7, label='Self-Sufficiency (%)')

    plt.xlabel('Month')
    plt.ylabel('Percentage (%)')
    plt.title('Monthly Average Self-Consumption and Self-Sufficiency')
    plt.legend()
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'monthly_self_consumption.png'))
    print(f"Monthly self-consumption visualization saved to {os.path.join(figures_dir, 'monthly_self_consumption.png')}")

    # 2. Pie chart showing energy balance
    plt.figure(figsize=(10, 10))

    # Calculate total values
    total_pv = results['total_pv_generation_kwh']
    total_demand = results['total_elec_demand_kwh']
    total_grid_import = results['total_grid_import_kwh']
    total_grid_export = results['total_grid_export_kwh']
    total_self_consumed = total_pv - total_grid_export

    # Create pie chart for energy sources
    labels = ['Self-Consumed PV', 'Grid Import']
    sizes = [total_self_consumed, total_grid_import]
    colors = ['#2ca02c', '#ff7f0e']
    explode = (0.1, 0)  # explode the 1st slice (Self-Consumed PV)

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Electricity Supply Sources')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'energy_sources_pie.png'))
    print(f"Energy sources pie chart saved to {os.path.join(figures_dir, 'energy_sources_pie.png')}")

    # 3. Monthly energy demand and generation
    monthly_elec = ts['elec_demand'].resample('ME').sum() / 1000  # Convert to MWh
    monthly_heat = ts['heat_demand'].resample('ME').sum() / 1000  # Convert to MWh
    monthly_pv = ts['pv_generation'].resample('ME').sum() / 1000  # Convert to MWh
    monthly_import = ts['grid_import'].resample('ME').sum() / 1000  # Convert to MWh

    plt.figure(figsize=(12, 6))

    # Convert index to month names for better visualization
    months = [d.strftime('%b') for d in monthly_elec.index]

    # Plot monthly electricity demand, PV generation, and grid import
    plt.bar(months, monthly_elec, color='#1f77b4', alpha=0.7, label='Electricity Demand')
    plt.bar(months, monthly_pv, color='#2ca02c', alpha=0.7, label='PV Generation')
    plt.bar(months, monthly_import, color='#ff7f0e', alpha=0.7, label='Grid Import')

    plt.xlabel('Month')
    plt.ylabel('Energy (MWh/month)')
    plt.title('Monthly Electricity Demand, PV Generation, and Grid Import')
    plt.legend()
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'monthly_electricity.png'))
    print(f"Monthly electricity visualization saved to {os.path.join(figures_dir, 'monthly_electricity.png')}")

    # 4. Monthly heat demand
    plt.figure(figsize=(12, 6))

    plt.bar(months, monthly_heat, color='#d62728', alpha=0.7, label='Heat Demand')

    plt.xlabel('Month')
    plt.ylabel('Energy (MWh/month)')
    plt.title('Monthly Heat Demand')
    plt.legend()
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'monthly_heat.png'))
    print(f"Monthly heat visualization saved to {os.path.join(figures_dir, 'monthly_heat.png')}")

    # 5. Monthly average electricity prices
    monthly_prices = ts['price_profile'].resample('ME').mean()

    plt.figure(figsize=(12, 6))

    plt.bar(months, monthly_prices, color='#9467bd', alpha=0.7)

    plt.xlabel('Month')
    plt.ylabel('Price (EUR/MWh)')
    plt.title('Monthly Average Electricity Prices')
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'monthly_electricity_prices.png'))
    print(f"Monthly electricity prices visualization saved to {os.path.join(figures_dir, 'monthly_electricity_prices.png')}")

    # 6. CO2 emissions visualization
    if 'grid_import_emissions' in results and 'heat_emissions' in results:
        # Create pie chart for CO2 emissions
        plt.figure(figsize=(10, 10))

        # Get emissions data
        grid_import_emissions = results['grid_import_emissions']
        heat_emissions = results['heat_emissions']

        # Create pie chart
        labels = ['Electricity (Grid Import)', 'Heat Production']
        sizes = [grid_import_emissions, heat_emissions]
        colors = ['#ff7f0e', '#d62728']
        explode = (0, 0.1)  # explode the 2nd slice (Heat Production)

        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('CO2 Emissions Sources')

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'co2_emissions_pie.png'))
        print(f"CO2 emissions pie chart saved to {os.path.join(figures_dir, 'co2_emissions_pie.png')}")

        # Create monthly CO2 emissions bar chart
        # Calculate monthly emissions
        monthly_grid_import = ts['grid_import'].resample('ME').sum() / 1000  # Convert to MWh
        monthly_heat_demand = ts['heat_demand'].resample('ME').sum() / 1000  # Convert to MWh

        # Apply emission factors
        # monthly_grid_import and monthly_heat_demand are already in MWh
        # Apply emission factor (kg CO2/MWh) and convert to tonnes (divide by 1000)
        monthly_grid_emissions = monthly_grid_import * results['grid_emission_factor'] / 1000  # tonnes CO2
        # Use a default heat efficiency of 0.9 if not available
        heat_efficiency = 0.9  # Default value for natural gas boiler
        monthly_heat_emissions = monthly_heat_demand * results['heat_emission_factor'] / heat_efficiency / 1000  # tonnes CO2

        plt.figure(figsize=(12, 6))

        # Create stacked bar chart
        plt.bar(months, monthly_grid_emissions, color='#ff7f0e', alpha=0.7, label='Electricity (Grid Import)')
        plt.bar(months, monthly_heat_emissions, color='#d62728', alpha=0.7, bottom=monthly_grid_emissions, label='Heat Production')

        plt.xlabel('Month')
        plt.ylabel('CO2 Emissions (tonnes)')
        plt.title('Monthly CO2 Emissions')
        plt.legend()
        plt.grid(True, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'monthly_co2_emissions.png'))
        print(f"Monthly CO2 emissions visualization saved to {os.path.join(figures_dir, 'monthly_co2_emissions.png')}")

    return None

def plot_comparison(baseline_results, high_pv_results, output_dir="data/output/demo"):
    """Create comparison plots for the two scenarios"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Energy balance comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = ['Baseline', 'High PV']
    load = [baseline_results['total_load_kwh'], high_pv_results['total_load_kwh']]
    pv_gen = [baseline_results['pv_generation_kwh'], high_pv_results['pv_generation_kwh']]
    grid_import = [baseline_results['grid_import_kwh'], high_pv_results['grid_import_kwh']]
    grid_export = [baseline_results['grid_export_kwh'], high_pv_results['grid_export_kwh']]

    x = np.arange(len(scenarios))
    width = 0.35

    ax.bar(x, load, width, label='Total Load')
    ax.bar(x, pv_gen, width, label='PV Generation')
    ax.bar(x, grid_import, width, label='Grid Import')
    ax.bar(x, grid_export, width, label='Grid Export', bottom=0)

    ax.set_ylabel('Energy (kWh)')
    ax.set_title('Energy Balance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_balance_comparison.png'))

    # 2. Time series comparison for a sample day
    # Extract 24 hours starting from 6 AM on the first day
    start_idx = 6  # 6 AM on first day
    end_idx = start_idx + 24

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Baseline scenario
    ts_baseline = baseline_results['time_series']
    timestamps = ts_baseline['timestamps'][start_idx:end_idx]
    axs[0].plot(timestamps, ts_baseline['total_load'][start_idx:end_idx], 'b-', label='Total Load')
    axs[0].plot(timestamps, ts_baseline['pv_generation'][start_idx:end_idx], 'y-', label='PV Generation')
    axs[0].plot(timestamps, ts_baseline['grid_import'][start_idx:end_idx], 'r-', label='Grid Import')
    axs[0].plot(timestamps, ts_baseline['grid_export'][start_idx:end_idx], 'g-', label='Grid Export')
    axs[0].set_ylabel('Power (kW)')
    axs[0].set_title('Baseline Scenario - 24 Hour Profile')
    axs[0].legend()
    axs[0].grid(True)

    # High PV scenario
    ts_high_pv = high_pv_results['time_series']
    axs[1].plot(timestamps, ts_high_pv['total_load'][start_idx:end_idx], 'b-', label='Total Load')
    axs[1].plot(timestamps, ts_high_pv['pv_generation'][start_idx:end_idx], 'y-', label='PV Generation')
    axs[1].plot(timestamps, ts_high_pv['grid_import'][start_idx:end_idx], 'r-', label='Grid Import')
    axs[1].plot(timestamps, ts_high_pv['grid_export'][start_idx:end_idx], 'g-', label='Grid Export')
    axs[1].set_ylabel('Power (kW)')
    axs[1].set_xlabel('Time')
    axs[1].set_title('High PV Scenario - 24 Hour Profile')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_series_comparison.png'))

    # 3. Self-sufficiency and self-consumption comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Self-Sufficiency (%)', 'Self-Consumption (%)']
    baseline_values = [baseline_results['self_sufficiency'], baseline_results['self_consumption']]
    high_pv_values = [high_pv_results['self_sufficiency'], high_pv_results['self_consumption']]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, baseline_values, width, label='Baseline')
    ax.bar(x + width/2, high_pv_values, width, label='High PV')

    ax.set_ylabel('Percentage (%)')
    ax.set_title('Self-Sufficiency and Self-Consumption Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'self_sufficiency_comparison.png'))

    # 4. Cost comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    cost_metrics = ['Import Cost (€)', 'Export Revenue (€)', 'Net Cost (€)']
    baseline_costs = [baseline_results['import_cost'], baseline_results['export_revenue'], baseline_results['net_cost']]
    high_pv_costs = [high_pv_results['import_cost'], high_pv_results['export_revenue'], high_pv_results['net_cost']]

    x = np.arange(len(cost_metrics))
    width = 0.35

    ax.bar(x - width/2, baseline_costs, width, label='Baseline')
    ax.bar(x + width/2, high_pv_costs, width, label='High PV')

    ax.set_ylabel('Cost (€)')
    ax.set_title('Cost Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(cost_metrics)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_comparison.png'))

    print(f"Plots saved to {output_dir}")

def main():
    """Main function to run the demonstration"""
    print("PED Social Building Model - Baseline Demonstration")
    print("=================================================")

    # Load configurations
    config, params = load_configs()
    print("Configurations loaded successfully")

    # Set simulation parameters
    n_hours = 8760  # Full year
    start_date = "2025-01-01"

    # Use absolute paths based on the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_path = os.path.join(base_dir, 'data', 'input')
    output_dir = os.path.join(base_dir, 'data', 'output', 'demo_social_building')

    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")

    # Load profiles
    profiles = load_profiles(data_path, n_hours, start_date)
    print(f"Loaded profiles for {n_hours} hours starting from {start_date}")

    # Simulate baseline scenario
    results = simulate_baseline(params, profiles, n_hours)

    # Create visualizations and save summary
    summary = create_visualizations(results, output_dir)

    # Print summary results
    print("\nBaseline Scenario Summary:")
    print("=========================")
    for i, row in summary.iterrows():
        print(f"{row['Metric']}: {row['Value']:.2f}" if isinstance(row['Value'], float) else f"{row['Metric']}: {row['Value']}")

    print("\nDemonstration completed successfully!")

if __name__ == "__main__":
    main()
