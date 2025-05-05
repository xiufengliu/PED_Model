#!/usr/bin/env python3
"""
Simplified demonstration script for PED Lyngby Model scenarios.
This script simulates the key differences between scenarios without requiring PyPSA.
"""

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_configs():
    """Load configuration files"""
    with open("config/config.yml", 'r') as f:
        config = yaml.safe_load(f)
    
    with open("config/component_params.yml", 'r') as f:
        params = yaml.safe_load(f)
    
    return config, params

def generate_profiles(n_hours=72, start_date="2025-01-01"):
    """Generate simplified load and generation profiles"""
    timestamps = pd.date_range(start=start_date, periods=n_hours, freq='h')
    
    # Create a dictionary to store all profiles
    profiles = {}
    
    # Generate PV profile (normalized to 0-1)
    # Simple sinusoidal pattern with zeros at night
    day_hours = np.arange(n_hours) % 24
    pv_profile = np.zeros(n_hours)
    for i, hour in enumerate(day_hours):
        if 6 <= hour <= 18:  # Daylight hours
            pv_profile[i] = np.sin(np.pi * (hour - 6) / 12)
    
    # Add some randomness for clouds
    np.random.seed(42)  # For reproducibility
    cloud_factor = 0.8 + 0.2 * np.random.random(n_hours)
    pv_profile = pv_profile * cloud_factor
    
    profiles['pv_profile'] = pd.Series(pv_profile, index=timestamps)
    
    # Generate load profiles
    # Stadium - higher during events (assume evenings)
    stadium_elec = np.ones(n_hours) * 0.3  # Base load
    for i, hour in enumerate(day_hours):
        if 18 <= hour <= 22:  # Evening events
            day_of_week = (timestamps[i].weekday() + 1) % 7  # 0=Monday, 6=Sunday
            if day_of_week >= 5:  # Weekend
                stadium_elec[i] = 0.8 + 0.2 * np.random.random()  # Higher load during weekend events
    
    profiles['stadium_elec'] = pd.Series(stadium_elec, index=timestamps)
    
    # Swimming pool - relatively constant with morning and evening peaks
    pool_elec = np.ones(n_hours) * 0.4  # Base load
    for i, hour in enumerate(day_hours):
        if 6 <= hour <= 9 or 17 <= hour <= 20:  # Morning and evening peaks
            pool_elec[i] = 0.7 + 0.3 * np.random.random()
    
    profiles['pool_elec'] = pd.Series(pool_elec, index=timestamps)
    
    # General district - typical residential/commercial pattern
    general_elec = np.ones(n_hours) * 0.3  # Base load
    for i, hour in enumerate(day_hours):
        if 7 <= hour <= 9:  # Morning peak
            general_elec[i] = 0.6 + 0.2 * np.random.random()
        elif 17 <= hour <= 21:  # Evening peak
            general_elec[i] = 0.8 + 0.2 * np.random.random()
        elif 0 <= hour <= 5:  # Night valley
            general_elec[i] = 0.2 + 0.1 * np.random.random()
    
    profiles['general_elec'] = pd.Series(general_elec, index=timestamps)
    
    return profiles

def simulate_baseline(params, profiles, n_hours=72):
    """Simulate the baseline scenario"""
    results = {}
    
    # Get parameters
    baseline_pv_capacity_kw = params['baseline_pv']['capacity_kw']
    grid_import_cost = params['grid']['import_cost_eur_per_mwh']
    grid_export_price = params['grid']['export_price_eur_per_mwh']
    
    # Scale profiles by capacity
    stadium_load = profiles['stadium_elec'] * params['loads']['stadium_elec_peak_mw'] * 1000  # Convert to kW
    pool_load = profiles['pool_elec'] * params['loads']['pool_elec_peak_mw'] * 1000  # Convert to kW
    general_load = profiles['general_elec'] * params['loads']['general_elec_peak_mw'] * 1000  # Convert to kW
    
    # Total load
    total_load = stadium_load + pool_load + general_load
    
    # PV generation
    pv_generation = profiles['pv_profile'] * baseline_pv_capacity_kw
    
    # Grid interaction
    net_load = total_load - pv_generation
    grid_import = net_load.clip(lower=0)
    grid_export = (-net_load).clip(lower=0)
    
    # Calculate energy values (kWh)
    total_load_kwh = total_load.sum()
    pv_generation_kwh = pv_generation.sum()
    grid_import_kwh = grid_import.sum()
    grid_export_kwh = grid_export.sum()
    
    # Calculate costs
    import_cost = grid_import_kwh * grid_import_cost / 1000  # Convert to MWh for cost calculation
    export_revenue = grid_export_kwh * grid_export_price / 1000  # Convert to MWh for revenue calculation
    net_cost = import_cost - export_revenue
    
    # Self-sufficiency and self-consumption
    self_sufficiency = (pv_generation_kwh / total_load_kwh) * 100 if total_load_kwh > 0 else 0
    self_consumption = ((pv_generation_kwh - grid_export_kwh) / pv_generation_kwh) * 100 if pv_generation_kwh > 0 else 0
    
    # Store results
    results['total_load_kwh'] = total_load_kwh
    results['pv_generation_kwh'] = pv_generation_kwh
    results['grid_import_kwh'] = grid_import_kwh
    results['grid_export_kwh'] = grid_export_kwh
    results['import_cost'] = import_cost
    results['export_revenue'] = export_revenue
    results['net_cost'] = net_cost
    results['self_sufficiency'] = self_sufficiency
    results['self_consumption'] = self_consumption
    
    # Store time series for plotting
    results['time_series'] = {
        'timestamps': profiles['pv_profile'].index,
        'total_load': total_load,
        'pv_generation': pv_generation,
        'net_load': net_load,
        'grid_import': grid_import,
        'grid_export': grid_export
    }
    
    return results

def simulate_high_pv(params, profiles, n_hours=72):
    """Simulate the high PV scenario"""
    results = {}
    
    # Get parameters
    stadium_pv_capacity_kw = params['high_pv']['stadium_pv_capacity_kw']
    pool_pv_capacity_kw = params['high_pv']['pool_pv_capacity_kw']
    general_pv_capacity_kw = params['high_pv']['general_pv_capacity_kw']
    total_pv_capacity_kw = stadium_pv_capacity_kw + pool_pv_capacity_kw + general_pv_capacity_kw
    
    grid_import_cost = params['grid']['import_cost_eur_per_mwh']
    grid_export_price = params['grid']['export_price_eur_per_mwh']
    
    # Scale profiles by capacity
    stadium_load = profiles['stadium_elec'] * params['loads']['stadium_elec_peak_mw'] * 1000  # Convert to kW
    pool_load = profiles['pool_elec'] * params['loads']['pool_elec_peak_mw'] * 1000  # Convert to kW
    general_load = profiles['general_elec'] * params['loads']['general_elec_peak_mw'] * 1000  # Convert to kW
    
    # Total load
    total_load = stadium_load + pool_load + general_load
    
    # PV generation for each location
    stadium_pv_generation = profiles['pv_profile'] * stadium_pv_capacity_kw
    pool_pv_generation = profiles['pv_profile'] * pool_pv_capacity_kw
    general_pv_generation = profiles['pv_profile'] * general_pv_capacity_kw
    
    # Total PV generation
    pv_generation = stadium_pv_generation + pool_pv_generation + general_pv_generation
    
    # Grid interaction
    net_load = total_load - pv_generation
    grid_import = net_load.clip(lower=0)
    grid_export = (-net_load).clip(lower=0)
    
    # Calculate energy values (kWh)
    total_load_kwh = total_load.sum()
    pv_generation_kwh = pv_generation.sum()
    grid_import_kwh = grid_import.sum()
    grid_export_kwh = grid_export.sum()
    
    # Calculate costs
    import_cost = grid_import_kwh * grid_import_cost / 1000  # Convert to MWh for cost calculation
    export_revenue = grid_export_kwh * grid_export_price / 1000  # Convert to MWh for revenue calculation
    net_cost = import_cost - export_revenue
    
    # Self-sufficiency and self-consumption
    self_sufficiency = (pv_generation_kwh / total_load_kwh) * 100 if total_load_kwh > 0 else 0
    self_consumption = ((pv_generation_kwh - grid_export_kwh) / pv_generation_kwh) * 100 if pv_generation_kwh > 0 else 0
    
    # Store results
    results['total_load_kwh'] = total_load_kwh
    results['pv_generation_kwh'] = pv_generation_kwh
    results['grid_import_kwh'] = grid_import_kwh
    results['grid_export_kwh'] = grid_export_kwh
    results['import_cost'] = import_cost
    results['export_revenue'] = export_revenue
    results['net_cost'] = net_cost
    results['self_sufficiency'] = self_sufficiency
    results['self_consumption'] = self_consumption
    
    # Store time series for plotting
    results['time_series'] = {
        'timestamps': profiles['pv_profile'].index,
        'total_load': total_load,
        'pv_generation': pv_generation,
        'net_load': net_load,
        'grid_import': grid_import,
        'grid_export': grid_export
    }
    
    return results

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
    print("PED Lyngby Model - Scenario Demonstration")
    print("=========================================")
    
    # Load configurations
    config, params = load_configs()
    print("Configurations loaded successfully")
    
    # Generate profiles
    n_hours = config['simulation_settings']['num_hours']
    start_date = config['simulation_settings']['start_date']
    profiles = generate_profiles(n_hours, start_date)
    print(f"Generated profiles for {n_hours} hours starting from {start_date}")
    
    # Simulate baseline scenario
    print("\nSimulating baseline scenario...")
    baseline_results = simulate_baseline(params, profiles, n_hours)
    
    # Simulate high PV scenario
    print("\nSimulating high PV scenario...")
    high_pv_results = simulate_high_pv(params, profiles, n_hours)
    
    # Print summary results
    print("\nSummary Results:")
    print("================")
    print(f"{'Metric':<30} {'Baseline':<15} {'High PV':<15} {'Difference':<15} {'% Change':<15}")
    print("-" * 90)
    
    metrics = [
        ('Total Load (kWh)', 'total_load_kwh'),
        ('PV Generation (kWh)', 'pv_generation_kwh'),
        ('Grid Import (kWh)', 'grid_import_kwh'),
        ('Grid Export (kWh)', 'grid_export_kwh'),
        ('Import Cost (€)', 'import_cost'),
        ('Export Revenue (€)', 'export_revenue'),
        ('Net Cost (€)', 'net_cost'),
        ('Self-Sufficiency (%)', 'self_sufficiency'),
        ('Self-Consumption (%)', 'self_consumption')
    ]
    
    for label, key in metrics:
        baseline_value = baseline_results[key]
        high_pv_value = high_pv_results[key]
        difference = high_pv_value - baseline_value
        pct_change = (difference / baseline_value) * 100 if baseline_value != 0 else float('inf')
        
        print(f"{label:<30} {baseline_value:<15.2f} {high_pv_value:<15.2f} {difference:<15.2f} {pct_change:<15.2f}")
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    plot_comparison(baseline_results, high_pv_results)
    
    print("\nDemonstration completed successfully!")

if __name__ == "__main__":
    main()
