#!/usr/bin/env python3
"""
Plotting utilities for the PED Lyngby Model.

This module contains functions for visualizing the results of the PED Lyngby Model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl




def setup_plotting_style():
    """Set up the plotting style for consistent visualizations."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except (OSError, IOError):
        plt.style.use('default')    
    mpl.rcParams['font.family']       = 'sans-serif'
    mpl.rcParams['font.sans-serif']   = ['DejaVu Sans', 'Arial']
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18


def plot_energy_balance(results, scenario_name, output_dir=None):
    """
    Plot energy balance for a scenario.

    Args:
        results (dict): Dictionary with energy balance results
        scenario_name (str): Name of the scenario
        output_dir (str, optional): Directory to save the plot

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    setup_plotting_style()

    # Extract data
    categories = ['PV Generation', 'Grid Import', 'Grid Export', 'Electric Load', 'Thermal Load','CO₂ Emissions']
    values = [
        results.get('pv_generation_mwh', 0),
        results.get('grid_import_mwh', 0),
        results.get('grid_export_mwh', 0),
        results.get('total_electric_load_mwh', 0),
        results.get('total_thermal_load_mwh', 0),
        results.get('co2_emissions_kg', 0)  # ← valore di emissioni (kg CO₂)
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', "#000000"]
    bars = ax.bar(categories, values, color=colors)

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')

    # Customize plot
    ax.set_ylabel('Energy (mwh)')
    ax.set_title(f'Energy Balance - {scenario_name}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Save if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{scenario_name}_energy_balance.png'), dpi=300, bbox_inches='tight')

    return fig
    for k, v in time_series.items():
        print(f"{k}: type={type(v)}, len={len(v) if hasattr(v, '__len__') else 'NA'}")




def plot_energy_balance_breakdown(results, scenario_name, output_dir=None):
    """
    Plot a 2x2 grid of bar charts, splitting the energy balance into four parts:
      1) PV Generation, Grid Import, Grid Export
      2) Thermal Load, Electric Load
      3) Heat import, Heat Pump Production
      4) CO₂ Emissions
    """
    setup_plotting_style()

    # Define groups and subplot titles
    groups = [
        (['PV Generation','PV→Load', 'Battery Dispatch', 'Grid Import', 'Grid Export'], 'Elettrico: PV & Rete'),
        (['Thermal Load', 'Electric Load'], 'Domanda Energetica'),
        (['Heat import', 'Heat Pump Production'], 'Flussi Termici'),
        (['CO₂ Emissions Grid', 'CO₂ Emissions DH'], 'Emissioni CO₂')
    ]

    # Map display labels to result keys
    key_map = {
        'PV Generation':            'pv_generation_mwh',
        'PV→Load':           'battery_discharge_mwh',
        'Battery Dispatch':  'battery_dispatch_mwh',
        'Grid Import':              'grid_import_mwh',
        'Grid Export':              'grid_export_mwh',
        'Thermal Load':             'total_thermal_load_mwh',
        'Electric Load':            'total_electric_load_mwh',
        'Heat import':              'DH Import to Building (MWh_th)',
        'Heat Pump Production':     'Heat Pump Production (MWh_th)',
        'CO₂ Emissions Grid':       'co2_emissions_kg',
        'CO₂ Emissions DH':          'dh_emissions_kg'
    }

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', "#000000"]
    for ax, (labels, subtitle) in zip(axes, groups):
        # Extract values for each label
        vals = [results.get(key_map[label], 0) for label in labels]
        # Plot bars
        bars = ax.bar(labels, vals, color=colors[:len(labels)])
        # Annotate bars
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
        ax.set_title(subtitle)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Super-title and layout adjustment
    fig.suptitle(f'Energy Balance Breakdown - {scenario_name}', fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f'{scenario_name}_energy_balance_breakdown.png')
        plt.savefig(fname, dpi=300, bbox_inches='tight')

    return fig

# (rest of your existing functions: plot_time_series, plot_scenario_comparison, plot_heatmap, save_hourly_results)

def plot_time_series(time_series, scenario_name, output_dir=None):
    """
    Plot time series data for a scenario.

    Args:
        time_series (dict): Dictionary with time series data
        scenario_name (str): Name of the scenario
        output_dir (str, optional): Directory to save the plot

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    setup_plotting_style()

    # Extract data
    timestamps = time_series.get('timestamps', pd.DatetimeIndex([]))
    electric_load = time_series.get('electric_load', pd.Series(np.zeros(len(timestamps)), index=timestamps))
    thermal_load = time_series.get('thermal_load', pd.Series(np.zeros(len(timestamps)), index=timestamps))
    pv_generation = time_series.get('pv_generation', pd.Series(np.zeros(len(timestamps)), index=timestamps))
    grid_import = time_series.get('grid_import', pd.Series(np.zeros(len(timestamps)), index=timestamps))
    grid_export = time_series.get('grid_export', pd.Series(np.zeros(len(timestamps)), index=timestamps))
    

    

    

    # DEBUG: controllo coerenza lunghezze
    for serie, nome in zip(
        [electric_load, thermal_load, pv_generation, grid_import, grid_export],
        ['electric_load', 'thermal_load', 'pv_generation', 'grid_import', 'grid_export']):
        if len(serie) != len(timestamps):
            print(f"[ERRORE] La serie '{nome}' ha lunghezza {len(serie)} invece di {len(timestamps)}")

# Fix per evitare crash se le serie temporali non sono allineate
    if any(len(serie) != len(timestamps) for serie in [
       electric_load, thermal_load, pv_generation, grid_import, grid_export
    ]):
       print("[ERRORE FATALE] Le serie non sono coerenti in lunghezza, il plotting verrà saltato.")
       return None



    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot time series
    ax.plot(timestamps, electric_load, color='#1b9e77', label='Electric Load')
    ax.plot(timestamps, thermal_load, color='#d95f02', label='Thermal Load')
    ax.plot(timestamps, pv_generation, color='#7570b3', label='PV Generation')
    ax.plot(timestamps, grid_import, color='#e7298a', label='Grid Import')
    ax.plot(timestamps, grid_export, color='#66a61e', label='Grid Export')
    ax.legend()  # per includere la nuova curva



    # Customize plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Power (MW)')
    ax.set_title(f'Time Series - {scenario_name}')
    ax.legend()
    ax.grid(True)

    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()

    # Save if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{scenario_name}_time_series.png'), dpi=300, bbox_inches='tight')

    return fig


def plot_scenario_comparison(scenarios_results, metric, output_dir=None):
    """
    Plot comparison of a specific metric across scenarios.

    Args:
        scenarios_results (dict): Dictionary with results for each scenario
        metric (str): Metric to compare
        output_dir (str, optional): Directory to save the plot

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    setup_plotting_style()

    # Extract data
    scenarios = list(scenarios_results.keys())
    values = [scenarios_results[scenario].get(metric, 0) for scenario in scenarios]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars
    bars = ax.bar(scenarios, values, color='#2196F3')

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')

    # Customize plot
    ax.set_ylabel(metric)
    ax.set_title(f'Scenario Comparison - {metric}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Save if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'scenario_comparison_{metric}.png'), dpi=300, bbox_inches='tight')

    return fig


def plot_heatmap(data, x_labels, y_labels, title, output_dir=None, filename=None):
    """
    Plot a heatmap.

    Args:
        data (numpy.ndarray): 2D array of data
        x_labels (list): Labels for x-axis
        y_labels (list): Labels for y-axis
        title (str): Title of the plot
        output_dir (str, optional): Directory to save the plot
        filename (str, optional): Filename for the saved plot

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    setup_plotting_style()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FFFFFF', '#E3F2FD', '#90CAF9', '#2196F3', '#1565C0'])

    # Plot heatmap
    im = ax.imshow(data, cmap=cmap)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotate x-axis labels if needed
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add values in cells
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            ax.text(j, i, f'{data[i, j]:.1f}', ha="center", va="center", color="black")

    # Set title
    ax.set_title(title)

    # Adjust layout
    fig.tight_layout()

    # Save if output_dir is provided
    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')

    return fig
def save_hourly_results(network, output_dir):
    """
    Estrae e salva i risultati orari del network in un CSV.

    Args:
        network: oggetto pypsa con risultati dello scenario.
        output_dir: cartella dove salvare i risultati.

    Returns:
        DataFrame contenente i dati orari.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamps = network.snapshots

    df_hourly = pd.DataFrame({
        "timestamps": timestamps,
        "electric_load": network.loads_t.p["electric_load"] if "electric_load" in network.loads_t.p else 0,
        "thermal_load": network.loads_t.p["thermal_load"] if "thermal_load" in network.loads_t.p else 0,
        "pv_generation": network.generators_t.p["pv"] if "pv" in network.generators_t.p else 0,
        "grid_import": network.generators_t.p["grid"] if "grid" in network.generators_t.p else 0,
        "grid_export": -network.loads_t.p["grid_export"] if "grid_export" in network.loads_t.p else 0,
    })

    df_hourly.to_csv(os.path.join(output_dir, "baseline_hourly_results.csv"), index=False)

    return df_hourly

def compute_pv_battery_breakdown(net):
    """
    Ritorna un dizionario con l’energia totale [MWh] per ciascun flusso:
      - PV → Building Elec  (link: 'PV Autoconsumo')
      - PV → Battery Storage (link: 'Battery Charge')
      - Battery Storage → Building Elec (link: 'Battery Dispatch')
    """
    flows = net.links_t.p0  # MW per ogni snapshot
    def safe_sum(col):
        return flows[col].clip(lower=0.0).sum() if col in flows.columns else 0.0

    return {
        'PV → Building Elec':              safe_sum('PV Autoconsumo'),
        'PV → Battery Storage':            safe_sum('Battery Charge'),
        'Battery Storage → Building Elec': safe_sum('Battery Dispatch'),
    }


def plot_pv_battery_breakdown(net, scenario_name='scenario', output_dir=None):
    """
    Plotta un istogramma con la scomposizione dei tre flussi.
    """
    breakdown = compute_pv_battery_breakdown(net)
    labels = list(breakdown.keys())
    values = list(breakdown.values())

    # ---- stile matplotlib ----
    fig, ax = plt.subplots(figsize=(10,6))
    bars = ax.bar(labels, values)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}',
                    xy=(bar.get_x()+bar.get_width()/2, h),
                    xytext=(0,4), textcoords='offset points',
                    ha='center', va='bottom')
    ax.set_ylabel('Energia [MWh]')
    ax.set_title(f'Flussi PV ↔ Battery ↔ Building – {scenario_name}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    # Salva su file se richiesto
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f"{output_dir}/{scenario_name}_pv_battery_breakdown.png", dpi=300, bbox_inches='tight')
    return fig



if __name__ == "__main__":
    # Example usage
    print("This module is not meant to be run directly.")
    print("Import it and use its functions in your scripts.")

    