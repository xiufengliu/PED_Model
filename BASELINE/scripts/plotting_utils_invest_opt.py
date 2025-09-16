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
        (['PV Generation', 'Grid Import', 'Grid Export'], 'Elettrico: PV & Rete'),
        (['Thermal Load', 'Electric Load'], 'Domanda Energetica'),
        (['Heat import', 'Heat Pump Production','DSM Heat Charge','DSM Heat Dispatch'], 'Flussi Termici'),
        (['CO₂ Emissions Grid', 'CO₂ Emissions DH'], 'Emissioni CO₂')
    ]

    # Map display labels to result keys
    key_map = {
        'PV Generation':            'pv_generation_mwh',
        'Grid Import':              'grid_import_mwh',
        'Grid Export':              'grid_export_mwh',
        'Thermal Load':             'total_thermal_load_mwh',
        'Electric Load':            'total_electric_load_mwh',
        'Heat import':              'DH Import to Building (MWh_th)',
        'Heat Pump Production':     'Heat Pump Production (MWh_th)',
        'DSM Heat Charge':    'DSM Heat Charge (MWh_th)',
        'DSM Heat Dispatch':  'DSM Heat Dispatch (MWh_th)',
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
###########################################################PV
def compute_pv_battery_breakdown(net):
    """
    Ritorna un dizionario con l’energia elettrica totale [MWh] per ciascun flusso:
      • Grid Import effettivo (inflexible + flexible − PV_direct − batterie_dispatch)
      • PV → Building Elec         (link "Battery Discharge")
      • PV → Battery Storage       (link "Battery Charge")
      • Battery → Building Elec    (link "Battery Dispatch")
    """
    flows = net.links_t.p0

    # 1) Carico elettrico inflessibile
    inf = net.loads_t.p_set['Building Elec Load'].sum()
    # 2) Carico elettrico flessibile (se presente)
    flex = net.loads_t.p_set.get('DSM Elec Flex Load',
             pd.Series(0.0, index=net.snapshots)).sum()
    total_load = inf + flex

    # 3) Flussi PV↔Building / Batterie
    pv_direct = flows['Battery Discharge'].sum()
    pv_charge = flows['Battery Charge'].sum()
    batt_disp = flows['Battery Dispatch'].sum()
    # 4) Flusso Grid → Battery Storage (link definito nel tuo scenario cost_self_1)
    #    Cambia 'Grid Charge' col nome esatto del tuo Link
    grid_to_batt = flows.get('Grid Import_Storage', pd.Series(0.0, index=net.snapshots)).sum()
    # 5) Quello che “deve” importare la rete
    effective_grid = total_load - pv_direct - batt_disp
    flows = net.links_t.p0  # Serie oraria MW per ogni Link
    return {
        'Grid Import (eff.)':         effective_grid,
        'PV → Building Elec':          pv_direct,
        'PV → Battery Storage':        pv_charge,
        'Grid→Battery Storage':        grid_to_batt,
        'Battery → Building Elec':     batt_disp,
        'PV → Heat Source Bus (Heat Pump)':    flows['Heat Pump'].sum(),

    }

def plot_pv_battery_breakdown(net, scenario_name='scenario', output_dir=None):
    """
    Plotta un bar‑plot con la scomposizione approfondita dell’elettrico:
      • Grid Import effettivo
      • PV → Building Elec
      • PV → Battery Storage
      • Battery → Building Elec
    """
    import os
    import matplotlib.pyplot as plt
    from scripts.plotting_utils_pv_battery_1 import setup_plotting_style

    breakdown = compute_pv_battery_breakdown(net)
    labels, values = zip(*breakdown.items())

    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom')

    ax.tick_params(axis='x', labelsize=9, labelrotation=30)
    ax.set_ylabel('Energia elettrica [MWh]')
    ax.set_title(f'Electric Energy Breakdown – {scenario_name}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'{scenario_name}_electric_breakdown.png'),
                    dpi=300, bbox_inches='tight')
    return fig


    ############################################################
    
def compute_thermal_breakdown(net):
    import pandas as pd

    flows = net.links_t.p0
    # 1) Carico inflessibile
    inf = net.loads_t.p_set['Building Heat Load'].sum()
    # 2) Carico flessibile DSM
    flex = net.loads_t.p_set.get('DSM Heat Flex Load',
               pd.Series(0.0, index=net.snapshots)).sum()
    # Totale domanda (inflex. + flex)
    total_load = inf + flex

    # 3) Flussi diretti e storage
    hp_direct   = flows['Thermal Discharge'].sum()
    charge      = flows['Thermal Charge'].sum()
    dispatch    = flows['Thermal Dispatch'].sum()
    # 4) Quello che “deve” importare DH per pareggiare il bilancio
    effective_dh = total_load - hp_direct - dispatch

    return {
        'Effective DH Import':            effective_dh,
        'Heat Source → Building Heat':    hp_direct,
        'Heat Source → Thermal Storage':  charge,
        'Thermal Storage → Building Heat':dispatch,
    }



def plot_thermal_breakdown(net, scenario_name='scenario', output_dir=None):
    """
    Plotta un bar‑plot con la scomposizione dei tre flussi termici.
    """
    from scripts.plotting_utils_pv_battery_1 import setup_plotting_style
    import os
    import matplotlib.pyplot as plt

    breakdown = compute_thermal_breakdown(net)
    labels, values = zip(*breakdown.items())

    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom')

    # Piccole etichette ruotate per non sovrapporre
    ax.tick_params(axis='x', labelsize=8, labelrotation=30)

    ax.set_ylabel('Energia termica [MWh_th]')
    ax.set_title(f'Flussi Termici – {scenario_name}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'{scenario_name}_thermal_breakdown.png'),
                    dpi=300, bbox_inches='tight')
    return fig

def plot_dsm_thermal_breakdown(hourly_df, scenario_name='scenario', output_dir=None):
    """
    Visualizza un bar‑plot con la scomposizione approfondita del DSM termico:
      • DSM Heat Charge
      • DSM Heat Dispatch
      • Heat Flexible Load   (profilo flessibile caricato)
      • Residual Heat Demand = Heat Inflexible + Charge - Dispatch
    """
    import os
    import matplotlib.pyplot as plt
    from scripts.plotting_utils_pv_battery_1 import setup_plotting_style

    # 1) DSM Heat Charge
    dsm_charge      = hourly_df['heat_charge_mwh'].sum()
    # 2) DSM Heat Dispatch
    dsm_dispatch    = hourly_df['heat_dispatch_mwh'].sum()
    # 3) Heat Flexible Load (è pari al carico DSM caricato)
    heat_flexible   = hourly_df['heat_charge_mwh'].sum()
    # 4) Heat Inflexible Load
    heat_inflexible = hourly_df['heat_inflexible_load_mwh'].sum()
    # Residual = inflexible + charge - dispatch
    residual        = heat_inflexible + dsm_charge - dsm_dispatch

    breakdown = {
        'DSM Heat Charge':      dsm_charge,
        'DSM Heat Dispatch':    dsm_dispatch,
        'Heat Flexible Load':   heat_flexible,
        'Inflexible Heat Demand': residual,
    }

    labels, values = zip(*breakdown.items())

    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0,4), textcoords='offset points',
                    ha='center', va='bottom')

    # Ruoto e riduco leggermente le etichette per chiarezza
    ax.tick_params(axis='x', labelsize=9, labelrotation=30)

    ax.set_ylabel('Energia termica [MWh_th]')
    ax.set_title(f'DSM Thermal Breakdown – {scenario_name}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'{scenario_name}_dsm_thermal_breakdown.png'),
                    dpi=300, bbox_inches='tight')
    return fig


def plot_investment_capacity_expansion(summary_df, scenario_name='invest_opt', output_dir=None):
    """
    Plot capacity expansion across investment periods for different technologies.

    Args:
        summary_df (pd.DataFrame): Investment summary DataFrame with Period, Technology, Capacity_MW columns
        scenario_name (str): Name of the scenario
        output_dir (str, optional): Directory to save the plot

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    setup_plotting_style()

    # Filter for capacity data
    capacity_data = summary_df[summary_df['Capacity_MW'] > 0].copy()

    if capacity_data.empty:
        print("No capacity data available for plotting")
        return None

    # Pivot data for plotting
    pivot_data = capacity_data.pivot(index='Period', columns='Technology', values='Capacity_MW').fillna(0)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create stacked bar chart
    bottom = np.zeros(len(pivot_data.index))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, tech in enumerate(pivot_data.columns):
        ax.bar(pivot_data.index, pivot_data[tech], bottom=bottom,
               label=tech, color=colors[i % len(colors)], alpha=0.8)
        bottom += pivot_data[tech]

    ax.set_xlabel('Investment Period')
    ax.set_ylabel('Installed Capacity (MW)')
    ax.set_title(f'Technology Capacity Expansion - {scenario_name}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{scenario_name}_capacity_expansion.png'),
                    dpi=300, bbox_inches='tight')

    return fig


def plot_investment_cost_breakdown(summary_df, scenario_name='invest_opt', output_dir=None):
    """
    Plot CAPEX breakdown by technology and period.

    Args:
        summary_df (pd.DataFrame): Investment summary DataFrame
        scenario_name (str): Name of the scenario
        output_dir (str, optional): Directory to save the plot

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    setup_plotting_style()

    # Filter for CAPEX data
    capex_data = summary_df[summary_df['CAPEX_EUR'] > 0].copy()

    if capex_data.empty:
        print("No CAPEX data available for plotting")
        return None

    # Convert to millions for better readability
    capex_data['CAPEX_MEUR'] = capex_data['CAPEX_EUR'] / 1e6

    # Create subplot for each period
    periods = capex_data['Period'].unique()
    fig, axes = plt.subplots(1, len(periods), figsize=(15, 6))

    if len(periods) == 1:
        axes = [axes]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, period in enumerate(periods):
        period_data = capex_data[capex_data['Period'] == period]

        # Create pie chart for this period
        sizes = period_data['CAPEX_MEUR']
        labels = period_data['Technology']

        _, _, autotexts = axes[i].pie(sizes, labels=labels, autopct='%1.1f%%',
                                      colors=colors[:len(labels)], startangle=90)
        axes[i].set_title(f'CAPEX Breakdown {period}\nTotal: {sizes.sum():.1f} M€')

        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

    plt.suptitle(f'Investment Cost Breakdown - {scenario_name}', fontsize=16)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{scenario_name}_cost_breakdown.png'),
                    dpi=300, bbox_inches='tight')

    return fig


def plot_technology_mix_optimization(summary_df, scenario_name='invest_opt', output_dir=None):
    """
    Plot technology mix evolution across investment periods.

    Args:
        summary_df (pd.DataFrame): Investment summary DataFrame
        scenario_name (str): Name of the scenario
        output_dir (str, optional): Directory to save the plot

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    setup_plotting_style()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Capacity evolution
    capacity_data = summary_df[summary_df['Capacity_MW'] > 0]
    if not capacity_data.empty:
        for tech in capacity_data['Technology'].unique():
            tech_data = capacity_data[capacity_data['Technology'] == tech]
            ax1.plot(tech_data['Period'], tech_data['Capacity_MW'],
                    marker='o', linewidth=2, label=tech)
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Capacity (MW)')
        ax1.set_title('Technology Capacity Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Storage capacity evolution
    storage_data = summary_df[summary_df['Capacity_MWh'] > 0]
    if not storage_data.empty:
        for tech in storage_data['Technology'].unique():
            tech_data = storage_data[storage_data['Technology'] == tech]
            ax2.plot(tech_data['Period'], tech_data['Capacity_MWh'],
                    marker='s', linewidth=2, label=tech)
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Energy Capacity (MWh)')
        ax2.set_title('Storage Capacity Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. CAPEX evolution
    capex_data = summary_df[summary_df['CAPEX_EUR'] > 0]
    if not capex_data.empty:
        capex_data['CAPEX_MEUR'] = capex_data['CAPEX_EUR'] / 1e6
        for tech in capex_data['Technology'].unique():
            tech_data = capex_data[capex_data['Technology'] == tech]
            ax3.plot(tech_data['Period'], tech_data['CAPEX_MEUR'],
                    marker='^', linewidth=2, label=tech)
        ax3.set_xlabel('Period')
        ax3.set_ylabel('CAPEX (M€)')
        ax3.set_title('Investment Cost Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Capacity factors
    cf_data = summary_df[summary_df['Capacity_Factor'] > 0]
    if not cf_data.empty:
        for tech in cf_data['Technology'].unique():
            tech_data = cf_data[cf_data['Technology'] == tech]
            ax4.plot(tech_data['Period'], tech_data['Capacity_Factor'],
                    marker='d', linewidth=2, label=tech)
        ax4.set_xlabel('Period')
        ax4.set_ylabel('Capacity Factor')
        ax4.set_title('Technology Utilization')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)

    plt.suptitle(f'Technology Mix Optimization Results - {scenario_name}', fontsize=16)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{scenario_name}_technology_mix.png'),
                    dpi=300, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    # Example usage
    print("This module is not meant to be run directly.")
    print("Import it and use its functions in your scripts.")

