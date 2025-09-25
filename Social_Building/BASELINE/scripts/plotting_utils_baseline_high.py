import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap


def setup_plotting_style():
    """Set up the plotting style for consistent visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18

def plot_pv_split(hourly_df: pd.DataFrame, scenario_name: str, output_dir: str | None = None):
    """
    Visualizza la PV annuale totale e come si suddivide in:
      - auto-consumata
      - esportata (verso rete, post-perdite)
      (facoltativo: curtailment)

    Usa le colonne create in generate_hourly_df:
      pv_generation_kwh (totale), pv_self_consumed_kwh, pv_exported_to_grid_kwh, pv_curtailed_kwh
    """
    setup_plotting_style()

    # Totali annui
    total_pv = hourly_df['pv_generation_kwh'].sum()
    pv_self  = hourly_df['pv_self_consumed_kwh'].sum()
    pv_exp   = hourly_df['pv_exported_to_grid_kwh'].sum()
    pv_curt  = hourly_df.get('pv_curtailed_kwh', pd.Series([0])).sum()

    # --- Grafico 1: Stack bar annuale ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.bar(['PV totale'], [total_pv], color='#90CAF9', label='Totale PV')
    # Stack a fianco per la ripartizione (self vs export; curtailment opzionale)
    cats = ['Auto-consumo', 'Export']
    vals = [pv_self, pv_exp]
    colors = ['#66BB6A', '#42A5F5']
    if pv_curt > 1e-6:
        cats.append('Curtailment')
        vals.append(pv_curt)
        colors.append('#BDBDBD')

    ax1.bar(cats, vals, color=colors)
    for x, v in zip(['PV totale'] + cats, [total_pv] + vals):
        ax1.annotate(f'{v/1000:.1f} MWh', xy=(x, v), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

    ax1.set_ylabel('Energia (kWh)')
    ax1.set_title(f'PV: totale e ripartizione – {scenario_name}')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig1.savefig(os.path.join(output_dir, f'{scenario_name}_pv_split_annual.png'), dpi=300, bbox_inches='tight')

    # --- Grafico 2: Area stack settimanale (esempio: prima settimana) ---
    if len(hourly_df) > 0:
        # prendi una finestra significativa (es. 1 settimana)
        start = hourly_df.index.min()
        end = start + pd.Timedelta(days=7)
        sl = hourly_df.loc[(hourly_df.index >= start) & (hourly_df.index < end)]

        fig2, ax2 = plt.subplots(figsize=(14, 6))
        ax2.stackplot(
            sl.index,
            sl['pv_self_consumed_kwh'],
            sl['pv_exported_to_grid_kwh'],
            labels=['Auto-consumo', 'Export'],
            alpha=0.9
        )
        ax2.plot(sl.index, sl['pv_generation_kwh'], label='PV totale', linewidth=1.5)
        ax2.set_ylabel('Energia per ora (kWh)')
        ax2.set_title(f'PV – auto-consumo vs export (prima settimana) – {scenario_name}')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        ax2.xaxis.set_major_locator(mdates.DayLocator())
        fig2.autofmt_xdate()

        if output_dir:
            fig2.savefig(os.path.join(output_dir, f'{scenario_name}_pv_split_week.png'), dpi=300, bbox_inches='tight')

    return fig1

def plot_energy_balance(results, scenario_name, output_dir=None):
    """
    Plot energy balance for a scenario in MWh.

    Args:
        results (dict): Dictionary with keys:
            - total_pv_mwh
            - total_import_mwh
            - total_export_mwh
            - total_electric_load_mwh
            - total_thermal_load_mwh
        scenario_name (str): Name of the scenario
        output_dir (str, optional): Directory to save the plot

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    setup_plotting_style()

    categories = ['PV Generation', 'Grid Import', 'Grid Export', 'Electric Load', 'Thermal Load']
    values = [
        results.get('total_pv_mwh', 0),
        results.get('total_import_mwh', 0),
        results.get('total_export_mwh', 0),
        results.get('total_electric_load_mwh', 0),
        results.get('total_thermal_load_mwh', 0)
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']
    bars = ax.bar(categories, values, color=colors)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_ylabel('Energy (MWh)')
    ax.set_title(f'Energy Balance - {scenario_name}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{scenario_name}_energy_balance.png'),
                    dpi=300, bbox_inches='tight')

    return fig


def plot_time_series(time_series, scenario_name, output_dir=None, unit='kWh'):
    """
    Plot delle serie temporali per lo scenario.

    NOTE: I dati attesi sono in kWh per snapshot (come generati da main.py):
      - electric_load (kWh)
      - thermal_load (kWh)
      - pv_generation (kWh)
      - grid_import (kWh)
      - grid_export (kWh)

    Args:
        time_series (dict): chiavi:
            - timestamps (DatetimeIndex)
            - electric_load
            - thermal_load
            - pv_generation
            - grid_import
            - grid_export
        scenario_name (str)
        output_dir (str, opzionale)
        unit (str): etichetta unità y-axis (default: 'kWh')
    """
    setup_plotting_style()

    timestamps    = time_series.get('timestamps', pd.DatetimeIndex([]))
    electric_load = time_series.get('electric_load',  pd.Series(np.zeros(len(timestamps)), index=timestamps))
    thermal_load  = time_series.get('thermal_load',   pd.Series(np.zeros(len(timestamps)), index=timestamps))
    pv_generation = time_series.get('pv_generation',  pd.Series(np.zeros(len(timestamps)), index=timestamps))
    grid_import   = time_series.get('grid_import',    pd.Series(np.zeros(len(timestamps)), index=timestamps))
    grid_export   = time_series.get('grid_export',    pd.Series(np.zeros(len(timestamps)), index=timestamps))

    for serie, nome in zip(
        [electric_load, thermal_load, pv_generation, grid_import, grid_export],
        ['electric_load', 'thermal_load', 'pv_generation', 'grid_import', 'grid_export']
    ):
        if len(serie) != len(timestamps):
            print(f"[ERROR] Series '{nome}' length {len(serie)} != timestamps length {len(timestamps)}")
    if any(len(serie) != len(timestamps) for serie in [electric_load, thermal_load, pv_generation, grid_import, grid_export]):
        print("[FATAL ERROR] Time series lengths mismatch, skipping plot.")
        return None

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(timestamps, electric_load, label='Electric Load')
    ax.plot(timestamps, thermal_load,  label='Thermal Load')
    ax.plot(timestamps, pv_generation, label='PV Generation')
    ax.plot(timestamps, grid_import,   label='Grid Import')
    ax.plot(timestamps, grid_export,   label='Grid Export')

    ax.set_xlabel('Time')
    ax.set_ylabel(f'Energy per snapshot ({unit})')
    ax.set_title(f'Time Series - {scenario_name}')
    ax.legend()
    ax.grid(True)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{scenario_name}_time_series.png'),
                    dpi=300, bbox_inches='tight')

    return fig



def plot_scenario_comparison(scenarios_results, metric, output_dir=None):
    setup_plotting_style()
    scenarios = list(scenarios_results.keys())
    values = [scenarios_results[sc].get(metric, 0) for sc in scenarios]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(scenarios, values, color='#2196F3')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
    ax.set_ylabel(metric)
    ax.set_title(f'Scenario Comparison - {metric}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'scenario_comparison_{metric}.png'),
                    dpi=300, bbox_inches='tight')
    return fig


def plot_heatmap(data, x_labels, y_labels, title, output_dir=None, filename=None):
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FFFFFF', '#E3F2FD', '#90CAF9', '#2196F3', '#1565C0'])
    im = ax.imshow(data, cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            ax.text(j, i, f'{data[i, j]:.1f}', ha="center", va="center")
    ax.set_title(title)
    fig.tight_layout()
    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    return fig
