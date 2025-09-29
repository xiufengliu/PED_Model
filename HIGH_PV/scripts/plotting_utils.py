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
    categories = ['PV Generation', 'Grid Import', 'Grid Export', 'Total Demand']
    values = [
        results.get('pv_generation_kwh', 0),
        results.get('grid_import_kwh', 0),
        results.get('grid_export_kwh', 0),
        results.get('total_load_kwh', 0)
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars
    colors = ['#FFC107', '#F44336', '#4CAF50', '#2196F3']
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
    ax.set_ylabel('Energy (kWh)')
    ax.set_title(f'Energy Balance - {scenario_name}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Save if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{scenario_name}_energy_balance.png'), dpi=300, bbox_inches='tight')

    return fig


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
    total_load = time_series.get('total_load', pd.Series([], index=timestamps))
    pv_generation = time_series.get('pv_generation', pd.Series([], index=timestamps))
    grid_import = time_series.get('grid_import', pd.Series([], index=timestamps))
    grid_export = time_series.get('grid_export', pd.Series([], index=timestamps))

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot time series
    ax.plot(timestamps, total_load, 'b-', label='Total Load')
    ax.plot(timestamps, pv_generation, 'y-', label='PV Generation')
    ax.plot(timestamps, grid_import, 'r-', label='Grid Import')
    ax.plot(timestamps, grid_export, 'g-', label='Grid Export')

    # Customize plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Power (kW)')
    ax.set_title(f'Time Series - {scenario_name}')
    ax.legend()
    ax.grid(True)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
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


if __name__ == "__main__":
    # Example usage
    print("This module is not meant to be run directly.")
    print("Import it and use its functions in your scripts.")