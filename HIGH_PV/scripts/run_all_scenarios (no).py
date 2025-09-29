#!/usr/bin/env python3
"""
Script to run all available scenarios and compare their results.
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the scenarios package
from scenarios import SCENARIO_FUNCTIONS


def run_all_scenarios(config_file, params_file, output_dir):
    """
    Run all available scenarios and collect their results.
    
    Args:
        config_file (str): Path to the main configuration file
        params_file (str): Path to the component parameters file
        output_dir (str): Directory to save the results
        
    Returns:
        dict: Dictionary with results for each scenario
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of available scenarios
    scenarios = list(SCENARIO_FUNCTIONS.keys())
    print(f"Found {len(scenarios)} scenarios: {', '.join(scenarios)}")
    
    # Run each scenario
    results = {}
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"Running scenario: {scenario}")
        print(f"{'='*80}")
        
        # Run the scenario using the main script
        cmd = [
            sys.executable,
            os.path.join('scripts', 'main.py'),
            '--scenario', scenario,
            '--config', config_file,
            '--params', params_file
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Scenario {scenario} completed successfully.")
            
            # Load the results
            scenario_output_dir = os.path.join('data', 'output', f'scenario_{scenario}')
            summary_file = os.path.join(scenario_output_dir, f'{scenario}_summary.csv')
            
            if os.path.exists(summary_file):
                summary_df = pd.read_csv(summary_file)
                # Convert to dictionary for easier access
                results[scenario] = summary_df.iloc[0].to_dict()
                print(f"Loaded results for scenario {scenario}.")
            else:
                print(f"Warning: No summary file found for scenario {scenario}.")
        
        except subprocess.CalledProcessError as e:
            print(f"Error running scenario {scenario}: {e}")
    
    return results


def compare_scenarios(results, output_dir):
    """
    Compare the results of all scenarios and create visualizations.
    
    Args:
        results (dict): Dictionary with results for each scenario
        output_dir (str): Directory to save the comparison results
    """
    if not results:
        print("No results to compare.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame with all results
    comparison_df = pd.DataFrame(results).T
    
    # Save the comparison to CSV
    comparison_file = os.path.join(output_dir, 'scenario_comparison.csv')
    comparison_df.to_csv(comparison_file)
    print(f"Saved scenario comparison to {comparison_file}")
    
    # Create visualizations
    try:
        # Import plotting utilities
        from plotting_utils import setup_plotting_style, plot_scenario_comparison
        
        # Set up plotting style
        setup_plotting_style()
        
        # Create comparison plots for key metrics
        metrics = [
            'Total Grid Import (MWh)',
            'Total Grid Export (MWh)',
            'Total PV Produced (MWh)',
            'Total Operational Cost (EUR)'
        ]
        
        for metric in metrics:
            if metric in comparison_df.columns:
                # Create a simple bar chart
                plt.figure(figsize=(10, 6))
                comparison_df[metric].plot(kind='bar', color='#2196F3')
                plt.title(f'Scenario Comparison - {metric}')
                plt.ylabel(metric)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Save the plot
                plot_file = os.path.join(output_dir, f'comparison_{metric.replace(" ", "_").lower()}.png')
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                print(f"Saved comparison plot for {metric} to {plot_file}")
    
    except ImportError as e:
        print(f"Warning: Could not create visualizations: {e}")
    
    print("\nScenario comparison completed.")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run all scenarios and compare their results.")
    parser.add_argument("-c", "--config", help="Path to the main configuration file", default="config/config.yml")
    parser.add_argument("-p", "--params", help="Path to the component parameters file", default="config/component_params.yml")
    parser.add_argument("-o", "--output", help="Directory to save the comparison results", default="data/output/comparison")
    
    args = parser.parse_args()
    
    # Run all scenarios
    results = run_all_scenarios(args.config, args.params, args.output)
    
    # Compare the results
    compare_scenarios(results, args.output)


if __name__ == "__main__":
    main()
