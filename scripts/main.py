# scripts/main.py
"""
Main script to run Lyngby PED Scenario simulations.
"""
import pypsa
import pandas as pd
import os
import argparse # For command-line arguments
import yaml
from build_ped_network import create_baseline_network # Import baseline function
# Import functions for other scenarios later (e.g., from build_scenario_X.py)

def run_scenario(scenario_name, config_file, params_file):
    """Runs the specified scenario."""

    print(f"\n--- Running Scenario: {scenario_name} ---")

    # --- Load Main Config ---
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # --- Determine Paths ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, config.get('paths', {}).get('data_input', 'data/input'))
    OUTPUT_BASE_PATH = os.path.join(PROJECT_ROOT, config.get('paths', {}).get('data_output', 'data/output'))

    # Get scenario-specific config if available
    scenario_config = config.get('scenarios', {}).get(scenario_name, {})
    output_subdir = scenario_config.get('output_subdir', f'scenario_{scenario_name}')
    OUTPUT_PATH = os.path.join(OUTPUT_BASE_PATH, output_subdir)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"Output directory: {OUTPUT_PATH}")

    # --- Build Network ---
    # Select the correct network building function based on scenario
    # Add more build functions later (e.g., create_pv_storage_network)
    if scenario_name == 'baseline':
        network = create_baseline_network(config_file, params_file, DATA_PATH)
    # elif scenario_name == 'high_pv':
        # network = create_high_pv_network(config_file, params_file, DATA_PATH) # Example
    else:
        print(f"Error: Network build function for scenario '{scenario_name}' not defined.")
        return

    # --- Run Simulation / Optimization ---
    print("Proceeding to simulation stage...")
    solver_name = config.get('simulation_settings', {}).get('solver', 'highs')
    solver_options = config.get('simulation_settings', {}).get('solver_options', {})

    try:
        print(f"Attempting LOPF using solver: {solver_name}...")
        network.lopf(solver_name=solver_name, solver_options=solver_options)
        print(f"LOPF completed successfully using {solver_name}.")

    except Exception as e:
         print(f"LOPF failed: {e}")
         print(f"Ensure the solver '{solver_name}' is installed and accessible.")
         print("Simulation aborted.")
         # Decide if you want to save the network structure even if LOPF fails
         # network_output_file = os.path.join(OUTPUT_PATH, f'{scenario_name}_network_setup.nc')
         # network.export_to_netcdf(network_output_file)
         return # Stop processing this scenario


    # --- Process and Save Results ---
    print("Saving network and results...")

    # Save the network structure and results
    network_output_file = os.path.join(OUTPUT_PATH, f'{scenario_name}_network_results.nc')
    try:
        network.export_to_netcdf(network_output_file)
        print(f"Network and results saved to: {network_output_file}")
    except Exception as e:
        print(f"Error saving network to NetCDF: {e}")


    # Example: Calculate and save some key summary results
    print("Calculating summary results...")
    try:
        # Adjust weightings based on your snapshot definition (hourly assumed here)
        # If snapshots are not hourly, adjust snapshot_weightings accordingly
        if not hasattr(network.snapshot_weightings, 'generators'):
             network.snapshot_weightings['generators'] = pd.Series(1.0, index=network.snapshots) # Assume hourly if missing

        hours_in_period = network.snapshot_weightings.generators.sum()

        # Grid Interaction
        grid_series = network.generators_t.p.get('Grid', pd.Series(0.0, index=network.snapshots)) # Default to 0 if no Grid generator
        total_grid_import = grid_series[grid_series > 1e-3].sum() * network.snapshot_weightings.generators.loc[grid_series > 1e-3].sum() # MWh
        total_grid_export = -grid_series[grid_series < -1e-3].sum() * network.snapshot_weightings.generators.loc[grid_series < -1e-3].sum() # MWh (make positive)

        # Heat Production
        heat_series = network.generators_t.p.get('Heat Source', pd.Series(0.0, index=network.snapshots))
        total_heat_produced = heat_series.sum() * hours_in_period # MWh_th

        # PV Production
        pv_series = network.generators_t.p.get('Existing PV', pd.Series(0.0, index=network.snapshots))
        total_pv_produced = pv_series.sum() * hours_in_period # MWh

        # Total Demand
        total_elec_demand = sum(network.loads_t.p_set[col].sum() for col in network.loads_t.p_set.columns if network.loads.loc[col].carrier == 'electricity') * hours_in_period
        total_heat_demand = sum(network.loads_t.p_set[col].sum() for col in network.loads_t.p_set.columns if network.loads.loc[col].carrier == 'heat') * hours_in_period


        # Costs (Simplified - assumes constant marginal costs)
        op_cost_grid_import = total_grid_import * network.generators.at['Grid', 'marginal_cost']
        # op_revenue_grid_export = total_grid_export * grid_export_price # Needs careful handling of export price
        op_cost_heat = total_heat_produced * network.generators.at['Heat Source', 'marginal_cost']
        total_op_cost = op_cost_grid_import + op_cost_heat # - op_revenue_grid_export

        results_summary = pd.DataFrame({
            'Scenario': [scenario_name],
            'Total Grid Import (MWh)': [total_grid_import],
            'Total Grid Export (MWh)': [total_grid_export],
            'Total Heat Produced (MWh_th)': [total_heat_produced],
            'Total PV Produced (MWh)': [total_pv_produced],
            'Total Elec Demand (MWh)': [total_elec_demand],
            'Total Heat Demand (MWh_th)': [total_heat_demand],
            'Total Operational Cost (EUR)': [total_op_cost] # Simplified cost
        })
        summary_file = os.path.join(OUTPUT_PATH, f'{scenario_name}_summary.csv')
        results_summary.to_csv(summary_file, index=False)
        print(f"Summary results saved to: {summary_file}")

    except Exception as e:
         print(f"Could not calculate/save summary results: {e}")

    print(f"--- Scenario {scenario_name} finished ---")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run PyPSA simulations for the Lyngby PED project.")
    parser.add_argument("-s", "--scenario", help="Name of the scenario to run (must match config.yml)", required=True)
    parser.add_argument("-c", "--config", help="Path to the main configuration file", default="config/config.yml")
    parser.add_argument("-p", "--params", help="Path to the component parameters file", default="config/component_params.yml")

    args = parser.parse_args()

    # Basic validation
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        exit()
    if not os.path.exists(args.params):
        print(f"Error: Component parameters file not found at {args.params}")
        exit()

    # Run the selected scenario
    run_scenario(args.scenario, args.config, args.params)

    print("\nMain script finished.")

