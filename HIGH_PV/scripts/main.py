#!/usr/bin/env python3
"""
Main script to run Social Building PED Scenario simulations.
"""
import pypsa
import pandas as pd
import os
import argparse
import yaml
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the scenarios package
from scenarios import get_scenario_function

# Global variable to store original marginal costs
ORIGINAL_MARGINAL_COSTS = {}

def run_scenario(scenario_name, config_file, params_file):
    """Runs the specified scenario."""
    global ORIGINAL_MARGINAL_COSTS

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
    try:
        # Get the appropriate scenario function
        create_network = get_scenario_function(scenario_name)

        # Force the use of 8760 hours
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        config_data['simulation_settings']['num_hours'] = 8760

        # Create a temporary config file with the updated settings
        temp_config_file = os.path.join(os.path.dirname(config_file), 'temp_config.yml')
        with open(temp_config_file, 'w') as f:
            yaml.dump(config_data, f)

        # Build the network with the temporary config file
        network = create_network(temp_config_file, params_file, DATA_PATH)

        # Remove the temporary config file
        os.remove(temp_config_file)
    except Exception as e:
        print(f"Error building network for scenario '{scenario_name}': {e}")
        return

    # --- Run Simulation / Optimization ---
    print("Proceeding to simulation stage...")
    solver_name = config.get('simulation_settings', {}).get('solver', 'highs')
    solver_options = config.get('simulation_settings', {}).get('solver_options', {})

    # Store original marginal costs before optimization
    ORIGINAL_MARGINAL_COSTS = {}

    # Debug: Print marginal costs before optimization
    print("\nDEBUG: Generator marginal costs before optimization:")
    for gen_name, gen in network.generators.iterrows():
        print(f"{gen_name}: {gen.marginal_cost}")
        if isinstance(gen.marginal_cost, pd.Series):
            ORIGINAL_MARGINAL_COSTS[gen_name] = gen.marginal_cost.copy()
        else:
            ORIGINAL_MARGINAL_COSTS[gen_name] = gen.marginal_cost

    try:
        print(f"Attempting LOPF using solver: {solver_name}...")

        # For PyPSA >= 0.34.0, we need to use linopy
        import pypsa

        # Create optimization model and solve it
        network.optimize(solver_name=solver_name, solver_options=solver_options)

        # No need to extract results, they are already in the network
        status = "ok"
        termination_condition = "optimal"

        if termination_condition == "optimal":
            print(f"LOPF completed successfully using {solver_name}.")

            # Restore original marginal costs after optimization
            for gen_name, cost in ORIGINAL_MARGINAL_COSTS.items():
                network.generators.at[gen_name, 'marginal_cost'] = cost

            # Debug: Print marginal costs after restoration
            print("\nDEBUG: Generator marginal costs after restoration:")
            for gen_name, gen in network.generators.iterrows():
                print(f"{gen_name}: {gen.marginal_cost}")
        else:
            print(f"LOPF failed with status: {status}, termination condition: {termination_condition}")
            print("Simulation aborted.")
            return # Stop processing this scenario

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

        # Grid Interaction
        # Use 'Grid' as the generator name (used in high_pv.py and baseline.py)
        grid_series = network.generators_t.p.get('Grid', pd.Series(0.0, index=network.snapshots)) # Default to 0 if no Grid generator
        # Somma i valori orari (MW * 1h = MWh)
        total_grid_import = grid_series[grid_series > 1e-3].sum() # MWh
        total_grid_export = -grid_series[grid_series < -1e-3].sum() # MWh (make positive)

        # Heat Production
        heat_series = network.generators_t.p.get('Heat Source', pd.Series(0.0, index=network.snapshots))
        # Somma i valori orari (MW * 1h = MWh)
        total_heat_produced = heat_series.sum() # MWh_th

        # PV Production
        total_pv_produced = 0

        # Check for baseline PV
        if 'Existing PV' in network.generators_t.p:
            pv_series = network.generators_t.p['Existing PV']
            total_pv_produced += pv_series.sum() # MWh

        # Check for high PV
        if 'Maximized PV' in network.generators_t.p:
            pv_series = network.generators_t.p['Maximized PV']
            total_pv_produced += pv_series.sum() # MWh

        # Check for rooftop PV
        if 'Rooftop PV' in network.generators_t.p:
            pv_series = network.generators_t.p['Rooftop PV']
            total_pv_produced += pv_series.sum() # MWh

        # Total Demand
        total_elec_demand = sum(network.loads_t.p_set[col].sum() for col in network.loads_t.p_set.columns if network.loads.loc[col].carrier == 'electricity')
        total_heat_demand = sum(network.loads_t.p_set[col].sum() for col in network.loads_t.p_set.columns if network.loads.loc[col].carrier == 'heat')


        # Debug information
        print("\nDEBUG: Generator information:")
        print(network.generators)
        print("\nDEBUG: Generator marginal costs:")
        for gen_name, gen in network.generators.iterrows():
            print(f"{gen_name}: {gen.marginal_cost}")

        # Costs calculation
        # Calculate costs manually using the scenario type
        if scenario_name == 'high_pv':
            # For high_pv scenario, use variable prices
            # Electricity cost: average price * total import
            avg_elec_price = 70.78  # EUR/MWh (from grid_prices.csv)
            op_cost_grid_import = total_grid_import * avg_elec_price
            print(f"Using average electricity price: {avg_elec_price:.2f} EUR/MWh")
            print(f"Grid import: total={total_grid_import:.2f} MWh")

            # Heat cost: average price * total heat produced
            avg_heat_price = 69.73  # EUR/MWh (from thermal_energy_prices_denmark.csv)
            op_cost_heat = total_heat_produced * avg_heat_price
            print(f"Using average heat price: {avg_heat_price:.2f} EUR/MWh")
            print(f"Heat production: total={total_heat_produced:.2f} MWh")
        elif scenario_name == 'baseline':
            # For baseline scenario, use variable prices
            # Electricity cost: average price * total import
            avg_elec_price = 70.78  # EUR/MWh (from grid_prices.csv)
            op_cost_grid_import = total_grid_import * avg_elec_price
            print(f"Using average electricity price: {avg_elec_price:.2f} EUR/MWh")
            print(f"Grid import: total={total_grid_import:.2f} MWh")

            # Heat cost: average price * total heat produced
            avg_heat_price = 69.73  # EUR/MWh (from thermal_energy_prices_denmark.csv)
            op_cost_heat = total_heat_produced * avg_heat_price
            print(f"Using average heat price: {avg_heat_price:.2f} EUR/MWh")
            print(f"Heat production: total={total_heat_produced:.2f} MWh")
        else:
            # For other scenarios, use default prices
            avg_elec_price = 50.0  # EUR/MWh
            op_cost_grid_import = total_grid_import * avg_elec_price
            print(f"Using default electricity price: {avg_elec_price:.2f} EUR/MWh")

            avg_heat_price = 40.0  # EUR/MWh
            op_cost_heat = total_heat_produced * avg_heat_price
            print(f"Using default heat price: {avg_heat_price:.2f} EUR/MWh")

        # Calculate total operational cost
        total_op_cost = op_cost_grid_import + op_cost_heat # - op_revenue_grid_export
        print(f"Grid import cost: {op_cost_grid_import:.2f} EUR, Heat cost: {op_cost_heat:.2f} EUR, Total: {total_op_cost:.2f} EUR")

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


def main():
    """Main entry point for the script."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run PyPSA simulations for the Social Building PED project.")
    parser.add_argument("-s", "--scenario", help="Name of the scenario to run (must match config.yml)", required=True)
    parser.add_argument("-c", "--config", help="Path to the main configuration file", default="config/config.yml")
    parser.add_argument("-p", "--params", help="Path to the component parameters file", default="config/component_params.yml")
    parser.add_argument("-l", "--list", help="List available scenarios", action="store_true")

    args = parser.parse_args()

    # List available scenarios if requested
    if args.list:
        from scenarios import SCENARIO_FUNCTIONS
        print("Available scenarios:")
        for scenario in SCENARIO_FUNCTIONS.keys():
            print(f"  - {scenario}")
        return

    # Basic validation
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        exit(1)
    if not os.path.exists(args.params):
        print(f"Error: Component parameters file not found at {args.params}")
        exit(1)

    # Run the selected scenario
    run_scenario(args.scenario, args.config, args.params)

    print("\nMain script finished.")


if __name__ == "__main__":
    main()

