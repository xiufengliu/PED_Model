#!/usr/bin/env python3
"""
Optimization utilities for the PED Lyngby Model.

This module contains functions for optimizing the operation of the PED Lyngby Model.
"""

import numpy as np
import pandas as pd


def calculate_operational_costs(network):
    """
    Calculate operational costs for a PyPSA network.

    Args:
        network (pypsa.Network): The PyPSA network

    Returns:
        dict: Dictionary with operational costs
    """
    # Initialize results dictionary
    costs = {
        'total': 0,
        'by_carrier': {},
        'by_component': {}
    }

    # Calculate costs by carrier
    for carrier in network.carriers.index:
        carrier_gens = network.generators[network.generators.carrier == carrier].index
        if len(carrier_gens) > 0:
            # Sum the marginal costs * generation for all generators of this carrier
            carrier_cost = sum(
                network.generators_t.p[gen].sum() * network.generators.at[gen, 'marginal_cost']
                for gen in carrier_gens
            )
            costs['by_carrier'][carrier] = carrier_cost
            costs['total'] += carrier_cost

    # Calculate costs by component
    for component in ['Generator', 'StorageUnit', 'Store', 'Link']:
        if component in network.components and len(getattr(network, component.lower() + 's')) > 0:
            component_df = getattr(network, component.lower() + 's')
            if 'marginal_cost' in component_df.columns:
                for idx in component_df.index:
                    if component == 'Generator':
                        # For generators, cost is marginal_cost * generation
                        if idx in network.generators_t.p:
                            cost = network.generators_t.p[idx].sum() * component_df.at[idx, 'marginal_cost']
                            costs['by_component'][idx] = cost
                    elif component == 'StorageUnit':
                        # For storage, cost might be related to cycles or throughput
                        # This is a simplified placeholder
                        if idx in network.storage_units_t.p_dispatch:
                            cost = network.storage_units_t.p_dispatch[idx].sum() * component_df.at[idx, 'marginal_cost']
                            costs['by_component'][idx] = cost
                    elif component == 'Link':
                        # For links, cost might be related to throughput
                        # This is a simplified placeholder
                        if idx in network.links_t.p0:
                            cost = network.links_t.p0[idx].sum() * component_df.at[idx, 'marginal_cost']
                            costs['by_component'][idx] = cost

    return costs


def calculate_self_sufficiency(network):
    """
    Calculate self-sufficiency metrics for a PyPSA network.

    Args:
        network (pypsa.Network): The PyPSA network

    Returns:
        dict: Dictionary with self-sufficiency metrics
    """
    # Initialize results dictionary
    results = {
        'total_demand': 0,
        'local_generation': 0,
        'grid_import': 0,
        'grid_export': 0,
        'self_sufficiency': 0,
        'self_consumption': 0
    }

    # Calculate total electrical demand
    for load in network.loads.index:
        if network.loads.at[load, 'carrier'] == 'electricity':
            results['total_demand'] += network.loads_t.p_set[load].sum()

    # Calculate local generation (excluding grid imports)
    for gen in network.generators.index:
        if gen != 'Grid' and network.generators.at[gen, 'carrier'] == 'electricity':
            results['local_generation'] += network.generators_t.p[gen].sum()

    # Calculate grid import/export
    if 'Grid' in network.generators.index:
        grid_series = network.generators_t.p['Grid']
        results['grid_import'] = grid_series[grid_series > 1e-3].sum()
        results['grid_export'] = -grid_series[grid_series < -1e-3].sum()

    # Calculate self-sufficiency and self-consumption
    if results['total_demand'] > 0:
        results['self_sufficiency'] = min(results['local_generation'] / results['total_demand'], 1.0)

    if results['local_generation'] > 0:
        results['self_consumption'] = min((results['local_generation'] - results['grid_export']) / results['local_generation'], 1.0)

    return results


def optimize_battery_operation(load_profile, pv_profile, battery_capacity_kwh, battery_power_kw,
                              grid_import_cost=0.15, grid_export_price=0.05, efficiency=0.9):
    """
    Optimize battery operation to minimize costs.

    Args:
        load_profile (pd.Series): Load profile in kW
        pv_profile (pd.Series): PV generation profile in kW
        battery_capacity_kwh (float): Battery capacity in kWh
        battery_power_kw (float): Battery power in kW
        grid_import_cost (float): Cost of importing from the grid in €/kWh
        grid_export_price (float): Price for exporting to the grid in €/kWh
        efficiency (float): Battery round-trip efficiency

    Returns:
        pd.DataFrame: Optimized battery operation
    """
    # This is a simplified placeholder function
    # In a real implementation, you would use an optimization library
    # like pyomo or cvxpy to solve this problem

    # Calculate net load (load - PV)
    net_load = load_profile - pv_profile

    # Initialize results
    n_steps = len(net_load)
    battery_energy = np.zeros(n_steps + 1)  # +1 to include final state
    battery_charge = np.zeros(n_steps)
    battery_discharge = np.zeros(n_steps)
    grid_import = np.zeros(n_steps)
    grid_export = np.zeros(n_steps)

    # Simple rule-based strategy (not optimal):
    # - Charge battery when PV > load (excess PV)
    # - Discharge battery when load > PV (deficit)
    for t in range(n_steps):
        if net_load[t] < 0:  # Excess PV
            # Charge battery with excess PV
            battery_charge[t] = min(-net_load[t], battery_power_kw,
                                   (battery_capacity_kwh - battery_energy[t]) / efficiency)
            # Export remaining excess
            grid_export[t] = -net_load[t] - battery_charge[t]
        else:  # Deficit
            # Discharge battery to meet deficit
            battery_discharge[t] = min(net_load[t], battery_power_kw,
                                     battery_energy[t] * efficiency)
            # Import remaining deficit
            grid_import[t] = net_load[t] - battery_discharge[t]

        # Update battery energy
        battery_energy[t+1] = battery_energy[t] + battery_charge[t] * efficiency - battery_discharge[t] / efficiency

    # Calculate costs
    total_cost = grid_import.sum() * grid_import_cost - grid_export.sum() * grid_export_price

    # Create results DataFrame
    results = pd.DataFrame({
        'load': load_profile.values,
        'pv': pv_profile.values,
        'net_load': net_load.values,
        'battery_charge': battery_charge,
        'battery_discharge': battery_discharge,
        'battery_energy': battery_energy[:-1],  # Remove the final state
        'grid_import': grid_import,
        'grid_export': grid_export
    }, index=load_profile.index)

    # Add cost as an attribute
    results.attrs['total_cost'] = total_cost

    return results


if __name__ == "__main__":
    # Example usage
    print("This module is not meant to be run directly.")
    print("Import it and use its functions in your scripts.")