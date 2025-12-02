# PED Lyngby Model - Scenarios Guide

This document provides a guide to the scenarios available in the PED Lyngby Model and how to create new ones.

## Available Scenarios

The following scenarios are currently implemented:

1. **baseline**
   - Reference case with existing loads and minimal local assets
   - Relies heavily on grid import for electricity and a primary heat source
   - Includes minimal PV capacity (50 kW)

2. **high_pv**
   - Maximizes solar PV deployment on available surfaces within the district
   - Includes distributed PV systems at the stadium (200 kW), pool (150 kW), and general district (300 kW)
   - Total PV capacity: 650 kW

## Running Scenarios

To run a scenario, use the main script:

```bash
# List available scenarios
python scripts/main.py --list

# Run a specific scenario
python scripts/main.py --scenario baseline
python scripts/main.py --scenario high_pv
```

## Creating New Scenarios

### Using the Scenario Creation Script

The easiest way to create a new scenario is to use the provided script:

```bash
python scripts/create_scenario.py pv_battery "Combines high PV penetration with electrical battery storage"
```

This will:
1. Create a new scenario file in the `scenarios` directory
2. Update the necessary configuration files
3. Provide guidance on next steps to implement your scenario

### Manual Creation

If you prefer to create a scenario manually:

1. Create a new Python file in the `scenarios` directory (e.g., `pv_battery.py`)
   - Use `scenarios/template.py` as a starting point
   - Implement the `create_network` function

2. Add your scenario to the `SCENARIO_FUNCTIONS` dictionary in `scenarios/__init__.py`:
   ```python
   SCENARIO_FUNCTIONS = {
       'baseline': baseline.create_network,
       'high_pv': high_pv.create_network,
       'pv_battery': pv_battery.create_network,  # Add your scenario here
   }
   ```

3. Add your scenario parameters to `config/component_params.yml`:
   ```yaml
   # PV Battery Scenario Assets
   pv_battery:
     # Stadium rooftop PV
     stadium_pv_capacity_kw: 200.0
     # Swimming pool rooftop PV
     pool_pv_capacity_kw: 150.0
     # General district PV
     general_pv_capacity_kw: 300.0
     # Battery storage
     battery_capacity_kwh: 500.0
     battery_power_kw: 250.0
     battery_efficiency: 0.9
     # All PV installations use the same generation profile
     marginal_cost: 0
   ```

4. Add your scenario configuration to `config/config.yml`:
   ```yaml
   scenarios:
     # ... existing scenarios ...
     pv_battery:
       description: "Combines high PV penetration with electrical battery storage"
       output_subdir: "scenario_pv_battery"
   ```

## Scenario Implementation Guidelines

1. Each scenario should be implemented in its own module
2. Use the `utils.py` module for common functionality
3. Each scenario module should have a `create_network` function with the following signature:
   ```python
   def create_network(config_file, params_file, data_path):
       # Implementation
       return network
   ```
4. Use descriptive names for components to make results analysis easier
5. Document the key differences from the baseline scenario in the module docstring

## Suggested Future Scenarios

The following scenarios are suggested for future implementation:

1. **pv_battery**
   - Combines high PV penetration with electrical battery storage
   - Different battery sizes can be tested

2. **thermal_storage**
   - Focuses on optimizing the heating system by incorporating thermal storage
   - Potentially coupled with heat pumps or optimized district heating interaction

3. **dsm** (Demand-Side Management)
   - Simulates the impact of demand-side management by shifting flexible loads
   - Based on local generation or price signals

4. **cost_opt** / **self_suff_max**
   - Compares operational strategies for a given asset configuration
   - Requires adjusting optimization objective

5. **future_prices**
   - Tests a chosen configuration against different future energy price assumptions
   - Sensitivity analysis for economic robustness

6. **invest_opt**
   - Allows the model to determine the optimal capacities of components
   - Requires multi-period investment setup in PyPSA
