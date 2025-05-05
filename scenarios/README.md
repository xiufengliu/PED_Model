# PED Lyngby Model - Scenarios Package

This package contains the different scenario implementations for the PED Lyngby Model.

## Available Scenarios

1. **baseline**: Reference case with existing loads and minimal local assets.
2. **high_pv**: Maximized solar PV deployment on available surfaces within the district.

## Adding New Scenarios

To add a new scenario:

1. Create a new Python file in this directory (e.g., `pv_battery.py`)
   - Use `template.py` as a starting point
   - Implement the `create_network` function

2. Add your scenario to the `SCENARIO_FUNCTIONS` dictionary in `__init__.py`:
   ```python
   SCENARIO_FUNCTIONS = {
       'baseline': baseline.create_network,
       'high_pv': high_pv.create_network,
       'your_scenario': your_scenario.create_network,  # Add your scenario here
   }
   ```

3. Add your scenario parameters to `config/component_params.yml`

4. Add your scenario configuration to `config/config.yml`:
   ```yaml
   scenarios:
     # ... existing scenarios ...
     your_scenario:
       description: "Description of your scenario"
       output_subdir: "scenario_your_scenario"
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
