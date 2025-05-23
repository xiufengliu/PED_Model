# PED Lyngby Model - Scenarios Package

This package contains the different scenario implementations for the PED Lyngby Model.

## Available Scenarios

1. **baseline**: Reference case with existing loads and minimal local assets. Uses variable electricity prices from `grid_prices.csv` instead of a fixed import cost.
2. **high_pv**: Maximized solar PV deployment on available surfaces within the district. Also uses variable electricity prices.

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

## Variable Electricity Prices

The model uses variable electricity prices from the file `grid_prices.csv` instead of a fixed import cost. This is configured in the `component_params.yml` file:

```yaml
grid:
  capacity_mw: 10.0
  import_cost_eur_per_mwh: variable  # Using variable electricity prices from grid_prices.csv
  price_profile: grid_prices.csv  # File containing hourly electricity prices
  export_price_eur_per_mwh: 20.0
  transformer_efficiency: 0.98
```

The variable prices are loaded in the `utils.py` file using the `load_electricity_price_profile` function:

```python
def load_electricity_price_profile(data_dir, index):
    """
    Load electricity price profile from CSV file.
    """
    filepath = os.path.join(data_dir, 'timeseries', 'grid_prices.csv')
    try:
        price_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        price_profile = price_df.iloc[:, 0].reindex(index).fillna(50.0)
        return price_profile
    except FileNotFoundError:
        print(f"Warning: Electricity price file not found '{filepath}'. Using default constant price.")
        return pd.Series(50.0, index=index)  # Default price: 50 EUR/MWh
```

These prices are then used as the marginal cost for the grid generator in the `setup_basic_network` function:

```python
network.add("Generator", "Grid",
            bus="Grid Connection",
            carrier="electricity",
            marginal_cost=price_profile,  # Use variable price profile
            p_nom=grid_capacity,
            p_nom_extendable=True,
            p_nom_max=20.0,
            p_min_pu=-1,
            p_max_pu=1)
```

This allows the model to account for the time-varying nature of electricity prices, which is important for realistic economic analysis.
