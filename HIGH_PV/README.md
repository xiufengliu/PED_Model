# PED_Social_Building_Model

Simulation model for the Positive Energy District concept for a social building in Copenhagen, featuring components like a social building with 165 apartments, solar PV, heating systems, and energy storage.

## Project Goal

To evaluate the technical feasibility and economic viability of different technology configurations and operational strategies for achieving a positive energy balance in a social building with 165 apartments in Copenhagen.

## Structure

* `/config`: Configuration files (model parameters, scenario settings).
    * `config.yml`: Main configuration file with simulation settings and scenario definitions.
    * `component_params.yml`: Parameters for energy system components, including:
        * Grid connection parameters (capacity, import/export prices, efficiency)
        * PV system parameters (capacity, marginal cost)
        * Heat source parameters (type, capacity, cost, efficiency)
        * Social building parameters (load profiles, number of apartments)
* `/data`: Input data and simulation outputs.
    * `/data/input`: Raw and processed input data (time series, parameters).
        * `/data/input/timeseries`: Time series data including:
            * `electricity_demand.csv`: Hourly electricity demand for the social building (kW).
            * `heat_demand.csv`: Hourly heat demand for the social building (kW).
            * `solar_pv_generation.csv`: Hourly PV generation profile (normalized to 1 kW capacity).
            * `grid_prices.csv`: Hourly electricity prices from the grid (EUR/MWh).
    * `/data/output`: Simulation results (ignored by Git by default).
        * `/data/output/scenario_baseline`: Results for the baseline scenario.
            * `baseline_hourly_results.csv`: Hourly results including energy flows and costs.
            * `baseline_summary.csv`: Summary statistics for the entire simulation period.
    * `/visualizations`: Visualizations of simulation results.
* `/environment.yml`: Conda environment specification.
* `/notebooks`: Jupyter notebooks for analysis, visualization, workflow steps.
* `/scenarios`: Scenario implementations, each in its own module.
    * `__init__.py`: Module initialization file that registers available scenarios.
    * `baseline.py`: Implementation of the baseline scenario.
    * `high_pv.py`: Implementation of the high PV scenario.
    * `template.py`: Template for creating new scenario implementations.
    * `utils.py`: Utility functions used across different scenarios, including:
        * Functions for loading configuration and data
        * Functions for setting up the basic network structure
        * Functions for loading electricity price profiles
* `/scripts`: Core Python scripts for the simulation model.
    * `main.py`: Main script for running scenario optimizations.
    * `simulate_baseline_annual.py`: Script for simulating the baseline scenario without optimization.
    * `create_scenario.py`: Script for creating new scenario files.
    * `data_processing.py`: Utilities for processing input data.
    * `plotting_utils.py`: Utilities for creating visualizations.
* `/tests`: Optional directory for automated tests.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd PED_Social_Building_Model
    ```
2.  **Create Conda Environment:** Requires Anaconda or Miniconda.
    ```bash
    conda env create -f environment.yml
    conda activate PED_Social_Building_Model
    ```
    *Note: You may need to install a suitable optimization solver separately (e.g., HiGHS, CBC, Gurobi). Check `environment.yml` and PyPSA documentation.*
3.  **Prepare Data:** Place necessary input data files for Copenhagen (weather, electricity and heat demand for the social building, PV potential, grid prices, etc.) in the `data/input/` subdirectories. Update configuration files (`component_params.yml`, `config.yml`) in `/config` as needed with realistic parameters.
    * **Note:** The model uses variable electricity prices from `data/input/timeseries/grid_prices.csv`. This file should contain hourly electricity prices in EUR/MWh.

## Model Description

### Baseline Scenario

The baseline scenario represents the current energy system of a social building with 165 apartments in Copenhagen. The key components of this scenario are:

1. **Social Building**: A residential building with 165 apartments, with electricity and heat demand profiles based on realistic data for Copenhagen.

2. **Grid Connection**: Connection to the electricity grid with variable import prices based on hourly data from the Danish electricity market. The grid connection has a capacity of 10 MW, which can be extended if necessary. The variable electricity prices are read from the file `grid_prices.csv` and used instead of a fixed import cost.

3. **Existing PV Installation**: A small solar PV installation with a capacity of 15 kW, representing the existing renewable energy generation on the building.

4. **Heat Source**: A gas boiler with a capacity of 2 MW and an efficiency of 90%, providing heat for the building.

The baseline scenario uses realistic constraints and may result in an infeasible optimization problem, which accurately represents the limitations of the current energy system. To analyze the baseline scenario even when optimization is infeasible, a simulation approach (without optimization) is provided in `scripts/simulate_baseline_annual.py`.

#### Capacity Constraints

The model includes the following capacity constraints:

1. **Grid Connection**:
   - Nominal capacity: 10 MW
   - Extendable: Yes
   - Maximum capacity: 20 MW
   - Import cost: Variable (from `grid_prices.csv`)
   - Export price: 20 EUR/MWh (fixed)
   - These values are set high enough to ensure the model can find a feasible solution if one exists.

2. **Heat Source (Gas Boiler)**:
   - Nominal capacity: 2 MW
   - Extendable: Yes
   - Maximum capacity: 10 MW
   - Efficiency: 90%
   - The capacity is set to ensure it can meet the maximum heat demand.

3. **PV Installation**:
   - Capacity: 15 kW (fixed, not extendable)
   - This represents the existing small-scale PV installation on the building.

4. **Building Connections**:
   - Nominal capacity: 10 MW
   - Extendable: Yes
   - Maximum capacity: 100 MW
   - These connections link the social building to the district electricity and heat buses.

These capacity constraints are designed to be realistic while still allowing the model to find a feasible solution if one exists. The fact that most capacities are extendable means that capacity constraints should not be the primary cause of infeasibility in the optimization problem.

### Simulation vs. Optimization

The model supports two approaches:

1. **Optimization**: Using PyPSA's optimization capabilities to find the optimal operation of the energy system, minimizing total system costs while satisfying all constraints. This is the approach used by the main script `scripts/main.py`.

2. **Simulation**: A simpler approach that simulates the operation of the energy system without optimization, using predefined rules for dispatch. This is the approach used by `scripts/simulate_baseline_annual.py` for the baseline scenario.

The simulation approach is particularly useful for the baseline scenario, as it allows us to analyze the current energy system even when the optimization problem is infeasible due to realistic constraints.

## How to Run

The main script `scripts/main.py` is used to run optimizations for specific scenarios defined in `config/config.yml`. For the baseline scenario, you can also use `scripts/simulate_baseline_annual.py` to run a simulation without optimization.

Example commands:

```bash
# List available scenarios
python scripts/main.py --list

# Run the baseline scenario
python scripts/main.py --scenario baseline

# Run the high PV scenario
python scripts/main.py --scenario high_pv

# Run with custom config and params files
python scripts/main.py --scenario baseline --config config/config.yml --params config/component_params.yml

# Run the baseline simulation (without optimization)
python scripts/simulate_baseline_annual.py
```

* The `--config` and `--params` arguments default to the standard file paths but can be overridden.

### Creating New Scenarios

You can create a new scenario using the provided script:

```bash
python scripts/create_scenario.py pv_battery "Combines high PV penetration with electrical battery storage"
```

This will:
1. Create a new scenario file in the `scenarios` directory
2. Update the necessary configuration files
3. Provide guidance on next steps to implement your scenario

## Simulation Scenarios

The following scenarios are defined or suggested for analysis (scenario names used in `--scenario` argument are indicative):

1.  **`baseline`**:
    * **Description:** Reference case representing the current situation or a minimal setup with existing loads (social building with 165 apartments) and minimal local assets. Relies heavily on grid import for electricity (with variable prices from grid_prices.csv) and a primary heat source (gas boiler).
    * **Purpose:** Establish reference energy consumption, costs, grid interaction, and emissions.
    * **Note:** This scenario uses realistic constraints and may result in an infeasible optimization problem, which accurately represents the limitations of the baseline scenario. A simulation approach (without optimization) is provided in `scripts/simulate_baseline_annual.py` to analyze the baseline scenario even when optimization is infeasible.

2.  **`high_pv`**:
    * **Description:** Explores maximizing solar PV deployment on available surfaces of the social building. Storage and heating systems might remain at baseline levels.
    * **Purpose:** Assess maximum local renewable electricity generation, self-consumption potential without significant storage, and increased grid export.

3.  **`pv_battery`**:
    * **Description:** Combines high PV penetration with significant electrical battery storage. Different battery sizes can be tested.
    * **Purpose:** Quantify the impact of batteries on increasing PV self-consumption, reducing grid dependency (especially peak imports), and potentially enabling peak shaving.

4.  **`thermal_storage`**:
    * **Description:** Focuses on optimizing the heating system by incorporating thermal storage (e.g., hot water tanks) potentially coupled with heat pumps or optimized district heating interaction.
    * **Purpose:** Evaluate flexibility in the heating sector, potential for sector coupling (using renewable electricity for heat), and reducing peak heat demand.

5.  **`dsm`**:
    * **Description:** Simulates the impact of demand-side management by shifting flexible electrical or thermal loads (e.g., EV charging, pre-heating/cooling, specific pool/stadium operations) based on local generation or price signals.
    * **Purpose:** Assess the value of load flexibility in reducing costs and integrating variable renewables.

6.  **`cost_opt` / `self_suff_max`**: (Requires adjusting optimization objective)
    * **Description:** Compares operational strategies for a given asset configuration. `cost_opt` minimizes total operational monetary costs. `self_suff_max` modifies the objective to prioritize using local resources, potentially at a slightly higher monetary cost.
    * **Purpose:** Understand the trade-off between economic efficiency and energy independence/self-sufficiency.

7.  **`future_prices`**: (Sensitivity analysis)
    * **Description:** Tests a chosen configuration (e.g., `pv_battery`) against different future energy price assumptions (e.g., higher carbon taxes, different grid tariff structures).
    * **Purpose:** Assess the economic robustness of the PED design under uncertain market conditions.

8.  **`invest_opt`**: (Requires multi-period investment setup in PyPSA)
    * **Description:** Allows the model to determine the optimal capacities of PV, batteries, thermal storage, etc., by minimizing total system costs (operational + annualized investment) over the project lifetime.
    * **Purpose:** Identify the most cost-effective mix and sizing of technologies from an investment perspective.

*Currently, only the `baseline` and `high_pv` scenarios are fully implemented. Use the scenario creation script and follow the modular structure to implement additional scenarios.*

## Simulation Results

The simulation results are stored in the `/data/output` directory, with a subdirectory for each scenario. For the baseline scenario, the results are in `/data/output/scenario_baseline`.

### Baseline Scenario Results

The baseline simulation (`scripts/simulate_baseline_annual.py`) produces the following results:

1. **Hourly Results**: A CSV file (`baseline_hourly_results.csv`) containing hourly values for:
   - Electricity demand (MW)
   - Heat demand (MW)
   - PV generation (MW)
   - Grid import (MW)
   - Grid export (MW)
   - Heat production (MW)
   - Electricity price (EUR/MWh)
   - Electricity cost (EUR)
   - Heat cost (EUR)

2. **Summary Statistics**: A CSV file (`baseline_summary.csv`) containing summary statistics for the entire simulation period:
   - Total electricity demand (MWh/year)
   - Total heat demand (MWh/year)
   - Total PV generation (MWh/year)
   - Total grid import (MWh/year)
   - Total grid export (MWh/year)
   - Total heat production (MWh/year)
   - Average electricity price (EUR/MWh)
   - Total electricity cost (EUR/year)
   - Total heat cost (EUR/year)
   - Total energy cost (EUR/year)
   - PV self-consumption (%)
   - Electricity self-sufficiency (%)
   - Grid Import CO2 Emissions (tonnes CO2/year)
   - Heat CO2 Emissions (tonnes CO2/year)
   - Total CO2 Emissions (tonnes CO2/year)
   - CO2 Emissions per Apartment (tonnes CO2/apartment/year)

3. **Visualizations**: Several PNG files in the `/visualizations` directory:
   - `monthly_electricity.png`: Monthly electricity demand, PV generation, and grid import
   - `monthly_heat.png`: Monthly heat demand and production
   - `monthly_costs.png`: Monthly electricity and heat costs
   - `monthly_electricity_prices.png`: Monthly average electricity prices

### High PV Scenario Results

The High PV scenario optimization (`scripts/main.py --scenario high_pv`) produces similar results:

1. **Summary Statistics**: A CSV file (`high_pv_summary_complete.csv`) containing the same summary statistics as the baseline scenario, but for the High PV scenario.

2. **Comparative Analysis**: A comprehensive analysis of the High PV scenario compared to the Baseline scenario is available in the `summary/summary.md` file.

3. **Visualizations**: Several PNG files in the `data/output/plots` directory:
   - `baseline_vs_highpv.png`: Comparison of key metrics between Baseline and High PV scenarios
   - `savings_analysis.png`: Analysis of cost savings and payback period
   - `environmental_impact.png`: Analysis of CO2 emissions reduction
   - `self_consumption.png`: Analysis of electricity self-sufficiency
   - `pv_seasonal_comparison.png`: Seasonal comparison of PV generation
   - `pv_monthly_generation.png`: Monthly PV generation
   - `pv_weather_day_2025-01-15.png`: Detailed analysis of a winter day
   - `pv_weather_day_2025-07-15.png`: Detailed analysis of a summer day

These results provide a comprehensive overview of the energy system's performance, including energy flows, costs, and key performance indicators.

### Variable Electricity Prices

The model uses variable electricity prices from the file `grid_prices.csv` instead of a fixed import cost. This is reflected in the configuration file `component_params.yml`, where the `import_cost_eur_per_mwh` parameter is set to "variable" and a new parameter `price_profile` is added to specify the file containing the hourly prices.

```yaml
grid:
  capacity_mw: 10.0
  import_cost_eur_per_mwh: variable  # Using variable electricity prices from grid_prices.csv
  price_profile: grid_prices.csv  # File containing hourly electricity prices
  export_price_eur_per_mwh: 20.0
  transformer_efficiency: 0.98
```

The variable prices are loaded in the `utils.py` file using the `load_electricity_price_profile` function and used as the marginal cost for the grid generator in the `setup_basic_network` function. This allows the model to account for the time-varying nature of electricity prices, which is important for realistic economic analysis.

## Contributing

[Add contribution guidelines if applicable, e.g., how to report bugs, suggest features, or submit code changes.]

## License

[Specify the chosen license, e.g., MIT License. See LICENSE file.]

## Contact

[Your Name/Organization and contact information.]