# PED_Lyngby_Model

Simulation model for the Positive Energy District concept in Lyngby, featuring components like a stadium, swimming pool, solar PV, heating systems, and energy storage.

## Project Goal

[Describe the main objectives of this modeling project. e.g., To evaluate the technical feasibility and economic viability of different technology configurations and operational strategies for achieving a positive energy balance in the specified Lyngby district.]

## Structure

* `/config`: Configuration files (model parameters, scenario settings).
* `/data`: Input data and simulation outputs.
    * `/data/input`: Raw and processed input data (time series, parameters).
    * `/data/output`: Simulation results (ignored by Git by default).
* `/environment.yml`: Conda environment specification.
* `/notebooks`: Jupyter notebooks for analysis, visualization, workflow steps.
* `/scripts`: Core Python scripts for the simulation model.
* `/tests`: Optional directory for automated tests.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd PED_Lyngby_Model
    ```
2.  **Create Conda Environment:** Requires Anaconda or Miniconda.
    ```bash
    conda env create -f environment.yml
    conda activate PED_Lyngby_Model
    ```
    *Note: You may need to install a suitable optimization solver separately (e.g., HiGHS, CBC, Gurobi). Check `environment.yml` and PyPSA documentation.*
3.  **Prepare Data:** Place necessary input data files for Lyngby (weather, loads for stadium/pool/general, PV potential, grid prices, etc.) in the `data/input/` subdirectories. Update configuration files (`component_params.yml`, `config.yml`) in `/config` as needed with realistic parameters.

## How to Run

The main script `scripts/main.py` is used to run simulations for specific scenarios defined in `config/config.yml`.

Example command to run the 'baseline' scenario:

```bash
python scripts/main.py --scenario baseline --config config/config.yml --params config/component_params.yml
```

* Replace `baseline` with the desired scenario name (see list below).
* The `--config` and `--params` arguments default to the standard file paths but can be overridden.

## Simulation Scenarios

The following scenarios are defined or suggested for analysis (scenario names used in `--scenario` argument are indicative):

1.  **`baseline`**:
    * **Description:** Reference case representing the current situation or a minimal setup with existing loads (stadium, pool, general) and minimal local assets. Relies heavily on grid import for electricity and a primary heat source (e.g., district heating or boiler).
    * **Purpose:** Establish reference energy consumption, costs, grid interaction, and emissions.

2.  **`high_pv`**:
    * **Description:** Explores maximizing solar PV deployment on available surfaces within the district. Storage and heating systems might remain at baseline levels.
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

*Modify `config/config.yml` and `config/component_params.yml` to reflect the specific parameters for each scenario you wish to run. You may need to create additional Python functions in `scripts/` to build the network for more complex scenarios.*

## Contributing

[Add contribution guidelines if applicable, e.g., how to report bugs, suggest features, or submit code changes.]

## License

[Specify the chosen license, e.g., MIT License. See LICENSE file.]

## Contact

[Your Name/Organization and contact information.]
```