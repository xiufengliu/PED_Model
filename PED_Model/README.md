# PED Social Building Model

## Overview
This repository contains a Positive Energy District (PED) model for a social building with 165 apartments. The model simulates the energy flows, costs, and CO2 emissions for the building over a full year (8760 hours) starting from January 1, 2025.

## Structure
The repository is organized as follows:

- `BASELINE/`: Contains the baseline scenario implementation
  - `config/`: Configuration files for the model
  - `data/`: Input and output data
    - `input/`: Input data for the model
      - `timeseries/`: Time series data (electricity demand, heat demand, PV generation, grid prices)
    - `output/`: Output data generated by the model
  - `scripts/`: Python scripts for running the model
  - `visualizations/`: Visualizations generated by the model

## Baseline Scenario
The baseline scenario represents the current energy situation of the social building. It includes:

- Electricity demand for 165 apartments
- Heat demand for 165 apartments
- PV generation from a 62 kW system
- Grid electricity prices
- CO2 emissions from electricity and heat consumption

## How to Run
To run the baseline scenario, execute the following command:

```bash
python BASELINE/scripts/demo_scenarios.py
```

## Results
The model generates the following results:

- Energy flows (electricity demand, heat demand, PV generation, grid import)
- Energy costs (electricity costs, heat costs)
- CO2 emissions (from electricity and heat consumption)
- Self-consumption and self-sufficiency metrics

## Visualizations
The model generates the following visualizations:

- Monthly electricity demand, PV generation, and grid import
- Monthly heat demand
- Monthly self-consumption and self-sufficiency
- Energy sources pie chart
- Monthly electricity prices
- CO2 emissions pie chart
- Monthly CO2 emissions

## Data Sources
- Electricity and heat demand data are based on realistic profiles for a social building with 165 apartments
- PV generation data is based on weather data from the Danish Meteorological Institute
- Grid electricity prices are based on historical data from the Danish electricity market
- CO2 emission factors are based on realistic values for Denmark

## Requirements
- Python 3.8 or higher
- pandas
- numpy
- matplotlib
- PyYAML
