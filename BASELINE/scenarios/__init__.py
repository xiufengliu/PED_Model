"""
PED Social Building Model - Scenarios Package

This package contains the different scenario implementations for the PED Social Building Model.
Each scenario is implemented in its own module.
"""

# Import all scenario modules to make them available when importing the package
from . import baseline
from . import high_pv
from . import pv_battery_1
from . import pv_battery_2
from . import pv_battery_3
from . import thermal_storage
from . import dsm
from . import cost_self_1
from . import cost_self_2
from . import future_prices_1
from . import future_prices_2
from . import invest_opt





# Dictionary mapping scenario names to their network creation functions
SCENARIO_FUNCTIONS = {
    'baseline': baseline.create_network,
    'high_pv': high_pv.create_network,
    'pv_battery_1': pv_battery_1.create_network,
    'pv_battery_2': pv_battery_2.create_network,
    'pv_battery_3': pv_battery_3.create_network,
    'thermal_storage': thermal_storage.create_network,
    'dsm': dsm.create_network,
    'cost_self_1': cost_self_1.create_network,
    'cost_self_2': cost_self_2.create_network,
    'future_prices_1': future_prices_1.create_network,
    'future_prices_2': future_prices_2.create_network,
    'invest_opt': invest_opt.create_network,










    # Add more scenarios here as they are implemented
}

def get_scenario_function(scenario_name):
    """
    Get the network creation function for a given scenario name.

    Args:
        scenario_name (str): Name of the scenario (must match keys in SCENARIO_FUNCTIONS)

    Returns:
        function: The network creation function for the specified scenario

    Raises:
        ValueError: If the scenario name is not recognized
    """
    if scenario_name not in SCENARIO_FUNCTIONS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available scenarios: {list(SCENARIO_FUNCTIONS.keys())}")

    return SCENARIO_FUNCTIONS[scenario_name]
