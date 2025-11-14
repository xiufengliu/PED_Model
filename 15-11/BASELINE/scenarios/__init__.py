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
from . import thermal_storage_COP1
from . import thermal_storage_COP1_HP
from . import thermal_storage_COP2
from . import thermal_storage_COP3
from . import dsm
from . import dsm_HP
from . import dsm_HP_th_priority
from . import dsm_HP_el_priority


from . import self_opt
from . import self_opt_HP
from . import cost_opt
from . import cost_opt_HP
from . import future_prices_cost_opt
from . import future_prices_self_opt
from . import invest_opt





# Dictionary mapping scenario names to their network creation functions
SCENARIO_FUNCTIONS = {
    'baseline': baseline.create_network,
    'high_pv': high_pv.create_network,
    'pv_battery_1': pv_battery_1.create_network,
    'pv_battery_2': pv_battery_2.create_network,
    'pv_battery_3': pv_battery_3.create_network,
    'thermal_storage_COP1': thermal_storage_COP1.create_network,
    'thermal_storage_COP1_HP': thermal_storage_COP1_HP.create_network,
    'thermal_storage_COP2': thermal_storage_COP2.create_network,
    'thermal_storage_COP3': thermal_storage_COP3.create_network,
    'dsm': dsm.create_network,
    'dsm_HP': dsm_HP.create_network,
    'dsm_HP_th_priority': dsm_HP_th_priority.create_network,
    'dsm_HP_el_priority': dsm_HP_el_priority.create_network,
    'self_opt': self_opt.create_network,
    'self_opt_HP': self_opt_HP.create_network,
    'cost_opt': cost_opt.create_network,
    'cost_opt_HP': cost_opt_HP.create_network,
    'future_prices_cost_opt': future_prices_cost_opt.create_network,
    'future_prices_self_opt': future_prices_self_opt.create_network,
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
