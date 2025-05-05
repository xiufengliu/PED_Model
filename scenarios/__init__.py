"""
PED Lyngby Model - Scenarios Package

This package contains the different scenario implementations for the PED Lyngby Model.
Each scenario is implemented in its own module.
"""

# Import all scenario modules to make them available when importing the package
from . import baseline
from . import high_pv

# Dictionary mapping scenario names to their network creation functions
SCENARIO_FUNCTIONS = {
    'baseline': baseline.create_network,
    'high_pv': high_pv.create_network,
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
