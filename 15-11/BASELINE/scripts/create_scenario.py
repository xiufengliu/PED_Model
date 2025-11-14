#!/usr/bin/env python3
"""
Script to create a new scenario for the PED Lyngby Model.
This script creates the necessary files and updates the configuration for a new scenario.
"""

import os
import sys
import argparse
import shutil
import yaml
import traceback # Assicurati che traceback sia importato per traceback.print_exc()




def create_scenario(scenario_name, description):
    """Create a new scenario with the given name and description."""
    # Validate scenario name
    if not scenario_name.isalnum() and not '_' in scenario_name:
        print(f"Error: Scenario name must be alphanumeric (with underscores allowed). Got: {scenario_name}")
        return False
    
    
    # Check if scenario already exists
    scenario_file = os.path.join('scenarios', f"{scenario_name}.py")
    if os.path.exists(scenario_file):
        print(f"Error: Scenario '{scenario_name}' already exists at {scenario_file}")
        return False
    
    # Create scenario file from template
    template_file = os.path.join('scenarios', 'template.py')
    if not os.path.exists(template_file):
        print(f"Error: Template file not found at {template_file}")
        return False
    
    # Copy template to new scenario file
    shutil.copy(template_file, scenario_file)
    
    # Update the new scenario file
    with open(scenario_file, 'r') as f:
        content = f.read()
    
    # Replace template content with scenario-specific content
    content = content.replace("PED Lyngby Model - Template for New Scenarios", 
                             f"PED Lyngby Model - {scenario_name.replace('_', ' ').title()} Scenario")
    content = content.replace("This is a template file for creating new scenarios. Copy this file and modify it\nto implement a new scenario.", 
                             description)
    content = content.replace("Steps to create a new scenario:\n1. Copy this file to a new file named after your scenario (e.g., pv_battery.py)\n2. Implement the create_network function with your scenario-specific logic\n3. Add your scenario to the SCENARIO_FUNCTIONS dictionary in __init__.py\n4. Add your scenario parameters to component_params.yml\n5. Add your scenario configuration to config.yml", 
                             f"This module implements the {scenario_name.replace('_', ' ')} scenario for the PED Lyngby Model.")
    content = content.replace("Building new scenario network...", 
                             f"Building {scenario_name.replace('_', ' ')} network...")
    content = content.replace("# scenario_params = params.get('your_scenario_name', {})", 
                             f"# {scenario_name}_params = params.get('{scenario_name}', {{}})")
    content = content.replace("New scenario network build complete.", 
                             f"{scenario_name.replace('_', ' ').title()} network build complete.")
    
    # Write updated content back to file
    with open(scenario_file, 'w') as f:
        f.write(content)
    
    # Update __init__.py to include the new scenario
    init_file = os.path.join('scenarios', '__init__.py')
    if not os.path.exists(init_file):
        print(f"Error: __init__.py file not found at {init_file}")
        return False
    
    with open(init_file, 'r') as f:
        init_content = f.read()
    
    # Add import statement
    import_section_end = init_content.find("# Dictionary mapping scenario names")
    if import_section_end == -1:
        print("Error: Could not find import section in __init__.py")
        return False
    
    import_section = init_content[:import_section_end]
    rest_of_file = init_content[import_section_end:]
    
    # Add new import
    if f"from . import {scenario_name}" not in import_section:
        import_lines = import_section.strip().split('\n')
        last_import_line = import_lines[-1]
        new_import_line = f"from . import {scenario_name}"
        import_section = import_section.replace(last_import_line, 
                                               f"{last_import_line}\n{new_import_line}")
    
    # Add scenario to SCENARIO_FUNCTIONS dictionary
    dict_start = rest_of_file.find("SCENARIO_FUNCTIONS = {")
    dict_end = rest_of_file.find("}", dict_start)
    
    if dict_start == -1 or dict_end == -1:
        print("Error: Could not find SCENARIO_FUNCTIONS dictionary in __init__.py")
        return False
    
    dict_content = rest_of_file[dict_start:dict_end+1]
    dict_lines = dict_content.strip().split('\n')
    
    # Check if scenario already in dictionary
    if f"'{scenario_name}': {scenario_name}.create_network" in dict_content:
        print(f"Warning: Scenario '{scenario_name}' already in SCENARIO_FUNCTIONS dictionary")
    else:
        # Add new scenario to dictionary
        # Find the last line with a comma
        last_entry_line = None
        for i, line in enumerate(dict_lines):
            if ',' in line and '}' not in line:
                last_entry_line = i
        
        if last_entry_line is not None:
            # Insert new entry after the last entry
            new_entry = f"    '{scenario_name}': {scenario_name}.create_network,"
            dict_lines.insert(last_entry_line + 1, new_entry)
            
            # Reconstruct the dictionary
            new_dict_content = '\n'.join(dict_lines)
            rest_of_file = rest_of_file.replace(dict_content, new_dict_content)
    
    # Write updated content back to file
    with open(init_file, 'w') as f:
        f.write(import_section + rest_of_file)
    
    # Update config.yml to include the new scenario
    config_file = os.path.join('config', 'config.yml')
    if not os.path.exists(config_file):
        print(f"Error: Config file not found at {config_file}")
        return False
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add new scenario to config
    if 'scenarios' not in config:
        config['scenarios'] = {}
    
    if scenario_name in config['scenarios']:
        print(f"Warning: Scenario '{scenario_name}' already in config.yml")
    else:
        config['scenarios'][scenario_name] = {
            'description': description,
            'output_subdir': f"scenario_{scenario_name}"
        }
    
    # Write updated config back to file
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Update component_params.yml to include the new scenario parameters
    params_file = os.path.join('config', 'component_params.yml')
    if not os.path.exists(params_file):
        print(f"Error: Component parameters file not found at {params_file}")
        return False
    
    with open(params_file, 'r') as f:
        params_content = f.read()
    
    # Add comment for new scenario parameters
    if f"# {scenario_name.replace('_', ' ').title()} Scenario Assets" not in params_content:
        params_content += f"\n# {scenario_name.replace('_', ' ').title()} Scenario Assets\n{scenario_name}:\n  # Add your parameters here\n"
    
    # Write updated params back to file
    with open(params_file, 'w') as f:
        f.write(params_content)
    
    print(f"Successfully created new scenario: {scenario_name}")
    print(f"- Scenario file: {scenario_file}")
    print(f"- Updated __init__.py")
    print(f"- Updated config.yml")
    print(f"- Updated component_params.yml")
    print("\nNext steps:")
    print(f"1. Edit {scenario_file} to implement your scenario")
    print(f"2. Add your scenario parameters to config/component_params.yml")
    print(f"3. Run your scenario with: python scripts/main.py --scenario {scenario_name}")
    
    return True

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Create a new scenario for the PED Lyngby Model.")
    parser.add_argument("name", help="Name of the new scenario (alphanumeric with underscores)")
    parser.add_argument("description", help="Brief description of the scenario")
    
    args = parser.parse_args()
    
    # Ensure we're in the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    success = create_scenario(args.name, args.description)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
