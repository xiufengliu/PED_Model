#!/usr/bin/env python3

"""
Tool to create and run scenarios for the PED Lyngby Model.

Commands:
  create   Create a new scenario module and update configs.
  run      Execute an existing scenario end-to-end.
"""
import os
import sys
import argparse
import shutil
import yaml
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scenarios import get_scenario_function
from scripts.plotting_utils import (
    plot_time_series,
    plot_energy_balance,
    plot_energy_balance_breakdown,
    plot_pv_battery_breakdown,
    plot_thermal_breakdown,
    plot_dsm_thermal_breakdown,
)

HOURS_IN_YEAR = 8760


def create_scenario(scenario_name: str, description: str) -> bool:
    if not (scenario_name.replace('_', '').isalnum()):
        print(f"Error: Scenario name must be alphanumeric (underscores allowed). Got: {scenario_name}")
        return False

    scenarios_dir = Path('scenarios')
    scenario_file = scenarios_dir / f"{scenario_name}.py"
    template_file = scenarios_dir / 'template.py'

    if scenario_file.exists():
        print(f"Error: Scenario '{scenario_name}' already exists at {scenario_file}")
        return False
    if not template_file.exists():
        print(f"Error: Template file not found at {template_file}")
        return False

    shutil.copy(template_file, scenario_file)
    content = scenario_file.read_text()
    header = f"PED Lyngby Model - {scenario_name.replace('_', ' ').title()} Scenario"
    content = content.replace("PED Lyngby Model - Template for New Scenarios", header)
    content = content.replace(
        "This is a template file for creating new scenarios. Copy this file and modify it\nto implement a new scenario.",
        description
    )
    content = content.replace("Building new scenario network...",
                              f"Building {scenario_name.replace('_', ' ')} network...")
    scenario_file.write_text(content)

    init_file = scenarios_dir / '__init__.py'
    if not init_file.exists():
        print(f"Error: __init__.py not found at {init_file}")
        return False
    init_text = init_file.read_text()

    import_marker = "# Dictionary mapping scenario names"
    idx = init_text.find(import_marker)
    if idx == -1:
        print("Error: Could not find import section in __init__.py")
        return False
    before, after = init_text[:idx], init_text[idx:]
    import_line = f"from . import {scenario_name}\n"
    if import_line not in before:
        lines = before.rstrip().splitlines()
        lines.append(import_line.strip())
        before = "\n".join(lines) + "\n"

    dict_start = after.find("SCENARIO_FUNCTIONS = {")
    dict_end = after.find("}", dict_start)
    if dict_start == -1 or dict_end == -1:
        print("Error: Could not find SCENARIO_FUNCTIONS dictionary in __init__.py")
        return False
    dict_block = after[dict_start:dict_end+1]
    entry = f"    '{scenario_name}': {scenario_name}.create_network,"
    if entry not in dict_block:
        lines = dict_block.splitlines()
        lines.insert(-1, entry)
        new_block = "\n".join(lines)
        after = after.replace(dict_block, new_block)
    init_file.write_text(before + after)

    config_path = Path('config') / 'config.yml'
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return False
    config = yaml.safe_load(config_path.read_text())
    config.setdefault('scenarios', {})
    if scenario_name not in config['scenarios']:
        config['scenarios'][scenario_name] = {
            'description': description,
            'output_subdir': f"scenario_{scenario_name}"
        }
        config_path.write_text(yaml.dump(config, default_flow_style=False))

    params_path = Path('config') / 'component_params.yml'
    if not params_path.exists():
        print(f"Error: Component params file not found at {params_path}")
        return False
    params_text = params_path.read_text()
    header = f"# {scenario_name.replace('_', ' ').title()} Scenario Assets"
    if header not in params_text:
        params_text += f"\n{header}\n{scenario_name}:\n  # Add your parameters here\n"
        params_path.write_text(params_text)

    print(f"Successfully created new scenario: {scenario_name}")
    return True


def load_config(config_file: str) -> dict:
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def determine_paths(config: dict, config_file: str) -> tuple[str, str]:
    config_abs_path = Path(config_file).resolve()
    project_root = config_abs_path.parent.parent
    data_in = project_root / config.get('paths', {}).get('data_input', 'data/input')
    data_out = project_root / config.get('paths', {}).get('data_output', 'data/output')
    return str(data_in), str(data_out)


def override_simulation_hours(config: dict, hours: int = HOURS_IN_YEAR) -> dict:
    config['simulation_settings'] = config.get('simulation_settings', {})
    config['simulation_settings']['num_hours'] = hours
    return config


def write_temp_config(base_config: dict, config_file: str) -> str:
    temp_path = Path(config_file).parent / 'temp_config.yml'
    with open(temp_path, 'w') as f:
        yaml.dump(base_config, f)
    return str(temp_path)


def build_network(scenario_name: str, config_file: str, params_file: str, data_path: str):
    try:
        fn = get_scenario_function(scenario_name)
        return fn(config_file, params_file, data_path)
    except Exception:
        raise RuntimeError(f"Error building network for '{scenario_name}': {traceback.format_exc()}")


def run_lopf(network, solver_name: str, solver_options: dict) -> None:
    try:
        network.optimize(solver_name=solver_name, solver_options=solver_options)
    except Exception as e:
        raise RuntimeError(f"LOPF failed ({solver_name}): {e}")


def calculate_summary(network) -> pd.DataFrame:
    # snapshot weightings
    if not hasattr(network.snapshot_weightings, 'generators'):
        network.snapshot_weightings['generators'] = pd.Series(1.0, index=network.snapshots)
    w = network.snapshot_weightings.generators

    Z = pd.Series(0.0, index=network.snapshots)

    # Import/Export (standardizzati sui nomi dello scenario DSM)
    imp_gridside = network.links_t.p0.get('Grid Import', Z).clip(lower=0.0)       # MWh lato rete
    imp_building = (-network.links_t.p1.get('Grid Import', Z)).clip(lower=0.0)    # MWh lato edificio
    exp_grid     = (-network.links_t.p1.get('PV Export', Z)).clip(lower=0.0)      # MWh esportati lato rete

    total_imp = (imp_building * w).sum()
    total_exp = (exp_grid * w).sum()

    # Emissioni grid (pagate sulla quantità acquistata lato rete)
    co2g = network.links.at["Grid Import", "co2_emission_factor_kg_per_mwh"]
    emis_imp = (imp_gridside * co2g * w).sum()
    emis_exp = (exp_grid * co2g * w).sum()  # avoided

    # Emissioni DH
    dh_ts  = network.links_t.p0.get("Heat import", Z).clip(lower=0.0)
    co2dh  = network.links.at["Heat import","co2_emission_factor_kg_per_mwh"]
    emis_dh = (dh_ts * co2dh * w).sum()

    # PV e Heating
    pv_ts   = network.generators_t.p.get('Rooftop PV', Z)
    total_pv = (pv_ts * w).sum()

    # Carichi (inflex + flex)
    elec_inf = network.loads_t.p.get('Building Elec Load',  Z)
    elec_ch  = network.loads_t.p.get('DSM Elec Flex Load', Z)
    heat_inf = network.loads_t.p.get('Building Heat Load',  Z)
    heat_ch  = network.loads_t.p.get('DSM Heat Flex Load', Z)

    # Dispatch DSM (convertiamo i segni: p0>0 = dal DSM verso Building)
    e_disp_raw = network.links_t.p0.get('DSM Elec Dispatch', Z)
    h_disp_raw = network.links_t.p0.get('DSM Heat Dispatch', Z)
    elec_disp = e_disp_raw.clip(lower=0.0)
    heat_disp = h_disp_raw.clip(lower=0.0)

    total_elec_demand = ((elec_inf + elec_ch - elec_disp) * w).sum()
    total_heat_demand = ((heat_inf + heat_ch - heat_disp) * w).sum()

    # Costi import elettrica (prezzo orario sul lato rete)
    grid_price = network.grid_price_series
    cost_grid_import = (imp_gridside * grid_price * w).sum()

    # Costi riscaldamento da DH (generator "Heating")
    heat_ts = network.generators_t.p.get('Heating', Z)
    mc_h = network.generators.at['Heating', 'marginal_cost']
    cost_heat = (heat_ts * mc_h * w).sum() if hasattr(mc_h, 'shape') else heat_ts.sum() * mc_h

    op_cost = cost_grid_import + cost_heat

    # Outputs ricercati dai plot 2x2
    hp_out = (-network.links_t.p1.get('Heat Pump', Z)).clip(lower=0.0)
    total_hp = (hp_out * w).sum()
    total_dh = (dh_ts * w).sum()

    return pd.DataFrame([{
        'Total Grid Import (MWh)':        total_imp,
        'Total Grid Export (MWh)':        total_exp,
        'Grid Import Cost (EUR)':         cost_grid_import,
        'Total Grid Import Cost (EUR)':   cost_grid_import,
        'Total Operational Cost (EUR)':   op_cost,
        'Total PV Produced (MWh)':        total_pv,
        'Total Heat Produced (MWh_th)':   total_heat_demand,
        'Total Elec Demand (MWh)':        total_elec_demand,
        'Heat Pump Production (MWh_th)':  total_hp,
        'DH Import to Building (MWh_th)': total_dh,
        'DSM Heat Dispatch (MWh_th)':     (h_disp_raw.clip(lower=0.0) * w).sum(),
        'Total Import Emissions (kgCO₂)': emis_imp,
        'Avoided Emissions (kgCO₂)':      emis_exp,
        'Total DH Emissions (kgCO₂)':     emis_dh,
    }])


def generate_hourly_df(network) -> pd.DataFrame:
    snaps = network.snapshots
    Z = pd.Series(0.0, index=snaps)

    # Carichi
    elec_inf = network.loads_t.p.get('Building Elec Load',  Z)
    elec_ch  = network.loads_t.p.get('DSM Elec Flex Load', Z)
    heat_inf = network.loads_t.p.get('Building Heat Load',  Z)

    # DSM termico
    h_charge_raw = network.links_t.p0.get('DSM Heat Charge',   Z)
    h_disp_raw   = network.links_t.p0.get('DSM Heat Dispatch', Z)
    heat_charge  = h_charge_raw.clip(lower=0.0)
    heat_disp    = h_disp_raw.clip(lower=0.0)

    # DSM elettrico (dispatch lato p0>0)
    e_disp_raw = network.links_t.p0.get('DSM Elec Dispatch', Z)
    elec_disp  = e_disp_raw.clip(lower=0.0)

    # Flussi principali
    pv      = network.generators_t.p.get('Rooftop PV', Z)
    grid_i  = (-network.links_t.p1.get('Grid Import', Z)).clip(lower=0.0)  # lato edificio
    grid_e  = (-network.links_t.p1.get('PV Export',   Z)).clip(lower=0.0)  # lato rete
    hp_out  = (-network.links_t.p1.get('Heat Pump',   Z)).clip(lower=0.0)
    dh_imp  =  network.links_t.p0.get('Heat import',  Z).clip(lower=0.0)

    df = pd.DataFrame({
        'timestamp':                       snaps,
        'electricity_inflexible_load_mwh': elec_inf.values,
        'electricity_charge_mwh':          elec_ch.values,
        'electricity_dispatch_mwh':        elec_disp.values,
        'electricity_load_mwh':            (elec_inf + elec_ch - elec_disp).values,
        'heat_inflexible_load_mwh':        heat_inf.values,
        'heat_charge_mwh':                 heat_charge.values,
        'heat_dispatch_mwh':               heat_disp.values,
        'heat_load_mwh':                   (heat_inf + heat_charge - heat_disp).values,
        'pv_generation_mwh':               pv.values,
        'grid_import_mwh':                 grid_i.values,
        'grid_export_mwh':                 grid_e.values,
        'heat_pump_output':                hp_out.values,
        'dh_import':                       dh_imp.values,
    })

    co2g = network.links.at["Grid Import","co2_emission_factor_kg_per_mwh"]
    df['grid_import_emissions_kg'] = df['grid_import_mwh'] * co2g
    df['grid_export_emissions_kg'] = df['grid_export_mwh'] * co2g

    co2d = network.links.at["Heat import","co2_emission_factor_kg_per_mwh"]
    df['dh_emissions_kg'] = df['dh_import'] * co2d

    df['total_load_mwh'] = df['electricity_load_mwh'] + df['heat_load_mwh']
    return df


def run_scenario(scenario_name: str, config_file: str, params_file: str) -> pd.DataFrame:
    print(f"\n--- Running Scenario: {scenario_name} ---")
    config_abs_path = Path(config_file).resolve()
    params_abs_path = Path(params_file).resolve()
    print(f"Using config file: {config_abs_path}")
    print(f"Using params file: {params_abs_path}")

    config = load_config(str(config_abs_path))
    data_in, data_out_base = determine_paths(config, str(config_abs_path))
    subdir = config.get('scenarios', {}).get(scenario_name, {}).get('output_subdir', f"scenario_{scenario_name}")
    out_dir = Path(data_out_base) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    override = override_simulation_hours(config)
    temp_cfg = write_temp_config(override, str(config_abs_path))

    try:
        net, _, _ = build_network(scenario_name, temp_cfg, str(params_abs_path), data_in)
        Path(temp_cfg).unlink()  # remove temp
    except RuntimeError as e:
        print(e)
        return pd.DataFrame()

    sim = override.get('simulation_settings', {})
    try:
        run_lopf(net, sim.get('solver', 'highs'), sim.get('solver_options', {}))
    except RuntimeError as e:
        print(e)
        return pd.DataFrame()

    # Salvataggio network (best-effort)
    net_file = out_dir / f"{scenario_name}_network_results.nc"
    try:
        net.export_to_netcdf(str(net_file))
    except Exception as e:
        print(f"Error saving network: {e}")

    # Summary + hourly
    summary_df = calculate_summary(net)
    summary_df.to_csv(out_dir / f"{scenario_name}_summary.csv", index=False)
    hourly_df = generate_hourly_df(net)
    hourly_df.to_csv(out_dir / f"{scenario_name}_hourly_results.csv", index=False)

    # Batterie – SoC e totali (informativi)
    try:
        soc = net.stores_t.e.get("Battery Store", pd.Series(0.0, index=net.snapshots))
        hourly_df["battery_soc_mwh"] = soc.values
    except Exception:
        pass

    # Logging DSM Heat
    heat_ch = hourly_df['heat_charge_mwh'].sum()
    heat_dc = hourly_df['heat_dispatch_mwh'].sum()
    print(f"DSM Heat Charge:              {heat_ch:.2f} MWh_th")
    print(f"DSM Heat Dispatch:            {heat_dc:.2f} MWh_th")

    # Plotting
    try:
        plot_time_series({
            'timestamps':    hourly_df['timestamp'],
            'electric_load': hourly_df['electricity_load_mwh'],
            'thermal_load':  hourly_df['heat_load_mwh'],
            'pv_generation': hourly_df['pv_generation_mwh'],
            'grid_import':   hourly_df['grid_import_mwh'],
            'grid_export':   hourly_df['grid_export_mwh'],
        }, scenario_name=scenario_name, output_dir=str(out_dir))

        plot_energy_balance({
            'pv_generation_mwh':       hourly_df['pv_generation_mwh'].sum(),
            'grid_import_mwh':         hourly_df['grid_import_mwh'].sum(),
            'grid_export_mwh':         hourly_df['grid_export_mwh'].sum(),
            'total_electric_load_mwh': hourly_df['electricity_load_mwh'].sum(),
            'total_thermal_load_mwh':  hourly_df['heat_load_mwh'].sum(),
            'co2_emissions_kg':        hourly_df['grid_import_emissions_kg'].sum(),
        }, scenario_name=scenario_name, output_dir=str(out_dir))

        hp_total = hourly_df['heat_pump_output'].sum()
        dh_total = hourly_df['dh_import'].sum()

        plot_energy_balance_breakdown({
            'pv_generation_mwh':               hourly_df['pv_generation_mwh'].sum(),
            'grid_import_mwh':                 hourly_df['grid_import_mwh'].sum(),
            'grid_export_mwh':                 hourly_df['grid_export_mwh'].sum(),
            'total_electric_load_mwh':         hourly_df['electricity_load_mwh'].sum(),
            'total_thermal_load_mwh':          hourly_df['heat_load_mwh'].sum(),
            'Heat Pump Production (MWh_th)':   hp_total,
            'DH Import to Building (MWh_th)':  dh_total,
            'DSM Heat Charge (MWh_th)':        heat_ch,
            'DSM Heat Dispatch (MWh_th)':      heat_dc,
            'co2_emissions_kg':                hourly_df['grid_import_emissions_kg'].sum(),
            'dh_emissions_kg':                 hourly_df['dh_emissions_kg'].sum(),
        }, scenario_name=scenario_name, output_dir=str(out_dir))

        plot_pv_battery_breakdown(net, scenario_name=scenario_name, output_dir=str(out_dir))
        plot_thermal_breakdown(net, scenario_name=scenario_name, output_dir=str(out_dir))
        plot_dsm_thermal_breakdown(hourly_df, scenario_name=scenario_name, output_dir=str(out_dir))
    except Exception as e:
        print(f"Plotting error: {e}")

    return hourly_df


def main():
    parser = argparse.ArgumentParser(description="PED Lyngby scenario tool")
    sub = parser.add_subparsers(dest='command', required=True)

    create_p = sub.add_parser('create', help='Create a new scenario')
    create_p.add_argument('name', help='Scenario name (alphanumeric, underscores)')
    create_p.add_argument('description', help='Brief description')

    run_p = sub.add_parser('run', help='Run an existing scenario')
    run_p.add_argument('scenario', help='Scenario name')
    run_p.add_argument('config', help='Path to config.yml')
    run_p.add_argument('params', help='Path to component_params.yml')

    args = parser.parse_args()
    print('*' * 40)
    print(f"Command: {args.command}")
    if args.command == 'run':
        print(f"Scenario: {args.scenario}")
    print('*' * 40)

    if args.command == 'create':
        success = create_scenario(args.name, args.description)
        if not success:
            sys.exit(1)
    elif args.command == 'run':
        df = run_scenario(args.scenario, args.config, args.params)
        if df.empty:
            sys.exit(1)


if __name__ == '__main__':
    main()
