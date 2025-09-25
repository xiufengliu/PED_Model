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


# ----------------------------
# Scenario creation utilities
# ----------------------------
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

    # Update scenarios/__init__.py
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

    # Add to config
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

    # Add header to component params
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


# ----------------------------
# Config helpers
# ----------------------------
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


# ----------------------------
# Build & solve
# ----------------------------
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


# ----------------------------
# Reporting (NET-side coherent)
# ----------------------------
def _weights(network) -> pd.Series:
    snaps = network.snapshots
    try:
        w = network.snapshot_weightings["objective"].reindex(snaps)
        w = w.fillna(1.0)
    except Exception:
        w = pd.Series(1.0, index=snaps)
    return w

import pandas as pd

def calculate_summary(network) -> pd.DataFrame:
    """
    Summary coerente NET lato building, allineato a main.py:
      - Import/Export: sia NET (p1) sia GROSS (p0) per confronto.
      - Prezzo DH: usa links_t.marginal_cost['Heat import'] se significativo,
        altrimenti generators_t.marginal_cost['Heating'] (nessuna divisione per efficienza).
      - Emissioni: da fattori sui link (fallback generator).
      - Stesse colonne e naming di main.py.
    """
    import pandas as pd

    n = network  # alias per compatibilità con main.py
    snaps = n.snapshots

    # Pesi orari
    if not hasattr(n.snapshot_weightings, "generators"):
        n.snapshot_weightings["generators"] = pd.Series(1.0, index=snaps)
    w = n.snapshot_weightings.generators.reindex(snaps).fillna(1.0)

    # Helper comodo
    def s_or(df, col, fill=0.0):
        return df.get(col, pd.Series(fill, index=snaps)).reindex(snaps).fillna(fill)

    # ---- Prezzo elettrico NET (usato per i costi import/export) ----
    links = n.links if hasattr(n, "links") else pd.DataFrame()
    eta_grid = float(links.at["Grid Import", "efficiency"]) \
        if ("Grid Import" in links.index and "efficiency" in links.columns) else 1.0

    Gmc = getattr(getattr(n, "generators_t", None), "marginal_cost", pd.DataFrame())
    if hasattr(n, "grid_price_series") and n.grid_price_series is not None:
        grid_price_net = pd.Series(n.grid_price_series, index=snaps).astype(float)
    else:
        if "Grid" in Gmc.columns:
            grid_price_gen = Gmc["Grid"].reindex(snaps).fillna(0.0).astype(float)
        elif ("generators" in dir(n)) and ("Grid" in getattr(n, "generators", pd.DataFrame()).index) \
             and ("marginal_cost" in n.generators.columns):
            grid_price_gen = pd.Series(float(n.generators.at["Grid", "marginal_cost"]), index=snaps)
        else:
            grid_price_gen = pd.Series(0.0, index=snaps)
        grid_price_net = (grid_price_gen / max(eta_grid, 1e-12)).astype(float)

    export_price_net = grid_price_net.copy()

    # ---- Flussi elettrici (NET/GROSS) ----
    p0 = getattr(getattr(n, "links_t", None), "p0", pd.DataFrame())
    p1 = getattr(getattr(n, "links_t", None), "p1", pd.DataFrame())

    imp_net   = (-s_or(p1, "Grid Import")).clip(lower=0.0)   # lato building
    imp_gross = ( s_or(p0, "Grid Import")).clip(lower=0.0)   # lato sorgente

    if "Grid Export" in getattr(p1, "columns", []):
        exp_col = "Grid Export"
    elif "PV Export" in getattr(p1, "columns", []):
        exp_col = "PV Export"
    else:
        exp_col = None

    exp_net   = ((-p1[exp_col]).reindex(snaps).clip(lower=0.0) if exp_col else pd.Series(0.0, index=snaps))
    exp_gross = (( p0[exp_col]).reindex(snaps).clip(lower=0.0) if exp_col else pd.Series(0.0, index=snaps))

    # ---- Costi elettrici ----
    cost_elec  = float((imp_net * grid_price_net * w).sum())
    rev_export = float((exp_net * export_price_net * w).sum())

    # ---- District Heating (NET) ----
    q_dh_net = (-s_or(p1, "Heat import")).clip(lower=0.0)

    # Prezzo DH NET: link se “significativo”, altrimenti generator "Heating" (NO / efficienza)
    lt_mc = getattr(getattr(n, "links_t", None), "marginal_cost", pd.DataFrame())
    dh_price_net = None
    if "Heat import" in getattr(lt_mc, "columns", []):
        dh_price_link = lt_mc["Heat import"].reindex(snaps).astype(float)
        if dh_price_link.notna().any() and dh_price_link.abs().sum() > 0:
            dh_price_net = dh_price_link.fillna(0.0)

    if dh_price_net is None:
        if "Heating" in Gmc.columns:
            dh_price_net = Gmc["Heating"].reindex(snaps).fillna(0.0).astype(float)
        elif ("generators" in dir(n)) and ("Heating" in getattr(n, "generators", pd.DataFrame()).index) \
             and ("marginal_cost" in n.generators.columns):
            dh_price_net = pd.Series(float(n.generators.at["Heating", "marginal_cost"]), index=snaps)
        else:
            dh_price_net = pd.Series(0.0, index=snaps)

    dh_price_net = dh_price_net.clip(lower=0.0)  # nessuna divisione per efficienza
    cost_dh = float((q_dh_net * dh_price_net * w).sum())

    # ---- Emissioni CO2 ----
    def read_co2_factor(link_name, gen_name):
        if (link_name in links.index) and ("co2_emission_factor_kg_per_mwh" in links.columns):
            v = links.at[link_name, "co2_emission_factor_kg_per_mwh"]
            return float(0.0 if v is None or v == "" else v)
        gens = getattr(n, "generators", pd.DataFrame())
        if (gen_name in gens.index) and ("co2_emission_factor_kg_per_mwh" in gens.columns):
            v = gens.at[gen_name, "co2_emission_factor_kg_per_mwh"]
            return float(0.0 if v is None or v == "" else v)
        return 0.0

    grid_co2_kg_per_mwh = read_co2_factor("Grid Import", "Grid")
    dh_co2_kg_per_mwh   = read_co2_factor("Heat import", "Heating")

    co2_elec_t = float((imp_net * w).sum()) * grid_co2_kg_per_mwh / 1000.0
    co2_dh_t   = float((q_dh_net * w).sum()) * dh_co2_kg_per_mwh / 1000.0
    co2_tot_t  = co2_elec_t + co2_dh_t

    # ---- PV info ----
    Gp = getattr(getattr(n, "generators_t", None), "p", pd.DataFrame())
    pv_total_mwh = float((Gp.get("Rooftop PV", pd.Series(0.0, index=snaps)) * w).sum())

    total_cost = cost_elec + cost_dh - rev_export

    return pd.DataFrame([{
        # energia
        "Total PV Produced (MWh)":            pv_total_mwh,
        "Total Grid Import (Net, MWh)":       float((imp_net   * w).sum()),
        "Total Grid Import (Gross, MWh)":     float((imp_gross * w).sum()),
        "Total Grid Export (Net, MWh)":       float((exp_net   * w).sum()),
        "Total Grid Export (Gross, MWh)":     float((exp_gross * w).sum()),
        "DH Heat Delivered (MWh_th)":         float((q_dh_net  * w).sum()),
        # costi/ricavi
        "Electricity Import Cost (EUR)":      cost_elec,
        "District Heating Cost (EUR)":        cost_dh,
        "Export Revenue (EUR)":               rev_export,
        "Total Operational Cost (EUR)":       total_cost,
        # emissioni
        "CO2 Electricity Import (t)":         co2_elec_t,
        "CO2 District Heating (t)":           co2_dh_t,
        "CO2 Total (t)":                      co2_tot_t,
    }])



def generate_hourly_df(network) -> pd.DataFrame:
    snaps = network.snapshots
    Z = pd.Series(0.0, index=snaps)

    # Loads
    elec_inf = network.loads_t.p.get('Building Elec Load', Z)
    elec_ch  = network.loads_t.p.get('DSM Elec Flex Load', Z)
    heat_inf = network.loads_t.p.get('Building Heat Load', Z)

    # DSM thermal links (if present)
    h_charge_raw = network.links_t.p0.get('DSM Heat Charge',   Z) if 'DSM Heat Charge'   in network.links.index else Z
    h_disp_raw   = network.links_t.p0.get('DSM Heat Dispatch', Z) if 'DSM Heat Dispatch' in network.links.index else Z
    heat_charge  = h_charge_raw.clip(lower=0.0)
    heat_disp    = h_disp_raw.clip(lower=0.0)

    # DSM electric dispatch (if present)
    e_disp_raw = network.links_t.p0.get('DSM Elec Dispatch', Z) if 'DSM Elec Dispatch' in network.links.index else Z
    elec_disp  = e_disp_raw.clip(lower=0.0)

    # Main flows (NET side)
    pv      = network.generators_t.p.get('Rooftop PV', Z)
    grid_i  = (-network.links_t.p1.get('Grid Import', Z)).clip(lower=0.0)
    grid_e  = (-network.links_t.p1.get('PV Export',   Z)).clip(lower=0.0)
    hp_out  = (-network.links_t.p1.get('Heat Pump',   Z)).clip(lower=0.0) if 'Heat Pump' in network.links.index else Z
    dh_imp  = (-network.links_t.p1.get('Heat import', Z)).clip(lower=0.0)

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

    # Emissions (NET quantities)
    co2g = float(network.links.at["Grid Import", "co2_emission_factor_kg_per_mwh"]) \
        if "co2_emission_factor_kg_per_mwh" in network.links.columns else 0.0
    df['grid_import_emissions_kg'] = df['grid_import_mwh'] * co2g
    df['grid_export_emissions_kg'] = df['grid_export_mwh'] * co2g

    co2d = float(network.links.at["Heat import", "co2_emission_factor_kg_per_mwh"]) \
        if "co2_emission_factor_kg_per_mwh" in network.links.columns else 0.0
    df['dh_emissions_kg'] = df['dh_import'] * co2d

    df['total_load_mwh'] = df['electricity_load_mwh'] + df['heat_load_mwh']
    return df


# ----------------------------
# Runner
# ----------------------------
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
        Path(temp_cfg).unlink(missing_ok=True)
    except RuntimeError as e:
        print(e)
        return pd.DataFrame()

    sim = override.get('simulation_settings', {})
    try:
        run_lopf(net, sim.get('solver', 'highs'), sim.get('solver_options', {}))
    except RuntimeError as e:
        print(e)
        return pd.DataFrame()

    # Save network (best-effort)
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

    # Battery SoC (if present)
    try:
        soc = net.stores_t.e.get("Battery Store", pd.Series(0.0, index=net.snapshots))
        hourly_df["battery_soc_mwh"] = soc.values
    except Exception:
        pass

    # Log DSM heat energy
    heat_ch = hourly_df['heat_charge_mwh'].sum()
    heat_dc = hourly_df['heat_dispatch_mwh'].sum()
    print(f"DSM Heat Charge:              {heat_ch:.2f} MWh_th")
    print(f"DSM Heat Dispatch:            {heat_dc:.2f} MWh_th")

    # Plotting (best-effort)
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


# ----------------------------
# CLI
# ----------------------------
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