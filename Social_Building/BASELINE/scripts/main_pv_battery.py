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

import pandas as pd
import matplotlib.pyplot as plt

# project_root/scripts/ → aggiunge project_root al sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# factory scenari e util per i plot
from scenarios import get_scenario_function
from scripts.plotting_utils import (
    plot_time_series,
    plot_energy_balance,
    plot_pv_battery_breakdown,
)

HOURS_IN_YEAR = 8760


# ----------------------------- Scenario creation -----------------------------

def create_scenario(scenario_name: str, description: str) -> bool:
    """Create a new scenario with the given name and description."""
    if not (scenario_name.replace("_", "").isalnum()):
        print(f"Error: Scenario name must be alphanumeric (underscores allowed). Got: {scenario_name}")
        return False

    scenarios_dir = Path("scenarios")
    scenario_file = scenarios_dir / f"{scenario_name}.py"
    template_file = scenarios_dir / "template.py"

    if scenario_file.exists():
        print(f"Error: Scenario '{scenario_name}' already exists at {scenario_file}")
        return False
    if not template_file.exists():
        print(f"Error: Template file not found at {template_file}")
        return False

    shutil.copy(template_file, scenario_file)
    content = scenario_file.read_text()

    header = f"PED Lyngby Model - {scenario_name.replace('_', ' ').title()} Scenario"
    content = content.replace(
        "PED Lyngby Model - Template for New Scenarios",
        header
    ).replace(
        "This is a template file for creating new scenarios. Copy this file and modify it\nto implement a new scenario.",
        description
    ).replace(
        "Building new scenario network...",
        f"Building {scenario_name.replace('_', ' ')} network..."
    )
    scenario_file.write_text(content)

    # aggiorna __init__.py
    init_file = scenarios_dir / "__init__.py"
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
    dict_block = after[dict_start:dict_end + 1]
    entry = f"    '{scenario_name}': {scenario_name}.create_network,"
    if entry not in dict_block:
        lines = dict_block.splitlines()
        lines.insert(-1, entry)
        new_block = "\n".join(lines)
        after = after.replace(dict_block, new_block)

    init_file.write_text(before + after)

    # aggiorna config.yml
    config_path = Path("config") / "config.yml"
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return False
    config = yaml.safe_load(config_path.read_text()) or {}
    config.setdefault("scenarios", {})
    if scenario_name not in config["scenarios"]:
        config["scenarios"][scenario_name] = {
            "description": description,
            "output_subdir": f"scenario_{scenario_name}",
        }
        config_path.write_text(yaml.dump(config, default_flow_style=False))

    # aggiorna component_params.yml
    params_path = Path("config") / "component_params.yml"
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


# ----------------------------- Config helpers --------------------------------

def load_config(config_file: str) -> dict:
    with open(config_file, "r") as f:
        return yaml.safe_load(f) or {}


def determine_paths(config: dict, config_file: str) -> tuple[str, str]:
    config_abs_path = Path(config_file).resolve()
    project_root = config_abs_path.parent.parent
    data_in = project_root / config.get("paths", {}).get("data_input", "data/input")
    data_out = project_root / config.get("paths", {}).get("data_output", "data/output")
    return str(data_in), str(data_out)


def override_simulation_hours(config: dict, hours: int = HOURS_IN_YEAR) -> dict:
    cfg = dict(config)  # shallow copy
    cfg["simulation_settings"] = cfg.get("simulation_settings", {})
    cfg["simulation_settings"]["num_hours"] = int(hours)
    return cfg


def write_temp_config(base_config: dict, config_file: str) -> str:
    temp_path = Path(config_file).parent / "temp_config.yml"
    with open(temp_path, "w") as f:
        yaml.dump(base_config, f)
    return str(temp_path)


# ----------------------------- Build & LOPF ----------------------------------

def build_network(scenario_name: str, config_file: str, params_file: str, data_path: str):
    """Ritorna SEMPRE un pypsa.Network (mai n[0])."""
    try:
        fn = get_scenario_function(scenario_name)
        net = fn(config_file, params_file, data_path)
        return net
    except Exception:
        raise RuntimeError(f"Error building network for '{scenario_name}': {traceback.format_exc()}")


def run_lopf(network, solver_name: str, solver_options: dict) -> None:
    try:
        network.optimize(solver_name=solver_name, solver_options=solver_options)
    except Exception as e:
        raise RuntimeError(f"LOPF failed ({solver_name}): {e}")


# ----------------------------- Summaries (NET lato building) -----------------
def calculate_summary(n) -> pd.DataFrame:
    """
    Calcola costi e emissioni NET lato building.
    - DH: costo = Σ( q_dh_net [MWh_th] * prezzo_dh_net [EUR/MWh_th] )
      dove q_dh_net = -p1("Heat import")≥0
      prezzo_dh_net: prima da links_t.marginal_cost["Heat import"], altrimenti (prezzo_gen/η_dh)
    - CO2 DH: fattore sul Link "Heat import" (kg/MWh_th) * q_dh_net
    - Elettricità: import NET e prezzo NET (marginal_cost Grid / η_grid o serie mercato)
    """
    snaps = n.snapshots

    # Pesi orari (se non definiti)
    if not hasattr(n.snapshot_weightings, "generators"):
        n.snapshot_weightings["generators"] = pd.Series(1.0, index=snaps)
    w = n.snapshot_weightings.generators.reindex(snaps).fillna(1.0)

    links = n.links
    Lp1   = getattr(getattr(n, "links_t", None), "p1", pd.DataFrame())
    Gmc   = getattr(getattr(n, "generators_t", None), "marginal_cost", pd.DataFrame())
    Gp    = getattr(getattr(n, "generators_t", None), "p", pd.DataFrame())

    def s_or(df, col, fill=0.0):
        return df.get(col, pd.Series(fill, index=snaps)).reindex(snaps).fillna(fill)

    # Efficienze (per eventuali fallback prezzi)
    eta_grid = float(links.at["Grid Import", "efficiency"]) \
        if ("Grid Import" in links.index and "efficiency" in links.columns) else 1.0
    eta_dh   = float(links.at["Heat import", "efficiency"]) \
        if ("Heat import" in links.index and "efficiency" in links.columns) else 1.0

    # Prezzo elettrico NET (EUR/MWh)
    if hasattr(n, "grid_price_series") and n.grid_price_series is not None:
        grid_price_net = pd.Series(n.grid_price_series, index=snaps).astype(float)
    else:
        if "Grid" in Gmc.columns:
            grid_price_gen = Gmc["Grid"].reindex(snaps).fillna(0.0).astype(float)
        elif "Grid" in getattr(n, "generators", pd.DataFrame()).index and "marginal_cost" in n.generators.columns:
            grid_price_gen = pd.Series(float(n.generators.at["Grid", "marginal_cost"]), index=snaps)
        else:
            grid_price_gen = pd.Series(0.0, index=snaps)
        grid_price_net = (grid_price_gen / max(eta_grid, 1e-12)).astype(float)

    export_price_net = grid_price_net.copy()

    # Import/Export NET (p1<0 lato building)
    imp_net = s_or(Lp1, "Grid Import").mul(-1.0).clip(lower=0.0)
    if "Grid Export" in Lp1.columns:
        exp_net = s_or(Lp1, "Grid Export").mul(-1.0).clip(lower=0.0)
    elif "PV Export" in Lp1.columns:
        exp_net = s_or(Lp1, "PV Export").mul(-1.0).clip(lower=0.0)
    else:
        exp_net = pd.Series(0.0, index=snaps)

    cost_elec  = float((imp_net * grid_price_net * w).sum())
    rev_export = float((exp_net * export_price_net * w).sum())

    # ---------------- DH: calore consegnato NET, prezzo NET e CO2 ----------------
    # q_dh_net = energia termica CONSEGNATA al building (MWh_th)
    q_dh_net = s_or(Lp1, "Heat import").mul(-1.0).clip(lower=0.0)

    # PREZZO DH NET:
    # 1) se esiste una serie sul link, usala (già netta a valle)
    # 2) ALTRIMENTI usa direttamente il marginal_cost del Generator "Heating"
    #    SENZA dividere per l'efficienza (si assume che il parametro dei component_params
    #    sia già espresso nel modo voluto dall’utente).
    if hasattr(n.links_t, "marginal_cost") and "Heat import" in getattr(n.links_t.marginal_cost, "columns", []):
        dh_price_net = n.links_t.marginal_cost["Heat import"].reindex(snaps).fillna(0.0).astype(float)
    else:
        if "Heating" in Gmc.columns:
            dh_price_net = Gmc["Heating"].reindex(snaps).fillna(0.0).astype(float)
        elif "Heating" in getattr(n, "generators", pd.DataFrame()).index and "marginal_cost" in n.generators.columns:
            dh_price_net = pd.Series(float(n.generators.at["Heating", "marginal_cost"]), index=snaps)
        else:
            dh_price_net = pd.Series(0.0, index=snaps)

    cost_dh = float((q_dh_net * dh_price_net * w).sum())

    # CO2 DH:
    # 1) prova a leggere il fattore dal Link "Heat import"
    # 2) fallback: se definito sui generators (es. sui component_params applicati al generatore),
    #    usa "Heating"
    if ("Heat import" in n.links.index) and ("co2_emission_factor_kg_per_mwh" in n.links.columns):
        dh_co2_kg_per_mwh = float(n.links.at["Heat import", "co2_emission_factor_kg_per_mwh"] or 0.0)
    else:
        if ("Heating" in getattr(n, "generators", pd.DataFrame()).index) and \
           ("co2_emission_factor_kg_per_mwh" in n.generators.columns):
            dh_co2_kg_per_mwh = float(n.generators.at["Heating", "co2_emission_factor_kg_per_mwh"] or 0.0)
        else:
            dh_co2_kg_per_mwh = 0.0

    co2_dh_t = float((q_dh_net * w).sum()) * dh_co2_kg_per_mwh / 1000.0


    # Emissioni (t) — fattori sul Link (kg/MWh → /1000)
    grid_co2_kg_per_mwh = float(links.at["Grid Import", "co2_emission_factor_kg_per_mwh"]) \
        if ("Grid Import" in links.index and "co2_emission_factor_kg_per_mwh" in links.columns) else 0.0


    co2_elec_t = float((imp_net * w).sum()) * grid_co2_kg_per_mwh / 1000.0
    co2_tot_t  = co2_elec_t + co2_dh_t

    pv_total_mwh = float((Gp.get("Rooftop PV", pd.Series(0.0, index=snaps)).reindex(snaps).fillna(0.0) * w).sum())
    total_cost   = cost_elec + cost_dh - rev_export

    return pd.DataFrame([{
        "Total PV Produced (MWh)":            pv_total_mwh,
        "Total Grid Import (MWh)":            float((imp_net * w).sum()),
        "Total Grid Export (MWh)":            float((exp_net * w).sum()),
        "DH Heat Delivered (MWh_th)":         float((q_dh_net * w).sum()),
        "Electricity Import Cost (EUR)":      cost_elec,
        "District Heating Cost (EUR)":        cost_dh,
        "Export Revenue (EUR)":               rev_export,
        "Total Operational Cost (EUR)":       total_cost,
        "CO2 Electricity Import (t)":         co2_elec_t,
        "CO2 District Heating (t)":           co2_dh_t,
        "CO2 Total (t)":                      co2_tot_t,
    }])



def generate_hourly_df(network) -> pd.DataFrame:
    """
    DataFrame orario con carichi, PV, import/export, calore DH, emissioni NET lato building.
    """
    snaps = network.snapshots

    # Carichi elettrici/termici (solo loads effettivamente esistenti)
    elec_cols, heat_cols = [], []
    if hasattr(network, "loads_t") and hasattr(network.loads_t, "p_set") and not network.loads_t.p_set.empty:
        if hasattr(network, "loads") and "carrier" in network.loads.columns:
            for c in network.loads_t.p_set.columns:
                if c in network.loads.index:
                    if network.loads.loc[c, "carrier"] == "electricity":
                        elec_cols.append(c)
                    elif network.loads.loc[c, "carrier"] == "heat":
                        heat_cols.append(c)

    elec = network.loads_t.p_set[elec_cols].sum(axis=1) if elec_cols else pd.Series(0.0, index=snaps)
    heat = network.loads_t.p_set[heat_cols].sum(axis=1) if heat_cols else pd.Series(0.0, index=snaps)

    # PV
    pv = network.generators_t.p.get("Rooftop PV", pd.Series(0.0, index=snaps)).reindex(snaps).fillna(0.0)

    # Import NET e LORDO (informativo)
    grid_imp_gross = network.links_t.p0.get("Grid Import", pd.Series(0.0, index=snaps)).reindex(snaps).clip(lower=0.0)
    grid_imp_net   = (-network.links_t.p1.get("Grid Import", pd.Series(0.0, index=snaps))).reindex(snaps).clip(lower=0.0)

    # Export NET con fallback nome
    if hasattr(network.links_t, "p0") and "Grid Export" in network.links_t.p0.columns:
        exp_col = "Grid Export"
    elif hasattr(network.links_t, "p0") and "PV Export" in network.links_t.p0.columns:
        exp_col = "PV Export"
    else:
        exp_col = None

    grid_exp_gross = (network.links_t.p0[exp_col].reindex(snaps).clip(lower=0.0) if exp_col else pd.Series(0.0, index=snaps))
    grid_exp_net   = ((-network.links_t.p1[exp_col]).reindex(snaps).clip(lower=0.0) if exp_col else pd.Series(0.0, index=snaps))

    # DH NET (calore consegnato)
    q_dh_net = (-network.links_t.p1.get("Heat import", pd.Series(0.0, index=snaps))).reindex(snaps).clip(lower=0.0)

    # Emissioni NET
    grid_co2_kg_per_mwh = float(network.links.at["Grid Import", "co2_emission_factor_kg_per_mwh"]) \
        if ("Grid Import" in network.links.index and "co2_emission_factor_kg_per_mwh" in network.links.columns) else 0.0
    dh_co2_kg_per_mwh = float(network.links.at["Heat import", "co2_emission_factor_kg_per_mwh"]) \
        if ("Heat import" in network.links.index and "co2_emission_factor_kg_per_mwh" in network.links.columns) else 0.0

    df = pd.DataFrame({
        "timestamp":                 snaps,
        "electricity_load_mwh":      elec.values,
        "heat_load_mwh":             heat.values,
        "pv_generation_mwh":         pv.values,
        "grid_import_mwh":           grid_imp_net.values,        # NET
        "grid_import_gross_mwh":     grid_imp_gross.values,      # LORDO (info)
        "grid_export_mwh":           grid_exp_net.values,        # NET
        "grid_export_gross_mwh":     grid_exp_gross.values,      # LORDO (info)
        "dh_heat_delivered_mwh_th":  q_dh_net.values,
    })

    df["total_load_mwh"] = df["electricity_load_mwh"] + df["heat_load_mwh"]

    # Emissioni NET lato building
    df["grid_import_emissions_kg"] = df["grid_import_mwh"] * grid_co2_kg_per_mwh
    df["dh_emissions_kg"]          = df["dh_heat_delivered_mwh_th"] * dh_co2_kg_per_mwh
    df["total_emissions_kg"]       = df["grid_import_emissions_kg"] + df["dh_emissions_kg"]

    return df


# ----------------------------- Run scenario ----------------------------------

def run_scenario(scenario_name: str, config_file: str, params_file: str) -> pd.DataFrame:
    print(f"\n--- Running Scenario: {scenario_name} ---")
    config_abs_path = Path(config_file).resolve()
    params_abs_path = Path(params_file).resolve()
    print(f"Using config file: {config_abs_path}")
    print(f"Using params file:  {params_abs_path}")

    # Config e percorsi
    config = load_config(str(config_abs_path))
    data_in, data_out_base = determine_paths(config, str(config_abs_path))
    subdir = config.get("scenarios", {}).get(scenario_name, {}).get("output_subdir", f"scenario_{scenario_name}")
    out_dir = Path(data_out_base) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Override ore simulazione
    override = override_simulation_hours(config)
    temp_cfg = write_temp_config(override, str(config_abs_path))

    # Build network
    try:
        net = build_network(scenario_name, temp_cfg, str(params_abs_path), data_in)
    except RuntimeError as e:
        Path(temp_cfg).unlink(missing_ok=True)
        print(e)
        return pd.DataFrame()
    finally:
        Path(temp_cfg).unlink(missing_ok=True)

    # LOPF
    sim = override.get("simulation_settings", {})
    try:
        run_lopf(net, sim.get("solver", "highs"), sim.get("solver_options", {}))
    except RuntimeError as e:
        print(e)
        return pd.DataFrame()

    # Salvataggio network su NetCDF
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

    # KPI picco
    df = hourly_df.copy()
    t_peak_load = df["electricity_load_mwh"].idxmax()
    peak_load = df.loc[t_peak_load, "electricity_load_mwh"]
    import_at_peak = df.loc[t_peak_load, "grid_import_mwh"]
    peak_shaving = peak_load - import_at_peak
    print(f"Picco domanda             = {peak_load:.3f} MWh")
    print(f"Import allo stesso istante= {import_at_peak:.3f} MWh")
    print(f"Peak-shaving              = {peak_shaving:.3f} MWh")

    # Batteria: SoC/carica/scarica (robusto ai nomi/scenari senza batteria)
    stores_df = getattr(net, "stores", pd.DataFrame())
    stores_e = getattr(getattr(net, "stores_t", None), "e", pd.DataFrame())

    battery_store_name = None
    if not stores_df.empty:
        for cand in ["Battery Store", "Battery", "Battery_Store"]:
            if cand in stores_df.index:
                battery_store_name = cand
                break
        if battery_store_name is None:
            pref = stores_df[
                stores_df.get("bus", "").eq("Battery Storage")
                | stores_df.get("carrier", "").eq("electricity")
            ]
            battery_store_name = pref.index[0] if not pref.empty else stores_df.index[0]

    if battery_store_name and battery_store_name in stores_e.columns:
        soc = stores_e[battery_store_name].reindex(net.snapshots)
        hourly_df["battery_soc_mwh"] = soc.values
        delta = soc.diff()
        total_charge = float(delta.clip(lower=0).sum())
        total_discharge = float((-delta.clip(upper=0)).sum())
    else:
        hourly_df["battery_soc_mwh"] = 0.0
        total_charge = 0.0
        total_discharge = 0.0

    # Plot
    try:
        plot_time_series(
            {
                "timestamps": hourly_df["timestamp"],
                "electric_load": hourly_df["electricity_load_mwh"],
                "thermal_load": hourly_df["heat_load_mwh"],
                "pv_generation": hourly_df["pv_generation_mwh"],
                "grid_import": hourly_df["grid_import_mwh"],
                "grid_export": hourly_df["grid_export_mwh"],
            },
            scenario_name=scenario_name,
            output_dir=str(out_dir),
        )

        plot_energy_balance(
            {
                "pv_generation_mwh": hourly_df["pv_generation_mwh"].sum(),
                "grid_import_mwh": hourly_df["grid_import_mwh"].sum(),
                "grid_export_mwh": hourly_df["grid_export_mwh"].sum(),
                "total_electric_load_mwh": hourly_df["electricity_load_mwh"].sum(),
                "total_thermal_load_mwh": hourly_df["heat_load_mwh"].sum(),
                "co2_emissions_kg": hourly_df["total_emissions_kg"].sum(),  # totale (grid + DH)
            },
            scenario_name=scenario_name,
            output_dir=str(out_dir),
        )

        plot_pv_battery_breakdown(
            net,
            scenario_name=scenario_name,
            output_dir=str(out_dir),
        )

    except Exception as e:
        print(f"Plotting error: {e}")

    return hourly_df


# ----------------------------- CLI -------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PED Lyngby scenario tool")
    sub = parser.add_subparsers(dest="command", required=True)

    create_p = sub.add_parser("create", help="Create a new scenario")
    create_p.add_argument("name", help="Scenario name (alphanumeric, underscores)")
    create_p.add_argument("description", help="Brief description")

    run_p = sub.add_parser("run", help="Run an existing scenario")
    run_p.add_argument("scenario", help="Scenario name")
    run_p.add_argument("config", help="Path to config.yml")
    run_p.add_argument("params", help="Path to component_params.yml")

    args = parser.parse_args()
    print("*" * 40)
    print(f"Command: {args.command}")
    if args.command == "run":
        print(f"Scenario: {args.scenario}")
    print("*" * 40)

    if args.command == "create":
        success = create_scenario(args.name, args.description)
        if not success:
            sys.exit(1)
    elif args.command == "run":
        df = run_scenario(args.scenario, args.config, args.params)
        if df.empty:
            sys.exit(1)


if __name__ == "__main__":
    main()
