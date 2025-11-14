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

# Add the project root directory to the Python path
# This assumes main.py is in 'project_root/scripts/'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

# Correct import for get_scenario_function from the scenarios package
from scenarios import get_scenario_function
from scripts.plotting_utils import plot_time_series, plot_energy_balance
from scripts.plotting_utils import plot_pv_split

HOURS_IN_YEAR = 8760


def create_scenario(scenario_name: str, description: str) -> bool:
    """Create a new scenario with the given name and description."""
    # Validate scenario name
    if not (scenario_name.replace('_', '').isalnum()):
        print(f"Error: Scenario name must be alphanumeric (underscores allowed). Got: {scenario_name}")
        return False

    # Paths
    scenarios_dir = Path('scenarios')
    scenario_file = scenarios_dir / f"{scenario_name}.py"
    template_file = scenarios_dir / 'template.py'

    # Check existence
    if scenario_file.exists():
        print(f"Error: Scenario '{scenario_name}' already exists at {scenario_file}")
        return False
    if not template_file.exists():
        print(f"Error: Template file not found at {template_file}")
        return False

    # Copy and replace
    shutil.copy(template_file, scenario_file)
    content = scenario_file.read_text()
    header = f"PED Lyngby Model - {scenario_name.replace('_', ' ').title()} Scenario"
    content = content.replace(
        "PED Lyngby Model - Template for New Scenarios",
        header
    )
    content = content.replace(
        "This is a template file for creating new scenarios. Copy this file and modify it\nto implement a new scenario.",
        description
    )
    # other replacements...
    content = content.replace(
        "Building new scenario network...",
        f"Building {scenario_name.replace('_', ' ')} network..."
    )
    scenario_file.write_text(content)

    # Update __init__.py
    init_file = scenarios_dir / '__init__.py'
    if not init_file.exists():
        print(f"Error: __init__.py not found at {init_file}")
        return False
    init_text = init_file.read_text()

    # Add import
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

    # Add to SCENARIO_FUNCTIONS
    dict_start = after.find("SCENARIO_FUNCTIONS = {")
    dict_end = after.find("}", dict_start)
    if dict_start == -1 or dict_end == -1:
        print("Error: Could not find SCENARIO_FUNCTIONS dictionary in __init__.py")
        return False
    dict_block = after[dict_start:dict_end+1]
    entry = f"    '{scenario_name}': {scenario_name}.create_network,"
    if entry not in dict_block:
        lines = dict_block.splitlines()
        # insert before closing brace
        lines.insert(-1, entry)
        new_block = "\n".join(lines)
        after = after.replace(dict_block, new_block)
    init_file.write_text(before + after)

    # Update config/config.yml
    config_path = Path('config') / 'config.yml'
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return False
    config = yaml.safe_load(config_path.read_text())
    config.setdefault('scenarios', {})
    if scenario_name in config['scenarios']:
        print(f"Warning: Scenario '{scenario_name}' already in config.yml")
    else:
        config['scenarios'][scenario_name] = {
            'description': description,
            'output_subdir': f"scenario_{scenario_name}"
        }
        config_path.write_text(yaml.dump(config, default_flow_style=False))

    # Update config/component_params.yml
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
    # Use the absolute path of the config_file to determine the project root
    # This makes it robust regardless of the initial working directory
    config_abs_path = Path(config_file).resolve()
    project_root = config_abs_path.parent.parent # Assuming config is in 'project_root/config'
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
    """
    Summary robusto per il baseline:
    - DH: costo su calore CONSEGNATO via 'Heat import';
           prezzo a valle da links_t.marginal_cost['Heat import']
           oppure ricostruito da prezzo del generatore 'Heating' / eta_dh.
    - Elettricità: costo import da grid_price_series (o ricostruito) e ricavo export.
    - CO2: da fattori sui link ('Grid Import' e 'Heat import').
    - KPI PV: autoconsumo ed export.
    """
    import pandas as pd
    import numpy as np

    snaps = network.snapshots

    # Pesi orari (default 1.0)
    if not hasattr(network.snapshot_weightings, "generators"):
        network.snapshot_weightings["generators"] = pd.Series(1.0, index=snaps)
    w = network.snapshot_weightings.generators.reindex(snaps).fillna(1.0)

    links = network.links
    Lp0   = getattr(getattr(network, "links_t", None), "p0", pd.DataFrame())
    Lp1   = getattr(getattr(network, "links_t", None), "p1", pd.DataFrame())
    Lmc_t = getattr(getattr(network, "links_t", None), "marginal_cost", pd.DataFrame())
    Gp_t  = getattr(getattr(network, "generators_t", None), "p", pd.DataFrame())
    Gmc_t = getattr(getattr(network, "generators_t", None), "marginal_cost", pd.DataFrame())

    # --- Helper sicuri ---
    def s_or(df, col, fill=0.0):
        return df.get(col, pd.Series(fill, index=snaps)).reindex(snaps).fillna(fill)

    def eff(lnk):
        try: return float(links.at[lnk, "efficiency"])
        except Exception: return 1.0

    # ================= PV e carichi =================
    pv_dispatch = s_or(Gp_t, "Rooftop PV")  # MW lato PV Bus
    # Flusso PV->Building (lato PV Bus)
    pv_to_build = s_or(Lp0, "PV Autoconsumo").clip(lower=0.0)
    # Export PV: tutti i link in uscita dal PV Bus tranne 'PV Autoconsumo'
    pv_out_links = [n for n, r in links.iterrows() if r.get("bus0","") == "PV Bus" and n != "PV Autoconsumo"]
    pv_export_p0 = sum(Lp0[l].clip(lower=0.0) for l in pv_out_links) if pv_out_links else pd.Series(0.0, index=snaps)
    # Export PV lato rete (p0 * efficienza)
    pv_export_grid = sum(Lp0[l].clip(lower=0.0) * eff(l) for l in pv_out_links) if pv_out_links else pd.Series(0.0, index=snaps)

    # ================= Import/Export elettrico =================
    # Import (MW lato Building) = -p1('Grid Import')
    imp_flow = s_or(Lp1, "Grid Import").mul(-1.0).clip(lower=0.0)
    # --- Peak Import (MW) e timestamp ---
    peak_import_mw = float(imp_flow.max())
    peak_import_ts = imp_flow.idxmax()

    # Export lato rete = tutti i link con bus1 == 'Grid Elec Buy' (include PV Export)
    grid_side_exports = [n for n, r in links.iterrows() if r.get("bus1","") == "Grid Elec Buy"]
    exp_flow = sum(Lp0[l].clip(lower=0.0) * eff(l) for l in grid_side_exports) if grid_side_exports else pd.Series(0.0, index=snaps)

    # Prezzo elettrico (EUR/MWh) — usa la serie salvata dal baseline
    if hasattr(network, "grid_price_series") and network.grid_price_series is not None:
        grid_price = pd.Series(network.grid_price_series, index=snaps).astype(float)
    else:
        # fallback da generatore "Grid" ed efficienza trasformatore
        if "Grid" in getattr(Gmc_t, "columns", []):
            gen_mcost = Gmc_t["Grid"].reindex(snaps).fillna(0.0)
        elif "Grid" in network.generators.index and "marginal_cost" in network.generators.columns:
            gen_mcost = pd.Series(float(network.generators.at["Grid","marginal_cost"]), index=snaps)
        else:
            gen_mcost = pd.Series(0.0, index=snaps)
        eta_grid = eff("Grid Import") if "Grid Import" in links.index else 1.0
        grid_price = (gen_mcost / max(eta_grid, 1e-12)).astype(float)

    import_cost   = float((imp_flow * grid_price * w).sum())
    export_revenue= float((exp_flow * grid_price * w).sum())  # export_price = import_price

    # ================= District Heating (DH) =================
    # Link di import DH (baseline crea proprio "Heat import")
    dh_link = "Heat import" if "Heat import" in links.index else None

    # Calore DH consegnato (MW_th lato Building) = -p1('Heat import')
    if dh_link and dh_link in Lp1.columns:
        heat_dh_delivered = (-Lp1[dh_link]).reindex(snaps).fillna(0.0).clip(lower=0.0)
    else:
        heat_dh_delivered = pd.Series(0.0, index=snaps)

        # --- Peak Import Termico (MW_th) e timestamp ---
    peak_heat_import_mwth = float(heat_dh_delivered.max())
    peak_heat_import_ts   = heat_dh_delivered.idxmax()


    # Efficienza DH per riportare i prezzi a valle
    eta_dh = eff(dh_link) if dh_link else 1.0
    # a) costo effettivo in obiettivo (se disponibile): p_gen * mc_gen
    if "Heating" in getattr(Gp_t, "columns", []) and "Heating" in getattr(Gmc_t, "columns", []):
        heat_cost_objective = float((Gp_t["Heating"].reindex(snaps).fillna(0.0)
                                    * Gmc_t["Heating"].reindex(snaps).fillna(0.0)
                                    * w).sum())
    else:
        heat_cost_objective = None

    # b) prezzo a valle (EUR/MWh_th) da usare come fallback robusto
    dh_price_net = None
    if dh_link and hasattr(network.links_t, "marginal_cost") and dh_link in Lmc_t.columns:
        _cand = Lmc_t[dh_link].reindex(snaps).astype(float)
        # usa il prezzo sul link solo se ha contenuto informativo (non tutto zero/NaN)
        if _cand.abs().sum(skipna=True) > 1e-9:
            dh_price_net = _cand.fillna(0.0)
    if dh_price_net is None:
        # ricava dal generatore: costo_consegna = costo_gen / eta_dh
        if "Heating" in getattr(Gmc_t, "columns", []):
            dh_price_gen = Gmc_t["Heating"].reindex(snaps).fillna(0.0).astype(float)
        elif "Heating" in network.generators.index and "marginal_cost" in network.generators.columns:
            dh_price_gen = pd.Series(float(network.generators.at["Heating","marginal_cost"]), index=snaps)
        else:
            dh_price_gen = pd.Series(0.0, index=snaps)
        dh_price_net = (dh_price_gen / max(eta_dh, 1e-12)).astype(float)

    heat_cost_dh_fallback = float((heat_dh_delivered * dh_price_net * w).sum())
    # c) scegli la contabilizzazione coerente con il modello
    heat_cost_dh = heat_cost_objective if heat_cost_objective is not None else heat_cost_dh_fallback

    # ================= CO2 =================
    co2_elec_kg = 0.0
    if "Grid Import" in links.index and "co2_emission_factor_kg_per_mwh" in links.columns:
        try:
            co2_elec_kg = float((imp_flow * w).sum()) * float(links.at["Grid Import","co2_emission_factor_kg_per_mwh"])
        except Exception:
            pass

    co2_heat_kg = 0.0
    if dh_link and "co2_emission_factor_kg_per_mwh" in links.columns:
        try:
            co2_heat_kg = float((heat_dh_delivered * w).sum()) * float(links.at[dh_link,"co2_emission_factor_kg_per_mwh"])
        except Exception:
            pass

    # ================= Totali e KPI =================
    tot_import_mwh  = float((imp_flow * w).sum())
    tot_export_mwh  = float((exp_flow * w).sum())
    tot_pv_mwh      = float((pv_dispatch * w).sum())
    pv_to_build_mwh = float((pv_to_build * w).sum())
    pv_export_mwh   = float((pv_export_grid * w).sum())  # export PV lato rete
    dh_delivered_mwh= float((heat_dh_delivered * w).sum())

    total_cost = float(import_cost + heat_cost_dh - export_revenue)

    return pd.DataFrame([{
        # Energia
        "Total PV Produced (MWh)":           tot_pv_mwh,
        "PV → Building (MWh)":               pv_to_build_mwh,
        "PV → Grid (MWh)":                   pv_export_mwh,
        "Total Grid Import (MWh)":           tot_import_mwh,
        "Total Grid Export (MWh)":           tot_export_mwh,
        "DH Heat Delivered (MWh_th)":        dh_delivered_mwh,
        "Peak Import (MW)":                   peak_import_mw,
        "Peak Import Timestamp":             peak_import_ts,
        "Thermal Peak Import (MW_th)":        peak_heat_import_mwth,
        "Thermal Peak Import Timestamp":      peak_heat_import_ts,



        # Costi/Ricavi
        "Electricity Import Cost (EUR)":     import_cost,
        "District Heating Cost (EUR)":       heat_cost_dh,
        "Export Revenue (EUR)":              export_revenue,
        "Total Operational Cost (EUR)":      total_cost,

        # CO2
        "CO2 Electricity Import (t)":        co2_elec_kg / 1000.0,
        "CO2 District Heating (t)":          co2_heat_kg / 1000.0,
        "CO2 Total (t)":                     (co2_elec_kg + co2_heat_kg) / 1000.0,
    }])


def generate_hourly_df(network) -> pd.DataFrame:
    snaps = network.snapshots

    # Domande (MW)
    elec_cols = [c for c in network.loads_t.p_set.columns
                 if network.loads.loc[c].carrier == 'electricity']
    heat_cols = [c for c in network.loads_t.p_set.columns
                 if network.loads.loc[c].carrier == 'heat']

    elec = network.loads_t.p_set[elec_cols].sum(axis=1) if elec_cols else pd.Series(0.0, index=snaps)
    heat = network.loads_t.p_set[heat_cols].sum(axis=1) if heat_cols else pd.Series(0.0, index=snaps)

    # PV dispatch (MW lato PV Bus)
    pv_dispatch = network.generators_t.p.get("Rooftop PV", pd.Series(0.0, index=snaps))

    # Import (MW lato Building) = -p1(Grid Import)
    if "Grid Import" in network.links.index:
        grid_import_mw = (-network.links_t.p1["Grid Import"]).clip(lower=0.0)
    else:
        grid_import_mw = pd.Series(0.0, index=snaps)

    # ======= EXPORT lato rete (robusto a nomi link) =======
    # - tutti i link con bus0 == 'Building Elec' (export via building)
    # - tutti i link con bus0 == 'PV Bus' tranne 'PV Autoconsumo' (export diretto da PV)
    links = network.links

    bld_export_links = links.index[links.bus0.eq("Building Elec")]
    pv_out_links     = links.index[links.bus0.eq("PV Bus")]
    pv_export_links  = [lnk for lnk in pv_out_links if lnk != "PV Autoconsumo"]

    # lato rete = p0 * eff per ciascun link
    grid_export_from_building_mw = sum(
        (network.links_t.p0[lnk].clip(lower=0.0) *
         float(links.at[lnk, "efficiency"]))
        for lnk in bld_export_links
    ) if len(bld_export_links) else pd.Series(0.0, index=snaps)

    grid_export_from_pv_mw = sum(
        (network.links_t.p0[lnk].clip(lower=0.0) *
         float(links.at[lnk, "efficiency"]))
        for lnk in pv_export_links
    ) if len(pv_export_links) else pd.Series(0.0, index=snaps)

    grid_export_mw = grid_export_from_building_mw + grid_export_from_pv_mw

    # DataFrame base in kWh
    df = pd.DataFrame({
        "timestamp": snaps,
        "electricity_load_kwh": (elec.values * 1000),
        "heat_load_kwh":        (heat.values * 1000),
        "pv_generation_kwh":    (pv_dispatch.values * 1000), 
        "grid_import_kwh":      (grid_import_mw.values * 1000),
        "grid_export_kwh":      (grid_export_mw.values * 1000),
    })

    # ======= PV split lato PV Bus =======
    # PV -> Building (p0 su 'PV Autoconsumo')
    pv_to_building_mw = network.links_t.p0.get("PV Autoconsumo", pd.Series(0.0, index=snaps)).clip(lower=0.0)

    # Autoconsumo = min(PV->Building, carico elettrico)
    pv_self_mw = pd.concat([pv_to_building_mw, elec], axis=1).min(axis=1)

    # Export da PV Bus (p0 somma di tutti i link in uscita dal PV Bus escluso 'PV Autoconsumo')
    pv_export_p0_mw = sum(
        network.links_t.p0[lnk].clip(lower=0.0)
        for lnk in pv_export_links
    ) if len(pv_export_links) else pd.Series(0.0, index=snaps)

    # Curtailment "di dispatch" = residuo (dovrebbe ~0 salvo tolleranze)
    pv_curt_dispatch_mw = (pv_dispatch - pv_self_mw - pv_export_p0_mw).clip(lower=0.0)

    # Export PV lato rete per il grafico split (coerente con energy balance)
    pv_export_gridside_mw = grid_export_from_pv_mw

    # Colonne in kWh
    df["pv_to_building_kwh"]      = pv_to_building_mw.values      * 1000
    df["pv_self_consumed_kwh"]    = pv_self_mw.values             * 1000
    df["pv_exported_to_grid_kwh"] = pv_export_gridside_mw.values  * 1000
    df["pv_curtailed_kwh"]        = pv_curt_dispatch_mw.values    * 1000

    # ======= Validazione bilancio elettrico =======
    hourly_supply = df["pv_generation_kwh"] + df["grid_import_kwh"]
    hourly_demand = df["electricity_load_kwh"] + df["grid_export_kwh"]
    err = (hourly_supply - hourly_demand).abs()
    print("ENERGY BALANCE VALIDATION:")
    print(f"  Max hourly error: {float(err.max()):.6f} kWh")
    print(f"  Mean hourly error: {float(err.mean()):.6f} kWh")
    annual_error = float(abs(hourly_supply.sum() - hourly_demand.sum()))
    rel = annual_error / max(hourly_demand.sum(), 1e-9) * 100
    print(f"  Annual balance error: {annual_error:.6f} kWh ({rel:.4f}%)")

    # Indice
    df = df.set_index("timestamp")
    df.index.name = "snapshot"
    df["timestamp"] = df.index
    return df


def quick_export_checks(net) -> None:
    """Controlli rapidi su prezzi, componenti e flussi export/surplus (robusti a export dal PV Bus)."""
    try:
        print("\n=== QUICK EXPORT/PRICE CHECK ===")
        snaps = net.snapshots
        w = getattr(net.snapshot_weightings, "generators", pd.Series(1.0, index=snaps))

        # Prezzi
        if hasattr(net, "grid_price_series"):
            gp = pd.Series(net.grid_price_series, index=snaps)
            print(f"Prezzo import/export (EUR/MWh): min={float(gp.min()):.2f}, max={float(gp.max()):.2f}")
        else:
            print("WARNING: net.grid_price_series non presente sul network")

        # Componenti chiave
        print("Presenza 'Export Revenue' (generatore ricavo):", "Export Revenue" in net.generators.index)
        print("Presenza 'PV Export' (link):", "PV Export" in net.links.index)
        if "PV Export" in net.links.index:
            print("  PV Export p_nom:", net.links.at["PV Export", "p_nom"])
            print("  PV Export efficiency:", net.links.at["PV Export", "efficiency"])
        if "Grid Import" in net.links.index:
            print("  Grid Import p_nom:", net.links.at["Grid Import", "p_nom"])
            print("  Grid Import efficiency:", net.links.at["Grid Import", "efficiency"])

        # Serie utili
        pv = net.generators_t.p.get("Rooftop PV", pd.Series(0.0, index=snaps))
        elec_load_cols = [c for c in net.loads.index if net.loads.loc[c, "carrier"] == "electricity"]
        load = net.loads_t.p_set[elec_load_cols].sum(axis=1) if elec_load_cols else pd.Series(0.0, index=snaps)

        # Link di export lato grid (robusto a nomi/posizione)
        links = net.links
        bld_export_links = links.index[links.bus0.eq("Building Elec")]
        pv_out_links     = links.index[links.bus0.eq("PV Bus")]
        pv_export_links  = [lnk for lnk in pv_out_links if lnk != "PV Autoconsumo"]

        grid_export_from_building = sum(
            (net.links_t.p0[lnk].clip(lower=0.0) * float(links.at[lnk, "efficiency"]))
            for lnk in bld_export_links
        ) if len(bld_export_links) else pd.Series(0.0, index=snaps)

        grid_export_from_pv = sum(
            (net.links_t.p0[lnk].clip(lower=0.0) * float(links.at[lnk, "efficiency"]))
            for lnk in pv_export_links
        ) if len(pv_export_links) else pd.Series(0.0, index=snaps)

        # Flusso PV→building (lato PV bus) e autoconsumo
        pv_to_build = net.links_t.p0.get("PV Autoconsumo", pd.Series(0.0, index=snaps)).clip(lower=0.0)
        pv_self     = pd.concat([pv_to_build, load], axis=1).min(axis=1)

        # Export lato PV-bus = eccedenza PV→Building + export diretti dal PV bus
        pv_export_p0 = (pv_to_build - pv_self).clip(lower=0.0)
        for lnk in pv_export_links:
            pv_export_p0 = pv_export_p0.add(net.links_t.p0[lnk].clip(lower=0.0), fill_value=0.0)

        # Curtailment di dispatch = residuo del dispatch PV
        pv_curt_dispatch = (pv - pv_self - pv_export_p0).clip(lower=0.0)

        # Stime aggregate
        exp_hours = int(((grid_export_from_building + grid_export_from_pv) > 1e-9).sum())
        print("Ore con PV > Load:", int((pv > load).sum()))
        print("Ore con export > 0:", exp_hours)
        print("PV curtailment di dispatch (MWh):", float((pv_curt_dispatch * w).sum()))

        # Info costi negativi sul link (se presenti)
        mc_links = getattr(net.links_t, "marginal_cost", pd.DataFrame())
        if "PV Export" in mc_links.columns:
            ser = mc_links["PV Export"]
            print("PV Export marginal_cost (EUR/MWh): min={:.2f}, max={:.2f}".format(float(ser.min()), float(ser.max())))
        else:
            print("INFO: marginal_cost orario per 'PV Export' non trovato (ok se usi altri link di export)")

    except Exception as e:
        print("Quick check failed:", e)

def run_scenario(scenario_name: str, config_file: str, params_file: str) -> pd.DataFrame:
    print(f"\n--- Running Scenario: {scenario_name} ---")
    
    # Resolve absolute paths
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
        result = build_network(scenario_name, temp_cfg, str(params_abs_path), data_in)
        Path(temp_cfg).unlink()

        if isinstance(result, tuple):
            net, _, _ = result
        else:
            net = result
    except RuntimeError as e:
        print(e)
        return pd.DataFrame()
    
    # ---- LOPF ----
    sim = override.get('simulation_settings', {})
    try:
        run_lopf(net, sim.get('solver', 'highs'), sim.get('solver_options', {}))
    except RuntimeError as e:
        print(e)
        return pd.DataFrame()
    
    # ---- VERIFICA SUBITO DOPO LOPF ----
    quick_export_checks(net)

    # ---- Salvataggi ----
    net_file = out_dir / f"{scenario_name}_network_results.nc"
    try:
        net.export_to_netcdf(str(net_file))
    except Exception as e:
        print(f"Error saving network: {e}")


    # --- Peak Shaving vs baseline, se non siamo nel baseline ---
    # 1) Calcola il summary dello scenario (contiene già Peak Import)
    summary_df = calculate_summary(net)

    # 2) Peak Shaving vs baseline: aggiungi nel summary PRIMA di salvarlo
    try:
        if scenario_name.lower() != "baseline":
            base_subdir = config.get('scenarios', {}).get('baseline', {}).get('output_subdir', 'scenario_baseline')
            base_dir = Path(data_out_base) / base_subdir
            base_summary_file = base_dir / "baseline_summary.csv"

            if base_summary_file.exists():
                base_sum = pd.read_csv(base_summary_file)
                if "Peak Import (MW)" in base_sum.columns and len(base_sum) > 0:
                    baseline_peak = float(base_sum.loc[0, "Peak Import (MW)"])
                    scenario_peak = float(summary_df.loc[0, "Peak Import (MW)"])
                    delta_mw = max(baseline_peak - scenario_peak, 0.0)
                    denom = max(baseline_peak, 1e-9)
                    shaving_pct = 100.0 * delta_mw / denom
                    summary_df["Peak Shaving (MW)"] = delta_mw
                    summary_df["Peak Shaving (%)"]  = shaving_pct
            else:
                print(f"[INFO] Baseline summary non trovato a: {base_summary_file} → salto il peak shaving.")
    except Exception as e:
        print(f"[WARN] Peak shaving non calcolato: {e}")



    try:
        if scenario_name.lower() != "baseline":
            base_subdir = config.get('scenarios', {}).get('baseline', {}).get('output_subdir', 'scenario_baseline')
            base_dir = Path(data_out_base) / base_subdir
            base_summary_file = base_dir / "baseline_summary.csv"

            if base_summary_file.exists():
               base_sum = pd.read_csv(base_summary_file)
                # Shaving elettrico (già presente)
               if "Peak Import (MW)" in base_sum.columns and len(base_sum) > 0:
                    baseline_peak_el = float(base_sum.loc[0, "Peak Import (MW)"])
                    scenario_peak_el = float(summary_df.loc[0, "Peak Import (MW)"])
                    delta_el = max(baseline_peak_el - scenario_peak_el, 0.0)
                    summary_df["Peak Shaving (MW)"] = delta_el
                    summary_df["Peak Shaving (%)"]  = 100.0 * delta_el / max(baseline_peak_el, 1e-9)

                # Shaving TERMICO
            if "Thermal Peak Import (MW_th)" in base_sum.columns and len(base_sum) > 0:
                baseline_peak_th = float(base_sum.loc[0, "Thermal Peak Import (MW_th)"])
                scenario_peak_th = float(summary_df.loc[0, "Thermal Peak Import (MW_th)"])
                delta_th = max(baseline_peak_th - scenario_peak_th, 0.0)
                summary_df["Thermal Peak Shaving (MW_th)"] = delta_th
                summary_df["Thermal Peak Shaving (%)"]     = 100.0 * delta_th / max(baseline_peak_th, 1e-9)
            else:
                print(f"[INFO] Baseline summary non trovato a: {base_summary_file} → salto il peak shaving termico.")
    except Exception as e:
        print(f"[WARN] Peak shaving termico non calcolato: {e}")


    # 3) Ora salva il summary (con le colonne di shaving se presenti)
    summary_df.to_csv(out_dir / f"{scenario_name}_summary.csv", index=False)
    
    hourly_df = generate_hourly_df(net)
    hourly_df.to_csv(out_dir / f"{scenario_name}_hourly_results.csv", index=False)

    # ---- Metriche annuali per il grafico energy balance ----
    metrics = {
        "total_pv_mwh":            hourly_df["pv_generation_kwh"].sum() / 1000,
        "total_import_mwh":        hourly_df["grid_import_kwh"].sum() / 1000,
        "total_export_mwh":        hourly_df["grid_export_kwh"].sum() / 1000,
        "total_electric_load_mwh": hourly_df["electricity_load_kwh"].sum() / 1000,
        "total_thermal_load_mwh":  hourly_df["heat_load_kwh"].sum() / 1000,
    }

    # ---- Plot ----
    try:
        plot_time_series({
            'timestamps': hourly_df['timestamp'],
            'electric_load': hourly_df['electricity_load_kwh'],
            'thermal_load':  hourly_df['heat_load_kwh'],
            'pv_generation': hourly_df['pv_generation_kwh'],
            'grid_import':   hourly_df['grid_import_kwh'],
            'grid_export':   hourly_df['grid_export_kwh'],
        }, scenario_name=scenario_name, output_dir=str(out_dir))
        
        plot_energy_balance(metrics, scenario_name=scenario_name, output_dir=str(out_dir))
        plot_pv_split(hourly_df, scenario_name=scenario_name, output_dir=str(out_dir))
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
    print(f"Command: {args.scenario}, ")
    print('*' * 40)
    
    # Removed the problematic os.chdir line.
    # The script should be run from the project root.
    # If not, the user needs to provide absolute paths or change directory manually.

    if args.command == 'create':
        success = create_scenario(args.name, args.description)
        if not success:
            sys.exit(1)
    elif args.command == 'run':
        # Pass the paths directly as they are provided from the command line,
        # assuming the user runs from the project root.
        df = run_scenario(args.scenario, args.config, args.params)
        if df.empty:
            sys.exit(1)


if __name__ == '__main__':
    main()
