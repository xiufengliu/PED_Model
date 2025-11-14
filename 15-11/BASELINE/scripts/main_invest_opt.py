#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run scenario (invest_opt) e genera KPI INVESTIMENTO PRO:
- CSV chiari (summary, investment_summary, finance_per_tech, finance_project)
- Grafici professionali: CAPEX upfront/annuo, payback, NPV, dimensionamento (MW/MWh)

Assunzioni & Config:
- Di default si assume che 'capital_cost' in PyPSA sia ANNUALIZZATO (€/unità/anno).
- Se invece nel dataset è "upfront" (€/unità una tantum), imposta nel config:
    finance:
      capital_cost_is_annualized: false
  e fornisci anche 'lifetime_years_by_tech' per ogni tecnologia (se diversi dai default).
"""

import sys
import argparse
import yaml
import traceback
from pathlib import Path
import pandas as pd

# percorso progetto (scripts/ è a un livello sotto la root)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenarios import get_scenario_function
from scenarios.invest_opt import calculate_economics
from scripts.plotting_utils import (
    # operativi minimi
    plot_time_series, plot_energy_balance,
    # finanza
    prepare_financials,
    plot_capex_upfront_bar, plot_investment_cost_pie, plot_payback_by_tech,
    plot_npv_waterfall, plot_capex_ann_bar, plot_capacity_sticks,
    plot_finance_summary_card,
    annuity_factor, annualized_from_upfront,
    # invest_opt publication-quality plots
    plot_economics_dashboard, plot_npv_sensitivity
)

HOURS_IN_YEAR = 8760


# ---------------- utils ----------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def determine_paths(config: dict, config_file: str) -> tuple[str, str]:
    cfg_abs = Path(config_file).resolve()
    root = cfg_abs.parent.parent
    data_in = root / config.get('paths', {}).get('data_input',  'data/input')
    data_out= root / config.get('paths', {}).get('data_output', 'data/output')
    return str(data_in), str(data_out)

def override_sim_hours(config: dict, hours: int = HOURS_IN_YEAR) -> dict:
    config['simulation_settings'] = config.get('simulation_settings', {})
    config['simulation_settings']['num_hours'] = hours
    return config

def write_temp_config(cfg: dict, config_file: str) -> str:
    temp = Path(config_file).parent / "temp_config.yml"
    with open(temp, "w") as f:
        yaml.dump(cfg, f)
    return str(temp)

def ensure_dir(p: Path) -> None:
    """Crea in modo sicuro la directory p (o il parent di un file) se mancante."""
    d = p if p.suffix == "" else p.parent
    d.mkdir(parents=True, exist_ok=True)


# ---------------- build + solve ----------------
def build_network(scenario: str, config_file: str, params_file: str, data_path: str):
    try:
        fn = get_scenario_function(scenario)
        return fn(config_file, params_file, data_path)
    except Exception:
        raise RuntimeError(f"Error building network for '{scenario}': {traceback.format_exc()}")

def run_lopf(network, solver_name: str, solver_options: dict):
    if hasattr(network, "optimize"):   # PyPSA >= 0.27
        network.optimize(solver_name=solver_name, solver_options=solver_options)
    elif hasattr(network, "lopf"):     # fallback PyPSA <= 0.26
        network.lopf(solver_name=solver_name, solver_options=solver_options)
    else:
        raise RuntimeError("Neither Network.optimize nor Network.lopf is available.")


# ---------------- summaries ----------------
def _calc_summaries(n, finance_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    summary_row (annuo) + tabella CAPEX per tecnologia.
    Interpreta 'capital_cost' come annualizzato, a meno di override in config.
    """
    snaps = n.snapshots
    try:
        w = n.snapshot_weightings["generators"].reindex(snaps).fillna(1.0)
    except Exception:
        w = pd.Series(1.0, index=snaps)

    def s_or(df, col, fill=0.0):
        if df is None or not hasattr(df, "reindex"):
            return pd.Series(fill, index=snaps)
        return df.get(col, pd.Series(fill, index=snaps)).reindex(snaps).fillna(fill)

    lt_p1 = getattr(getattr(n, "links_t", None), "p1", pd.DataFrame())
    Gp    = getattr(getattr(n, "generators_t", None), "p", pd.DataFrame())
    Gmc   = getattr(getattr(n, "generators_t", None), "marginal_cost", pd.DataFrame())
    links = getattr(n, "links", pd.DataFrame())

    # prezzi elettricità & DH lato generatore
    if hasattr(n, "grid_price_series") and n.grid_price_series is not None:
        price_elec_gen = pd.Series(n.grid_price_series, index=snaps).astype(float)
    elif "Grid" in Gmc.columns:
        price_elec_gen = Gmc["Grid"].reindex(snaps).fillna(0.0).astype(float)
    else:
        price_elec_gen = pd.Series(0.0, index=snaps)

    if "Heating" in Gmc.columns:
        price_dh_gen = Gmc["Heating"].reindex(snaps).fillna(0.0).astype(float)
    else:
        price_dh_gen = pd.Series(0.0, index=snaps)

    # efficienze (scalari)
    def eff(name, default=1.0):
        try:
            return max(float(n.links.at[name, "efficiency"]), 1e-12)
        except Exception:
            return max(default, 1e-12)

    eta_grid = eff("Grid Import", 1.0)
    hp_eff   = eff("Heat Pump", 3.5)
    eta_dh   = eff("Heat import", 1.0)

    # flussi net lato arrivo (MWh/anno sugli snapshot considerati)
    imp_building_net = (-s_or(lt_p1, "Grid Import")).clip(lower=0.0)
    imp_batt_net     = (-s_or(lt_p1, "Battery Charge")).clip(lower=0.0)
    hp_net_th        = (-s_or(lt_p1, "Heat Pump")).clip(lower=0.0)
    q_dh_net         = (-s_or(lt_p1, "Heat import")).clip(lower=0.0)
    exp_net          = (-s_or(lt_p1, "PV Export")).clip(lower=0.0)

    imp_hp_gen_ts = (hp_net_th / hp_eff).fillna(0.0)           # MWh_e equivalenti
    imp_building_gen = ((imp_building_net / eta_grid) * w).sum()
    imp_batt_gen     = ((imp_batt_net     / eta_grid) * w).sum()
    imp_hp_gen       = (imp_hp_gen_ts * w).sum()
    total_grid_buy_gen = float(imp_building_gen + imp_batt_gen + imp_hp_gen)

    q_dh_gen = (q_dh_net / max(eta_dh, 1e-12))

    # costi e ricavi (€/anno)
    cost_grid_building = float(((imp_building_net/eta_grid) * price_elec_gen * w).sum())
    cost_grid_batt     = float(((imp_batt_net    /eta_grid) * price_elec_gen * w).sum())
    cost_grid_hp       = float(((imp_hp_gen_ts)            * price_elec_gen * w).sum())
    electricity_import_cost = cost_grid_building + cost_grid_batt + cost_grid_hp

    cost_dh = float((q_dh_gen * price_dh_gen * w).sum())
    export_revenue = float((exp_net * price_elec_gen * w).sum())
    total_opex = electricity_import_cost + cost_dh - export_revenue

    # emissioni (t/anno)
    def co2_factor(link_name, gen_name):
        if (link_name in links.index) and ("co2_emission_factor_kg_per_mwh" in links.columns):
            v = links.at[link_name, "co2_emission_factor_kg_per_mwh"]
        else:
            gens = getattr(n, "generators", pd.DataFrame())
            if (gen_name in gens.index) and ("co2_emission_factor_kg_per_mwh" in gens.columns):
                v = gens.at[gen_name, "co2_emission_factor_kg_per_mwh"]
            else:
                v = 0.0
        try:
            return float(0.0 if v in [None,""] else v)
        except Exception:
            return 0.0

    grid_co2 = co2_factor("Grid Import", "Grid")
    dh_co2   = co2_factor("Heat import", "Heating")

    co2_building_kg = float(((imp_building_net * (grid_co2 / eta_grid)) * w).sum())
    co2_batt_kg     = float(((imp_batt_net     * (grid_co2 / eta_grid)) * w).sum())
    co2_hp_kg       = float(((imp_hp_gen_ts    *  grid_co2)             * w).sum())
    co2_elec_t = (co2_building_kg + co2_batt_kg + co2_hp_kg) / 1000.0
    co2_dh_t   = float(((q_dh_net * (dh_co2 / max(eta_dh,1e-12))) * w).sum()) / 1000.0
    co2_tot_t  = co2_elec_t + co2_dh_t

    # PV e carichi (MWh/anno)
    pv_total_mwh = float((Gp.get("Rooftop PV", pd.Series(0.0, index=snaps)) * w).sum())
    Lp = getattr(n, "loads_t", None)
    elec_load = Lp.p.get("Building Elec Load", pd.Series(0.0, index=snaps))
    elec_load_tot = float((elec_load * w).sum())
    grid_to_building_net = float((imp_building_net * w).sum())
    el_self_suff = 1.0 - (grid_to_building_net / max(elec_load_tot, 1e-9))
    pv_self_cons = (pv_total_mwh - float((exp_net * w).sum())) / max(pv_total_mwh, 1e-9)
    nab_mwh = pv_total_mwh - float((exp_net * w).sum()) - grid_to_building_net

    # --------- CAPEX per tecnologia ---------
    # Estrazione capacità ottime
    PV_p = n.generators.at["Rooftop PV", "p_nom_opt"] if "Rooftop PV" in n.generators.index else 0.0
    TR_p = n.links.at["Grid Import", "p_nom_opt"] if "Grid Import" in n.links.index else 0.0
    EXP_p = n.links.at["PV Export", "p_nom_opt"] if "PV Export" in n.links.index else 0.0
    BAT_p = n.links.at["Battery Charge", "p_nom_opt"] if "Battery Charge" in n.links.index else 0.0
    BAT_e = n.stores.at["Battery Store", "e_nom_opt"] if "Battery Store" in n.stores.index else 0.0
    TES_p = n.links.at["Thermal Charge", "p_nom_opt"] if "Thermal Charge" in n.links.index else 0.0
    TES_e = n.stores.at["Thermal Store", "e_nom_opt"] if "Thermal Store" in n.stores.index else 0.0
    HP_p  = n.links.at["Heat Pump", "p_nom_opt"] if "Heat Pump" in n.links.index else 0.0

    # capital_cost per unità come dato PyPSA
    cc_PV   = n.generators.at["Rooftop PV", "capital_cost"] if "Rooftop PV" in n.generators.index else 0.0
    cc_TR   = n.links.at["Grid Import", "capital_cost"] if "Grid Import" in n.links.index else 0.0
    cc_EXP  = n.links.at["PV Export", "capital_cost"] if "PV Export" in n.links.index else 0.0
    cc_BATp = n.links.at["Battery Charge", "capital_cost"] if "Battery Charge" in n.links.index else 0.0
    cc_BATe = n.stores.at["Battery Store", "capital_cost"] if "Battery Store" in n.stores.index else 0.0
    cc_TESp = n.links.at["Thermal Charge", "capital_cost"] if "Thermal Charge" in n.links.index else 0.0
    cc_TESe = n.stores.at["Thermal Store", "capital_cost"] if "Thermal Store" in n.stores.index else 0.0
    cc_HP   = n.links.at["Heat Pump", "capital_cost"] if "Heat Pump" in n.links.index else 0.0

    # flag: il capital_cost è annualizzato (default) o upfront?
    r = float(finance_cfg.get("discount_rate", 0.05))
    lifetimes = finance_cfg.get("lifetime_years_by_tech", {})
    def life(tech, default):
        return lifetimes.get(tech, default)

    capital_cost_is_annualized = bool(finance_cfg.get("capital_cost_is_annualized", True))

    cap_items = []  # (Technology, Capacity, CAPEX_ann_EUR)
    def add_item(tech, capacity, cc, default_life):
        if capital_cost_is_annualized:
            cap_ann = float(cc) * float(capacity)
        else:
            # cc è upfront → converti a annualizzato con AF(r, life)
            cap_ann = annualized_from_upfront(float(cc) * float(capacity), r, life(tech, default_life))
        cap_items.append((tech, float(capacity), cap_ann))

    add_item("PV (MW)", PV_p, cc_PV, 25)
    add_item("Transformer (MW)", TR_p, cc_TR, 30)
    add_item("Export Link (MW)", EXP_p, cc_EXP, 20)
    add_item("Battery Power (MW)", BAT_p, cc_BATp, 15)
    add_item("Battery Energy (MWh)", BAT_e, cc_BATe, 15)
    add_item("TES Power (MW_th)", TES_p, cc_TESp, 20)
    add_item("TES Energy (MWh_th)", TES_e, cc_TESe, 20)
    add_item("Heat Pump (MW_th)", HP_p, cc_HP, 15)

    capex_ann_df = pd.DataFrame(cap_items, columns=["Technology","Capacity","CAPEX_ann_EUR"])
    total_capex_ann = float(capex_ann_df["CAPEX_ann_EUR"].sum())

    summary = {
        # costi/ricavi annuali
        "Electricity Import Cost (EUR)": electricity_import_cost,
        "District Heating Cost (EUR)":   cost_dh,
        "Export Revenue (EUR)":          export_revenue,
        "Total Operational Cost (EUR)":  total_opex,
        "Annualized CAPEX (EUR)":        total_capex_ann,
        "Total Annual System Cost (EUR)": total_opex + total_capex_ann,

        # emissioni annue
        "CO2 Electricity Import (t)":    co2_elec_t,
        "CO2 District Heating (t)":      co2_dh_t,
        "CO2 Total (t)":                 co2_tot_t,

        # capacità ottime (dimensionamento)
        "PV p_nom (MW)": PV_p,
        "Battery p_nom (MW)": BAT_p,
        "Battery e_nom (MWh)": BAT_e,
        "TES p_nom (MW_th)": TES_p,
        "TES e_nom (MWh_th)": TES_e,
        "HP p_nom (MW_th)": HP_p,
        "Transformer p_nom (MW)": TR_p,
        "Export p_nom (MW)": EXP_p,

        # KPI energetici
        "PV generation (MWh/yr)": pv_total_mwh,
        "Electric Self-Sufficiency (%)": 100.0 * max(min(el_self_suff, 1.0), 0.0),
        "PV Self-Consumption (%)":       100.0 * max(min(pv_self_cons, 1.0), 0.0),
        "Net Annual Energy Balance (MWh/yr)": nab_mwh
    }
    return pd.DataFrame([summary]), capex_ann_df


def _hourly_df(n) -> pd.DataFrame:
    snaps = n.snapshots
    Z = pd.Series(0.0, index=snaps)

    elec = n.loads_t.p.get('Building Elec Load', Z)
    heat = n.loads_t.p.get('Building Heat Load', Z)
    pv   = n.generators_t.p.get('Rooftop PV', Z)
    gi   = (-n.links_t.p1.get('Grid Import', Z)).clip(lower=0.0)
    ge   = (-n.links_t.p1.get('PV Export', Z)).clip(lower=0.0)

    df = pd.DataFrame({
        "timestamp": snaps,
        "electricity_load_mwh": elec.values,
        "heat_load_mwh": heat.values,
        "pv_generation_mwh": pv.values,
        "grid_import_mwh": gi.values,
        "grid_export_mwh": ge.values,
    })
    return df


# ---------------- run scenario ----------------
def run_scenario(scenario: str, config_file: str, params_file: str) -> pd.DataFrame:
    print("\n****************************************")
    print("Command: run"); print(f"Scenario: {scenario}")
    print("****************************************")

    cfg_abs = Path(config_file).resolve()
    par_abs = Path(params_file).resolve()
    cfg = load_config(str(cfg_abs))

    data_in, data_out_base = determine_paths(cfg, str(cfg_abs))
    out_root = Path(data_out_base)
    ensure_dir(out_root)

    subdir = cfg.get('scenarios', {}).get(scenario, {}).get('output_subdir', f"scenario_{scenario}")
    out_dir = out_root / subdir
    ensure_dir(out_dir)

    override = override_sim_hours(cfg)
    temp_cfg = write_temp_config(override, str(cfg_abs))

    try:
        n, _, _ = build_network(scenario, temp_cfg, str(par_abs), data_in)
    except RuntimeError as e:
        print(e)
        Path(temp_cfg).unlink(missing_ok=True)
        return pd.DataFrame()
    finally:
        Path(temp_cfg).unlink(missing_ok=True)

    sim = override.get('simulation_settings', {})
    run_lopf(n, sim.get('solver','highs'), sim.get('solver_options', {}))

    # salva network grezzo (verifica)
    net_file = out_dir / f"{scenario}_network_results.nc"
    ensure_dir(net_file)
    try:
        n.export_to_netcdf(str(net_file))
    except Exception as e:
        print(f"[WARN] saving network: {e}")

    # summary + capex_ann (interpreta annualizzato vs upfront secondo config)
    finance_cfg = override.get("finance", {})
    summary_df, capex_ann_df = _calc_summaries(n, finance_cfg)

    summary_path = out_dir / f"{scenario}_summary.csv"
    ensure_dir(summary_path)
    summary_df.to_csv(summary_path, index=False)

    # investment summary con capacità e CAPEX_ann per tecnologia
    over = summary_df.iloc[0]
    rows = []

    def capex_ann_for(label):
        s = capex_ann_df.loc[capex_ann_df["Technology"]==label, "CAPEX_ann_EUR"]
        return float(s.iloc[0]) if not s.empty else 0.0

    def add_row(tech, cap_mw=0.0, cap_mwh=0.0, capex_ann=0.0):
        rows.append({"Technology": tech, "Capacity_MW": cap_mw, "Capacity_MWh": cap_mwh, "CAPEX_ann_EUR": capex_ann})

    add_row("PV (MW)",             float(over.get("PV p_nom (MW)", 0.0)),              0.0, capex_ann_for("PV (MW)"))
    add_row("Battery Power (MW)",  float(over.get("Battery p_nom (MW)", 0.0)),         0.0, capex_ann_for("Battery Power (MW)"))
    add_row("Battery Energy (MWh)",0.0,                                                float(over.get("Battery e_nom (MWh)", 0.0)), capex_ann_for("Battery Energy (MWh)"))
    add_row("TES Power (MW_th)",   float(over.get("TES p_nom (MW_th)", 0.0)),          0.0, capex_ann_for("TES Power (MW_th)"))
    add_row("TES Energy (MWh_th)", 0.0,                                                float(over.get("TES e_nom (MWh_th)", 0.0)), capex_ann_for("TES Energy (MWh_th)"))
    add_row("Heat Pump (MW_th)",   float(over.get("HP p_nom (MW_th)", 0.0)),           0.0, capex_ann_for("Heat Pump (MW_th)"))
    add_row("Transformer (MW)",    float(over.get("Transformer p_nom (MW)", 0.0)),     0.0, capex_ann_for("Transformer (MW)"))
    add_row("Export Link (MW)",    float(over.get("Export p_nom (MW)", 0.0)),          0.0, capex_ann_for("Export Link (MW)"))

    investment_df = pd.DataFrame(rows)
    inv_path = out_dir / f"{scenario}_investment_summary.csv"
    ensure_dir(inv_path)
    investment_df.to_csv(inv_path, index=False)

    # hourly (opzionale)
    hourly_df = _hourly_df(n)
    hourly_path = out_dir / f"{scenario}_hourly_results.csv"
    ensure_dir(hourly_path)
    hourly_df.to_csv(hourly_path, index=False)

    # ---------------- KPI FINANZIARI (NPV, payback) ----------------
    r  = float(finance_cfg.get("discount_rate", 0.05))
    Y  = int(finance_cfg.get("project_years", 20))
    lt = finance_cfg.get("lifetime_years_by_tech", None)

    # baseline (se disponibile) per i risparmi annui
    baseline_dir  = Path(data_out_base) / "scenario_baseline"
    baseline_file = baseline_dir / "baseline_summary.csv"
    baseline_row  = None
    if baseline_file.exists():
        try:
            baseline_row = pd.read_csv(baseline_file).iloc[0]
        except Exception as e:
            print(f"[WARN] baseline_summary.csv unreadable: {e}")

    pertech_fin, project_kpis = prepare_financials(
        investment_df=investment_df,
        summary_row=over,
        baseline_summary_row=baseline_row,
        discount_rate=r,
        project_years=Y,
        lifetime_by_tech=lt,
    )

    pertech_path = out_dir / f"{scenario}_finance_per_tech.csv"
    ensure_dir(pertech_path)
    pertech_fin.to_csv(pertech_path, index=False)

    project_path = out_dir / f"{scenario}_finance_project.csv"
    ensure_dir(project_path)
    pd.DataFrame([project_kpis]).to_csv(project_path, index=False)

    # ---------------- Grafici chiave (anni/tasso sempre in titolo) ----------------
    try:
        # Operativi minimi (annuali)
        eb_vals = {
            'pv_generation_mwh':           hourly_df['pv_generation_mwh'].sum(),
            'grid_import_mwh':             hourly_df['grid_import_mwh'].sum(),
            'grid_export_mwh':             hourly_df['grid_export_mwh'].sum(),
            'total_electric_load_mwh':     hourly_df['electricity_load_mwh'].sum(),
            'total_thermal_load_mwh':      hourly_df['heat_load_mwh'].sum(),
            'co2_emissions_kg':            1000.0 * float(over.get("CO2 Total (t)", 0.0)),
        }
        ensure_dir(out_dir)
        plot_energy_balance(eb_vals, scenario, str(out_dir))

        # Investimento
        plot_capex_upfront_bar(pertech_fin, scenario, str(out_dir), years=Y, r=r)
        plot_capex_ann_bar(investment_df, scenario, str(out_dir), years=Y, r=r)
        plot_investment_cost_pie(pertech_fin, scenario, str(out_dir), years=Y, r=r)
        plot_capacity_sticks(investment_df, scenario, str(out_dir))
        plot_payback_by_tech(pertech_fin, scenario, str(out_dir), years=Y, r=r)
        plot_npv_waterfall(project_kpis, scenario, str(out_dir))
        plot_finance_summary_card(project_kpis, scenario, str(out_dir))

        # ---------------- INVEST_OPT: Advanced Economic Analysis ----------------
        if scenario == "invest_opt":
            print("\n" + "="*60)
            print("CALCULATING ADVANCED ECONOMIC METRICS (NPV, CAPEX, PAYBACK)")
            print("="*60)

            # Load component parameters for economic calculations
            import yaml
            with open(params_file, 'r', encoding='utf-8') as f:
                params = yaml.safe_load(f)

            # Calculate comprehensive economics
            economics = calculate_economics(n, params, cfg)

            # Save economics results to JSON
            import json
            economics_path = out_dir / f"{scenario}_economics.json"
            with open(economics_path, 'w') as f:
                json.dump(economics, f, indent=2, default=float)
            print(f"✓ Economics results saved: {economics_path}")

            # Print summary to console
            print("\n" + "-"*60)
            print("ECONOMIC SUMMARY")
            print("-"*60)
            print(f"Total CAPEX: {economics['total_capex_eur']/1e6:.2f} M€")
            print(f"  - PV: {economics['capex_breakdown']['pv']/1e6:.2f} M€")
            print(f"  - Battery: {economics['capex_breakdown']['battery']/1e6:.2f} M€")
            print(f"  - Thermal Storage: {economics['capex_breakdown']['thermal_storage']/1e6:.2f} M€")
            print(f"  - Heat Pump: {economics['capex_breakdown']['heat_pump']/1e6:.2f} M€")
            print(f"\nNPV ({economics['financial_parameters']['project_lifetime_years']} years, {economics['financial_parameters']['discount_rate']*100:.1f}% discount): {economics['npv_eur']/1e6:.2f} M€")
            print(f"\nPayback Time:")
            print(f"  - Simple: {economics['payback_time_years']['simple']:.1f} years")
            print(f"  - Discounted: {economics['payback_time_years']['discounted']:.1f} years")
            print(f"\nAnnual Costs:")
            print(f"  - Baseline (Grid+DH): {economics['annual_costs']['baseline_eur_per_year']/1e3:.1f} k€/year")
            print(f"  - Optimized: {economics['annual_costs']['optimized_eur_per_year']/1e3:.1f} k€/year")
            print(f"  - Annual Savings: {economics['annual_costs']['annual_savings_eur']/1e3:.1f} k€/year")
            print("-"*60 + "\n")

            # Generate publication-quality plots
            print("Generating publication-quality visualizations...")
            plot_economics_dashboard(economics, scenario, str(out_dir))
            plot_npv_sensitivity(economics, scenario, str(out_dir))
            print("✓ All economic visualizations generated successfully!\n")

    except Exception as e:
        print(f"[WARN] plotting: {e}")
        import traceback
        traceback.print_exc()

    return hourly_df


# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Run invest_opt scenario with finance KPIs")
    run_p = parser.add_subparsers(dest='command', required=True).add_parser('run', help='Run scenario')
    run_p.add_argument('scenario', help='Scenario name (e.g., invest_opt)')
    run_p.add_argument('config', help='Path to config.yml')
    run_p.add_argument('params', help='Path to component_params.yml')
    args = parser.parse_args()

    if args.command == 'run':
        print('*'*40); print("Command: run"); print('*'*40)
        df = run_scenario(args.scenario, args.config, args.params)
        if df.empty:
            sys.exit(1)

if __name__ == "__main__":
    main()
