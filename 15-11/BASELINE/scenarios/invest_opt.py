# -*- coding: utf-8 -*-
# PED Lyngby Model — Scenario: invest_opt
# Multi-period investment optimization with economic analysis (NPV, Payback, CAPEX)
# Optimizes capacities of PV, Battery, TES, HP, Transformer, and Export connections
# Based on realistic Danish energy system costs (2024-2025)

import os
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
import pypsa

from .utils import load_config, load_or_generate_profile, load_electricity_price_profile
from scripts import data_processing


# ---------------------------
# Financial Functions
# ---------------------------
def annuity_factor(r: float, n: float) -> float:
    """Calculate annuity factor for converting CAPEX to annual costs."""
    r = float(r); n = float(n)
    if n <= 0:
        return 1.0
    if abs(r) < 1e-12:
        return 1.0 / n
    return r / (1.0 - (1.0 + r) ** (-n))


def annualized(capex_per_unit: float, lifetime_y: float, discount: float) -> float:
    """Convert total CAPEX to annualized cost."""
    return float(capex_per_unit) * annuity_factor(discount, lifetime_y)


def calculate_npv(cash_flows: np.ndarray, discount_rate: float) -> float:
    """
    Calculate Net Present Value of cash flows.

    Args:
        cash_flows: Array of annual cash flows (negative for costs, positive for savings)
        discount_rate: Annual discount rate (e.g., 0.04 for 4%)

    Returns:
        NPV in EUR
    """
    years = np.arange(len(cash_flows))
    discount_factors = 1.0 / (1.0 + discount_rate) ** years
    return np.sum(cash_flows * discount_factors)


def calculate_payback_time(initial_investment: float, annual_savings: np.ndarray,
                          discount_rate: float = 0.0) -> float:
    """
    Calculate payback time in years.

    Args:
        initial_investment: Initial CAPEX in EUR
        annual_savings: Array of annual operational savings in EUR
        discount_rate: Discount rate for discounted payback (0 for simple payback)

    Returns:
        Payback time in years (np.inf if never pays back)
    """
    if initial_investment <= 0:
        return 0.0

    cumulative = 0.0
    for year, saving in enumerate(annual_savings):
        if discount_rate > 0:
            cumulative += saving / (1.0 + discount_rate) ** year
        else:
            cumulative += saving

        if cumulative >= initial_investment:
            # Linear interpolation for fractional year
            if year == 0:
                return 0.0
            prev_cumulative = cumulative - saving / (1.0 + discount_rate) ** year if discount_rate > 0 else cumulative - saving
            fraction = (initial_investment - prev_cumulative) / (cumulative - prev_cumulative)
            return year + fraction

    return np.inf  # Never pays back within the period


# ---------------------------
# Lettura serie
# ---------------------------
def _read_weather_and_temp(data_path: str, idx: pd.DatetimeIndex) -> Tuple[pd.DataFrame, pd.Series]:
    fp = os.path.join(data_path, "timeseries", "weather_data.csv")
    tdf = pd.read_csv(fp, index_col=0, parse_dates=True)
    tdf = tdf.reindex(idx)
    T_amb_C = tdf["temperature"].astype(float)
    weather_df = pd.DataFrame({"solar_radiation": tdf["solar_radiation"].astype(float)}, index=idx)
    return weather_df, T_amb_C


def _pv_profile(params: dict, weather_df: pd.DataFrame) -> pd.Series:
    pv_params = params.get("pv", params.get("pv_battery_2", {}))
    eff_pv = float(pv_params.get("efficiency_pv", 0.19))
    eff_inv = float(pv_params.get("inverter_efficiency", 0.97))
    area_m2 = float(pv_params.get("surface_m2", 1000.0))

    pv_gen_kw = data_processing.calculate_pv_generation(
        weather_df=weather_df,
        pv_surface_m2=area_m2,
        pv_efficiency_pv=eff_pv,
        pv_inverter_efficiency=eff_inv,
    )
    pv_gen_mw = pv_gen_kw / 1000.0
    denom = max(pv_gen_mw.max(), 1e-9)
    pv_pu = (pv_gen_mw / denom).clip(lower=0.0)
    pv_pu.name = "pv_pu"
    return pv_pu


# ---------------------------
# Snapshots multi-anno
# ---------------------------
def _build_multi_year_index(start_date: str, num_hours: int, years: List[int]) -> pd.DatetimeIndex:
    """Crea indice datetime concatenando le stesse ore per ciascun anno."""
    if not years:
        return pd.date_range(start=start_date, periods=num_hours, freq="h")
    blocks = []
    base = pd.to_datetime(start_date)
    for y in years:
        start = pd.Timestamp(year=int(y), month=base.month, day=base.day, hour=base.hour)
        block = pd.date_range(start=start, periods=num_hours, freq="h")
        blocks.append(block)
    return blocks[0].append(blocks[1:]) if len(blocks) > 1 else blocks[0]


def _per_year_weights(snaps: pd.DatetimeIndex, years: List[int], year_weights: List[float]) -> pd.Series:
    """Serie di pesi per snapshot: usa weight per anno su colonna 'objective'."""
    w = pd.Series(1.0, index=snaps)
    if not years:
        return w
    if not year_weights or len(year_weights) != len(years):
        year_weights = [1.0] * len(years)
    for y, wy in zip(years, year_weights):
        w.loc[snaps.year == int(y)] = float(wy)
    return w


# ---------------------------
# Scenario
# ---------------------------
def create_network(config_file: str, params_file: str, data_path: str):
    # 1) Config & params
    config, params = load_config(config_file, params_file)

    # Multi-anno
    sim = config.get("simulation_settings", {})
    start_date = sim.get("start_date", "2018-01-01")
    n_hours    = int(sim.get("num_hours", 8760))
    years      = list(sim.get("years", []))  # es: [2018, 2019, 2020]
    year_w     = list(sim.get("year_objective_weights", []))  # opzionale

    # 2) Network + snapshots
    n = pypsa.Network()
    snapshots = _build_multi_year_index(start_date, n_hours, years)
    n.set_snapshots(snapshots)

    # Pesi snapshot: **deve esistere 'objective'** per PyPSA >= 0.27
    w = _per_year_weights(n.snapshots, years, year_w)
    n.snapshot_weightings = pd.DataFrame(
        {
            "objective": w,     # <— fondamentale per l’optimizer
            "generators": w,
            "stores": w,
            "lines": w,
        },
        index=n.snapshots,
    )

    # 3) Serie meteo e profili
    weather_df, T_amb_C = _read_weather_and_temp(data_path, n.snapshots)
    pv_pu = _pv_profile(params, weather_df)

    # 4) Carrier & Bus (niente DSM orfani)
    for c in ["electricity", "heat", "heat_distribution", "heat_pump"]:
        if c not in n.carriers.index:
            n.add("Carrier", c)

    n.add("Bus", "Grid Elec Sell",  carrier="electricity")
    n.add("Bus", "Grid Elec Buy",   carrier="electricity")
    n.add("Bus", "Building Elec",   carrier="electricity")
    n.add("Bus", "Battery Storage", carrier="electricity")
    n.add("Bus", "PV Bus",          carrier="electricity")
    n.add("Bus", "District Heating", carrier="heat")
    n.add("Bus", "Building Heat",    carrier="heat")
    n.add("Bus", "Heat Source Bus",  carrier="heat")
    n.add("Bus", "Thermal Storage",  carrier="heat")

    # 5) Finanza / CAPEX annualizzato
    fin = params.get("finance", {})
    discount = float(fin.get("discount_rate", 0.05))
    cap = params.get("capex", {})

    def ann(key_cost, key_life, default_cost, default_life):
        c = float(cap.get(key_cost, default_cost))
        L = float(cap.get(key_life, default_life))
        return annualized(c, L, discount)

    pv_capex_a_per_MW      = ann("pv_per_mw",           "pv_life_y",            700e3, 25)
    bat_e_capex_a_per_MWh  = ann("battery_e_per_mwh",   "battery_life_y",       150e3, 12)
    bat_p_capex_a_per_MW   = ann("battery_p_per_mw",    "battery_life_y",        60e3, 12)
    tes_e_capex_a_per_MWh  = ann("tes_e_per_mwh",       "tes_life_y",            25e3, 20)
    tes_p_capex_a_per_MW   = ann("tes_p_per_mw",        "tes_life_y",            10e3, 20)
    hp_capex_a_per_MWth    = ann("hp_per_mwth",         "hp_life_y",            350e3, 18)
    trafo_capex_a_per_MW   = ann("trafo_per_mw",        "trafo_life_y",          40e3, 30)
    export_capex_a_per_MW  = ann("export_link_per_mw",  "export_link_life_y",    15e3, 30)

    # 6) Limiti superiori
    lim = params.get("limits", {})
    pv_p_nom_max         = float(lim.get("pv_p_nom_max_mw",          2.0))
    bat_e_nom_max        = float(lim.get("battery_e_nom_max_mwh",    6.0))
    bat_p_nom_max        = float(lim.get("battery_p_nom_max_mw",     3.0))
    tes_e_nom_max        = float(lim.get("tes_e_nom_max_mwh",        12.0))
    tes_p_nom_max        = float(lim.get("tes_p_nom_max_mw",         5.0))
    hp_p_nom_max         = float(lim.get("hp_p_nom_max_mwth",        3.0))
    grid_import_p_max    = float(lim.get("grid_import_p_nom_max_mw", 5.0))
    export_p_nom_max     = float(lim.get("export_p_nom_max_mw",      5.0))

    # 7) Prezzi rete / emissioni
    grid_params = params.get("grid", {})
    transformer_eff = float(grid_params.get("transformer_efficiency", 0.985))
    grid_co2_kg = float(grid_params.get("CO2_emissions_kg_per_mwh", 0.0))

    # Prezzo import: variabile o fisso (EUR/kWh) → EUR/MWh
    if str(grid_params.get("import_cost_eur_per_kwh", "variable")).lower() == "variable":
        base_idx = pd.date_range(start=start_date, periods=n_hours, freq="h")
        base_price_kwh = load_electricity_price_profile(data_path, base_idx)  # EUR/kWh
        base_price_mwh = base_price_kwh.astype(float) * 1000.0               # EUR/MWh
        if years:
            prices = []
            for y in years:
                prices.append(base_price_mwh.rename(lambda ts: ts.replace(year=int(y))))
            import_price_series = pd.concat(prices).reindex(n.snapshots)
        else:
            import_price_series = base_price_mwh.reindex(n.snapshots)
    else:
        fixed = float(grid_params.get("import_cost_eur_per_kwh", 0.22))
        import_price_series = pd.Series(fixed * 1000.0, index=n.snapshots)

    export_price_series = import_price_series.copy()
    # salva per summary
    n.grid_price_series = import_price_series

    # 8) Carichi edificio (ripetuti per anni se necessario)
    bld = params.get("social_building", {})
    base_idx = pd.date_range(start=start_date, periods=n_hours, freq="h")

    elec_one_year = load_or_generate_profile(
        bld.get("electricity_load_profile", "electricity_demand.csv"),
        bld.get("electricity_peak_mw", 0.23),
        data_path,
        base_idx
    ) / 1000.0  # kW -> MW

    heat_one_year = load_or_generate_profile(
        bld.get("heat_load_profile", "heat_demand.csv"),
        bld.get("heat_peak_mw", 0.32),
        data_path,
        base_idx
    ) / 1000.0  # kW -> MW

    if years:
        def rep(series):
            lst = []
            for y in years:
                idx = series.index.map(lambda ts: ts.replace(year=int(y)))
                lst.append(pd.Series(series.values, index=idx))
            return pd.concat(lst).reindex(n.snapshots)
        elec_mw = rep(elec_one_year)
        heat_mw = rep(heat_one_year)
    else:
        elec_mw = elec_one_year.reindex(n.snapshots)
        heat_mw = heat_one_year.reindex(n.snapshots)

    n.add("Load", "Building Elec Load", bus="Building Elec", p_set=elec_mw, carrier="electricity")
    n.add("Load", "Building Heat Load", bus="Building Heat", p_set=heat_mw, carrier="heat")

    # 9) Generatore Grid + trafo (investibile)
    n.add("Generator", "Grid",
          bus="Grid Elec Sell", carrier="electricity",
          p_nom_extendable=True, p_nom_max=grid_import_p_max,
          marginal_cost=import_price_series, efficiency=1.0)

    n.add("Link", "Grid Import",
          bus0="Grid Elec Sell", bus1="Building Elec",
          p_nom_extendable=True, p_nom_max=grid_import_p_max,
          efficiency=transformer_eff,
          capital_cost=trafo_capex_a_per_MW,
          marginal_cost=0.0)
    n.links.at["Grid Import", "co2_emission_factor_kg_per_mwh"] = grid_co2_kg

    # 10) PV investibile
    n.add("Generator", "Rooftop PV",
          bus="PV Bus", carrier="electricity",
          p_nom_extendable=True, p_nom_max=pv_p_nom_max,
          p_max_pu=pv_pu.reindex(n.snapshots).fillna(0.0),
          capital_cost=pv_capex_a_per_MW,
          marginal_cost=0.0, efficiency=1.0)

    n.add("Link", "PV Autoconsumo",
          bus0="PV Bus", bus1="Building Elec",
          p_nom_extendable=True, p_nom_max=pv_p_nom_max,
          efficiency=1.0, capital_cost=0.0, marginal_cost=0.0)

    # 11) Batteria (energia + potenza)
    bat_cfg = params.get("battery", params.get("pv_battery_2", {}).get("battery", {}))
    eta_rt = float(bat_cfg.get("efficiency_round_trip", 0.90))
    eff_store = eta_rt ** 0.5
    eff_dispatch = eta_rt ** 0.5

    n.add("Store", "Battery Store",
          bus="Battery Storage",
          e_nom_extendable=True, e_nom_max=bat_e_nom_max,
          e_min_pu=0.15, e_max_pu=0.85,
          e_cyclic=True, standing_loss=0.0,
          capital_cost=bat_e_capex_a_per_MWh,
          marginal_cost=0.0)

    n.add("Link", "Battery Charge",
          bus0="Building Elec", bus1="Battery Storage",
          p_nom_extendable=True, p_nom_max=bat_p_nom_max,
          efficiency=eff_store,
          capital_cost=bat_p_capex_a_per_MW, marginal_cost=0.0)

    n.add("Link", "Battery Dispatch",
          bus0="Battery Storage", bus1="Building Elec",
          p_nom_extendable=True, p_nom_max=bat_p_nom_max,
          efficiency=eff_dispatch,
          capital_cost=0.0, marginal_cost=0.0)

    # 12) Heat Pump (COP fisso da params, nessun time-varying!)
    hp_cfg = params.get("hp", {})
    hp_cop_fixed = float(hp_cfg.get("cop_fixed", 3.5))  # default 3.5

    n.add("Link", "Heat Pump",
          bus0="Building Elec", bus1="Heat Source Bus",
          p_nom_extendable=True, p_nom_max=hp_p_nom_max,
          efficiency=hp_cop_fixed,
          capital_cost=hp_capex_a_per_MWth,
          marginal_cost=0.0, carrier="heat_pump")

    # 13) TES + distribuzione
    tes_standing = float(params.get("tes", {}).get("standing_loss_per_hour", 0.005))
    ETA_TES_CH, ETA_TES_DIS = 0.98, 0.98

    n.add("Store", "Thermal Store",
          bus="Thermal Storage",
          e_nom_extendable=True, e_nom_max=tes_e_nom_max,
          e_min_pu=0.15, e_max_pu=0.85,
          e_cyclic=True,
          standing_loss=tes_standing,
          capital_cost=tes_e_capex_a_per_MWh,
          marginal_cost=0.0)

    n.add("Link", "Thermal Charge",
          bus0="Heat Source Bus", bus1="Thermal Storage",
          p_nom_extendable=True, p_nom_max=tes_p_nom_max,
          efficiency=ETA_TES_CH,
          capital_cost=tes_p_capex_a_per_MW, marginal_cost=0.0)

    n.add("Link", "Thermal Dispatch",
          bus0="Thermal Storage", bus1="Building Heat",
          p_nom_extendable=True, p_nom_max=tes_p_nom_max,
          efficiency=ETA_TES_DIS,
          capital_cost=0.0, marginal_cost=0.0)

    n.add("Link", "Thermal Discharge",
          bus0="Heat Source Bus", bus1="Building Heat",
          p_nom_extendable=True, p_nom_max=tes_p_nom_max,
          efficiency=1.0,
          capital_cost=tes_p_capex_a_per_MW, marginal_cost=0.0,
          carrier="heat_distribution")

    # 14) District Heating (solo OPEX)
    heat_params = params.get("district_heating", params.get("baseline_heat_source", {}))
    heat_eff = float(heat_params.get("efficiency_heating", 0.95))
    heat_cost = float(heat_params.get("cost_eur_per_mwh_th", 45.0))
    dh_co2 = float(heat_params.get("co2_emissions_dh_kg_per_mwh", 0.0))

    n.add("Link", "Heat import",
          bus0="District Heating", bus1="Building Heat",
          p_nom_extendable=True, p_nom_max=tes_p_nom_max,
          efficiency=heat_eff,
          capital_cost=0.0, marginal_cost=0.0)
    n.links.at["Heat import", "co2_emission_factor_kg_per_mwh"] = dh_co2

    n.add("Generator", "Heating",
          bus="District Heating", carrier="heat",
          p_nom_extendable=True, p_nom_max=tes_p_nom_max,
          marginal_cost=heat_cost)

    # 15) Export elettrico (ricavo + CAPEX + limiti)
    n.add("Link", "PV Export",
          bus0="PV Bus", bus1="Grid Elec Buy",
          p_nom_extendable=True, p_nom_max=export_p_nom_max,
          efficiency=0.95,
          capital_cost=export_capex_a_per_MW,
          marginal_cost=-export_price_series)  # ricavo

    # NIENTE links_t.set_panel(...) — efficienze time-varying non supportate. COP fisso.

    return n, {}, pd.DataFrame(index=n.snapshots)


def calculate_economics(n: pypsa.Network, params: dict, config: dict) -> Dict:
    """
    Calculate economic metrics after optimization: CAPEX, NPV, Payback Time.

    Args:
        n: Optimized PyPSA network
        params: Component parameters dictionary
        config: Configuration dictionary

    Returns:
        Dictionary with economic metrics
    """
    fin = params.get("finance", {})
    discount_rate = float(fin.get("discount_rate", 0.04))
    project_lifetime = int(fin.get("project_lifetime_years", 25))
    cap = params.get("capex", {})

    # Extract optimal capacities
    pv_capacity_mw = n.generators.at["Rooftop PV", "p_nom_opt"] if "Rooftop PV" in n.generators.index else 0.0
    battery_e_mwh = n.stores.at["Battery Store", "e_nom_opt"] if "Battery Store" in n.stores.index else 0.0
    battery_p_mw = n.links.at["Battery Charge", "p_nom_opt"] if "Battery Charge" in n.links.index else 0.0
    tes_e_mwh = n.stores.at["Thermal Store", "e_nom_opt"] if "Thermal Store" in n.stores.index else 0.0
    tes_p_mw = n.links.at["Thermal Charge", "p_nom_opt"] if "Thermal Charge" in n.links.index else 0.0
    hp_p_mw = n.links.at["Heat Pump", "p_nom_opt"] if "Heat Pump" in n.links.index else 0.0
    grid_import_mw = n.links.at["Grid Import", "p_nom_opt"] if "Grid Import" in n.links.index else 0.0
    export_mw = n.links.at["PV Export", "p_nom_opt"] if "PV Export" in n.links.index else 0.0

    # Calculate total CAPEX (not annualized)
    capex_pv = pv_capacity_mw * float(cap.get("pv_per_mw", 650000))
    capex_battery_e = battery_e_mwh * float(cap.get("battery_e_per_mwh", 200000))
    capex_battery_p = battery_p_mw * float(cap.get("battery_p_per_mw", 80000))
    capex_battery = capex_battery_e + capex_battery_p
    capex_tes_e = tes_e_mwh * float(cap.get("tes_e_per_mwh", 30000))
    capex_tes_p = tes_p_mw * float(cap.get("tes_p_per_mw", 15000))
    capex_tes = capex_tes_e + capex_tes_p
    capex_hp = hp_p_mw * float(cap.get("hp_per_mwth", 400000))
    capex_trafo = grid_import_mw * float(cap.get("trafo_per_mw", 50000))
    capex_export = export_mw * float(cap.get("export_link_per_mw", 20000))

    total_capex = capex_pv + capex_battery + capex_tes + capex_hp + capex_trafo + capex_export

    # Calculate annual operational costs and revenues
    # Get total system cost from objective function
    total_system_cost = n.objective  # This is the optimized total cost

    # Estimate baseline cost (without investments - grid only scenario)
    # This would be the cost if we only used grid electricity and district heating
    baseline_annual_cost = estimate_baseline_cost(n, params)

    # Annual savings = baseline cost - optimized cost
    annual_savings = baseline_annual_cost - (total_system_cost / len(set(n.snapshots.year)))

    # Create cash flow array for NPV calculation
    # Year 0: -CAPEX, Years 1-N: annual savings
    cash_flows = np.zeros(project_lifetime + 1)
    cash_flows[0] = -total_capex  # Initial investment
    cash_flows[1:] = annual_savings  # Annual savings

    # Calculate NPV
    npv = calculate_npv(cash_flows, discount_rate)

    # Calculate payback time
    annual_savings_array = np.full(project_lifetime, annual_savings)
    payback_simple = calculate_payback_time(total_capex, annual_savings_array, discount_rate=0.0)
    payback_discounted = calculate_payback_time(total_capex, annual_savings_array, discount_rate=discount_rate)

    # Compile results
    economics = {
        "total_capex_eur": total_capex,
        "capex_breakdown": {
            "pv": capex_pv,
            "battery": capex_battery,
            "thermal_storage": capex_tes,
            "heat_pump": capex_hp,
            "transformer": capex_trafo,
            "export_connection": capex_export,
        },
        "optimal_capacities": {
            "pv_mw": pv_capacity_mw,
            "battery_energy_mwh": battery_e_mwh,
            "battery_power_mw": battery_p_mw,
            "tes_energy_mwh": tes_e_mwh,
            "tes_power_mw": tes_p_mw,
            "heat_pump_mw": hp_p_mw,
            "grid_import_mw": grid_import_mw,
            "export_capacity_mw": export_mw,
        },
        "annual_costs": {
            "baseline_eur_per_year": baseline_annual_cost,
            "optimized_eur_per_year": total_system_cost / len(set(n.snapshots.year)),
            "annual_savings_eur": annual_savings,
        },
        "npv_eur": npv,
        "payback_time_years": {
            "simple": payback_simple,
            "discounted": payback_discounted,
        },
        "financial_parameters": {
            "discount_rate": discount_rate,
            "project_lifetime_years": project_lifetime,
        },
    }

    return economics


def estimate_baseline_cost(n: pypsa.Network, params: dict) -> float:
    """
    Estimate baseline annual cost without renewable investments.
    This represents the cost of meeting all demand with grid electricity and district heating.

    Args:
        n: PyPSA network (with loads and price data)
        params: Parameters dictionary

    Returns:
        Estimated annual baseline cost in EUR
    """
    # Get annual electricity and heat demand
    elec_load = n.loads_t.p_set.get("Building Elec Load", pd.Series(0.0, index=n.snapshots))
    heat_load = n.loads_t.p_set.get("Building Heat Load", pd.Series(0.0, index=n.snapshots))

    # Get prices
    grid_price = n.grid_price_series if hasattr(n, 'grid_price_series') else pd.Series(220.0, index=n.snapshots)  # EUR/MWh
    heat_params = params.get("district_heating", params.get("baseline_heat_source", {}))
    heat_price = float(heat_params.get("cost_eur_per_mwh_th", 64.34))  # EUR/MWh_th

    # Calculate costs
    # Electricity: direct from grid
    elec_cost = (elec_load * grid_price).sum()

    # Heat: from district heating
    heat_cost = (heat_load * heat_price).sum()

    # Total annual cost (average over years if multi-year)
    total_cost = elec_cost + heat_cost
    num_years = len(set(n.snapshots.year))
    annual_cost = total_cost / num_years if num_years > 0 else total_cost
    # --- in fondo a invest_opt.py ---

def estimate_baseline_cost(n, params) -> float:
    """
    Stima un costo annuo baseline (solo Grid + DH) usando i profili e i prezzi
    presenti nel network (nessun investimento).
    """
    snaps = n.snapshots
    w = getattr(n, "snapshot_weightings", None)
    if w is None:
        w = pd.Series(1.0, index=snaps)
    elif isinstance(w, pd.DataFrame) and "generators" in w:
        w = w["generators"].reindex(snaps).fillna(1.0)
    elif isinstance(w, pd.Series):
        w = w.reindex(snaps).fillna(1.0)
    else:
        w = pd.Series(1.0, index=snaps)

    Gmc = getattr(n, "generators_t", None)
    price_elec = Gmc.marginal_cost.get("Grid", pd.Series(0.0, index=snaps)) if Gmc is not None else pd.Series(0.0, index=snaps)
    price_heat = Gmc.marginal_cost.get("Heating", pd.Series(0.0, index=snaps)) if Gmc is not None else pd.Series(0.0, index=snaps)

    Lp = n.loads_t
    elec = Lp.p.get('Building Elec Load', pd.Series(0.0, index=snaps))
    heat = Lp.p.get('Building Heat Load', pd.Series(0.0, index=snaps))

    # Efficienze di riferimento (trafo e DH)
    eta_grid = float(n.links.at["Grid Import","efficiency"]) if ("Grid Import" in n.links.index and "efficiency" in n.links.columns) else 1.0
    eta_dh   = float(n.links.at["Heat import","efficiency"]) if ("Heat import" in n.links.index and "efficiency" in n.links.columns) else 1.0

    cost_grid = float(((elec/eta_grid) * price_elec * w).sum())
    cost_dh   = float(((heat/eta_dh)  * price_heat * w).sum())
    return cost_grid + cost_dh


def calculate_economics(n, params: dict, cfg: dict) -> dict:
    """
    Calcola CAPEX (upfront), NPV e payback su orizzonte 'finance.project_years'
    con tasso 'finance.discount_rate', usando le capacità ottime di PyPSA.
    """
    finance = cfg.get("finance", {})
    discount_rate = float(finance.get("discount_rate", 0.05))
    project_lifetime = int(finance.get("project_years", 20))

    # Capacità ottime
    pv_mw   = n.generators.at["Rooftop PV","p_nom_opt"] if "Rooftop PV" in n.generators.index else 0.0
    bat_pmw = n.links.at["Battery Charge","p_nom_opt"]  if "Battery Charge" in n.links.index else 0.0
    bat_emw = n.stores.at["Battery Store","e_nom_opt"]  if "Battery Store" in n.stores.index else 0.0
    tes_pmw = n.links.at["Thermal Charge","p_nom_opt"]  if "Thermal Charge" in n.links.index else 0.0
    tes_emw = n.stores.at["Thermal Store","e_nom_opt"]  if "Thermal Store" in n.stores.index else 0.0
    hp_pmw  = n.links.at["Heat Pump","p_nom_opt"]       if "Heat Pump" in n.links.index else 0.0
    trafo_mw= n.links.at["Grid Import","p_nom_opt"]     if "Grid Import" in n.links.index else 0.0
    export_mw= n.links.at["PV Export","p_nom_opt"]      if "PV Export" in n.links.index else 0.0

    cap = params.get("capex", {})
    cap_pv      = pv_mw   * float(cap.get("pv_per_mw",            650000))
    cap_bat_e   = bat_emw * float(cap.get("battery_e_per_mwh",    200000))
    cap_bat_p   = bat_pmw * float(cap.get("battery_p_per_mw",      80000))
    cap_bat     = cap_bat_e + cap_bat_p
    cap_tes_e   = tes_emw * float(cap.get("tes_e_per_mwh",         30000))
    cap_tes_p   = tes_pmw * float(cap.get("tes_p_per_mw",          15000))
    cap_tes     = cap_tes_e + cap_tes_p
    cap_hp      = hp_pmw  * float(cap.get("hp_per_mwth",          400000))
    cap_trafo   = trafo_mw* float(cap.get("trafo_per_mw",          50000))
    cap_export  = export_mw*float(cap.get("export_link_per_mw",    20000))

    total_capex = cap_pv + cap_bat + cap_tes + cap_hp + cap_trafo + cap_export

    # OPEX annuo ottimo (dalla tua summary)
    # Se hai appena ottimizzato, puoi usare n.objective come costo annuale sul periodo simulato.
    annual_optimized_cost = float(getattr(n, "objective", 0.0)) / max(len(set(n.snapshots.year)), 1)

    # baseline annua (senza investimenti)
    baseline_annual_cost = estimate_baseline_cost(n, params)

    annual_savings = max(baseline_annual_cost - annual_optimized_cost, 0.0)

    # cash-flow: anno 0 -CAPEX, poi risparmi annui
    cash_flows = np.zeros(project_lifetime + 1)
    cash_flows[0] = -total_capex
    cash_flows[1:] = annual_savings

    # NPV e payback
    years = np.arange(len(cash_flows))
    discount_factors = 1.0 / (1.0 + discount_rate) ** years
    npv = float((cash_flows * discount_factors).sum())

    def _payback(discount: float) -> float:
        cum = 0.0
        for y in range(1, project_lifetime+1):
            add = annual_savings/((1+discount)**y) if discount>0 else annual_savings
            if (cum + add) >= total_capex:
                # interp lineare fra y-1 e y
                prev = cum
                frac = (total_capex - prev) / max(add, 1e-9)
                return (y-1) + frac
            cum += add
        return float("inf")

    pb_simple     = _payback(0.0)
    pb_discounted = _payback(discount_rate)

    return {
        "financial_parameters": {
            "discount_rate": discount_rate,
            "project_lifetime_years": project_lifetime,
        },
        "optimal_capacities": {
            "pv_mw": pv_mw, "battery_energy_mwh": bat_emw, "battery_power_mw": bat_pmw,
            "tes_energy_mwh": tes_emw, "tes_power_mw": tes_pmw, "heat_pump_mwth": hp_pmw,
            "transformer_mw": trafo_mw, "export_connection_mw": export_mw,
        },
        "capex_breakdown": {
            "pv": cap_pv, "battery": cap_bat, "thermal_storage": cap_tes,
            "heat_pump": cap_hp, "transformer": cap_trafo, "export_connection": cap_export
        },
        "total_capex_eur": total_capex,
        "annual_costs": {
            "baseline_eur_per_year": baseline_annual_cost,
            "optimized_eur_per_year": annual_optimized_cost,
            "annual_savings_eur": annual_savings
        },
        "npv_eur": npv,
        "payback_time_years": {
            "simple": pb_simple,
            "discounted": pb_discounted
        }
    }


    return annual_cost
