# -*- coding: utf-8 -*-
# PED Lyngby Model — Scenario: invest_opt (con DSM elettrico e termico)
# Co-ottimizzazione di investimenti (PV, Battery, TES, HP, Trafo, Export) e dispacciamento.
# Restituisce: (network, results_dict, df_hourly) per essere compatibile con scripts/main.py

from typing import Tuple
import os
import pandas as pd
import pypsa

from .utils import load_config, load_or_generate_profile, load_electricity_price_profile
from scripts import data_processing


def annuity_factor(r: float, n: float) -> float:
    r = float(r); n = float(n)
    if n <= 0:
        return 1.0
    if abs(r) < 1e-12:
        return 1.0 / n
    return r / (1.0 - (1.0 + r) ** (-n))


def annualized(capex_per_unit: float, lifetime_y: float, discount: float) -> float:
    return float(capex_per_unit) * annuity_factor(discount, lifetime_y)


def _read_series(data_path: str, snapshots: pd.DatetimeIndex) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    weather_fp = os.path.join(data_path, "timeseries", "weather_data.csv")

    # radiazione (usa il tuo helper per stare allineato agli altri scenari)
    raw_solar = data_processing.load_timeseries(weather_fp, data_path, index=snapshots)
    weather_df = pd.DataFrame({"solar_radiation": raw_solar}, index=snapshots)

    # temperatura esterna
    tdf = pd.read_csv(weather_fp, index_col=0, parse_dates=True)
    T_amb_C = tdf.reindex(snapshots)["temperature"].astype(float)

    return raw_solar, weather_df, T_amb_C


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


def _cop_timeseries(params: dict, T_amb_C: pd.Series) -> pd.Series:
    eta_carnot = float(params.get("hp", {}).get("eta_carnot", 0.5))
    T_sink_C = float(params.get("hp", {}).get("supply_temp_C", 50.0))
    T_sink_K = T_sink_C + 273.15
    T_src_K = (T_amb_C + 2.0) + 273.15
    COP_ts = eta_carnot * (T_sink_K / (T_sink_K - T_src_K))
    COP_ts = COP_ts.clip(lower=1.75, upper=5.2)
    COP_ts.name = "COP"
    return COP_ts


def create_network(config_file: str, params_file: str, data_path: str):
    # -----------------------------
    # 1. Config & snapshots
    # -----------------------------
    config, params = load_config(config_file, params_file)

    n = pypsa.Network()
    start_date = config["simulation_settings"]["start_date"]
    n_hours = int(config["simulation_settings"]["num_hours"])
    snapshots = pd.date_range(start=start_date, periods=n_hours, freq="h")
    n.set_snapshots(snapshots)

    # -----------------------------
    # 2. Serie necessarie
    # -----------------------------
    _, weather_df, T_amb_C = _read_series(data_path, n.snapshots)
    pv_pu = _pv_profile(params, weather_df)
    COP_ts = _cop_timeseries(params, T_amb_C)

    # -----------------------------
    # 3. Carrier & Buses
    # -----------------------------
    for c in ["electricity", "heat", "heat_distribution", "heat_pump", "dsm_elec", "dsm_heat"]:
        if c not in n.carriers.index:
            n.add("Carrier", c)

    # elettrico
    n.add("Bus", "Grid Elec Sell",  carrier="electricity")
    n.add("Bus", "Grid Elec Buy",   carrier="electricity")
    n.add("Bus", "Building Elec",   carrier="electricity")
    n.add("Bus", "Battery Storage", carrier="electricity")
    n.add("Bus", "PV Bus",          carrier="electricity")
    n.add("Bus", "DSM Elec",        carrier="dsm_elec")
    # termico
    n.add("Bus", "District Heating", carrier="heat")
    n.add("Bus", "Building Heat",    carrier="heat")
    n.add("Bus", "Heat Source Bus",  carrier="heat")
    n.add("Bus", "Thermal Storage",  carrier="heat")
    n.add("Bus", "DSM Heat",         carrier="dsm_heat")

    # -----------------------------
    # 4. Finanza / CAPEX annualizzato
    # -----------------------------
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

    # -----------------------------
    # 5. Limiti superiori (tutti finiti → niente unbounded)
    # -----------------------------
    lim = params.get("limits", {})
    pv_p_nom_max         = float(lim.get("pv_p_nom_max_mw",          2.0))
    bat_e_nom_max        = float(lim.get("battery_e_nom_max_mwh",    6.0))
    bat_p_nom_max        = float(lim.get("battery_p_nom_max_mw",     3.0))
    tes_e_nom_max        = float(lim.get("tes_e_nom_max_mwh",        12.0))
    tes_p_nom_max        = float(lim.get("tes_p_nom_max_mw",         5.0))
    hp_p_nom_max         = float(lim.get("hp_p_nom_max_mwth",        3.0))
    grid_import_p_max    = float(lim.get("grid_import_p_nom_max_mw", 5.0))
    export_p_nom_max     = float(lim.get("export_p_nom_max_mw",      5.0))

    # DSM limiti
    dsm = params.get("dsm", {})
    # elettrico
    dsm_e_e_nom_max      = float(dsm.get("elec_e_nom_max_mwh",       1.0))
    dsm_e_p_nom_max      = float(dsm.get("elec_p_nom_max_mw",        1.0))
    dsm_e_eta_ch         = float(dsm.get("elec_eta_charge",          0.97))
    dsm_e_eta_dis        = float(dsm.get("elec_eta_discharge",       0.97))
    dsm_e_soc_min        = float(dsm.get("elec_soc_min_pu",          0.0))
    dsm_e_soc_max        = float(dsm.get("elec_soc_max_pu",          1.0))
    dsm_e_standing       = float(dsm.get("elec_standing_loss_h",     0.0))
    dsm_e_cost_ch        = float(dsm.get("elec_cost_eur_per_mwh_ch", 2.0))
    dsm_e_cost_dis       = float(dsm.get("elec_cost_eur_per_mwh_dis",2.0))
    # termico
    dsm_h_e_nom_max      = float(dsm.get("heat_e_nom_max_mwh",       2.0))
    dsm_h_p_nom_max      = float(dsm.get("heat_p_nom_max_mw",        1.5))
    dsm_h_eta_ch         = float(dsm.get("heat_eta_charge",          0.98))
    dsm_h_eta_dis        = float(dsm.get("heat_eta_discharge",       0.98))
    dsm_h_soc_min        = float(dsm.get("heat_soc_min_pu",          0.0))
    dsm_h_soc_max        = float(dsm.get("heat_soc_max_pu",          1.0))
    dsm_h_standing       = float(dsm.get("heat_standing_loss_h",     0.002))
    dsm_h_cost_ch        = float(dsm.get("heat_cost_eur_per_mwh_ch", 1.5))
    dsm_h_cost_dis       = float(dsm.get("heat_cost_eur_per_mwh_dis",1.5))

    # -----------------------------
    # 6. Prezzi rete / emissioni
    # -----------------------------
    grid_params = params.get("grid", {})
    transformer_eff = float(grid_params.get("transformer_efficiency", 0.985))
    grid_co2_kg = float(grid_params.get("CO2_emissions_kg_per_mwh", 0.0))

    if str(grid_params.get("import_cost_eur_per_kwh", "variable")).lower() == "variable":
        import_price_series = load_electricity_price_profile(data_path, n.snapshots)
    else:
        fixed = float(grid_params.get("import_cost_eur_per_kwh", 0.22))
        import_price_series = pd.Series(fixed * 1000.0, index=n.snapshots)

    # prezzo export = stesso prezzo import (se hai un file dedicato, cambialo qui)
    export_price_series = import_price_series.copy()

    # per i tuoi summary
    n.grid_price_series = import_price_series

    # -----------------------------
    # 7. Carichi edificio
    # -----------------------------
    bld = params.get("social_building", {})
    elec_mw = load_or_generate_profile(
        bld.get("electricity_load_profile", "electricity_demand.csv"),
        bld.get("electricity_peak_mw", 0.23),
        data_path,
        n.snapshots
    ) / 1000.0
    elec_mw.index = n.snapshots

    heat_mw = load_or_generate_profile(
        bld.get("heat_load_profile", "heat_demand.csv"),
        bld.get("heat_peak_mw", 0.32),
        data_path,
        n.snapshots
    ) / 1000.0
    heat_mw.index = n.snapshots

    n.add("Load", "Building Elec Load", bus="Building Elec", p_set=elec_mw, carrier="electricity")
    n.add("Load", "Building Heat Load", bus="Building Heat", p_set=heat_mw, carrier="heat")

    # -----------------------------
    # 8. Generatore Grid + trafo (investibile)
    # -----------------------------
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

    # -----------------------------
    # 9. PV investibile + autoconsumo
    # -----------------------------
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

    # -----------------------------
    # 10. Batteria (energia + potenza investibili)
    # -----------------------------
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

    # -----------------------------
    # 11. Heat Pump (investibile, η(t)=COP(t))
    # -----------------------------
    n.add("Link", "Heat Pump",
          bus0="Building Elec", bus1="Heat Source Bus",
          p_nom_extendable=True, p_nom_max=hp_p_nom_max,
          efficiency=1.0,  # sovrascritto sotto
          capital_cost=hp_capex_a_per_MWth,
          marginal_cost=0.0, carrier="heat_pump")

    # -----------------------------
    # 12. TES + distribuzione
    # -----------------------------
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

    # -----------------------------
    # 13. District Heating (solo OPEX)
    # -----------------------------
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

    # -----------------------------
    # 14. Export elettrico (ricavo ma con CAPEX e limiti)
    # -----------------------------
    n.add("Link", "PV Export",
          bus0="PV Bus", bus1="Grid Elec Buy",
          p_nom_extendable=True, p_nom_max=export_p_nom_max,
          efficiency=0.95,
          capital_cost=export_capex_a_per_MW,
          marginal_cost=-export_price_series)

    # -----------------------------
    # 15. DSM ELETTRICO
    # -----------------------------
    n.add("Store", "DSM Elec Store",
          bus="DSM Elec",
          e_nom_extendable=True, e_nom_max=dsm_e_e_nom_max,
          e_min_pu=dsm_e_soc_min, e_max_pu=dsm_e_soc_max,
          e_cyclic=True, standing_loss=dsm_e_standing,
          capital_cost=0.0, marginal_cost=0.0)

    n.add("Link", "DSM Elec Charge",
          bus0="Building Elec", bus1="DSM Elec",
          p_nom_extendable=True, p_nom_max=dsm_e_p_nom_max,
          efficiency=dsm_e_eta_ch,
          capital_cost=0.0, marginal_cost=dsm_e_cost_ch)

    n.add("Link", "DSM Elec Dispatch",
          bus0="DSM Elec", bus1="Building Elec",
          p_nom_extendable=True, p_nom_max=dsm_e_p_nom_max,
          efficiency=dsm_e_eta_dis,
          capital_cost=0.0, marginal_cost=dsm_e_cost_dis)

    # -----------------------------
    # 16. DSM TERMICO
    # -----------------------------
    n.add("Store", "DSM Heat Store",
          bus="DSM Heat",
          e_nom_extendable=True, e_nom_max=dsm_h_e_nom_max,
          e_min_pu=dsm_h_soc_min, e_max_pu=dsm_h_soc_max,
          e_cyclic=True, standing_loss=dsm_h_standing,
          capital_cost=0.0, marginal_cost=0.0)

    n.add("Link", "DSM Heat Charge",
          bus0="Heat Source Bus", bus1="DSM Heat",
          p_nom_extendable=True, p_nom_max=dsm_h_p_nom_max,
          efficiency=dsm_h_eta_ch,
          capital_cost=0.0, marginal_cost=dsm_h_cost_ch)

    n.add("Link", "DSM Heat Dispatch",
          bus0="DSM Heat", bus1="Building Heat",
          p_nom_extendable=True, p_nom_max=dsm_h_p_nom_max,
          efficiency=dsm_h_eta_dis,
          capital_cost=0.0, marginal_cost=dsm_h_cost_dis)

    # -----------------------------
    # 17. Efficienze orarie: COP sulla Heat Pump
    # -----------------------------
    if (not hasattr(n.links_t, "efficiency")) or n.links_t.efficiency is None or n.links_t.efficiency.empty:
        n.links_t.efficiency = pd.DataFrame(index=n.snapshots, columns=n.links.index, dtype=float)
        for ln in n.links.index:
            base_eta = float(n.links.at[ln, "efficiency"]) if "efficiency" in n.links.columns else 1.0
            n.links_t.efficiency[ln] = base_eta

    # usa .ffill() / .bfill() per evitare il FutureWarning
    cop_series = COP_ts.reindex(n.snapshots).ffill().bfill()
    n.links_t.efficiency.loc[:, "Heat Pump"] = cop_series.values

    # -----------------------------
    # 18. Meta
    # -----------------------------
    n.meta = {"scenario": "invest_opt_with_dsm"}

    # -----------------------------
    # 19. Risultati "vuoti" per compatibilità con main.py
    #     (il solve lo fa main.py; qui mettiamo placeholder)
    # -----------------------------
    results = {
        "description": "invest_opt_with_dsm — network built, to be solved in main.py"
    }
    df_hourly = pd.DataFrame(index=n.snapshots)
    df_hourly.index.name = "snapshot"

    # >>> QUESTO È IL PUNTO CHIAVE <<<
    # deve restituire 3 oggetti, NON solo la rete
    return n, results, df_hourly
