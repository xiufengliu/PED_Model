def create_network(config_file, params_file, data_path):
    """
    DSM scenario (reworked, PRIORITÀ FINALIZZATE):
    - Stessa architettura di `thermal_storage` + shifting DSM (store virtuali).
    - Priorità (ε: più negativo = più prioritario) e dispatch con costo positivo.

    Ordine unico delle priorità (più → meno):
      1) PV → Heat Pump → Building Heat
      2) PV → Heat Pump → DSM Heat (carica, pre-riscaldo)
      3) PV → Building Elec (autoconsumo)
      4) PV/Building → DSM Elec Store (carica flessibile)
      5) PV → Heat Pump → Thermal Storage (carica)
      6) PV → Battery (carica)
      7) PV → Export (solo da PV; costo dinamico = −prezzo + micro-offset)

    Dispatch (solo quando serve, con piccolo costo positivo):
      8) DSM Elec Store → Building Elec    ( +1e-4 )
      9) Battery → Building Elec           ( +2e-4 )
     10) DSM Heat Store → Building Heat    ( +1e-4 )
     11) Thermal Storage → Building Heat   ( +2e-4 )

    Ultima risorsa (prezzi reali, nessun ε):
      - Grid Import (Generator "Grid" con serie prezzi)
      - District Heating (Generator "Heating")

    Obiettivo DSM:
      Valutare il valore della flessibilità (spostamento carichi elettrici/termici)
      per ridurre costi/emissioni e integrare rinnovabili variabili.
    """
    import os
    import pypsa
    import pandas as pd
    import numpy as np

    from .utils import load_config, load_or_generate_profile
    from scripts import data_processing

    # -----------------------------
    # Config & snapshots
    # -----------------------------
    config, params = load_config(config_file, params_file)

    n = pypsa.Network()
    start_date = config["simulation_settings"]["start_date"]
    n_hours    = int(config["simulation_settings"]["num_hours"])
    snapshots  = pd.date_range(start=start_date, periods=n_hours, freq="h")
    n.set_snapshots(snapshots)

    # -----------------------------
    # Meteo & PV
    # -----------------------------
    timestamps  = n.snapshots
    weather_fp  = os.path.join(data_path, "timeseries", "weather_data.csv")
    raw_solar   = data_processing.load_timeseries(weather_fp, data_path, index=timestamps)
    weather_df  = pd.DataFrame({"solar_radiation": raw_solar}, index=timestamps)

    pv_params   = params["pv_battery_2"]
    eff_pv      = float(pv_params["efficiency_pv"])
    eff_inv     = float(pv_params["inverter_efficiency"])
    area_m2     = float(pv_params["surface_m2"])

    pv_generation_kw = data_processing.calculate_pv_generation(
        weather_df=weather_df,
        pv_surface_m2=area_m2,
        pv_efficiency_pv=eff_pv,
        pv_inverter_efficiency=eff_inv,
    )
    pv_generation_mw = pv_generation_kw / 1000.0
    pv_p_nom_mw      = float(pv_generation_mw.max()) if pv_generation_mw.max() > 0 else 0.001
    pv_profile_pu    = (pv_generation_mw / pv_p_nom_mw).clip(lower=0.0)

    # -----------------------------
    # Carrier & Buses
    # -----------------------------
    for c in ["electricity", "heat", "heat_distribution", "heat_pump"]:
        if c not in n.carriers.index:
            n.add("Carrier", c)

    # Elettricità (Sell/Buy separati per evitare loop)
    n.add("Bus", "Grid Elec Sell",  carrier="electricity")
    n.add("Bus", "Grid Elec Buy",   carrier="electricity")
    n.add("Bus", "Building Elec",   carrier="electricity")
    n.add("Bus", "Battery Storage", carrier="electricity")
    n.add("Bus", "PV Bus",          carrier="electricity")

    # Termico
    n.add("Bus", "District Heating", carrier="heat")
    n.add("Bus", "Building Heat",    carrier="heat")
    n.add("Bus", "Heat Source Bus",  carrier="heat")
    n.add("Bus", "Thermal Storage",  carrier="heat")

    # DSM: bus dedicati (domanda flessibile)
    n.add("Bus", "DSM Elec Store", carrier="electricity")
    n.add("Bus", "DSM Heat Store", carrier="heat")

    # -----------------------------
    # Generatore PV
    # -----------------------------
    n.add("Generator", "Rooftop PV",
          bus="PV Bus", carrier="electricity",
          p_nom=pv_p_nom_mw, p_max_pu=pv_profile_pu,
          capital_cost=0.0, marginal_cost=0.0, efficiency=1.0)

    # -----------------------------
    # GRID: prezzi, generator di import e link
    # -----------------------------
    grid_params           = params["grid"]
    grid_co2_kg           = float(grid_params.get("CO2_emissions_kg_per_mwh", 0.0))
    grid_capacity_mw      = float(grid_params["capacity_mw"])
    transformer_eff       = float(grid_params["transformer_efficiency"])

    if str(grid_params["import_cost_eur_per_kwh"]).lower() == "variable":
        from .utils import load_electricity_price_profile
        grid_price_series = load_electricity_price_profile(data_path, n.snapshots)  # EUR/MWh
    else:
        fixed = float(grid_params["import_cost_eur_per_kwh"])
        grid_price_series = pd.Series(fixed * 1000.0, index=n.snapshots)

    export_price_series = grid_price_series.copy()
    n.grid_price_series = grid_price_series

    n.add("Generator", "Grid",
          bus="Grid Elec Sell", carrier="electricity",
          p_nom_extendable=True, marginal_cost=grid_price_series, efficiency=1.0)

    n.add("Link", "Grid Import",
          bus0="Grid Elec Sell", bus1="Building Elec",
          p_nom=grid_capacity_mw, efficiency=transformer_eff,
          marginal_cost=0.0, capital_cost=0.0)
    n.links.at["Grid Import", "co2_emission_factor_kg_per_mwh"] = grid_co2_kg

    # -----------------------------
    # Carichi edificio (split inflessibile/flessibile)
    # -----------------------------
    bld = params.get("social_building", {})
    elec_profile_mw = load_or_generate_profile(
        bld.get("electricity_load_profile", "electricity_demand.csv"),
        bld.get("electricity_peak_mw", 0.23), data_path, n.snapshots
    ) / 1000.0
    heat_profile_mw = load_or_generate_profile(
        bld.get("heat_load_profile", "heat_demand.csv"),
        bld.get("heat_peak_mw", 0.32), data_path, n.snapshots
    ) / 1000.0

    dsm_cfg   = params.get("dsm", {})
    flex_share = float(dsm_cfg.get("flexible_load_share", 0.3))
    max_shift  = float(dsm_cfg.get("max_shift_hours", 3))
    dsm_eta    = float(dsm_cfg.get("efficiency", 1.0))

    elec_flex = elec_profile_mw * flex_share
    elec_infl = elec_profile_mw - elec_flex
    heat_flex = heat_profile_mw * flex_share
    heat_infl = heat_profile_mw - heat_flex

    n.add("Load", "Building Elec Load", bus="Building Elec", p_set=elec_infl, carrier="electricity")
    n.add("Load", "DSM Elec Flex Load", bus="Building Elec", p_set=elec_flex, carrier="electricity")

    n.add("Load", "Building Heat Load", bus="Building Heat", p_set=heat_infl, carrier="heat")
    n.add("Load", "DSM Heat Flex Load", bus="Building Heat", p_set=heat_flex, carrier="heat")

    # -----------------------------
    # Dimensionamento store DSM (da profili flessibili)
    # -----------------------------
    # Elettrico DSM
    p_elec_peak = float(elec_flex.max()) if len(elec_flex) else 0.0
    e_elec_nom  = p_elec_peak * max_shift
    n.add("Store", "DSM Elec Store",
          bus="DSM Elec Store",
          e_nom=e_elec_nom,
          e_min_pu=0.0, e_max_pu=1.0,
          e_initial=0.0, e_cyclic=False,
          marginal_cost=0.0)

    # Termico DSM
    p_heat_peak = float(heat_flex.max()) if len(heat_flex) else 0.0
    e_heat_nom  = p_heat_peak * max_shift
    n.add("Store", "DSM Heat Store",
          bus="DSM Heat Store",
          e_nom=e_heat_nom,
          e_min_pu=0.0, e_max_pu=1.0,
          e_initial=0.0, e_cyclic=False,
          marginal_cost=0.0)

    # -----------------------------
    # Parametri termici (HP, DH, TES)
    # -----------------------------
    heat_params   = params.get("baseline_heat_source", {})
    heat_capacity = float(heat_params.get("capacity_mw_th", 10.0))
    heat_eff      = float(heat_params.get("efficiency_heating", 0.95))
    heat_cost     = float(heat_params.get("cost_eur_per_mwh_th", 45.0))
    heat_cop      = float(heat_params.get("cop", 3.0))
    standing_loss = float(heat_params.get("standing_loss_per_hour", 0.005))

    # -----------------------------
    # ε: scala e priorità (finali)
    # -----------------------------
    export_abs_max = float(pd.Series(export_price_series).max()) if len(export_price_series) else 0.0
    BASE = max(50.0, export_abs_max + 10.0)
    eps = {
        # Termico
        "hp":               -10.0*BASE,
        "th_discharge":     -9.0*BASE,
        "dsm_heat_charge":  -8.5*BASE,
        "th_charge":        -5.0*BASE,

        # Elettrico
        "pv_to_building":   -7.0*BASE,
        "dsm_elec_charge":  -6.5*BASE,
        "pv_to_batt":       -4.0*BASE,

        # Export (micro-offset)
        "pv_export_micro":  -1e-6,
    }

    # -----------------------------
    # TERMICO
    # -----------------------------
    # HP solo da PV
    n.add("Link", "Heat Pump",
          bus0="PV Bus", bus1="Heat Source Bus",
          efficiency=heat_cop, p_nom_extendable=True, p_nom=0.0,
          marginal_cost=eps["hp"], carrier="heat_pump")

    # 1) Heat Source → Building Heat (calore ora)
    n.add("Link", "Thermal Discharge",
          bus0="Heat Source Bus", bus1="Building Heat",
          p_nom=heat_capacity, efficiency=1.0,
          marginal_cost=eps["th_discharge"], carrier="heat_distribution")

    # 2) DSM Heat (pre-riscaldo) — carica (PV→HP→Building Heat→DSM)
    n.add("Link", "DSM Heat Charge",
          bus0="Building Heat", bus1="DSM Heat Store",
          p_nom=p_heat_peak, efficiency=1.0,
          marginal_cost=eps["dsm_heat_charge"])

    # TES store
    TES_E_NOM_MWH = 1.0
    TES_SOC_MIN, TES_SOC_MAX = 0.15, 0.85
    n.add("Store", "Thermal Store",
          bus="Thermal Storage",
          e_nom=TES_E_NOM_MWH,
          e_min_pu=TES_SOC_MIN, e_max_pu=TES_SOC_MAX,
          e_initial=0.50*TES_E_NOM_MWH,
          e_cyclic=True,
          standing_loss=standing_loss,
          marginal_cost=0.0)

    # 5) Heat Source → TES (carica)
    ETA_TES_CH, ETA_TES_DIS = 0.98, 0.98
    P_TES_CH_MW  = max(p_heat_peak, 1.0)
    P_TES_DIS_MW = max(p_heat_peak, heat_capacity)
    n.add("Link", "Thermal Charge",
          bus0="Heat Source Bus", bus1="Thermal Storage",
          p_nom=P_TES_CH_MW, efficiency=ETA_TES_CH,
          p_nom_extendable=False, marginal_cost=eps["th_charge"])

    # Dispatch termico (solo quando serve): DSM prima del TES
    n.add("Link", "DSM Heat Dispatch",
          bus0="DSM Heat Store", bus1="Building Heat",
          p_nom=p_heat_peak, efficiency=1.0,
          marginal_cost=1e-4)

    n.add("Link", "Thermal Dispatch",
          bus0="Thermal Storage", bus1="Building Heat",
          p_nom=P_TES_DIS_MW, efficiency=ETA_TES_DIS,
          marginal_cost=2e-4)

    # DH backup (prezzo sul generator)
    n.add("Link", "Heat import",
          bus0="District Heating", bus1="Building Heat",
          p_nom=heat_capacity, p_nom_extendable=False,
          marginal_cost=0.0, efficiency=heat_eff)
    if "co2_emissions_dh_kg_per_mwh" in heat_params:
        dh_factor = float(heat_params["co2_emissions_dh_kg_per_mwh"])
        n.links.at["Heat import", "co2_emission_factor_kg_per_mwh"] = dh_factor

    n.add("Generator", "Heating",
          bus="District Heating", carrier="heat",
          p_nom=heat_capacity, marginal_cost=heat_cost)

    # -----------------------------
    # ELETTRICO
    # -----------------------------
    # 3) PV → Building (autoconsumo)
    n.add("Link", "PV Autoconsumo",
          bus0="PV Bus", bus1="Building Elec",
          p_nom=pv_p_nom_mw, efficiency=1.0,
          capital_cost=0.0, marginal_cost=eps["pv_to_building"])

    # 4) Building/PV → DSM Elec (carica flessibile)
    n.add("Link", "DSM Elec Charge",
          bus0="Building Elec", bus1="DSM Elec Store",
          p_nom=p_elec_peak, efficiency=dsm_eta,
          marginal_cost=eps["dsm_elec_charge"], capital_cost=0.0)

    # Batteria
    batt_cfg       = params["pv_battery_2"]["battery"]
    battery_p_nom  = float(batt_cfg.get("p_nom_mw",   0.05))
    battery_e_nom  = float(batt_cfg.get("e_nom_mwh",  0.2))
    eta_rt         = float(batt_cfg.get("efficiency_round_trip", 0.9))
    export_eff     = float(batt_cfg.get("efficiency_export",     0.95))
    eff_store      = eta_rt ** 0.5
    eff_dispatch   = eta_rt ** 0.5

    n.add("Store", "Battery Store",
          bus="Battery Storage",
          e_nom=battery_e_nom,
          e_min_pu=0.15, e_max_pu=0.85,
          e_initial=0.50*battery_e_nom,
          e_cyclic=True, marginal_cost=0.0)

    # 6) PV → Battery (carica)
    n.add("Link", "Battery Charge",
          bus0="PV Bus", bus1="Battery Storage",
          p_nom=battery_p_nom, efficiency=eff_store,
          capital_cost=0.0, marginal_cost=eps["pv_to_batt"])

    # Dispatch elettrico (solo quando serve): DSM prima della batteria
    n.add("Link", "DSM Elec Dispatch",
          bus0="DSM Elec Store", bus1="Building Elec",
          p_nom=p_elec_peak, efficiency=dsm_eta,
          marginal_cost=1e-4, capital_cost=0.0)

    n.add("Link", "Battery Dispatch",
          bus0="Battery Storage", bus1="Building Elec",
          p_nom=battery_p_nom, efficiency=eff_dispatch,
          capital_cost=0.0, marginal_cost=2e-4)

    # -----------------------------
    # EXPORT (solo da PV)
    # -----------------------------
    n.add("Link", "PV Export",
          bus0="PV Bus", bus1="Grid Elec Buy",
          p_nom=pv_p_nom_mw, efficiency=export_eff,
          capital_cost=0.0, marginal_cost=0.0)

    if not hasattr(n.links_t, "marginal_cost") or n.links_t.marginal_cost is None or n.links_t.marginal_cost.empty:
        n.links_t.marginal_cost = pd.DataFrame(0.0, index=n.snapshots, columns=n.links.index)
    n.links_t.marginal_cost["PV Export"] = -pd.Series(export_price_series, index=n.snapshots) + eps["pv_export_micro"]
    n.links.at["PV Export", "co2_emission_factor_kg_per_mwh"] = grid_co2_kg

    # Export sink per chiudere il bilancio sul bus di export
    n.add("StorageUnit", "Export Sink",
          bus="Grid Elec Buy",
          p_nom=1e3, p_nom_extendable=True, max_hours=1e6,
          p_min_pu=-1.0, p_max_pu=0.0,
          efficiency_store=1.0, efficiency_dispatch=1.0,
          cyclic_state_of_charge=False, capital_cost=0.0, marginal_cost=0.0)

    # -----------------------------
    # OUTPUTS
    # -----------------------------
    results = {
        "pv_generation":      pv_generation_mw,
        "electric_load":      (elec_infl + elec_flex),
        "heat_load":          (heat_infl + heat_flex),

        "grid_import":        (-n.links_t.p1.get("Grid Import", pd.Series(0.0, index=n.snapshots))).clip(lower=0.0),
        "pv_export":          (-n.links_t.p1.get("PV Export",  pd.Series(0.0, index=n.snapshots))).clip(lower=0.0),

        "hp_output_th":       (-n.links_t.p1.get("Heat Pump",  pd.Series(0.0, index=n.snapshots))).clip(lower=0.0),

        "th_store_soc":       n.stores_t.e.get("Thermal Store", pd.Series(0.0, index=n.snapshots)),
        "th_store_charge":    (-n.links_t.p1.get("Thermal Charge",   pd.Series(0.0, index=n.snapshots))).clip(lower=0.0),
        "th_store_dispatch":  (-n.links_t.p1.get("Thermal Dispatch", pd.Series(0.0, index=n.snapshots))).clip(lower=0.0),

        "battery_soc":        n.stores_t.e.get("Battery Store", pd.Series(0.0, index=n.snapshots)),
        "battery_charge":     (-n.links_t.p1.get("Battery Charge",   pd.Series(0.0, index=n.snapshots))).clip(lower=0.0),
        "battery_dispatch":   (-n.links_t.p1.get("Battery Dispatch", pd.Series(0.0, index=n.snapshots))).clip(lower=0.0),

        "dsm_elec_soc":       n.stores_t.e.get("DSM Elec Store", pd.Series(0.0, index=n.snapshots)),
        "dsm_elec_charge":    (-n.links_t.p1.get("DSM Elec Charge",   pd.Series(0.0, index=n.snapshots))).clip(lower=0.0),
        "dsm_elec_dispatch":  (-n.links_t.p1.get("DSM Elec Dispatch", pd.Series(0.0, index=n.snapshots))).clip(lower=0.0),

        "dsm_heat_soc":       n.stores_t.e.get("DSM Heat Store", pd.Series(0.0, index=n.snapshots)),
        "dsm_heat_charge":    (-n.links_t.p1.get("DSM Heat Charge",   pd.Series(0.0, index=n.snapshots))).clip(lower=0.0),
        "dsm_heat_dispatch":  (-n.links_t.p1.get("DSM Heat Dispatch", pd.Series(0.0, index=n.snapshots))).clip(lower=0.0),
    }
    df_hourly = pd.DataFrame(results); df_hourly.index.name = "snapshot"

    # Log priorità e prezzi (debug)
    print({k: float(v) for k, v in eps.items()})
    print("max export price:", float(pd.Series(n.grid_price_series).max()))

    # Sanity checks minimi
    assert len(n.snapshots) == n_hours, "❌ Numero snapshot errato."
    for ln in n.loads.index:
        s = n.loads_t.p_set[ln]
        assert len(s) == n_hours and s.notna().all(), f"❌ Load '{ln}' non coerente."

    return n, results, df_hourly
