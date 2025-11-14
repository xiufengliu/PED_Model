def create_network(config_file, params_file, data_path):

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
    timestamps   = snapshots
    weather_path = os.path.join(data_path, "timeseries", "weather_data.csv")
    raw_solar    = data_processing.load_timeseries(weather_path, data_path, index=timestamps)
    weather_df   = pd.DataFrame({"solar_radiation": raw_solar}, index=timestamps)

    pv_params = params["pv_battery_2"]
    eff_pv  = float(pv_params["efficiency_pv"])
    eff_inv = float(pv_params["inverter_efficiency"])
    area_m2 = float(pv_params["surface_m2"])

    # PV potenziale (AC) in kW → MW
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

    # Elettrico
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

    # -----------------------------
    # Generatore PV
    # -----------------------------
    n.add("Generator", "Rooftop PV",
          bus="PV Bus", carrier="electricity",
          p_nom=pv_p_nom_mw, p_max_pu=pv_profile_pu,
          capital_cost=0.0, marginal_cost=0.0, efficiency=1.0)

    # -----------------------------
    # GRID prezzi & generator
    # -----------------------------
    grid_params          = params["grid"]
    grid_co2_kg          = float(grid_params.get("CO2_emissions_kg_per_mwh", 0.0))
    grid_capacity_mw     = float(grid_params["capacity_mw"])
    transformer_eff      = float(grid_params["transformer_efficiency"])

    if str(grid_params["import_cost_eur_per_kwh"]).lower() == "variable":
        from .utils import load_electricity_price_profile
        grid_price_series = load_electricity_price_profile(data_path, n.snapshots)  # EUR/MWh
    else:
        fixed = float(grid_params["import_cost_eur_per_kwh"])
        grid_price_series = pd.Series(fixed * 1000.0, index=n.snapshots)  # EUR/MWh

    # Se non hai una serie export distinta, usa sconto sull’acquisto (più realistica dell’acquisto)
    export_price_series = grid_price_series
    n.export_price_series = export_price_series

    n.grid_price_series = grid_price_series

    # Generatore "Grid" (costo = prezzo d'import)
    n.add("Generator", "Grid",
          bus="Grid Elec Sell", carrier="electricity",
          p_nom_extendable=True, marginal_cost=grid_price_series, efficiency=1.0)

    # Link di import (trasformatore verso il building)
    n.add("Link", "Grid Import",
          bus0="Grid Elec Sell", bus1="Building Elec",
          p_nom=grid_capacity_mw, efficiency=transformer_eff,
          marginal_cost=0.0, capital_cost=0.0)
    n.links.at["Grid Import", "co2_emission_factor_kg_per_mwh"] = grid_co2_kg

    # -----------------------------
    # Carichi edificio
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

    n.add("Load", "Building Elec Load", bus="Building Elec", p_set=elec_profile_mw, carrier="electricity")
    n.add("Load", "Building Heat Load", bus="Building Heat", p_set=heat_profile_mw,  carrier="heat")

    # -----------------------------
    # Parametri HEAT (HP, DH, Thermal Storage)
    # -----------------------------
    heat_params   = params.get("baseline_heat_source", {})
    heat_capacity = float(heat_params.get("capacity_mw_th", 10.0))  # potenza termica max HP & rete DH
    heat_eff      = float(heat_params.get("efficiency_heating", 0.95))
    heat_cost     = float(heat_params.get("cost_eur_per_mwh_th", 45.0))
    heat_cop      = float(heat_params.get("cop", 3.0))
    standing_loss = float(heat_params.get("standing_loss_per_hour", 0.005))



    # -----------------------------
    # HEAT: Heat pump & Thermal network
    # -----------------------------
    # HP SOLO da PV (non può usare la rete) + p_nom FINITO (non estensibile)
    n.add("Link", "Heat Pump",
          bus0="PV Bus", bus1="Heat Source Bus",
          efficiency=2.5,
          p_nom=heat_capacity, p_nom_extendable=False,
          marginal_cost=0.0, carrier="heat_pump")

    # HP → Building Heat (prima del TES)
    n.add("Link", "Thermal Discharge",
          bus0="Heat Source Bus", bus1="Building Heat",
          p_nom=heat_capacity, efficiency=1.0,
          marginal_cost=0, carrier="heat_distribution")

    # TES con ciclicità e finestra SoC
    TES_E_NOM_MWH = 1.0
    TES_SOC_MIN, TES_SOC_MAX = 0.15, 0.85
    TES_STANDING  = standing_loss

    n.add("Store", "Thermal Store",
          bus="Thermal Storage",
          e_nom=TES_E_NOM_MWH,
          e_min_pu=TES_SOC_MIN, e_max_pu=TES_SOC_MAX,
          e_initial=0.50*TES_E_NOM_MWH,
          e_cyclic=True,
          standing_loss=TES_STANDING,
          marginal_cost=0.0)

    # Potenze TES
    heat_peak_mw = float(bld.get("heat_peak_mw", 0.32))
    P_TES_CH_MW  = max(heat_peak_mw, 1.0)
    P_TES_DIS_MW = max(heat_peak_mw, heat_capacity)
    ETA_TES_CH, ETA_TES_DIS = 0.98, 0.98

    # HP → TES (carica) dopo aver servito edificio (ma sempre SOLO PV)
    n.add("Link", "Thermal Charge",
          bus0="Heat Source Bus", bus1="Thermal Storage",
          p_nom=P_TES_CH_MW, efficiency=ETA_TES_CH,
          p_nom_extendable=False, marginal_cost=0)

    # Dispatch termico: piccolo costo positivo per evitare cicli
    n.add("Link", "Thermal Dispatch",
          bus0="Thermal Storage", bus1="Building Heat",
          p_nom=P_TES_DIS_MW, efficiency=ETA_TES_DIS,
          marginal_cost=0)

    # District heating come backup (costo sul generator)
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
    # ELETTRICO (PV-first “economico” senza import extra)
    # -----------------------------
    # PV → Building (autoconsumo)
    n.add("Link", "PV Autoconsumo",
          bus0="PV Bus", bus1="Building Elec",
          p_nom=pv_p_nom_mw, efficiency=1.0,
          capital_cost=0.0, marginal_cost=0.0)

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

    # PV → Battery (carica)
    n.add("Link", "Battery Charge",
          bus0="PV Bus", bus1="Battery Storage",
          p_nom=battery_p_nom, efficiency=eff_store,
          capital_cost=0.0, marginal_cost=0.0)

    # Battery → Building (dispatch) – costo positivo piccolo
    n.add("Link", "Battery Dispatch",
          bus0="Battery Storage", bus1="Building Elec",
          p_nom=battery_p_nom, efficiency=eff_dispatch,
          capital_cost=0.0, marginal_cost=0)

    # -----------------------------
    # EXPORT (solo da PV)
    # -----------------------------
    n.add("Link", "PV Export",
          bus0="PV Bus", bus1="Grid Elec Buy",
          p_nom=pv_p_nom_mw, efficiency=export_eff,
          capital_cost=0.0, marginal_cost=-0.1)
    n.links.at["PV Export", "co2_emission_factor_kg_per_mwh"] = grid_co2_kg





    if (not hasattr(n.links_t, "marginal_cost")) or (n.links_t.marginal_cost is None):
      import pandas as pd
      n.links_t.marginal_cost = pd.DataFrame(index=n.snapshots, columns=n.links.index)




    # Assicurati che i link di dispatch restino con piccolo costo positivo (evita cicli gratuiti)
    if "Battery Dispatch" in n.links.index:
        n.links.at["Battery Dispatch", "marginal_cost"] = 0
        n.links_t.marginal_cost["Battery Dispatch"] = 0

    if "Thermal Dispatch" in n.links.index:
        n.links.at["Thermal Dispatch", "marginal_cost"] = 0
        n.links_t.marginal_cost["Thermal Dispatch"] = 0



    # -----------------------------
    # Export Sink: assorbe l'energia esportata
    # -----------------------------
    n.add("StorageUnit", "Export Sink",
          bus="Grid Elec Buy",
          p_nom=1e3, p_nom_extendable=True, max_hours=1e6,
          p_min_pu=-1.0, p_max_pu=0.0,
          efficiency_store=1.0, efficiency_dispatch=1.0,
          cyclic_state_of_charge=False, capital_cost=0.0, marginal_cost=0.0)

    # -----------------------------
    # OUTPUTS per main/plot (flussi netti lato bus di arrivo)
    # -----------------------------
    results = {
        "pv_generation":     pv_generation_mw,
        "electric_load":     n.loads_t.p.get("Building Elec Load", pd.Series(0.0, index=n.snapshots)),
        "heat_load":         n.loads_t.p.get("Building Heat Load", pd.Series(0.0, index=n.snapshots)),

        "grid_import":       (-n.links_t.p1.get("Grid Import", pd.Series(0.0, index=n.snapshots))).clip(lower=0.0),
        "pv_export":         (-n.links_t.p1.get("PV Export",  pd.Series(0.0, index=n.snapshots))).clip(lower=0.0),

        # output termico HP (bus1 = Heat Source Bus, positivo verso bus1)
        "hp_output_th":      (-n.links_t.p1.get("Heat Pump",  pd.Series(0.0, index=n.snapshots))).clip(lower=0.0),

        "th_store_soc":      n.stores_t.e.get("Thermal Store", pd.Series(0.0, index=n.snapshots)),
        "th_store_charge":   (-n.links_t.p1.get("Thermal Charge",   pd.Series(0.0, index=n.snapshots))).clip(lower=0.0),
        "th_store_dispatch": (-n.links_t.p1.get("Thermal Dispatch", pd.Series(0.0, index=n.snapshots))).clip(lower=0.0),

        "battery_soc":       n.stores_t.e.get("Battery Store", pd.Series(0.0, index=n.snapshots)),
        "battery_charge":    (-n.links_t.p1.get("Battery Charge",   pd.Series(0.0, index=n.snapshots))).clip(lower=0.0),
        "battery_dispatch":  (-n.links_t.p1.get("Battery Dispatch", pd.Series(0.0, index=n.snapshots))).clip(lower=0.0),
    }
    df_hourly = pd.DataFrame(results); df_hourly.index.name = "snapshot"

    # Debug veloce (facoltativo)

    print("max grid price:", float(pd.Series(n.grid_price_series).max()))
    print("max export price:", float(pd.Series(export_price_series).max()))

    # Sanity checks
    assert len(n.snapshots) == n_hours, "❌ Numero snapshot errato."
    for ln in n.loads.index:
        s = n.loads_t.p_set[ln]
        assert len(s) == n_hours and s.notna().all(), f"❌ Load '{ln}' non coerente."

    return n, results, df_hourly
