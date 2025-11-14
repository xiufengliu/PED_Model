"""
PED Lyngby Model - Baseline Scenario (PV + Battery, export-only-from-PV)

Priorità dei flussi elettrici:
1) PV → Building Elec
2) PV → Battery (carica)
3) PV → Export (verso Export Sink)
4) Battery → Building Elec (scarica)
5) Grid Import (costo reale via Generator "Grid")
"""

import os
import pypsa
import pandas as pd
import numpy as np

from .utils import load_config, load_or_generate_profile


def create_network(config_file, params_file, data_path):
    """
    Costruisce il network PyPSA per lo scenario baseline (con batteria reale)
    rispettando le priorità richieste. La nomenclatura è allineata a main.py.

    Ritorna:
        pypsa.Network
    """
    print("Building baseline network for social housing building...")

    # -----------------------------
    # Config & snapshots
    # -----------------------------
    config, params = load_config(config_file, params_file)

    n = pypsa.Network()
    start_date = config["simulation_settings"]["start_date"]
    n_hours = config["simulation_settings"]["num_hours"]
    snapshots = pd.date_range(start=start_date, periods=n_hours, freq="h")
    timestamps = pd.date_range(start=start_date, periods=n_hours, freq='h')
    n.set_snapshots(snapshots)
    print(f"Set network snapshots: {len(n.snapshots)} hours, starting {start_date}")

    # -----------------------------
    # Dati PV e meteo
    # -----------------------------
    pv_params = params["pv_battery_3"]
    eff_pv = pv_params["efficiency_pv"]
    eff_inv = pv_params["inverter_efficiency"]
    area_m2 = pv_params["surface_m2"] 

    # Estrai parametri batteria da component_params.yml
    battery_params = params["pv_battery_3"]["battery"]
    battery_p_nom = battery_params["p_nom_mw"]
    battery_e_nom = battery_params["e_nom_mwh"]
    round_trip_efficiency = battery_params["efficiency_round_trip"]
    export_efficiency = battery_params["efficiency_export"]
    desired_duration = battery_params["autonomy_hours"]

    # APPROACH A: p_nom = STC capacity, p_max_pu = irradiance/1000
    pv_p_nom_mw = (area_m2 * eff_pv * eff_inv) / 1000  # STC capacity in MW

    # Import data_processing to get PV profile
    from scripts import data_processing

    # Load weather data for PV calculation
    weather_filename = "weather_data.csv"
    raw_solar_radiation_series = data_processing.load_timeseries(weather_filename, data_path, index=timestamps)
    weather_df = pd.DataFrame({'solar_radiation': raw_solar_radiation_series}, index=timestamps)

    # Calculate p_max_pu as irradiance/1000 (STC normalization)
    pv_profile_pu = raw_solar_radiation_series / 1000  # Normalized on STC (1000 W/m²)

    print(f"DEBUG: Maximum irradiance in data: {raw_solar_radiation_series.max():.1f} W/m²")
    print(f"DEBUG: PV p_nom (STC capacity): {pv_p_nom_mw:.6f} MW ({pv_p_nom_mw*1000:.1f} kW)")
    print(f"DEBUG: p_max_pu normalized on STC: max = {pv_profile_pu.max():.6f}")

    # Calculate actual generation for verification
    pv_generation_kw = data_processing.calculate_pv_generation(
        weather_df=weather_df,
        pv_surface_m2=area_m2,
        pv_efficiency_pv=eff_pv,
        pv_inverter_efficiency=eff_inv
    )
    pv_generation_mw = pv_generation_kw / 1000

    print(f"DEBUG: PV generation - Max: {pv_generation_mw.max():.6f} MW, Annual: {pv_generation_mw.sum():.1f} MWh")
    print(f"DEBUG: PV profile p_max_pu - Max: {pv_profile_pu.max():.6f}, Mean: {pv_profile_pu.mean():.6f}")
    print(f"DEBUG: p_max_pu normalized on STC (1000 W/m²) - physically meaningful")

    # -----------------------------
    # Carrier
    # -----------------------------
    for c in ["electricity", "heat", "heat_distribution", "heat_pump"]:
        if c not in n.carriers.index:
            n.add("Carrier", c)

    # -----------------------------
    # Buses (coerenti con main.py)
    # -----------------------------
    n.add("Bus", "Grid Elec Sell", carrier="electricity")
    n.add("Bus", "Grid Elec Buy",  carrier="electricity")
    n.add("Bus", "Building Elec",  carrier="electricity")
    n.add("Bus", "Battery Storage", carrier="electricity")
    n.add("Bus", "PV Bus", carrier="electricity")

    # Heat buses (come in baseline)
    n.add("Bus", "District Heating", carrier="heat")
    n.add("Bus", "Building Heat",    carrier="heat")
    n.add("Bus", "Heat Source Bus",  carrier="heat")
    n.add("Bus", "Thermal Storage",  carrier="heat")

    # -----------------------------
    # Generatore PV (marginal_cost=0)
    # -----------------------------
    n.add("Generator", "Rooftop PV",
          bus="PV Bus",
          carrier="electricity",
          p_nom=pv_p_nom_mw,
          p_max_pu=pv_profile_pu,
          capital_cost=0.0,
          marginal_cost=0.0,
          efficiency=1.0)

    # -----------------------------
    # Rete elettrica e prezzi
    # Prezzi import (EUR/MWh): identici a high_pv
    grid_params = params["grid"]
    grid_co2_kg = grid_params.get('CO2_emissions_kg_per_mwh', 0)
    grid_capacity_mw = grid_params['capacity_mw']
    transformer_efficiency = grid_params['transformer_efficiency']


    if str(grid_params["import_cost_eur_per_kwh"]).lower() == "variable":
        from .utils import load_electricity_price_profile
        grid_price_series = load_electricity_price_profile(data_path, n.snapshots)  # EUR/MWh
        print("✔️ Prezzi elettrici variabili caricati da data_dir/timeseries/grid_prices.csv.")
    else:
        fixed = float(grid_params["import_cost_eur_per_kwh"])
        grid_price_series = pd.Series(fixed * 1000.0, index=n.snapshots)  # EUR/MWh
        print(f"✔️ Prezzo elettrico fisso: {fixed} EUR/kWh → {fixed*1000:.2f} EUR/MWh.")


    # Salva per summary/analisi
    n.grid_price_series = grid_price_series

    # Generatore "Grid" (costo = prezzo di import)
    n.add("Generator", "Grid",
          bus="Grid Elec Sell",
          carrier="electricity",
          p_nom_extendable=True,
          marginal_cost=grid_price_series,
          efficiency=1.0)


    # Link di import (costo sul link = 0; costo vero sta nel generator "Grid")
    n.add("Link", "Grid Import",
          bus0="Grid Elec Sell",
          bus1="Building Elec",
          p_nom=grid_capacity_mw,
          efficiency=transformer_efficiency,
          marginal_cost=0.0,
          capital_cost=0.0)
    n.links.at["Grid Import", "co2_emission_factor_kg_per_mwh"] = grid_co2_kg

    # -----------------------------
    # Carichi edificio (MW)
    # -----------------------------
    bld = params.get("social_building", {})
    elec_profile_mw = load_or_generate_profile(
        bld.get("electricity_load_profile", "electricity_demand.csv"),
        bld.get("electricity_peak_mw", 0.23),
        data_path,
        n.snapshots
    ) / 1000.0

    heat_profile_mw = load_or_generate_profile(
        bld.get("heat_load_profile", "heat_demand.csv"),
        bld.get("heat_peak_mw", 0.32),
        data_path,
        n.snapshots
    ) / 1000.0

    n.add("Load", "Building Elec Load", bus="Building Elec", p_set=elec_profile_mw, carrier="electricity")
    n.add("Load", "Building Heat Load", bus="Building Heat", p_set=heat_profile_mw, carrier="heat")

    # -----------------------------
    # Sorgente di calore (come baseline)
    # -----------------------------
    heat_params = params.get("baseline_heat_source", {})
    heat_capacity_mw = heat_params.get("capacity_mw_th", 10.0)
    heat_eff = heat_params.get("efficiency_heating", 0.95)

    from .utils import load_thermal_price_profile
    price_profile = heat_params.get("price_profile")
    if price_profile:
        heat_cost = load_thermal_price_profile(os.path.join(data_path, "timeseries", price_profile), n.snapshots)
        print("✔️ Prezzo termico variabile caricato (EUR/MWh).")
    else:
        heat_cost = float(heat_params.get("cost_eur_per_mwh_th", 45.0))
        print(f"✔️ Prezzo termico fisso: {heat_cost:.2f} EUR/MWh.")

    heat_params = params.get("baseline_heat_source", {})
    dh_co2_kg_per_mwh = float(heat_params.get("co2_emissions_dh_kg_per_mwh", 0.0))

    n.add("Link", "Heat import",
          bus0="District Heating",
          bus1="Building Heat",
          p_nom=1.0,
          p_nom_extendable=False,
          marginal_cost=0,
          efficiency=heat_eff)

    n.links.at["Heat import", "co2_emission_factor_kg_per_mwh"] = dh_co2_kg_per_mwh


    n.add("Generator", "Heating",
          bus="District Heating",
          carrier="heat",
          p_nom=heat_capacity_mw,
          marginal_cost=heat_cost)

    # ============================================================
    #            PRIORITÀ ELETTRICHE (via micro-ε)
    # ============================================================
    # Più NEGATIVO = più prioritario (rispetto a 0 del curtailment).
    # 1) PV→Building  2) PV→Battery  3) PV→ExportSink  4) Batt→Building  5) Grid Import
    eps_pv_to_load  = -1e-6    # PV→Building: massima priorità
    eps_batt_charge = -5e-9    # PV→Battery: seconda
    eps_export      = -1e-9    # PV→Export: terza (comunque meglio di curtailment=0)

    # Efficienze batteria (round-trip spezzato)
    # Se non hai batteria nei params, inventa un blocco minimo safe
    batt_cfg = params.get("pv_battery_3", {}).get("battery", {})
    battery_p_nom = float(batt_cfg.get("p_nom_mw", 0.05))         # default 50 kW
    battery_e_nom = float(batt_cfg.get("e_nom_mwh", 0.2))         # default 0.2 MWh
    eta_rt        = float(batt_cfg.get("efficiency_round_trip", 0.9))
    eff_store     = eta_rt ** 0.5
    eff_dispatch  = eta_rt ** 0.5

    # 1) PV → Building
    n.add("Link", "PV Autoconsumo",
          bus0="PV Bus",
          bus1="Building Elec",
          p_nom=pv_p_nom_mw,
          efficiency=1.0,
          capital_cost=0.0,
          marginal_cost=0)

    # Store (batteria reale)
    n.add("Store", "Battery Store",
          bus="Battery Storage",
          e_nom=battery_e_nom,
          p_nom=battery_p_nom,
          e_min_pu=0.15,            # SoC_min 15%
          e_max_pu=0.85,            # SoC_max 85%
          e_initial=0.50*battery_e_nom, # SoC iniziale 50% di E_nom
          e_cyclic=True,            # SoC(0) = SoC(T)          capital_cost=0.0,
          marginal_cost=0.0)

    # 2) PV → Battery (carica)
    n.add("Link", "Battery Charge",
          bus0="PV Bus",
          bus1="Battery Storage",
          p_nom=battery_p_nom,
          efficiency=eff_store,
          capital_cost=0.0,
          marginal_cost=0)

    # 4) Battery → Building (scarica) — preferita all’import
    n.add("Link", "Battery Dispatch",
          bus0="Battery Storage",
          bus1="Building Elec",
          p_nom=battery_p_nom,
          efficiency=eff_dispatch,
          capital_cost=0.0,
          marginal_cost=0.0)

    # 3) PV → Export (solo dal PV Bus)
    n.add("Link", "PV Export",
          bus0="PV Bus",
          bus1="Grid Elec Buy",
          p_nom=pv_p_nom_mw,
          efficiency=export_efficiency,
          capital_cost=0.0,
          marginal_cost=-0.1)
    n.links.at["PV Export", "co2_emission_factor_kg_per_mwh"] = grid_co2_kg

    # Export Sink: può solo CARICARE (simula la vendita), non restituisce mai energia
    n.add("StorageUnit", "Export Sink",
          bus="Grid Elec Buy",
          p_nom=1e3,                 # grande capacità di assorbimento
          p_nom_extendable=True,
          max_hours=1e6,             # capacità energetica enorme (virtuale)
          p_min_pu=-1.0,             # può assorbire fino a p_nom
          p_max_pu=0.0,              # non può erogare
          efficiency_store=1.0,
          efficiency_dispatch=1.0,   # irrilevante (p_max_pu=0)
          cyclic_state_of_charge=False,
          capital_cost=0.0,
          marginal_cost=0.0)



    # -----------------------------
    # Verifiche di coerenza
    # -----------------------------
    assert len(n.snapshots) == n_hours, "❌ Numero snapshot errato."
    for load_name in n.loads.index:
        s = n.loads_t.p_set[load_name]
        assert len(s) == n_hours and s.notna().all(), f"❌ Load '{load_name}' non coerente."

    # Se il generatore ha la colonna p_max_pu, controlla la lunghezza
    if "Rooftop PV" in n.generators.index and "Rooftop PV" in getattr(n.generators_t, "p_max_pu", pd.DataFrame()).columns:
        assert len(n.generators_t.p_max_pu["Rooftop PV"]) == n_hours, "❌ p_max_pu PV non coerente con snapshots."

    return n
