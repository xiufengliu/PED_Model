"""
PED Lyngby Model - Baseline Scenario

This module implements the baseline scenario for the PED Lyngby Model.
The baseline scenario represents the reference case for the specific social housing
building shown in the building_of_study.png image. This is a 13-floor multi-apartment
residential building with 165 apartments, built around 1970, with a total floor area
of approximately 9500 m² and a usable roof area of 700 m² for PV installation.
"""

import pypsa
import pandas as pd
import numpy as np
import os

from .utils import load_config, load_or_generate_profile

def create_network(config_file, params_file, data_path):
    """
    Builds the PyPSA network for the baseline scenario.

    The baseline scenario represents the reference case for the social housing building with:
    - 165 apartments in a 13-floor residential building from 1970
    - 9500 m² floor area and 962.94 m² usable roof area
    - Electricity and heat demand for the building
    - No storage components

    Args:
        config_file (str): Path to the main config file (e.g., config.yml).
        params_file (str): Path to the component parameters file (e.g., component_params.yml).
        data_path (str): Path to the input data directory.

    Returns:
        pypsa.Network: The configured PyPSA network object.
    """
    print("Building baseline network for social housing building...")

    # Load configuration
    config, params = load_config(config_file, params_file)

    # Create network and set time resolution
    network = pypsa.Network()
    start_date = config['simulation_settings']['start_date']
    n_hours = config['simulation_settings']['num_hours']
    timestamps = pd.date_range(start=start_date, periods=n_hours, freq='h')
    network.set_snapshots(timestamps)
    print(f"Set network snapshots: {len(network.snapshots)} hours, starting {start_date}")

    # ✅ PV generation from data_processing
    pv_params = params["baseline_pv"]
    area_m2 = pv_params["surface_m2"]
    eff_pv = pv_params["efficiency_pv"]
    eff_inv = pv_params["inverter_efficiency"]
    export_efficiency = pv_params["efficiency_export"]

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

    # Define energy carriers
    network.add("Carrier", "electricity")
    network.add("Carrier", "heat")
    network.add("Carrier", "heat_distribution")
    network.add("Carrier", "heat_pump")
    print("Added carriers: electricity, heat, heat_distribution, heat_pump")

    # Electricity buses (NON MODIFICATI)
    network.add("Bus", "Grid Elec Sell", carrier="electricity")
    network.add("Bus", "Grid Elec Buy", carrier="electricity")
    network.add("Bus", "Building Elec", carrier="electricity")
    network.add("Bus", "Battery Storage", carrier="electricity")
    network.add("Bus", "PV Bus", carrier="electricity")

    # Heat buses (NON MODIFICATI)
    network.add("Bus", "District Heating", carrier="heat")
    network.add("Bus", "Building Heat", carrier="heat")
    network.add("Bus", "Heat Source Bus", carrier="heat")
    network.add("Bus", "Thermal Storage", carrier="heat")

    # --- PV: generatore a costo marginale ~0 sul bus "PV Bus" ---
    network.add("Generator", "Rooftop PV",
        bus="PV Bus",
        carrier="electricity",
        p_nom=pv_p_nom_mw,          # MW
        p_max_pu=pv_profile_pu,     # [0..1], da irradianza/1000
        capital_cost=0,
        marginal_cost=0.0,
        efficiency=1.0
    )
    print(f"DEBUG: Rooftop PV - p_nom={pv_p_nom_mw:.6f} MW, marginal_cost=0.0 EUR/MWh")
    print("CHECK generator:")
    print(network.generators.loc["Rooftop PV"])
    print("CHECK snapshots PV max:", network.generators_t.p_max_pu["Rooftop PV"].describe() if "Rooftop PV" in network.generators_t.p_max_pu else "N/A")

    # Add grid connection
    grid_params    = params['grid']
    grid_co2_kg = grid_params.get('CO2_emissions_kg_per_mwh', 0)
    grid_capacity_mw = grid_params['capacity_mw']

    # — Import cost: fisso o variabile da CSV —
    if str(grid_params['import_cost_eur_per_kwh']).lower() == 'variable':
        from .utils import load_electricity_price_profile
        grid_price_series = load_electricity_price_profile(data_path, network.snapshots)
        print(f"✔️ Prezzi elettrici variabili caricati da data_dir/timeseries/grid_prices.csv.")
    else:
        fixed = float(grid_params['import_cost_eur_per_kwh'])
        grid_price_series = pd.Series(fixed * 1000, index=network.snapshots)  # EUR/MWh
        print(f"✔️ Prezzo elettrico fisso: {fixed} EUR/kWh → {fixed*1000} EUR/MWh.")

    # export_price = import_price
    export_price_series = grid_price_series.copy()


    # Salvo la serie sul network per il summary
    network.grid_price_series = grid_price_series

    transformer_efficiency = grid_params['transformer_efficiency']

    # ✅ Aggiungi generator chiamato 'Grid' (costo = prezzo di import)
    network.add("Generator", "Grid",
        bus="Grid Elec Sell",
        carrier="electricity",
        p_nom_extendable=True,
        marginal_cost=grid_price_series,
        efficiency=1.0
    )
    print("✔️ Generatore 'Grid' aggiunto alla rete.")

    # Link di import (nessun costo sul link; costo già sul generatore "Grid")
    network.add("Link", "Grid Import",
        bus0="Grid Elec Sell",
        bus1="Building Elec",
        p_nom=grid_capacity_mw,
        efficiency=transformer_efficiency,
        marginal_cost=0.0,
        capital_cost=0.0
    )
    print("DEBUG: Link 'Grid Import' p_nom =", network.links.at["Grid Import", "p_nom"])
    network.links.at["Grid Import", "co2_emission_factor_kg_per_mwh"] = grid_co2_kg

    # Add building parameters
    building_params = params.get('social_building', {})
    num_apartments = building_params.get('num_apartments', 165)
    num_floors = building_params.get('num_floors', 13)
    floor_area_m2 = building_params.get('floor_area_m2', 9500)
    roof_area_m2 = building_params.get('roof_area_m2', 962.94)
    construction_year = building_params.get('construction_year', 1970)

    # Add building loads (MW)
    elec_load_profile = load_or_generate_profile(
        building_params.get('electricity_load_profile', 'electricity_demand.csv'),
        building_params.get('electricity_peak_mw', 0.23),  # placeholder
        data_path,
        timestamps
    ) / 1000

    elec_peak_mw = elec_load_profile.max()
    print(f"DEBUG: Max domanda elettrica (usata come elec_peak_mw): {elec_peak_mw:.6f} MW")
    print("DEBUG: Profili elettrici caricati (primi 10 valori):", elec_load_profile[:10])
    print("DEBUG: Max domanda elettrica:", elec_load_profile.max())
    print("DEBUG: Min domanda elettrica:", elec_load_profile.min())
    print("DEBUG: Len domanda elettrica:", len(elec_load_profile))

    heat_load_profile = load_or_generate_profile(
        building_params.get('heat_load_profile', 'heat_demand.csv'),
        building_params.get('heat_peak_mw', 0.32),  # placeholder
        data_path,
        timestamps
    ) / 1000

    heat_peak_mw = heat_load_profile.max()
    print(f"DEBUG: Max domanda termica (usata come heat_peak_mw): {heat_peak_mw:.6f} MW")
    print("DEBUG: Profili termici caricati (primi 10 valori):", heat_load_profile[:10])
    print("DEBUG: Max domanda termica:", heat_load_profile.max())
    print("DEBUG: Min domanda termica:", heat_load_profile.min())
    print("DEBUG: Len domanda termica:", len(heat_load_profile))

    print(f"DEBUG: Elec load profile - max: {elec_load_profile.max()} MW, min: {elec_load_profile.min()} MW, len: {len(elec_load_profile)}")
    print(f"DEBUG: Heat load profile - max: {heat_load_profile.max()} MW, min: {heat_load_profile.min()} MW, len: {len(heat_load_profile)}")
    print(f"DEBUG: FINAL Elec load profile (MW) - max: {elec_load_profile.max()} MW, min: {elec_load_profile.min()} MW")
    print(f"DEBUG: FINAL Heat load profile (MW) - max: {heat_load_profile.max()} MW, min: {heat_load_profile.min()} MW")

    network.add("Load", "Building Elec Load", bus="Building Elec", p_set=elec_load_profile, carrier="electricity")
    network.add("Load", "Building Heat Load", bus="Building Heat", p_set=heat_load_profile, carrier="heat")

    print(f"Added loads for social housing building with {num_apartments} apartments, {num_floors} floors, {floor_area_m2} m² floor area")
    print(f"Electricity peak: {elec_peak_mw} MW, Heat peak: {heat_peak_mw} MW")
    print("DEBUG: Network loads (tabella):")
    print(network.loads)
    print("DEBUG: Network loads_t.p_set (prime righe):")
    print(network.loads_t.p_set.head())

    # Add heat source
    heat_params = params.get('baseline_heat_source', {})
    heat_capacity_mw = heat_params.get('capacity_mw_th', 10)
    heat_source_type = heat_params.get('type', 'district_heating_import')
    heat_cost = heat_params.get('cost_eur_per_mwh_th', 45)
    heat_efficiency = heat_params.get('efficiency_heating', 0.95)

    from .utils import load_thermal_price_profile
    price_profile = heat_params.get('price_profile')
    if price_profile:
        heat_cost = load_thermal_price_profile(os.path.join(data_path, 'timeseries', price_profile), network.snapshots)
        print("✔️ Prezzo termico variabile caricato")
    else:
        heat_cost = float(heat_params['cost_eur_per_mwh_th'])
        print(f"✔️ Prezzo termico fisso: {heat_cost} EUR/MWh")

    # Link da DH a edificio
    network.add("Link", "Heat import",
            bus0="District Heating",
            bus1="Building Heat",
            p_nom=1,
            p_nom_extendable=False,
            marginal_cost=0,
            efficiency=heat_efficiency
    )

    # ✅ Fattore CO2 per il teleriscaldamento (kg/MWh_th) sul link di import
    dh_co2 = float(params.get('baseline_heat_source', {}).get('co2_emissions_dh_kg_per_mwh', 0.0))
    network.links.at["Heat import", "co2_emission_factor_kg_per_mwh"] = dh_co2
    # Generator "Heating": prezzo a monte = prezzo consegna × η_DH  🔹 CHANGED
    if isinstance(heat_cost, pd.Series):
        # Generatore con costo orario: set base a 0 e popola generators_t  (minima estensione)
        network.add("Generator", "Heating",
            bus="District Heating",
            carrier="heat",
            p_nom=heat_capacity_mw,
            marginal_cost=0.0
        )
        if not hasattr(network.generators_t, "marginal_cost") or network.generators_t.marginal_cost is None or network.generators_t.marginal_cost.empty:
            network.generators_t.marginal_cost = pd.DataFrame(0.0, index=network.snapshots, columns=network.generators.index)
        network.generators_t.marginal_cost.loc[:, "Heating"] = (heat_cost * heat_efficiency).values  # CHANGED
    else:
        heat_cost_at_source = float(heat_cost) * heat_efficiency  # CHANGED
        network.add("Generator", "Heating",
            bus="District Heating",
            carrier="heat",
            p_nom=heat_capacity_mw,
            marginal_cost=heat_cost_at_source  # CHANGED
        )

    print(f"Added Heat Source generator: Capacity={heat_capacity_mw} MWth, Cost=variabile")

    # --- FLUSSI E PRIORITÀ ELETTRICI ---

    # PV → Building: priorità (nessun costo), capacità ≤ p_nom PV
    network.add("Link", "PV Autoconsumo",
        bus0="PV Bus",
        bus1="Building Elec",
        p_nom=pv_p_nom_mw,
        efficiency=1.0,
        capital_cost=0.0,
        marginal_cost=-1e-6
    )
    print(f"DEBUG: PV Autoconsumo p_nom={pv_p_nom_mw:.6f} MW, marginal_cost=0.0")

    # Building → Grid (export del surplus)
    # Prima rimuoviamo eventuali duplicati (non dovrebbero esserci in un network nuovo, ma sicuro è meglio)
    if "PV Export" in network.links.index:
        network.remove("Link", "PV Export")

    network.add("Link", "PV Export",
        bus0="PV Bus",
        bus1="Grid Elec Buy",
        p_nom=pv_p_nom_mw,             # limite ragionevole pari alla PV nominale (aumentabile)
        efficiency=export_efficiency,
        capital_cost=0.0,
        marginal_cost=0.0
    )
    network.links.at["PV Export", "co2_emission_factor_kg_per_mwh"] = grid_co2_kg
    print("DEBUG: Creato link 'PV Export' (Building Elec → Grid Elec Buy)")

    # Ricavo da export: generatore con costo negativo sul bus "Grid Elec Buy"
    # 1) Togli il generatore di ricavo se presente
    if "Export Revenue" in network.generators.index:
        network.remove("Generator", "Export Revenue")

    # 2) Ricavo sul link: prezzo export come costo marginale negativo
    # Assicurati che esista la matrice oraria dei costi marginali dei Link
    if not hasattr(network.links_t, "marginal_cost") or network.links_t.marginal_cost is None or network.links_t.marginal_cost.empty:
        network.links_t.marginal_cost = pd.DataFrame(
            0.0, index=network.snapshots, columns=network.links.index
        )

    # Assegna la serie oraria (negativa = ricavo) alla colonna del link di export
    network.links_t.marginal_cost["PV Export"] = pd.Series(
        -export_price_series, index=network.snapshots
    )

    # (opzionale) verifica veloce
    print("Link 'PV Export' marginal_cost head:")
    print(network.links_t.marginal_cost["PV Export"].head())

    # 3) Sink variabile sul bus di export: StorageUnit che può solo CARICARE
    #    - p_nom: capacità di assorbimento; metti un valore alto o estendibile
    #    - efficiency_dispatch=0.0 ⇒ non potrà mai restituire energia
    #    - cyclic_state_of_charge=False ⇒ non deve svuotarsi a fine orizzonte
    network.add(
        "StorageUnit", "Export Sink",
        bus="Grid Elec Buy",
        p_nom=1e3,
        p_nom_extendable=True,
        max_hours=1e6,
        # ⬇️ consenti SOLO carica (p negativo) e vieta scarica
        p_min_pu=-1.0,          # può assorbire fino a p_nom
        p_max_pu=0.0,           # non può erogare
        efficiency_store=1.0,
        efficiency_dispatch=1.0,  # irrilevante perché p_max_pu=0
        cyclic_state_of_charge=False,
        capital_cost=0.0,
        marginal_cost=0.0
    )

    # Assicurati che esista la matrice oraria dei costi dei Link
    if (not hasattr(network.links_t, "marginal_cost")
        or network.links_t.marginal_cost is None
        or network.links_t.marginal_cost.empty):
        network.links_t.marginal_cost = pd.DataFrame(
            0.0, index=network.snapshots, columns=network.links.index
        )

    # Serie oraria negativa (= ricavo) allineata agli snapshot
    export_price_series = pd.Series(export_price_series, index=network.snapshots)
    network.links_t.marginal_cost.loc[:, "PV Export"] = -export_price_series.values

    print("Link 'PV Export' marginal_cost head:")
    print(network.links_t.marginal_cost["PV Export"].head())

    print("Baseline network build complete.")

    print("DEBUG: p_nom PV =", network.generators.loc["Rooftop PV", "p_nom"])
    print("DEBUG: p_max_pu PV (first 10):", network.generators_t.p_max_pu["Rooftop PV"].head(10))
    print("DEBUG: p_max_pu PV stats:", network.generators_t.p_max_pu["Rooftop PV"].describe())
    print("DEBUG: APPROACH A - p_nom based on STC")
    print("DEBUG: p_max_pu = irradiance/1000 (physically meaningful)")

    # Controlli di consistenza finali
    print("✅ Verifica finale dati:")
    print("  → Numero snapshot:", len(network.snapshots))
    print("  → Prime date:", network.snapshots[:3])
    print("  → Ultime date:", network.snapshots[-3:])

    assert len(network.snapshots) == 8760, "❌ Gli snapshot non sono 8760!"
    assert all(isinstance(s, pd.Timestamp) for s in network.snapshots), "❌ Snapshot non sono timestamp!"

    for load_name in network.loads.index:
        load_series = network.loads_t.p_set[load_name]
        assert len(load_series) == 8760, f"❌ Carico '{load_name}' ha {len(load_series)} elementi, non 8760!"
        assert load_series.notna().all(), f"❌ Carico '{load_name}' contiene valori NaN!"

    for gen_name in network.generators.index:
        if gen_name in network.generators_t.p:
            gen_series = network.generators_t.p[gen_name]
            assert len(gen_series) == 8760, f"❌ Generatore '{gen_name}' ha {len(gen_series)} elementi, non 8760!"

    return network
