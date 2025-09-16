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
from scripts import data_processing

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



    # ✅ Caricamento dati meteo per il calcolo PV
    weather_path = os.path.join(data_path, "timeseries", "weather_data.csv")

    # Chiama data_processing.load_timeseries che ora leggerà la colonna 'solar_radiation'
    # e la restituirà come una Series.
    raw_solar_radiation_series = data_processing.load_timeseries(weather_path, data_path, index=timestamps)

    # Crea un DataFrame 'weather_df' con la colonna 'solar_radiation'
    # Questo è il formato atteso dalla funzione calculate_pv_generation
    weather_df = pd.DataFrame({'solar_radiation': raw_solar_radiation_series}, index=timestamps)

    # Usa data_processing.load_timeseries per caricare il DataFrame meteo
    weather_df = data_processing.load_timeseries(weather_path, data_path, index=timestamps).to_frame(name='solar_radiation')
    # Assicurati che la colonna sia nominata 'solar_radiation' per coerenza con la funzione calculate_pv_generation
    weather_df.columns = ['solar_radiation'] # Rinomina la colonna se necessario
    print("Colonne di weather_df:", weather_df.columns.tolist())

    # ✅ Estrai parametri PV dal file component_params (usa i tuoi parametri esistenti)
    pv_params = params["pv_battery_2"]
    eff_pv = pv_params["efficiency_pv"]
    eff_inv = pv_params["inverter_efficiency"]
    area_m2 = pv_params["surface_m2"] 

    # Estrai parametri batteria da component_params.yml
    battery_params = params["pv_battery_2"]["battery"]
    battery_p_nom = battery_params["p_nom_mw"]
    battery_e_nom = battery_params["e_nom_mwh"]
    round_trip_efficiency = battery_params["efficiency_round_trip"]
    export_efficiency = battery_params["efficiency_export"]
    desired_duration = battery_params["autonomy_hours"]
    # Parametri dsm
    dsm_params = params.get('dsm', {})
    flex_share = float(dsm_params.get('flexible_load_share', 0.3))
    max_shift = float(dsm_params.get('max_shift_hours', 3))
    eff = float(dsm_params.get('efficiency', 1))




    # Calcola la produzione PV in kW usando la funzione da data_processing.py
    pv_generation_kw = data_processing.calculate_pv_generation(
        weather_df=weather_df,
        pv_surface_m2=area_m2,
        pv_efficiency_pv=eff_pv,
        pv_inverter_efficiency=eff_inv
    )

    # Converti la generazione PV da kW a MW per PyPSA
    pv_generation_mw = pv_generation_kw / 1000 # <--- CONVERSIONE A MW

    # Stampa il potenziale PV annuo (MWh)
    total_pv_potential = pv_generation_mw.sum()
    print(f"TOTAL PV POTENTIAL (raw): {total_pv_potential:.1f} MWh/anno")


 #  Prepara il profilo per PyPSA
    # Assicurati che il valore massimo non sia zero per evitare divisione per zero in p_max_pu
    pv_p_nom_mw = pv_generation_mw.max()
    print("DEBUG: Max irradiance (W/m2):", weather_df['solar_radiation'].max()) # DEVE STAMPARE VALORI FINO A ~800

    # p_max_pu è il profilo normalizzato (valori tra 0 e 1)
    pv_profile_pu = pv_generation_mw / pv_p_nom_mw 





    # Define energy carriers
    network.add("Carrier", "electricity")
    network.add("Carrier", "heat")
    network.add("Carrier", "heat_distribution")
    network.add("Carrier", "heat_pump")
    print("Added carriers: electricity, heat, heat_distribution, heat_pump")

    # Electricity buses
    network.add("Bus", "Grid Elec", carrier="electricity")  # unico per grid import
    network.add("Bus", "Building Elec", carrier="electricity")
    # Bus della batteria: collegato solo allo Store e ai due link (carica/scarica)
    network.add("Bus", "Battery Storage", carrier="electricity")
    network.add("Bus", "PV Bus", carrier="electricity")
    network.add("Bus", "DSM Elec Store", carrier="electricity")
    network.add("Bus", "DSM Heat Store", carrier="heat")


    # Heat buses
    network.add("Bus", "District Heating", carrier="heat")  # può fare da interconnessione
    network.add("Bus", "Building Heat", carrier="heat")
    network.add("Bus", "Heat Source Bus", carrier="heat")
    network.add("Bus", "Thermal Storage", carrier="heat")

    ε1, ε2, ε3, ε4, ε5, ε6, ε7 = 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3, 1e-1, 2e-1




    # Aggiungi il generatore 'Rooftop PV' alla rete
    network.add("Generator", "Rooftop PV",
        bus="PV Bus",
        carrier="electricity",
        p_nom = pv_p_nom_mw, # <--- ORA QUESTO p_nom È IN MW
        p_max_pu=pv_profile_pu,
        capital_cost=0, # Costi di capitale zero per PV esistente nel baseline
        marginal_cost=0, # Costi marginali zero per PV
        efficiency=1 # Efficienza del generatore dopo la conversione in AC
    )

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
        # Carichiamo il CSV usando la cartella base: utils farà da sé os.path.join(data_dir, 'timeseries', 'grid_prices.csv')
        grid_price_series = load_electricity_price_profile(data_path, network.snapshots)
        print(f"✔️ Prezzi elettrici variabili caricati da data_dir/timeseries/grid_prices.csv.")

    else:
        # valore fisso in EUR/kWh → moltiplico per 1000 per avere EUR/MWh
        fixed = float(grid_params['import_cost_eur_per_kwh'])
        grid_price_series = pd.Series(fixed * 1000, index=network.snapshots)
        print(f"✔️ Prezzo elettrico fisso: {fixed} EUR/kWh → {fixed*1000} EUR/MWh.")
    print("DEBUG: prime 5 ore di prezzo rete:", grid_price_series.head())
    print("DEBUG: ultime 5 ore di prezzo rete:", grid_price_series.tail())

    # Salvo la serie e gli overhead sul network per il summary
    network.grid_price_series = grid_price_series
    network.epsilon_grid_import     = ε6
    network.epsilon_grid_storage    = ε7

    # Export price uguale all'import (serie oraria)
    grid_export_price = grid_price_series  # Usa la stessa serie del prezzo di import/export
    transformer_efficiency = grid_params['transformer_efficiency']




    heat_params = params.get('baseline_heat_source', {})
    heat_capacity_mw = heat_params.get('capacity_mw_th', 10)
    heat_source_type = heat_params.get('type', 'district_heating_import')
    heat_cost = heat_params.get('cost_eur_per_mwh_th', 45)
    heat_efficiency = heat_params.get('efficiency_heating', 0.95)
    heat_cop = heat_params.get('cop', 1.0)
    standing_loss = heat_params.get('standing_loss_per_hour', 0.005)


    # Connect heat source to building
    network.add("Link", "Thermal Discharge",
                bus0="Heat Source Bus",
                bus1="Building Heat",
                p_nom=1,
                efficiency=1,
                marginal_cost=ε1,
                carrier="heat_distribution")
    

    ######INTRODUCO IO

    network.add("Store", "Thermal Store",
        bus="Thermal Storage",
        e_nom=1,               # capacità termica in MWh (adatta al tuo caso)
        p_nom=1,
        e_initial=0,
        e_cyclic=False,
        marginal_cost=0.0,      # nessun costo operativo
        efficiency=1,
        standing_loss=standing_loss
    )

    network.add("Link", "Thermal Charge",
        bus0="Heat Source Bus",
        bus1="Thermal Storage",
        efficiency=1.0,
        p_nom=1,
        p_nom_extendable=False,
        marginal_cost=ε2
    )


    # — Link per scaricare il serbatoio termico verso l’edificio —
    network.add("Link", "Thermal Dispatch",
        bus0="Thermal Storage",   # da dove esce l’energia termica
        bus1="Building Heat",      # a dove la mandi
        p_nom=1.0,                 # scegli la capacità adeguata
        efficiency=1.0,            # o l’efficienza che preferisci
        marginal_cost=ε1          # costo nullo, come gli altri
    )





    # ✅ Aggiungi generator chiamato 'Grid' per supportare il calcolo in main.py
    network.add("Generator", "Grid",
        bus="Grid Elec",
        carrier="electricity",
        p_nom_extendable=True,
        marginal_cost=grid_price_series,
        efficiency=1
    )
    print("✔️ Generatore 'Grid' aggiunto alla rete.")


    # Link da rete elettrica a edificio
    network.add("Link", "Grid Import",
        bus0="Grid Elec",
        bus1="Building Elec",
        p_nom=grid_capacity_mw,          # capacità massima di connessione
        efficiency=transformer_efficiency,
        marginal_cost=grid_price_series+ε6,  # costo orario di import
        capital_cost=0
    )
    print("DEBUG: Link 'Grid Import' p_nom =", network.links.at["Grid Import", "p_nom"])
    network.links.at["Grid Import", "co2_emission_factor_kg_per_mwh"] = grid_co2_kg



    network.add("Link", "Grid Import_Storage",
        bus0="Grid Elec",
        bus1="Battery Storage",
        p_nom=battery_p_nom,          # capacità massima di connessione
        efficiency=1,
        marginal_cost=grid_price_series+ε7,  # costo orario di import
        capital_cost=0
    )



    # Add building parameters
    building_params = params.get('social_building', {})
    num_apartments = building_params.get('num_apartments', 165)
    num_floors = building_params.get('num_floors', 13)
    floor_area_m2 = building_params.get('floor_area_m2', 9500)
    roof_area_m2 = building_params.get('roof_area_m2', 962.94)
    construction_year = building_params.get('construction_year', 1970)

    # Add building loads
    # Electricity load: creiamo prima il profilo (MW)
    elec_load_profile = load_or_generate_profile(
        building_params.get('electricity_load_profile', 'electricity_demand.csv'),
        building_params.get('electricity_peak_mw', 0.23),  # serve solo come placeholder
        data_path,
        timestamps
    ) / 1000

    # Ora ricalcoliamo il picco reale dal profilo
    elec_peak_mw = elec_load_profile.max()
    print(f"DEBUG: Max domanda elettrica (usata come elec_peak_mw): {elec_peak_mw:.6f} MW")
    print("DEBUG: Profili elettrici caricati (primi 10 valori):", elec_load_profile[:10])
    print("DEBUG: Max domanda elettrica:", elec_load_profile.max())
    print("DEBUG: Min domanda elettrica:", elec_load_profile.min())
    print("DEBUG: Len domanda elettrica:", len(elec_load_profile))

    heat_load_profile = load_or_generate_profile(
        building_params.get('heat_load_profile', 'heat_demand.csv'),
        building_params.get('heat_peak_mw', 0.32),  # placeholder per load_or_generate_profile
        data_path,
        timestamps
    ) / 1000

    # Ora ricalcoliamo il picco reale dal profilo termico
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

    # Split DSM
    elec_flex = elec_load_profile * flex_share
    elec_infl = elec_load_profile - elec_flex
    heat_flex = heat_load_profile * flex_share
    heat_infl = heat_load_profile - heat_flex


    # Carichi inflessibili
    network.add("Load", "Building Elec Load", bus="Building Elec", p_set=elec_infl, carrier="electricity", controllable=False)
    network.add("Load", "Building Heat Load", bus="Building Heat", p_set=heat_infl, carrier="heat", controllable=False)


    print(f"Added loads for social housing building with {num_apartments} apartments, {num_floors} floors, {floor_area_m2} m² floor area")
    print(f"Electricity peak: {elec_peak_mw} MW, Heat peak: {heat_peak_mw} MW")
    # --- DEBUG DEL NETWORK SUI LOADS ---
    print("DEBUG: Network loads (tabella):")
    print(network.loads)
    print("DEBUG: Network loads_t.p_set (prime righe):")
    print(network.loads_t.p_set.head())


    ################################ DSM virtual storage ################################################
    p_elec_peak = elec_flex.max()
    p_heat_peak = heat_flex.max()
    e_elec_nom = p_elec_peak * max_shift
    e_heat_nom = p_heat_peak * max_shift

    # ===== DSM elettrico =====
    # Store “virtuale” su un bus dedicato
    network.add("Store", "DSM Elec Store",
        bus="DSM Elec Store",
        e_nom=e_elec_nom,        # capacità [MWh]
        p_nom=p_elec_peak,       # potenza massima di shift [MW]
        efficiency_store=eff,    # efficienza in entrata
        efficiency_dispatch=eff, # efficienza in uscita
        carrier="electricity",
        cyclic=False
    )
    # Link di “carica”: preleva il carico flessibile dall’edificio
    network.add("Link", "DSM Elec Charge",
        bus0="Building Elec",
        bus1="DSM Elec Store",
        p_nom=p_elec_peak,
        efficiency=eff,
        marginal_cost=0
    )
    # Link di “scarica”: rilascia il carico dal serbatoio all’edificio
    network.add("Link", "DSM Elec Dispatch",
        bus0="DSM Elec Store",
        bus1="Building Elec",
        p_nom=p_elec_peak,
        efficiency=eff,
        marginal_cost=0
    )


    # ===== DSM termico =====
    network.add("Store", "DSM Heat Store",
        bus="DSM Heat Store",
        e_nom=e_heat_nom,
        p_nom=p_heat_peak,
        efficiency_store=eff,
        efficiency_dispatch=eff,
        carrier="heat",
        cyclic=False
    )
    network.add("Link", "DSM Heat Charge",
        bus0="Building Heat",
        bus1="DSM Heat Store",
        p_nom=p_heat_peak,
        efficiency=eff,
        marginal_cost=0
    )
    network.add("Link", "DSM Heat Dispatch",
        bus0="DSM Heat Store",
        bus1="Building Heat",
        p_nom=p_heat_peak,
        efficiency=eff,
        marginal_cost=0
    )


    # — Aggiungo i profili flessibili come veri carichi (Load) vincolati —
    network.add("Load", "DSM Elec Flex Load",
                bus="Building Elec",
                p_set=elec_flex.values,
                carrier="electricity")
    network.add("Load", "DSM Heat Flex Load",
                bus="Building Heat",
                p_set=heat_flex.values,
                carrier="heat")



    print(f"✔️ DSM virtual storage aggiunto: elec peak={p_elec_peak:.3f} MW, heat peak={p_heat_peak:.3f} MW, e_nom_elec={e_elec_nom:.3f} MWh, e_nom_heat={e_heat_nom:.3f} MWh")





    from .utils import load_thermal_price_profile
    price_profile = heat_params.get('price_profile')
    if price_profile:
        heat_cost = load_thermal_price_profile(os.path.join(data_path, 'timeseries', price_profile), network.snapshots)
        print("✔️ Prezzo termico variabile caricato")
    else:
        heat_cost = float(heat_params['cost_eur_per_mwh_th'])
        print(f"✔️ Prezzo termico fisso: {heat_cost} EUR/MWh")

    hp_net_cost_pv = 0 - (heat_cop * heat_cost)
    hp_net_cost_grid = grid_price_series - (heat_cop * heat_cost)


    network.add("Link", "Heat Pump",
        bus0="PV Bus",     # ingresso elettrico
        bus1="Heat Source Bus",   # uscita termica
        efficiency=heat_cop,         # COP = 3.5
        p_nom_extendable=True,
        marginal_cost=hp_net_cost_pv+ε1,
        carrier="heat_pump"
    )
    


    # Link da centrale di calore a edificio
    network.add("Link", "Heat import",
            bus0="District Heating", 
            bus1="Building Heat", 
            p_nom=1,
            p_nom_extendable=False,
            marginal_cost=heat_cost+ε6,
            efficiency=heat_efficiency
    )

    # ← Imposto il fattore CO2 (kg/MWh) per il district heating
    dh_factor = float(params['baseline_heat_source']['co2_emissions_dh_kg_per_mwh'])
    network.links.at["Heat import", "co2_emission_factor_kg_per_mwh"] = dh_factor




    network.add("Generator", "Heating",
            bus="District Heating",
            carrier="heat",
            p_nom=heat_capacity_mw,
            marginal_cost=heat_cost)

    print(f"Added Heat Source generator: Capacity={heat_capacity_mw} MWth, Cost=variabile")



    eff_store     = round_trip_efficiency ** 0.5                          # ≃ 0.949
    eff_dispatch  = round_trip_efficiency ** 0.5                          # ≃ 0.949
    # ——— Dimensionamento dinamico della batteria ———
    # Calcolo del net load orario [MWh] = domanda [MWh] – PV [MWh]
    net_load = elec_load_profile - pv_generation_mw




    # 1) Definisci uno Store (solo energia)
    network.add("Store", "Battery Store",
        bus="Battery Storage",
        e_nom=battery_e_nom,           # energia che la batteria deve coprire [MWh]
        p_nom=battery_p_nom,           # potenza necessaria per 4 h di durata [MW]        capital_cost=0,
        state_of_charge_initial=0.0,
        marginal_cost=0
    )

    # 2) Link per caricare lo Store (from grid/bus into Store)
    network.add("Link", "Battery Charge",
        bus0="PV Bus",      # dove prendi l'energia
        bus1="Battery Storage",      # dove la immagazzini
        p_nom=battery_p_nom,       
        efficiency=eff_store,
        capital_cost=0,
        marginal_cost=ε4
    )

    # 3) Link per scaricare lo Store (from Store back to bus)
    network.add("Link", "Battery Dispatch",
        bus0="Battery Storage",      # da dove esce l'energia
        bus1="Building Elec",      # a dove la mandi
        p_nom=battery_p_nom,
        efficiency=eff_dispatch,
        capital_cost=0,
        marginal_cost=ε3
    )

    # 4) Link per scaricare lo Store (from Store back to bus)
    network.add("Link", "Battery Discharge",
        bus0="PV Bus",      # da dove esce l'energia
        bus1="Building Elec",      # a dove la mandi
        p_nom=grid_capacity_mw,
        efficiency=1,
        capital_cost=0,
        marginal_cost=ε3
    )





    




    export_profit = grid_price_series * export_efficiency

    network.add("Link", "Grid Export",
        bus0="PV Bus",
        bus1="Grid Elec",
        p_nom=pv_p_nom_mw,
        efficiency=0.1,
        marginal_cost=-grid_price_series+ε5
    )
    network.links.at["Grid Export", "co2_emission_factor_kg_per_mwh"] = grid_co2_kg

    
 


    # Add building envelope information (for reference only, not used in the model)
    envelope_params = params.get('building_envelope', {})
    if envelope_params:
        wall_area_m2 = envelope_params.get('wall_area_m2', 2700)
        window_area_m2 = envelope_params.get('window_area_m2', 1000)
        wall_u_value = building_params.get('wall_u_value', 0.6)
        window_u_value = building_params.get('window_u_value', 1.7)
        roof_u_value = envelope_params.get('roof_u_value', 0.3)

        print(f"Building envelope: Wall area={wall_area_m2} m², Window area={window_area_m2} m²")
        print(f"U-values: Wall={wall_u_value} W/(m²K), Window={window_u_value} W/(m²K), Roof={roof_u_value} W/(m²K)")

    print("Baseline network build complete.")



    # load totale = inflessibile + flessibile
    electric_load_series = elec_infl + elec_flex
    heat_load_series    = heat_infl + heat_flex


    grid_link_series = network.links_t.p0["Grid Import"] if "Grid Import" in network.links_t.p0 else pd.Series(0.0, index=network.snapshots)
    grid_export_series = network.links_t.p0["Grid Export"] if "Grid Export" in network.links_t.p0 else pd.Series(0.0, index=network.snapshots)
    # Serie oraria di calore prodotto dalla Heat Pump (p1 = output termico già moltiplicato per COP)
    heat_pump_series = network.links_t.p1.get(
        'Heat Pump',
        pd.Series(0.0, index=network.snapshots)
    )
    # Serie oraria di calore in ingresso da District Heating (p0 = import dal bus di origine)
    dh_import_series = network.links_t.p0.get(
        'Heat import',
        pd.Series(0.0, index=network.snapshots)
    )
    print("DEBUG: p_nom PV =", network.generators.loc["Rooftop PV", "p_nom"])
    print("DEBUG: p_max_pu PV (first 10):", network.generators_t.p_max_pu["Rooftop PV"].head(10))
    print("DEBUG: Max irradiance (W/m2):", weather_df['solar_radiation'].max()) # Variabile corretta
    print("DEBUG: pv_generation_mw.describe():") # Nome della variabile aggiornato
    print(pv_generation_mw.describe()) # Variabile corretta
    





    results = {
        "pv_generation": pv_generation_mw,
        "grid_import": grid_link_series,
        "grid_export": grid_export_series,
        "electric_load": elec_infl + elec_flex,    # somma degli array
        "heat_load":     heat_infl + heat_flex,
        "elec_inflex":   elec_infl,
        "elec_flex":     elec_flex,
        "heat_inflex":   heat_infl,
        "heat_flex":     heat_flex,
        "heat_pump_output":   heat_pump_series,
        "dh_import":          dh_import_series,

    }

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
    df_hourly = pd.DataFrame(results)
    df_hourly.index.name = "snapshot"
    return network, results, df_hourly