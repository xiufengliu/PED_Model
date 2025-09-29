"""
PED Lyngby Model - High PV Scenario

This module implements the high PV scenario for the PED Lyngby Model.
The high PV scenario maximizes solar PV deployment on available surfaces within the district.
It uses the high_pv_generator parameters from component_params.yml to configure the PV system.
"""

import pypsa
import pandas as pd
import numpy as np
import os

from .utils import load_config, load_or_generate_profile, setup_basic_network

def create_network(config_file, params_file, data_path):
    """
    Builds the PyPSA network for the high PV scenario.

    The high PV scenario maximizes solar PV deployment on the social housing building with:
    - 165 apartments in a 13-floor residential building from 1970
    - 9500 m² floor area and 700 m² usable roof area
    - Electricity and heat demand for the building
    - Maximized rooftop PV system (165 kW capacity)
    - Standard grid connection (10 MW capacity)
    - District heating connection (0.5 MW thermal capacity)
    - No storage components

    Args:
        config_file (str): Path to the main config file (e.g., config.yml).
        params_file (str): Path to the component parameters file (e.g., component_params.yml).
        data_path (str): Path to the input data directory.

    Returns:
        pypsa.Network: The configured PyPSA network object.
    """
    print("Building high PV network for social housing building...")

    # Load configuration
    config, params = load_config(config_file, params_file)

    # Create network and set time resolution
    network = pypsa.Network()
    start_date = config['simulation_settings']['start_date']
    n_hours = config['simulation_settings']['num_hours']
    timestamps = pd.date_range(start=start_date, periods=n_hours, freq='h')
    network.set_snapshots(timestamps)
    print(f"Set network snapshots: {len(network.snapshots)} hours, starting {start_date}")

    # Define energy carriers
    network.add("Carrier", "electricity")
    network.add("Carrier", "heat")
    print("Added carriers: electricity, heat")

    # Add buses (connection points)
    network.add("Bus", "Grid Connection", carrier="electricity")
    network.add("Bus", "Building Elec", carrier="electricity")
    network.add("Bus", "Building Heat", carrier="heat")
    network.add("Bus", "Heat Source", carrier="heat")

    # Add grid connection
    # Parametri della connessione alla rete:
    # - capacity_mw: capacità della connessione in MW (default: 10.0)
    # - import_cost_eur_per_mwh: costo di importazione in EUR/MWh (default: 50.0 o 'variable' per prezzi variabili)
    # - export_price_eur_per_mwh: prezzo di esportazione in EUR/MWh (default: 20.0)
    # - transformer_efficiency: efficienza del trasformatore (default: 0.98)
    # I prezzi variabili vengono caricati dal file grid_prices.csv
    grid_params = params['grid']
    grid_capacity_mw = grid_params.get('capacity_mw', 10.0)
    grid_import_cost = grid_params.get('import_cost_eur_per_mwh', 50.0)
    grid_export_price = grid_params.get('export_price_eur_per_mwh', 20.0)
    transformer_efficiency = grid_params.get('transformer_efficiency', 0.98)

    # Check if we're using variable electricity prices
    if grid_import_cost == 'variable':
        from .utils import load_electricity_price_profile
        grid_import_cost = load_electricity_price_profile(data_path, timestamps)
        cost_display = "Variable (from grid_prices.csv)"
        print(f"Loaded variable electricity prices: min={grid_import_cost.min():.2f}, max={grid_import_cost.max():.2f}, mean={grid_import_cost.mean():.2f} EUR/MWh")
    else:
        cost_display = f"{grid_import_cost} EUR/MWh"

    # Create the generator with the marginal cost
    network.add("Generator", "Grid",
                bus="Grid Connection",
                carrier="electricity",
                p_nom=grid_capacity_mw,
                p_min_pu=-1,  # Allow export
                p_max_pu=1,   # Allow import
                marginal_cost=grid_import_cost)  # Imposta direttamente il costo marginale come nel baseline

    network.add("Link", "Building Connection",
                bus0="Grid Connection",
                bus1="Building Elec",
                p_nom=grid_capacity_mw,
                efficiency=transformer_efficiency)
    print(f"Added Grid connection: Capacity={grid_capacity_mw} MW, Import Cost={cost_display}, Export Price={grid_export_price} EUR/MWh")

    # Add building parameters
    # Parametri dell'edificio sociale:
    # - num_apartments: numero di appartamenti (default: 165)
    # - num_floors: numero di piani (default: 13)
    # - floor_area_m2: superficie totale in m² (default: 9500)
    # - roof_area_m2: superficie del tetto utilizzabile per PV in m² (default: 700)
    # - construction_year: anno di costruzione (default: 1970)
    # - electricity_peak_mw: picco di domanda elettrica in MW (default: 0.23)
    # - heat_peak_mw: picco di domanda termica in MW (default: 0.32)
    # - electricity_load_profile: profilo di carico elettrico (default: electricity_demand.csv)
    # - heat_load_profile: profilo di carico termico (default: social_building_heat_demand_denmark.csv)
    # - pv_generation_profile: profilo di generazione fotovoltaica (default: solar_pv_generation.csv)
    building_params = params.get('social_building', {})
    num_apartments = building_params.get('num_apartments', 165)
    num_floors = building_params.get('num_floors', 13)
    floor_area_m2 = building_params.get('floor_area_m2', 9500)
    roof_area_m2 = building_params.get('roof_area_m2', 700)
    construction_year = building_params.get('construction_year', 1970)

    # Add building loads
    elec_peak_mw = building_params.get('electricity_peak_mw', 0.23)
    elec_load_profile = load_or_generate_profile(
        building_params.get('electricity_load_profile', 'electricity_demand.csv'),
        elec_peak_mw,
        data_path,
        timestamps
    )
    heat_peak_mw = building_params.get('heat_peak_mw', 0.32)
    heat_load_profile = load_or_generate_profile(
        building_params.get('heat_load_profile', 'social_building_heat_demand_denmark.csv'),
        heat_peak_mw,
        data_path,
        timestamps
    )
    network.add("Load", "Building Elec Load", bus="Building Elec", p_set=elec_load_profile)
    network.add("Load", "Building Heat Load", bus="Building Heat", p_set=heat_load_profile)
    print(f"Added loads for social housing building with {num_apartments} apartments, {num_floors} floors, {floor_area_m2} m² floor area")
    print(f"Electricity peak: {elec_peak_mw} MW, Heat peak: {heat_peak_mw} MW")

    # Connect heat source to building
    network.add("Link", "Building Heat Link",
                bus0="Heat Source",
                bus1="Building Heat",
                p_nom=1.0)  # 1 MW capacity should be sufficient

    # Add maximized PV system
    # Parametri del generatore fotovoltaico ad alta capacità:
    # - capacity_kwp: capacità installata in kWp (default: 165.0)
    # - inverter_efficiency: efficienza dell'inverter (default: 0.97, uguale al baseline)
    # - cost_eur_per_kwp: costo di installazione in EUR/kWp (default: 900)
    # - lifetime_years: vita utile in anni (default: 25)
    # - degradation_pct_per_year: degradazione annuale in percentuale (default: 0.5%)
    # Il profilo di generazione viene caricato dal file solar_pv_generation.csv e scalato in base alla capacità
    high_pv_params = params.get('high_pv_generator', {})
    pv_capacity_kw = high_pv_params.get('capacity_kwp', 165.0)
    pv_capacity_mw = pv_capacity_kw / 1000.0
    pv_inverter_efficiency = high_pv_params.get('inverter_efficiency', 0.97)

    # Load PV generation profile
    # Il profilo di generazione fotovoltaica è influenzato dai seguenti parametri meteorologici:
    # 1. Irradiazione solare (W/m²): è il fattore più importante e determina direttamente la produzione
    #    - Radiazione diretta: componente che arriva direttamente dal sole
    #    - Radiazione diffusa: componente che arriva dal cielo dopo essere stata diffusa dall'atmosfera
    #    - Radiazione riflessa: componente che arriva dopo essere stata riflessa dal terreno
    # 2. Temperatura ambiente (°C): influisce sull'efficienza delle celle fotovoltaiche
    #    - All'aumentare della temperatura, l'efficienza diminuisce (circa -0.4% per ogni °C sopra 25°C)
    # 3. Velocità del vento (m/s): influisce sulla temperatura dei pannelli
    #    - Un vento più forte raffredda i pannelli e ne migliora l'efficienza
    # 4. Copertura nuvolosa (%): influisce sulla radiazione diffusa
    # 5. Umidità relativa (%): influisce sulla trasmissione della radiazione
    # 6. Precipitazioni (mm): possono ridurre temporaneamente la produzione
    # 7. Neve e ghiaccio: possono coprire i pannelli e ridurre drasticamente la produzione
    # 8. Polvere e inquinamento: riducono la quantità di luce che raggiunge i pannelli
    #
    # Il file solar_pv_generation.csv contiene un profilo normalizzato che tiene conto di questi fattori
    # per la località di Lyngby, Danimarca, basato su dati meteorologici storici.
    pv_profile = load_or_generate_profile(
        building_params.get('pv_generation_profile', 'solar_pv_generation.csv'),
        1.0,  # Normalized profile
        data_path,
        timestamps
    )

    # Apply inverter efficiency to the PV profile
    pv_profile = pv_profile * pv_inverter_efficiency

    network.add("Generator", "Maximized PV",
                bus="Building Elec",
                carrier="electricity",
                p_nom=pv_capacity_mw,
                p_max_pu=pv_profile,
                marginal_cost=0)
    print(f"Added Maximized PV: {pv_capacity_kw} kWp, {roof_area_m2} m² roof area, Inverter efficiency: {pv_inverter_efficiency}")

    # Add heat source
    # Parametri della fonte di calore:
    # - type: tipo di fonte di calore ('district_heating_import' o 'gas_boiler')
    # - capacity_mw_th: capacità termica in MW (default: 0.5)
    # - cost_eur_per_mwh_th: costo in EUR/MWh (default: 40.0 o 'variable' per prezzi variabili)
    # - efficiency_if_boiler: efficienza se è una caldaia a gas (default: 0.9)
    # I prezzi variabili vengono caricati dal file thermal_energy_prices_denmark.csv
    heat_params = params.get('baseline_heat_source', {})
    heat_capacity_mw = heat_params.get('capacity_mw_th', 0.5)
    heat_cost = heat_params.get('cost_eur_per_mwh_th', 40.0)
    heat_source_type = heat_params.get('type', 'district_heating_import')

    # Check if we're using variable heat prices
    if heat_cost == 'variable':
        from .utils import load_thermal_price_profile
        heat_cost = load_thermal_price_profile(data_path, timestamps)
        heat_cost_display = "Variable (from thermal_energy_prices_denmark.csv)"
        print(f"Loaded variable heat prices: min={heat_cost.min():.2f}, max={heat_cost.max():.2f}, mean={heat_cost.mean():.2f} EUR/MWh")
    else:
        heat_cost_display = f"{heat_cost} EUR/MWh"

    if heat_source_type == 'gas_boiler':
        efficiency = heat_params.get('efficiency_if_boiler', 0.9)
        network.add("Generator", "Heat Source",
                    bus="Heat Source",
                    carrier="heat",
                    p_nom=heat_capacity_mw,
                    marginal_cost=heat_cost / efficiency if not isinstance(heat_cost, pd.Series) else heat_cost / efficiency)  # Adjust cost for efficiency
        print(f"Added Gas Boiler: Capacity={heat_capacity_mw} MWth, Cost={heat_cost_display}, Efficiency={efficiency}")
    else:
        network.add("Generator", "Heat Source",
                    bus="Heat Source",
                    carrier="heat",
                    p_nom=heat_capacity_mw,
                    marginal_cost=heat_cost)  # Imposta direttamente il costo marginale come nel baseline

        print(f"Added District Heating Connection: Capacity={heat_capacity_mw} MWth, Cost={heat_cost_display}")

    # Add placeholder storage components (zero capacity)
    network.add("StorageUnit", "Placeholder Battery",
                bus="Building Elec",
                p_nom=0,
                max_hours=0)
    network.add("Store", "Placeholder Thermal Storage",
                bus="Building Heat",
                e_nom=0)
    print("Added placeholder storage components (zero capacity for high_pv)")

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

    print("High PV network build complete.")
    return network
