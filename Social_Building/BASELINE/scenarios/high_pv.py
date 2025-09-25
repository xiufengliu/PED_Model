# high_pv.py

"""
PED Lyngby Model - High PV Scenario (allineato alla logica di baseline)

- Prezzo elettrico: generatore "Grid" con marginal_cost = prezzo orario (non scalato)
  e link "Grid Import" con sola efficienza (marginal_cost=0). Il summary calcola i costi a valle.
- Teleriscaldamento: costo impostato sul generatore "Heating" al "lato sorgente" = prezzo_a_valle * eta,
  mentre il link "Heat import" ha marginal_cost=0 ed efficienza = eta. In questo modo:
    cost_summary = (Q_gen * prezzo_sorgente) = (Q_bld/eta * prezzo_a_valle*eta) = Q_bld * prezzo_a_valle.
- Export FV: ricavo modellato come costo marginale NEGATIVO sul link "PV Export"
  (prezzo orario; epsilon negativo se il prezzo è 0) e sink di assorbimento.
"""

import pypsa
import pandas as pd
import numpy as np
import os

from .utils import load_config, load_or_generate_profile

def create_network(config_file, params_file, data_path):
    print("Building HIGH-PV network (aligned with baseline logic)...")

    # --- Config ---
    config, params = load_config(config_file, params_file)

    n = pypsa.Network()
    start_date = config['simulation_settings']['start_date']
    n_hours = config['simulation_settings']['num_hours']
    ts = pd.date_range(start=start_date, periods=n_hours, freq='h')
    n.set_snapshots(ts)

    # --- PV params (HIGH PV) ---
    pv_params = params["high_pv"]
    eff_pv = pv_params["efficiency_pv"]
    eff_inv = pv_params["inverter_efficiency"]
    area_m2 = pv_params["surface_m2"]
    export_efficiency = pv_params["efficiency_export"]

    # STC approach: p_nom=area*eta*inv/1000 (MW); p_max_pu = irr/1000
    from scripts import data_processing
    raw_solar = data_processing.load_timeseries("weather_data.csv", data_path, index=ts)
    pv_profile_pu = raw_solar / 1000.0

    pv_p_nom_mw = (area_m2 * eff_pv * eff_inv) / 1000.0
    pv_generation_kw = data_processing.calculate_pv_generation(
        weather_df=pd.DataFrame({"solar_radiation": raw_solar}, index=ts),
        pv_surface_m2=area_m2, pv_efficiency_pv=eff_pv, pv_inverter_efficiency=eff_inv
    )
    pv_generation_mw = pv_generation_kw / 1000.0

    # --- Carriers ---
    for c in ["electricity", "heat", "heat_distribution", "heat_pump"]:
        if c not in n.carriers.index:
            n.add("Carrier", c)

    # --- Buses ---
    n.add("Bus", "Grid Elec Sell",  carrier="electricity")
    n.add("Bus", "Grid Elec Buy",   carrier="electricity")
    n.add("Bus", "Building Elec",   carrier="electricity")
    n.add("Bus", "Battery Storage", carrier="electricity")
    n.add("Bus", "PV Bus",          carrier="electricity")

    n.add("Bus", "District Heating", carrier="heat")
    n.add("Bus", "Building Heat",    carrier="heat")
    n.add("Bus", "Heat Source Bus",  carrier="heat")
    n.add("Bus", "Thermal Storage",  carrier="heat")

    # --- PV generator ---
    n.add("Generator", "Rooftop PV",
          bus="PV Bus", carrier="electricity",
          p_nom=pv_p_nom_mw, p_max_pu=pv_profile_pu,
          capital_cost=0.0, marginal_cost=0.0, efficiency=1.0)

    # --- Grid (prezzi non scalati) ---
    grid_params = params['grid']
    grid_co2_kg = float(grid_params.get('CO2_emissions_kg_per_mwh', 0.0))
    grid_capacity_mw = float(grid_params['capacity_mw'])
    trafo_eta = float(grid_params['transformer_efficiency'])

    if str(grid_params['import_cost_eur_per_kwh']).lower() == 'variable':
        from .utils import load_electricity_price_profile
        grid_price = load_electricity_price_profile(data_path, n.snapshots)  # EUR/MWh
    else:
        fixed = float(grid_params['import_cost_eur_per_kwh'])
        grid_price = pd.Series(fixed * 1000.0, index=n.snapshots)            # EUR/MWh

    export_price = grid_price.copy()
    n.grid_price_series = grid_price

    # Generatore 'Grid' con costo = prezzo orario (non scalato)
    n.add("Generator", "Grid",
          bus="Grid Elec Sell", carrier="electricity",
          p_nom_extendable=True, marginal_cost=grid_price, efficiency=1.0)

    # Link di import (solo efficienza, nessun costo sul link)
    n.add("Link", "Grid Import",
          bus0="Grid Elec Sell", bus1="Building Elec",
          p_nom=grid_capacity_mw, efficiency=trafo_eta,
          marginal_cost=0.0, capital_cost=0.0)
    n.links.at["Grid Import", "co2_emission_factor_kg_per_mwh"] = grid_co2_kg

    # --- Carichi edificio ---
    bld = params.get('social_building', {})
    elec_profile_mw = load_or_generate_profile(
        bld.get('electricity_load_profile', 'electricity_demand.csv'),
        bld.get('electricity_peak_mw', 0.23),
        data_path, ts
    ) / 1000.0
    heat_profile_mw = load_or_generate_profile(
        bld.get('heat_load_profile', 'heat_demand.csv'),
        bld.get('heat_peak_mw', 0.32),
        data_path, ts
    ) / 1000.0

    n.add("Load", "Building Elec Load", bus="Building Elec", p_set=elec_profile_mw, carrier="electricity")
    n.add("Load", "Building Heat Load", bus="Building Heat", p_set=heat_profile_mw,  carrier="heat")

    # --- District heating (allineato a baseline) ---
    heat_params = params.get('baseline_heat_source', {})
    heat_capacity_mw = float(heat_params.get('capacity_mw_th', 10.0))
    heat_eta = float(heat_params.get('efficiency_heating', 0.95))

    # prezzo termico: fisso o serie oraria (prezzo a valle / consegna)
    from .utils import load_thermal_price_profile
    price_profile = heat_params.get('price_profile')
    if price_profile:
        heat_price_valle = load_thermal_price_profile(os.path.join(data_path, 'timeseries', price_profile), n.snapshots)
    else:
        heat_price_valle = float(heat_params.get('cost_eur_per_mwh_th', 45.0))

    # Link DH→Building: solo efficienza, costo = 0 (come baseline)
    n.add("Link", "Heat import",
          bus0="District Heating", bus1="Building Heat",
          p_nom=1.0, p_nom_extendable=False,
          marginal_cost=0.0, efficiency=heat_eta)

    # CO2 sul link (kg/MWh_th a valle)
    dh_co2 = float(heat_params.get('co2_emissions_dh_kg_per_mwh', 0.0))
    n.links.at["Heat import", "co2_emission_factor_kg_per_mwh"] = dh_co2

    # Generatore 'Heating': costo a monte = prezzo_a_valle * eta (serie o scalare)
    if isinstance(heat_price_valle, pd.Series):
        n.add("Generator", "Heating",
              bus="District Heating", carrier="heat",
              p_nom=heat_capacity_mw, marginal_cost=0.0)
        if (not hasattr(n.generators_t, "marginal_cost")
            or n.generators_t.marginal_cost is None
            or n.generators_t.marginal_cost.empty):
            n.generators_t.marginal_cost = pd.DataFrame(
                0.0, index=n.snapshots, columns=n.generators.index
            )
        n.generators_t.marginal_cost.loc[:, "Heating"] = (heat_price_valle * heat_eta).values
    else:
        n.add("Generator", "Heating",
              bus="District Heating", carrier="heat",
              p_nom=heat_capacity_mw, marginal_cost=float(heat_price_valle) * heat_eta)

    # --- Flussi elettrici: autoconsumo + export ---
    # PV → Building (prioritario)
    n.add("Link", "PV Autoconsumo",
          bus0="PV Bus", bus1="Building Elec",
          p_nom=pv_p_nom_mw, efficiency=1.0,
          capital_cost=0.0, marginal_cost=-1e-6)

    # PV → Grid (export) con ricavo sul link
    if "PV Export" in n.links.index:
        n.remove("Link", "PV Export")
    n.add("Link", "PV Export",
          bus0="PV Bus", bus1="Grid Elec Buy",
          p_nom=pv_p_nom_mw, efficiency=export_efficiency,
          capital_cost=0.0, marginal_cost=0.0)
    n.links.at["PV Export", "co2_emission_factor_kg_per_mwh"] = grid_co2_kg

    # marginal_cost orario NEGATIVO sul link di export (epsilon se prezzo=0)
    if (not hasattr(n.links_t, "marginal_cost")
        or n.links_t.marginal_cost is None
        or n.links_t.marginal_cost.empty):
        n.links_t.marginal_cost = pd.DataFrame(0.0, index=n.snapshots, columns=n.links.index)
    eps = 1e-6
    _ser = pd.Series(export_price, index=n.snapshots)
    n.links_t.marginal_cost.loc[:, "PV Export"] = -_ser.where(_ser > 0, eps).to_numpy()

    # Sink di export (assorbe e non restituisce)
    n.add("StorageUnit", "Export Sink",
          bus="Grid Elec Buy",
          p_nom=1e3, p_nom_extendable=True, max_hours=1e6,
          p_min_pu=-1.0, p_max_pu=0.0,
          efficiency_store=1.0, efficiency_dispatch=1.0,
          cyclic_state_of_charge=False, capital_cost=0.0, marginal_cost=0.0)

    # --- Sanity checks minimi ---
    assert len(n.snapshots) == n_hours, "❌ Numero snapshot errato."
    for ln in n.loads.index:
        s = n.loads_t.p_set[ln]
        assert len(s) == n_hours and s.notna().all(), f"❌ Load '{ln}' non coerente."

    return n
