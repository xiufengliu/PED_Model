#!/usr/bin/env python3  # Shebang: usa l’interprete Python 3 dell’ambiente

"""
Tool to create and run scenarios for the PED Lyngby Model.

Commands:
  create   Create a new scenario module and update configs.
  run      Execute an existing scenario end-to-end.
"""
import os                 # Modulo per operazioni con il sistema operativo
import sys                # Modulo per interagire con l’interprete Python
import argparse           # Modulo per il parsing degli argomenti da linea di comando
import shutil             # Modulo per copie, spostamenti e rimozioni di file/cartelle
import yaml               # Modulo per leggere e scrivere file YAML
import traceback          # Modulo per gestire e formattare gli stack trace
from pathlib import Path  # Classe per gestire percorsi di file in modo indipendente dal SO

# Aggiunge la cartella principale del progetto al percorso di ricerca dei moduli
# Assumendo che questo script sia in project_root/scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd       # Importa pandas per la manipolazione di dati tabellari

# Import corretto della funzione che restituisce la factory per gli scenari
from scenarios import get_scenario_function
# Import delle funzioni di utilità per il plotting
from scripts.plotting_utils import (
     plot_time_series,
     plot_energy_balance,
     plot_energy_balance_breakdown,
     plot_pv_battery_breakdown,
     plot_thermal_breakdown,
     plot_dsm_thermal_breakdown,
)


HOURS_IN_YEAR = 8760      # Costante: numero di ore in un anno

def create_scenario(scenario_name: str, description: str) -> bool:
    """Create a new scenario with the given name and description."""
    # Validazione del nome scenario: solo caratteri alfanumerici e underscore
    if not (scenario_name.replace('_', '').isalnum()):
        print(f"Error: Scenario name must be alphanumeric (underscores allowed). Got: {scenario_name}")
        return False

    # Definizione dei percorsi utili
    scenarios_dir = Path('scenarios')                                # cartella degli scenari: Path('scenarios') crea un oggetto Path (dalla libreria pathlib) che rappresenta il percorso relativo alla cartella scenarios all’interno del tuo progetto. Assegnando questo Path a scenarios_dir, hai un riferimento che puoi poi usare per costruire altri percorsi (file o sottocartelle) all’interno di scenarios.
    scenario_file = scenarios_dir / f"{scenario_name}.py"            # file scenario da creare
    template_file = scenarios_dir / 'template.py'                    # file template di partenza: costruisce il percorso scenarios/template.py, cioè il file di template usato come base per generare nuovi scenari.

    # Verifica che non esista già e che esista il template
    if scenario_file.exists():
        print(f"Error: Scenario '{scenario_name}' already exists at {scenario_file}")
        return False
    if not template_file.exists():
        print(f"Error: Template file not found at {template_file}")
        return False

    # Copia il template e legge il contenuto
    shutil.copy(template_file, scenario_file)
    content = scenario_file.read_text()

    # Sostituisce intestazione e descrizione nel file di scenario
    header = f"PED Lyngby Model - {scenario_name.replace('_', ' ').title()} Scenario"
    content = content.replace(
        "PED Lyngby Model - Template for New Scenarios",
        header
    )
    content = content.replace(
        "This is a template file for creating new scenarios. Copy this file and modify it\nto implement a new scenario.",
        description
    )
    # Adatta anche la stringa di log
    content = content.replace(
        "Building new scenario network...",
        f"Building {scenario_name.replace('_', ' ')} network..."
    )
    scenario_file.write_text(content)  # Salva il file modificato

    # Aggiorna __init__.py per importare il nuovo scenario
    init_file = scenarios_dir / '__init__.py' # costruisce il percorso completo al file __init__.py dentro la cartella scenarios.
    if not init_file.exists():
        print(f"Error: __init__.py not found at {init_file}")
        return False
    init_text = init_file.read_text()

    # Inserisce il nuovo import nella sezione giusta
    import_marker = "# Dictionary mapping scenario names"
    idx = init_text.find(import_marker)
    if idx == -1:
        print("Error: Could not find import section in __init__.py")
        return False
    before, after = init_text[:idx], init_text[idx:]
    import_line = f"from . import {scenario_name}\n"
    if import_line not in before:
        lines = before.rstrip().splitlines()
        lines.append(import_line.strip())
        before = "\n".join(lines) + "\n"

    # Inserisce il riferimento alla funzione di creazione nel dizionario
    dict_start = after.find("SCENARIO_FUNCTIONS = {")
    dict_end = after.find("}", dict_start)
    if dict_start == -1 or dict_end == -1:
        print("Error: Could not find SCENARIO_FUNCTIONS dictionary in __init__.py")
        return False
    dict_block = after[dict_start:dict_end+1] #Nella parte after trova l’inizio (dict_start) e la fine (dict_end) del blocco testuale che definisce SCENARIO_FUNCTIONS = { … }
    entry = f"    '{scenario_name}': {scenario_name}.create_network,"
    if entry not in dict_block:
        lines = dict_block.splitlines()
        lines.insert(-1, entry)  # prima della parentesi chiudente
        new_block = "\n".join(lines)
        after = after.replace(dict_block, new_block)
    init_file.write_text(before + after)  # Riscrive __init__.py

    # Aggiorna config/config.yml aggiungendo lo scenario
    config_path = Path('config') / 'config.yml'
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return False
    config = yaml.safe_load(config_path.read_text())
    config.setdefault('scenarios', {})
    if scenario_name in config['scenarios']:
        print(f"Warning: Scenario '{scenario_name}' already in config.yml")
    else:
        config['scenarios'][scenario_name] = {
            'description': description,
            'output_subdir': f"scenario_{scenario_name}"
        }
        config_path.write_text(yaml.dump(config, default_flow_style=False))

    # Aggiorna config/component_params.yml con sezione stub
    params_path = Path('config') / 'component_params.yml'
    if not params_path.exists():
        print(f"Error: Component params file not found at {params_path}")
        return False
    params_text = params_path.read_text()
    header = f"# {scenario_name.replace('_', ' ').title()} Scenario Assets"
    if header not in params_text:
        params_text += f"\n{header}\n{scenario_name}:\n  # Add your parameters here\n"
        params_path.write_text(params_text)

    print(f"Successfully created new scenario: {scenario_name}")  # Conferma
    return True

def load_config(config_file: str) -> dict:
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)  # Ritorna il contenuto YAML come dict

def determine_paths(config: dict, config_file: str) -> tuple[str, str]:
    config_abs_path = Path(config_file).resolve()       # Percorso assoluto del config
    project_root = config_abs_path.parent.parent        # Cartella radice del progetto
    data_in = project_root / config.get('paths', {}).get('data_input', 'data/input')
    data_out = project_root / config.get('paths', {}).get('data_output', 'data/output')
    return str(data_in), str(data_out)                  # Percorsi input e output dati

def override_simulation_hours(config: dict, hours: int = HOURS_IN_YEAR) -> dict:
    config['simulation_settings'] = config.get('simulation_settings', {})
    config['simulation_settings']['num_hours'] = hours  # Imposta ore di simulazione
    return config

def write_temp_config(base_config: dict, config_file: str) -> str:
    temp_path = Path(config_file).parent / 'temp_config.yml'
    with open(temp_path, 'w') as f:
        yaml.dump(base_config, f)  # Scrive config temporaneo
    return str(temp_path)

def build_network(scenario_name: str, config_file: str, params_file: str, data_path: str):
    try:
        fn = get_scenario_function(scenario_name)             # Ottiene factory scenario
        return fn(config_file, params_file, data_path)       # Costruisce la rete
    except Exception:
        raise RuntimeError(f"Error building network for '{scenario_name}': {traceback.format_exc()}")

def run_lopf(network, solver_name: str, solver_options: dict) -> None:
    try:
        network.optimize(solver_name=solver_name, solver_options=solver_options)  # Esegue ottimizzazione
    except Exception as e:
        raise RuntimeError(f"LOPF failed ({solver_name}): {e}")

def calculate_investment_summary(network) -> pd.DataFrame:
    """
    Calculate summary for multi-period investment optimization scenarios.

    Returns:
        pd.DataFrame: Summary with investment results by period and technology
    """
    periods = network.investment_periods

    # Initialize results dictionary
    results = {
        'Period': [],
        'Technology': [],
        'Capacity_MW': [],
        'Capacity_MWh': [],
        'CAPEX_EUR': [],
        'Annual_OPEX_EUR': [],
        'Total_Generation_MWh': [],
        'Capacity_Factor': []
    }

    # Extract optimal capacities for each technology and period
    for period in periods:
        # PV capacity
        if 'PV' in network.generators.index:
            pv_cap = network.generators.at['PV', 'p_nom_opt']
            pv_capex = network.generators.at['PV', 'capital_cost'] * pv_cap
            results['Period'].append(period)
            results['Technology'].append('Solar PV')
            results['Capacity_MW'].append(pv_cap)
            results['Capacity_MWh'].append(0)  # N/A for generators
            results['CAPEX_EUR'].append(pv_capex)
            results['Annual_OPEX_EUR'].append(0)  # Marginal cost handled separately

            # Calculate generation for this period
            period_snaps = network.snapshots[network.snapshots.get_level_values(0) == period]
            if 'PV' in network.generators_t.p.columns:
                pv_gen = network.generators_t.p.loc[period_snaps, 'PV'].sum()
                cf = pv_gen / (pv_cap * len(period_snaps)) if pv_cap > 0 else 0
            else:
                pv_gen = 0
                cf = 0
            results['Total_Generation_MWh'].append(pv_gen)
            results['Capacity_Factor'].append(cf)

        # Battery capacity
        if 'Battery Store' in network.stores.index:
            bat_e_cap = network.stores.at['Battery Store', 'e_nom_opt']
            bat_e_capex = network.stores.at['Battery Store', 'capital_cost'] * bat_e_cap

            # Battery power capacity from links
            bat_p_cap = 0
            bat_p_capex = 0
            if 'Battery Charge' in network.links.index:
                bat_p_cap = network.links.at['Battery Charge', 'p_nom_opt']
                bat_p_capex = network.links.at['Battery Charge', 'capital_cost'] * bat_p_cap

            results['Period'].append(period)
            results['Technology'].append('Battery Storage')
            results['Capacity_MW'].append(bat_p_cap)
            results['Capacity_MWh'].append(bat_e_cap)
            results['CAPEX_EUR'].append(bat_e_capex + bat_p_capex)
            results['Annual_OPEX_EUR'].append(0)
            results['Total_Generation_MWh'].append(0)  # Storage doesn't generate
            results['Capacity_Factor'].append(0)  # N/A for storage

        # Heat pump capacity
        if 'Heat Pump' in network.links.index:
            hp_cap = network.links.at['Heat Pump', 'p_nom_opt']
            hp_capex = network.links.at['Heat Pump', 'capital_cost'] * hp_cap
            results['Period'].append(period)
            results['Technology'].append('Heat Pump')
            results['Capacity_MW'].append(hp_cap)
            results['Capacity_MWh'].append(0)
            results['CAPEX_EUR'].append(hp_capex)
            results['Annual_OPEX_EUR'].append(0)

            # Calculate heat pump operation
            period_snaps = network.snapshots[network.snapshots.get_level_values(0) == period]
            if 'Heat Pump' in network.links_t.p0.columns:
                hp_gen = network.links_t.p0.loc[period_snaps, 'Heat Pump'].sum()
                cf = hp_gen / (hp_cap * len(period_snaps)) if hp_cap > 0 else 0
            else:
                hp_gen = 0
                cf = 0
            results['Total_Generation_MWh'].append(hp_gen)
            results['Capacity_Factor'].append(cf)

        # Thermal storage capacity
        if 'Thermal Store' in network.stores.index:
            ts_cap = network.stores.at['Thermal Store', 'e_nom_opt']
            ts_capex = network.stores.at['Thermal Store', 'capital_cost'] * ts_cap
            results['Period'].append(period)
            results['Technology'].append('Thermal Storage')
            results['Capacity_MW'].append(0)  # Power rating handled by links
            results['Capacity_MWh'].append(ts_cap)
            results['CAPEX_EUR'].append(ts_capex)
            results['Annual_OPEX_EUR'].append(0)
            results['Total_Generation_MWh'].append(0)
            results['Capacity_Factor'].append(0)

    return pd.DataFrame(results)


#Prende in input l’oggetto rete post-ottimizzazione.



def calculate_summary(network) -> pd.DataFrame:
    # Check if this is a multi-period investment optimization
    is_multi_period = hasattr(network, 'investment_periods') and len(network.investment_periods) > 1

    if is_multi_period:
        return calculate_investment_summary(network)

    # snapshot weightings
    if not hasattr(network.snapshot_weightings, 'generators'):
        network.snapshot_weightings['generators'] = pd.Series(1.0, index=network.snapshots)
    w = network.snapshot_weightings.generators

    # Grid import/export
    # Grid import → Building Elec  +  Grid import → Battery Storage
    imp_link    = network.links_t.p0.get('Grid Import', pd.Series(0.0, index=network.snapshots))
    # Somma dei due flussi orari dalla rete
    imp = imp_link 
    exp = network.links_t.p0.get('Grid Export', pd.Series(0.0, index=network.snapshots)).clip(lower=0)
    total_imp = (imp * w).sum()
    total_exp = (exp * w).sum()

    # Emissioni da grid
    co2g   = network.links.at["Grid Import","co2_emission_factor_kg_per_mwh"]
    emis_imp = (imp * co2g * w).sum()
    emis_exp = (exp * co2g * w).sum()

    # Emissioni da DH
    dh_ts = network.links_t.p0.get("Heat import", pd.Series(0.0, index=network.snapshots))
    co2dh = network.links.at["Heat import","co2_emission_factor_kg_per_mwh"]
    emis_dh = (dh_ts * co2dh * w).sum()
    # ————————————————————————————————
    #  DSM Heat Dispatch (termico rilasciato dallo storage)
    # ————————————————————————————————
    dsm_h_disp_ts    = network.links_t.p0.get("DSM Heat Dispatch", pd.Series(0.0, index=network.snapshots))
    total_dsm_h_disp = (dsm_h_disp_ts * w).sum()


    # Produzione PV e Heating
    pv_ts   = network.generators_t.p.get('Rooftop PV', pd.Series(0.0,index=network.snapshots))
    heat_ts = network.generators_t.p.get('Heating',    pd.Series(0.0,index=network.snapshots))
    total_pv   = pv_ts.sum()
    total_heat = (heat_ts * w).sum()

    # Carichi inflessibile + flessibile (charge)
    elec_inf  = network.loads_t.p.get('Building Elec Load',  pd.Series(0.0, index=network.snapshots))
    elec_ch   = network.loads_t.p.get('DSM Elec Flex Load', pd.Series(0.0, index=network.snapshots))
    heat_inf  = network.loads_t.p.get('Building Heat Load',  pd.Series(0.0, index=network.snapshots))
    heat_ch   = network.loads_t.p.get('DSM Heat Flex Load', pd.Series(0.0, index=network.snapshots))

    # Dispatch: prendo p0 e lo ribalto se negativo
    raw_e_disp = network.links_t.p0.get('DSM Elec Dispatch', pd.Series(0.0, index=network.snapshots))
    elec_disp  = (-raw_e_disp).clip(lower=0)
    raw_h_disp = network.links_t.p0.get('DSM Heat Dispatch', pd.Series(0.0, index=network.snapshots))
    heat_disp  = (-raw_h_disp).clip(lower=0)

    # Net demand = inflessibile + charge – dispatch
    total_elec_demand = ((elec_inf + elec_ch - elec_disp) * w).sum()
    total_heat_demand = ((heat_inf + heat_ch - heat_disp) * w).sum()

    # Costi operativi
    # 2’) Prendi il marginal_cost dagli stream di rete→edificio e rete→batteria
    grid_price = network.grid_price_series
    ε6 = network.epsilon_grid_import

    # 3’) Calcolo i costi orari applicando gli overhead originali
    cost_grid_import  = (imp_link    * (grid_price + ε6) * w).sum()
    total_grid_import_cost = cost_grid_import


    # ————————————————————————————————
    #  Costo riscaldamento (unchanged)
    # ————————————————————————————————
    mc_h      = network.generators.at['Heating','marginal_cost']
    cost_heat = (heat_ts * mc_h * w).sum() if hasattr(mc_h, 'shape') else total_heat * mc_h
    # Costo operativo complessivo
    op_cost  = total_grid_import_cost + cost_heat

    # Heat Pump & DH import
    hp_out = -network.links_t.p1.get('Heat Pump', pd.Series(0.0,index=network.snapshots))
    total_hp = (hp_out * w).sum()
    total_dh = (dh_ts * w).sum()
    print("DEBUG: mc_imp_link =", network.links.at['Grid Import','marginal_cost'])

    return pd.DataFrame([{
        'Total Grid Import (MWh)':        total_imp,
        'Total Grid Export (MWh)':        total_exp,
        'Grid Import Cost (EUR)':         cost_grid_import,
        'Total Grid Import Cost (EUR)':   total_grid_import_cost,
        'Total Operational Cost (EUR)':   op_cost,
        'Total PV Produced (MWh)':        total_pv,
        'Total Heat Produced (MWh_th)':   total_heat_demand,
        'Total Elec Demand (MWh)':        total_elec_demand,
        'Heat Pump Production (MWh_th)':  total_hp,
        'DH Import to Building (MWh_th)': total_dh,
        'DSM Heat Dispatch (MWh_th)':     total_dsm_h_disp,
        'Total Import Emissions (kgCO₂)': emis_imp,
        'Avoided Emissions (kgCO₂)':      emis_exp,
        'Total DH Emissions (kgCO₂)':     emis_dh,
    }])

def generate_hourly_df(network) -> pd.DataFrame:
    snaps = network.snapshots

    # inflessibile + flex
    elec_inf  = network.loads_t.p.get('Building Elec Load',  pd.Series(0.0,index=snaps))
    elec_ch   = network.loads_t.p.get('DSM Elec Flex Load', pd.Series(0.0,index=snaps))
    # carico inflessibile (resta qui)  
    heat_inf = network.loads_t.p.get('Building Heat Load', pd.Series(0.0, index=snaps))

    # 1) VERO “charge” sullo storage DSM  
    raw_h_charge = network.links_t.p0.get('DSM Heat Charge', pd.Series(0.0, index=snaps))
    heat_charge  = raw_h_charge.clip(lower=0)

    # 2) VERO “dispatch” dal storage DSM  
    raw_h_disp   = network.links_t.p0.get('DSM Heat Dispatch', pd.Series(0.0, index=snaps))
    heat_disp    = raw_h_disp.clip(lower=0)

    # dispatch normalizzato
    raw_e_disp = network.links_t.p0.get('DSM Elec Dispatch', pd.Series(0.0,index=snaps))
    elec_disp  = (-raw_e_disp).clip(lower=0)
 

    # altri flussi
    pv      = network.generators_t.p.get('Rooftop PV', pd.Series(0.0,index=snaps))
    grid_i  = network.links_t.p0.get('Grid Import',   pd.Series(0.0,index=snaps))
    grid_e  = network.links_t.p0.get('Grid Export',   pd.Series(0.0,index=snaps)).clip(lower=0)
    hp_out  = -network.links_t.p1.get('Heat Pump',    pd.Series(0.0,index=snaps))
    dh_imp  = network.links_t.p0.get('Heat import',   pd.Series(0.0,index=snaps))

    df = pd.DataFrame({
        'timestamp':                       snaps,
        'electricity_inflexible_load_mwh': elec_inf.values,
        'electricity_charge_mwh':          elec_ch.values,
        'electricity_dispatch_mwh':        elec_disp.values,
        'electricity_load_mwh':            (elec_inf + elec_ch - elec_disp).values,
        'heat_inflexible_load_mwh':        heat_inf.values,
        'heat_charge_mwh':   heat_charge.values,
        'heat_dispatch_mwh': heat_disp.values,
        'heat_load_mwh':                   (heat_inf + heat_charge - heat_disp).values,
        'pv_generation_mwh':               pv.values,
        'grid_import_mwh':                 grid_i.values,
        'grid_export_mwh':                 grid_e.values,
        'heat_pump_output':                hp_out.values,
        'dh_import':                       dh_imp.values,
    })

    co2g = network.links.at["Grid Import","co2_emission_factor_kg_per_mwh"]
    df['grid_import_emissions_kg'] = df['grid_import_mwh'] * co2g
    df['grid_export_emissions_kg'] = df['grid_export_mwh'] * co2g

    # Handle different heat source naming conventions
    try:
        co2d = network.links.at["Heat import","co2_emission_factor_kg_per_mwh"]
    except KeyError:
        try:
            co2d = network.generators.at["District Heating Backup","co2_emissions"]
        except KeyError:
            co2d = 33.9  # Default CO2 factor for district heating

    df['dh_emissions_kg'] = df['dh_import'] * co2d

    df['total_load_mwh'] = df['electricity_load_mwh'] + df['heat_load_mwh']
    return df


    

#####Chiama tutte le funzioni di supporto per caricare la configurazione, costruire la rete, eseguire l’ottimizzazione, salvare risultati e infine restituire un DataFrame orario
def run_scenario(scenario_name: str, config_file: str, params_file: str) -> pd.DataFrame:
    print(f"\n--- Running Scenario: {scenario_name} ---")  # Intestazione
    # Risolve percorsi assoluti dei file config e params
    config_abs_path = Path(config_file).resolve() #Path(...).resolve() converte eventuali percorsi relativi in un percorso assoluto, evitando problemi se lo script viene lanciato da directory diverse.
    params_abs_path = Path(params_file).resolve()
    print(f"Using config file: {config_abs_path}")
    print(f"Using params file: {params_abs_path}")
    

    # Carica config e determina percorsi dati
    config = load_config(str(config_abs_path))
    data_in, data_out_base = determine_paths(config, str(config_abs_path))
    subdir = config.get('scenarios', {}).get(scenario_name, {}).get('output_subdir', f"scenario_{scenario_name}")
    out_dir = Path(data_out_base) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)  # Crea cartella output

    # Override ore simulazione e scrive config temporaneo
    override = override_simulation_hours(config)
    temp_cfg = write_temp_config(override, str(config_abs_path))

    # Costruzione rete
    try:
        net, _, _ = build_network(scenario_name, temp_cfg, str(params_abs_path), data_in)
        Path(temp_cfg).unlink()  # Rimuove config temporaneo
    except RuntimeError as e:
        print(e)
        return pd.DataFrame()
    

    # Esecuzione LOPF
    sim = override.get('simulation_settings', {})
    try:
        run_lopf(net, sim.get('solver', 'highs'), sim.get('solver_options', {}))
    except RuntimeError as e:
        print(e)
        return pd.DataFrame()
    # ——— Make the “Grid Export Sink” load flexible ———
    if "Grid Export Sink" in net.loads.index:
        net.loads.at["Grid Export Sink", "p_set"] = 0.0
        net.loads.loc["Grid Export Sink", "sign"] = 1





    # Salvataggio network su NetCDF: Prova a esportare l’oggetto rete (di solito un file .nc NetCDF) nella cartella di output.
    net_file = out_dir / f"{scenario_name}_network_results.nc"
    try:
        net.export_to_netcdf(str(net_file))
    except Exception as e:
        print(f"Error saving network: {e}")

    # Calcola e salva summary e risultati orari
    summary_df = calculate_summary(net) #calculate_summary restituisce un DataFrame con le metriche aggregate (import/export, costi, ecc.) e lo salva in CSV.
    summary_df.to_csv(out_dir / f"{scenario_name}_summary.csv", index=False)
    hourly_df = generate_hourly_df(net) #generate_hourly_df crea un DataFrame orario dettagliato e lo salva anch’esso in CSV.
    hourly_df.to_csv(out_dir / f"{scenario_name}_hourly_results.csv", index=False)

    # Handle battery data if present (not all scenarios have batteries)
    if "Battery Store" in net.stores.index and not net.stores_t.e.empty:
        soc = net.stores_t.e["Battery Store"]      # Serie SoC in MWh: Estrae la serie temporale dello Stato di Carica (SoC) della batteria.
        hourly_df["battery_soc_mwh"] = soc.values                 # Aggiunge colonna SoC: La aggiunge come colonna al DataFrame orario.
        total_charge    = soc.diff().clip(lower=0).sum()          # MWh caricati: Calcola MWh totali caricati e scaricati nel periodo basandosi sulle differenze giornaliere.
        total_discharge = (-soc.diff().clip(upper=0)).sum()       # MWh scaricati
    else:
        hourly_df["battery_soc_mwh"] = 0
        total_charge = 0
        total_discharge = 0
    heat_ch = hourly_df['heat_charge_mwh'].sum()
    heat_dc = hourly_df['heat_dispatch_mwh'].sum()

    # 1) DSM Heat Charge (MWh_th)
    dsm_heat_charge = hourly_df['heat_charge_mwh'].sum()

    # 2) DSM Heat Discharge (MWh_th)
    dsm_heat_discharge = hourly_df['heat_dispatch_mwh'].sum()

    # 3) Heat Flexible Load (MWh_th)
    #    corrisponde al profilo di carico flessibile che hai in hourly_df:
    heat_flexible_load = hourly_df['heat_charge_mwh'].sum()

    # 4) Heat Inflexible + Charge − Discharge (MWh_th)
    heat_inflexible = hourly_df['heat_inflexible_load_mwh'].sum()
    net_heat_demand = heat_inflexible + dsm_heat_charge - dsm_heat_discharge

    print(f"DSM Heat Charge:           {dsm_heat_charge:.2f} MWh_th")
    print(f"DSM Heat Discharge:        {dsm_heat_discharge:.2f} MWh_th")
    print(f"Heat Flexible Load:        {heat_flexible_load:.2f} MWh_th")
    print(f"Inflexible + Charge−Dispatch: {net_heat_demand:.2f} MWh_th")



    # Generazione grafici
    try:
        plot_time_series({
            'timestamps': hourly_df['timestamp'],  #chiave:  pandas.Series che contiene, per ciascun timestamp, il valore del carico elettrico in kilowatt-ora.
            'electric_load': hourly_df['electricity_load_mwh'],
            'thermal_load': hourly_df['heat_load_mwh'],
            'pv_generation': hourly_df['pv_generation_mwh'],
            'grid_import': hourly_df['grid_import_mwh'],
            'grid_export': hourly_df['grid_export_mwh'],

        }, scenario_name=scenario_name, output_dir=str(out_dir))
        plot_energy_balance({
            'pv_generation_mwh': hourly_df['pv_generation_mwh'].sum(),
            'grid_import_mwh': hourly_df['grid_import_mwh'].sum(),
            'grid_export_mwh': hourly_df['grid_export_mwh'].sum(),
            'total_electric_load_mwh': hourly_df['electricity_load_mwh'].sum(),
            'total_thermal_load_mwh': hourly_df['heat_load_mwh'].sum(),
            'co2_emissions_kg': hourly_df['grid_import_emissions_kg'].sum(),      # ← qui il dato delle emissioni

        }, scenario_name=scenario_name, output_dir=str(out_dir))

         # --- breakdown 2×2 dello energy balance ---
        # calcolo i totali termici da hourly_df
        hp_total = hourly_df['heat_pump_output'].sum()
        dh_total = hourly_df['dh_import'].sum()

        plot_energy_balance_breakdown({
            'pv_generation_mwh':           hourly_df['pv_generation_mwh'].sum(),
            'grid_import_mwh':             hourly_df['grid_import_mwh'].sum(),
            'grid_export_mwh':             hourly_df['grid_export_mwh'].sum(),
            'total_electric_load_mwh':     hourly_df['electricity_load_mwh'].sum(),
            'total_thermal_load_mwh':      hourly_df['heat_load_mwh'].sum(),
            'Heat Pump Production (MWh_th)':    hp_total,
            'DH Import to Building (MWh_th)':   dh_total,
            'DSM Heat Charge (MWh_th)':        heat_ch,
            'DSM Heat Dispatch (MWh_th)':      heat_dc,
            'co2_emissions_kg':            hourly_df['grid_import_emissions_kg'].sum(),
            'dh_emissions_kg':   hourly_df['dh_emissions_kg'].sum(),   # 📈 emissioni CO2 DH
        }, scenario_name=scenario_name, output_dir=str(out_dir))
        # — Breakdown PV↔Battery↔Building Elec —
        plot_pv_battery_breakdown(
            net,
            scenario_name=scenario_name,
            output_dir=str(out_dir)
        )
        # — Breakdown flussi termici —
        plot_thermal_breakdown(
            net,
            scenario_name=scenario_name,
            output_dir=str(out_dir)
        )
        # — DSM Thermal Breakdown —
        plot_dsm_thermal_breakdown(
            hourly_df,
            scenario_name=scenario_name,
            output_dir=str(out_dir)
        )

    except Exception as e:
        print(f"Plotting error: {e}")
        

    return hourly_df  # Restituisce DataFrame orario

def main():
    parser = argparse.ArgumentParser(description="PED Lyngby scenario tool")  # Parser CLI
    sub = parser.add_subparsers(dest='command', required=True)             # Sottocomandi

    create_p = sub.add_parser('create', help='Create a new scenario')       # Comando create
    create_p.add_argument('name', help='Scenario name (alphanumeric, underscores)')
    create_p.add_argument('description', help='Brief description')

    run_p = sub.add_parser('run', help='Run an existing scenario')         # Comando run
    run_p.add_argument('scenario', help='Scenario name')
    run_p.add_argument('config', help='Path to config.yml')
    run_p.add_argument('params', help='Path to component_params.yml')

    args = parser.parse_args()  # Parsing degli argomenti
    print('*' * 40)
    print(f"Command: {args.scenario}, ")
    print('*' * 40)

    # Esecuzione in base al comando
    if args.command == 'create':
        success = create_scenario(args.name, args.description)
        if not success:
            sys.exit(1)
    elif args.command == 'run':
        df = run_scenario(args.scenario, args.config, args.params)
        if df.empty:
            sys.exit(1)

if __name__ == '__main__':
    main()  # Punto di ingresso: esegue main se lanciato come script
