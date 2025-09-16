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
import matplotlib.pyplot as plt


# Aggiunge la cartella principale del progetto al percorso di ricerca dei moduli
# Assumendo che questo script sia in project_root/scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd       # Importa pandas per la manipolazione di dati tabellari

# Import corretto della funzione che restituisce la factory per gli scenari
from scenarios import get_scenario_function
# Import delle funzioni di utilità per il plotting
from scripts.plotting_utils import plot_time_series, plot_energy_balance, plot_pv_battery_breakdown


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
    

def calculate_summary(network) -> pd.DataFrame:
    # Pesi e indice temporale
    if not hasattr(network.snapshot_weightings, 'generators'):
        network.snapshot_weightings['generators'] = pd.Series(1.0, index=network.snapshots)
    w = network.snapshot_weightings.generators
    snaps = network.snapshots

    # ---------------------------
    # Grid Import: lordo vs netto
    # ---------------------------
    imp_gross = network.links_t.p0.get('Grid Import', pd.Series(0.0, index=snaps)).clip(lower=0.0)
    imp_net   = (-network.links_t.p1.get('Grid Import', pd.Series(0.0, index=snaps))).clip(lower=0.0)

    # ---------------------------
    # Grid Export: lordo vs netto
    # ---------------------------
    if 'Grid Export' in network.links_t.p0.columns:
        exp_col = 'Grid Export'
    elif 'PV Export' in network.links_t.p0.columns:
        exp_col = 'PV Export'
    else:
        exp_col = None

    exp_gross = (network.links_t.p0[exp_col].clip(lower=0.0) if exp_col else pd.Series(0.0, index=snaps))
    exp_net   = ((-network.links_t.p1[exp_col]).clip(lower=0.0) if exp_col else pd.Series(0.0, index=snaps))

    # Totali (MWh) — usiamo i NETTI per i bilanci energetici
    total_import_net = (imp_net   * w).sum()
    total_export_net = (exp_net   * w).sum()
    total_import_gross = (imp_gross * w).sum()
    total_export_gross = (exp_gross * w).sum()

    # Emissioni e costi: su quanto ACQUISTI (lordo)
    co2_factor = network.links.at['Grid Import', 'co2_emission_factor_kg_per_mwh'] if 'Grid Import' in network.links.index else 0.0
    mc_gross = (network.links_t.marginal_cost['Grid Import']
                if hasattr(network.links_t, 'marginal_cost') and 'Grid Import' in network.links_t.marginal_cost.columns
                else (network.links.at['Grid Import', 'marginal_cost'] if 'Grid Import' in network.links.index else 0.0))

    total_imp_emis = ((imp_gross * co2_factor) * w).sum()
    cost_import = ((imp_gross * mc_gross * w).sum() if isinstance(mc_gross, pd.Series)
                   else total_import_gross * float(mc_gross))

    # Produzione/carichi (per completezza)
    heat = network.generators_t.p.get('Heating', pd.Series(0.0, index=snaps))
    pv   = network.generators_t.p.get('Rooftop PV', pd.Series(0.0, index=snaps))
    total_heat = (heat * w).sum()
    total_pv   = (pv   * w).sum()

    elec_ld = network.loads_t.p_set.get('Building Elec Load', pd.Series(0.0, index=snaps))
    heat_ld = network.loads_t.p_set.get('Building Heat Load', pd.Series(0.0, index=snaps))
    total_elec_demand = (elec_ld * w).sum()
    total_heat_demand = (heat_ld * w).sum()

    # Costo calore (scalare o serie)
    heat_mc = network.generators.at['Heating', 'marginal_cost'] if 'Heating' in network.generators.index else 0.0
    heat_cost_total = ((heat * heat_mc * w).sum() if isinstance(heat_mc, pd.Series)
                       else total_heat * float(heat_mc))

    operational_cost = cost_import + heat_cost_total

    # Evitate CO₂: calcolale su export NETTO (energia realmente consegnata alla Grid)
    avoid_co2 = (exp_net * co2_factor * w).sum()

    return pd.DataFrame([{
        'Total Grid Import (Net, MWh)':     total_import_net,
        'Total Grid Import (Gross, MWh)':   total_import_gross,
        'Total Grid Export (Net, MWh)':     total_export_net,
        'Total Grid Export (Gross, MWh)':   total_export_gross,
        'Total Heat Produced (MWh_th)':     total_heat,
        'Total PV Produced (MWh)':          total_pv,
        'Total Elec Demand (MWh)':          total_elec_demand,
        'Total Heat Demand (MWh_th)':       total_heat_demand,
        'Total Operational Cost (EUR)':     operational_cost,
        'Grid Cost (EUR)':                  cost_import,
        'Total Import Emissions (kgCO₂)':   total_imp_emis,
        'Avoid CO₂ Emissions (kgCO₂)':      avoid_co2,
    }])

######La funzione generate_hourly_df(network) → pd.DataFrame costruisce un DataFrame “orario” che mette insieme, per ciascun timestamp (snapshot), i principali flussi di energia in mwh
def generate_hourly_df(network) -> pd.DataFrame:
    snaps = network.snapshots

    # Carichi elettrici/termici
    if not network.loads_t.p_set.empty:
        elec_cols = [c for c in network.loads_t.p_set.columns if network.loads.loc[c].carrier == 'electricity']
        heat_cols = [c for c in network.loads_t.p_set.columns if network.loads.loc[c].carrier == 'heat']
    else:
        elec_cols, heat_cols = [], []

    elec = network.loads_t.p_set[elec_cols].sum(axis=1) if elec_cols else pd.Series(0.0, index=snaps)
    heat = network.loads_t.p_set[heat_cols].sum(axis=1) if heat_cols else pd.Series(0.0, index=snaps)

    # PV
    pv = network.generators_t.p.get('Rooftop PV', pd.Series(0.0, index=snaps))

    # Import lordo/netto
    grid_imp_gross = network.links_t.p0.get('Grid Import', pd.Series(0.0, index=snaps)).clip(lower=0.0)
    grid_imp_net   = (-network.links_t.p1.get('Grid Import', pd.Series(0.0, index=snaps))).clip(lower=0.0)

    # Export lordo/netto con fallback nome
    if 'Grid Export' in network.links_t.p0.columns:
        exp_col = 'Grid Export'
    elif 'PV Export' in network.links_t.p0.columns:
        exp_col = 'PV Export'
    else:
        exp_col = None

    grid_exp_gross = (network.links_t.p0[exp_col].clip(lower=0.0) if exp_col else pd.Series(0.0, index=snaps))
    grid_exp_net   = ((-network.links_t.p1[exp_col]).clip(lower=0.0) if exp_col else pd.Series(0.0, index=snaps))

    df = pd.DataFrame({
        'timestamp':               snaps,
        'electricity_load_mwh':    elec.values,
        'heat_load_mwh':           heat.values,
        'pv_generation_mwh':       pv.values,
        'grid_import_mwh':         grid_imp_net.values,      # NETTO
        'grid_import_gross_mwh':   grid_imp_gross.values,    # LORDO (per costi/emissioni)
        'grid_export_mwh':         grid_exp_net.values,      # NETTO (richiesto)
        'grid_export_gross_mwh':   grid_exp_gross.values,    # LORDO (facoltativo)
    })

    df['total_load_mwh'] = df['electricity_load_mwh'] + df['heat_load_mwh']

    # Emissioni: usiamo il LORDO acquistato
    co2_factor = network.links.at['Grid Import', 'co2_emission_factor_kg_per_mwh'] if 'Grid Import' in network.links.index else 0.0
    df['grid_import_emissions_kg'] = df['grid_import_gross_mwh'] * co2_factor

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
        net = build_network(scenario_name, temp_cfg, str(params_abs_path), data_in)
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
    # Se vuoi lavorare direttamente sul DataFrame già in memoria:
    df = hourly_df.copy()

    # timestamp in cui la domanda è massima
    t_peak_load = df["electricity_load_mwh"].idxmax()

    # valore di load e import in quello stesso istante
    peak_load          = df.loc[t_peak_load, "electricity_load_mwh"]
    import_at_peak     = df.loc[t_peak_load, "grid_import_mwh"]

    # vero peak-shaving sul picco di domanda
    peak_shaving       = peak_load - import_at_peak

    print(f"Picco domanda      = {peak_load:.3f} MWh")
    print(f"Import allo stesso istante = {import_at_peak:.3f} MWh")
    print(f"Peak-shaving (su picco domanda) = {peak_shaving:.3f} MWh")


    # Batterie: SoC e carica/scarica totali
    soc = net.stores_t.e["Battery Store"]      # Serie SoC in MWh: Estrae la serie temporale dello Stato di Carica (SoC) della batteria.
    hourly_df["battery_soc_mwh"] = soc.values                 # Aggiunge colonna SoC: La aggiunge come colonna al DataFrame orario.
    total_charge    = soc.diff().clip(lower=0).sum()          # MWh caricati: Calcola MWh totali caricati e scaricati nel periodo basandosi sulle differenze giornaliere.
    total_discharge = (-soc.diff().clip(upper=0)).sum()       # MWh scaricati

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
        # ——— dopo plot_energy_balance(…) ———
        plot_pv_battery_breakdown(
            net,
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
