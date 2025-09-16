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
    


    
def validate_priorities(n, tol=1e-3, max_show=10):
    import pandas as pd

    snaps = n.snapshots
    z = lambda: pd.Series(0.0, index=snaps)

    # --- Flussi NETTI (al nodo di arrivo, -p1) ---
    pv_exp_net    = (-n.links_t.p1.get("PV Export",        z())).clip(lower=0.0)
    grid_imp_net  = (-n.links_t.p1.get("Grid Import",      z())).clip(lower=0.0)
    batt_ch_net   = (-n.links_t.p1.get("Battery Charge",   z())).clip(lower=0.0)
    batt_dis_net  = (-n.links_t.p1.get("Battery Dispatch", z())).clip(lower=0.0)
    th_ch_net     = (-n.links_t.p1.get("Thermal Charge",   z())).clip(lower=0.0)
    th_dis_net    = (-n.links_t.p1.get("Thermal Dispatch", z())).clip(lower=0.0)
    dh_net        = (-n.links_t.p1.get("Heat import",      z())).clip(lower=0.0)

    elec_load     = n.loads_t.p_set.get("Building Elec Load", z())
    heat_load     = n.loads_t.p_set.get("Building Heat Load", z())

    # PV→Building netto e residuo elettrico dopo PV
    pv_to_build_net = (-n.links_t.p1.get("PV Autoconsumo", z())).clip(lower=0.0)
    elec_residual   = (elec_load - pv_to_build_net).clip(lower=0.0)

    # --- utilità: capacità effettiva (considera p_nom_opt se presente) ---
    def link_nominal(name):
        if name not in n.links.index:
            return 0.0
        try:
            if "p_nom_opt" in n.links.columns:
                val = n.links.at[name, "p_nom_opt"]
                if pd.notna(val) and float(val) > 0:
                    return float(val)
        except Exception:
            pass
        return float(n.links.at[name, "p_nom"])

    def at_cap_series(name):
        if name not in n.links.index:
            return pd.Series(False, index=snaps)
        p_nom_eff = link_nominal(name)
        if p_nom_eff <= 0:
            return pd.Series(False, index=snaps)
        return n.links_t.p0[name].abs() >= (p_nom_eff - 1e-9)

    cap_pv2bld  = at_cap_series("PV Autoconsumo")
    cap_batt_ch = at_cap_series("Battery Charge")
    cap_th_dc   = at_cap_series("Thermal Dispatch")

    # --- utilità: SoC con margini e_nom_opt se presente ---
    def store_nominal(name):
        if name not in n.stores.index:
            return 0.0
        try:
            if "e_nom_opt" in n.stores.columns:
                val = n.stores.at[name, "e_nom_opt"]
                if pd.notna(val) and float(val) > 0:
                    return float(val)
        except Exception:
            pass
        return float(n.stores.at[name, "e_nom"])

    def soc_series(name):
        if name not in n.stores.index:
            return pd.Series(0.0, index=snaps), 0.0, 0.0, 0.0
        e     = n.stores_t.e[name]
        e_nom = store_nominal(name)
        e_min = float(n.stores.at[name, "e_min_pu"])*e_nom if "e_min_pu" in n.stores.columns else 0.0
        e_max = float(n.stores.at[name, "e_max_pu"])*e_nom if "e_max_pu" in n.stores.columns else e_nom
        return e, e_nom, e_min, e_max

    batt_e, batt_enom, batt_emin, batt_emax = soc_series("Battery Store")
    tes_e,  tes_enom,  tes_emin,  tes_emax  = soc_series("Thermal Store")

    # Margini “significativi” (evita falsi positivi quando appoggiato ai limiti)
    eps_soc = max(1e-3, 0.01)  # 1% E_nom o 1e-3 MWh
    batt_above_min_margin = (batt_e - batt_emin) > max(1e-3, 0.01*batt_enom)
    batt_below_max_margin = (batt_emax - batt_e) > max(1e-3, 0.01*batt_enom)
    batt_near_max         = (batt_emax - batt_e) <= max(1e-3, 0.01*batt_enom)  # ✅ consente scarica durante export

    # TES: energia disponibile “non trascurabile” e collo di potenza
    P_TES_DIS = link_nominal("Thermal Dispatch")
    tes_energy_available   = (tes_e - tes_emin)
    tes_has_meaningful_energy = tes_energy_available > max(1e-3, 0.05*max(P_TES_DIS, 0.0))  # >5% di p_nom·1h

    # --- Check (con logica raffinata) ---
    v1 = (grid_imp_net > tol) & (pv_exp_net > tol)                          # import & export insieme
    v2_batt = (batt_ch_net > tol) & (batt_dis_net > tol)                    # batt charge & discharge
    v2_tes  = (th_ch_net   > tol) & (th_dis_net   > tol)                    # TES charge & dispatch

    # Export mentre scarico batteria -> VIOLAZIONE solo se:
    # - residuo elettrico nullo (PV copre il carico),
    # - link PV→Building non saturo,
    # - batteria NON quasi al massimo e sopra il minimo di margine.
    v3 = (pv_exp_net > tol) & (batt_dis_net > tol) \
         & (elec_residual <= tol) & (~cap_pv2bld) \
         & batt_above_min_margin & (~batt_near_max)

    # DH usato mentre TES non scarica -> VIOLAZIONE solo se:
    # - c'è domanda termica, TES ha energia significativa,
    # - link TES→Building non saturo e flusso TES→Building ~ 0.
    v4 = (dh_net > tol) & (heat_load > tol) & tes_has_meaningful_energy \
         & (~cap_th_dc) & (th_dis_net <= tol)

    # Export mentre carico batteria -> VIOLAZIONE solo se:
    # - batteria non è quasi al massimo e link carica non saturo.
    v5 = (pv_exp_net > tol) & (batt_ch_net > tol) & batt_below_max_margin & (~cap_batt_ch)

    def head_times(mask):
        return list(mask[mask].index[:max_show])

    problems = {
        "import_and_export_same_hour": (v1.sum(), head_times(v1)),
        "battery_charge_and_discharge_same_hour": (v2_batt.sum(), head_times(v2_batt)),
        "tes_charge_and_dispatch_same_hour": (v2_tes.sum(), head_times(v2_tes)),
        "export_while_import_or_batt_discharge": (v3.sum(), head_times(v3)),
        "dh_used_while_tes_not_dispatching": (v4.sum(), head_times(v4)),
        "export_while_battery_charging": (v5.sum(), head_times(v5)),
    }

    print("\n=== Priority validation summary ===")
    for k,(cnt,ts) in problems.items():
        print(f"{k}: {cnt} violations")
        if cnt:
            print("  examples:", ts)

    return problems

def explain_export(n, tol=1e-6):
    import pandas as pd

    s = n.snapshots
    Z = pd.Series(0.0, index=s)

    # --- Flussi netti (al bus di arrivo: -p1) ---
    exp_net    = (-n.links_t.p1.get("PV Export",        Z)).clip(lower=0.0)
    imp_net    = (-n.links_t.p1.get("Grid Import",      Z)).clip(lower=0.0)
    pv2b_net   = (-n.links_t.p1.get("PV Autoconsumo",   Z)).clip(lower=0.0)
    batt_ch    = (-n.links_t.p1.get("Battery Charge",   Z)).clip(lower=0.0)
    batt_dis   = (-n.links_t.p1.get("Battery Dispatch", Z)).clip(lower=0.0)

    # catena termica
    hp_el_in   = ( n.links_t.p0.get("Heat Pump",        Z)).clip(lower=0.0)  # input elettrico da PV Bus
    hp_out_th  = (-n.links_t.p1.get("Heat Pump",        Z)).clip(lower=0.0)  # uscita termica su Heat Source Bus
    th_ch      = (-n.links_t.p1.get("Thermal Charge",   Z)).clip(lower=0.0)  # HeatSource->TES
    th_dis     = (-n.links_t.p1.get("Thermal Dispatch", Z)).clip(lower=0.0)  # TES->Building Heat
    dh_to_bld  = (-n.links_t.p1.get("Heat import",      Z)).clip(lower=0.0)  # DH->Building Heat

    # --- Domande ---
    elec_load  = n.loads_t.p_set.get("Building Elec Load", Z)
    heat_load  = n.loads_t.p_set.get("Building Heat Load", Z)

    # --- Residuo elettrico "vero" (dopo tutte le forniture lato Building Elec) ---
    # load - (PV->building + Battery->building + Import)
    elec_residual_after = elec_load - pv2b_net - batt_dis - imp_net

    # Tolleranza robusta: assoluta + relativa
    elec_eps = tol + 1e-3 * elec_load.abs()
    elec_ok  = elec_residual_after.abs() <= elec_eps

    # --- Utility: p_nom effettiva (usa p_nom_opt se presente) ---
    def link_nom(name):
        if name not in n.links.index:
            return 0.0
        try:
            if "p_nom_opt" in n.links.columns:
                v = n.links.at[name, "p_nom_opt"]
                if pd.notna(v) and float(v) > 0:
                    return float(v)
        except Exception:
            pass
        return float(n.links.at[name, "p_nom"])

    def at_cap(name):
        if name not in n.links.index:
            return pd.Series(False, index=s)
        pnom = link_nom(name)
        if pnom <= 0:
            return pd.Series(False, index=s)
        return n.links_t.p0[name].abs() >= (pnom - 1e-9)

    cap_HP_to_build  = at_cap("Thermal Discharge")   # HeatSource -> Building Heat
    cap_HP_to_TES    = at_cap("Thermal Charge")      # HeatSource -> TES
    cap_TES_to_build = at_cap("Thermal Dispatch")    # TES       -> Building Heat
    cap_PV_to_build  = at_cap("PV Autoconsumo")
    cap_batt_charge  = at_cap("Battery Charge")

    # --- Utility: SoC & margini ---
    def e_nom(name):
        if name not in n.stores.index:
            return 0.0
        try:
            if "e_nom_opt" in n.stores.columns:
                v = n.stores.at[name, "e_nom_opt"]
                if pd.notna(v) and float(v) > 0:
                    return float(v)
        except Exception:
            pass
        return float(n.stores.at[name, "e_nom"])

    def soc(name):
        e     = n.stores_t.e.get(name, pd.Series(0.0, index=s))
        enom  = e_nom(name)
        emin  = float(n.stores.at[name, "e_min_pu"]) * enom if "e_min_pu" in n.stores.columns else 0.0
        emax  = float(n.stores.at[name, "e_max_pu"]) * enom if "e_max_pu" in n.stores.columns else enom
        return e, enom, emin, emax

    batt_e, batt_enom, batt_emin, batt_emax = soc("Battery Store")
    tes_e,  tes_enom,  tes_emin,  tes_emax  = soc("Thermal Store")

    soc_margin_batt = max(1e-3, 0.01 * max(1.0, batt_enom))
    soc_margin_tes  = max(1e-3, 0.01 * max(1.0, tes_enom))

    batt_near_max = (batt_emax - batt_e) <= soc_margin_batt
    tes_near_max  = (tes_emax  - tes_e)  <= soc_margin_tes
    tes_has_space = (tes_emax  - tes_e)  >  soc_margin_tes

    # --- Catena termica: può ancora assorbire PV? ---
    # (i) Serve direttamente l'edificio se c'è domanda e ALMENO uno dei due link verso Building non è saturo
    #     - TES->Building (Thermal Dispatch) oppure HeatSource->Building (Thermal Discharge)
    can_push_heat_to_build = (heat_load > tol) & (~(cap_TES_to_build & cap_HP_to_build))

    # (ii) Oppure può caricare TES se c'è spazio energetico e il link HeatSource->TES non è saturo
    can_charge_tes_more = tes_has_space & (~cap_HP_to_TES)

    heat_chain_can_absorb = can_push_heat_to_build | can_charge_tes_more
    heat_chain_blocked    = ~heat_chain_can_absorb

    # --- Batteria: può caricare? ---
    batt_has_space    = (batt_emax - batt_e) > soc_margin_batt
    batt_can_charge   = batt_has_space & (~cap_batt_charge)
    batt_full_or_cap  = ~batt_can_charge   # piena o limitata in potenza

    # --- Criterio Export GIUSTIFICATO ---
    # 1) nessuna domanda elettrica residua (entro tolleranza robusta)
    # 2) catena termica NON può assorbire altro
    # 3) batteria NON può essere caricata (piena o a cappio di potenza)
    export_justified = elec_ok & heat_chain_blocked & batt_full_or_cap

    # --- Report (solo ore con export) ---
    df = pd.DataFrame(index=s)
    df["export_mwh"]         = exp_net
    df["elec_residual_mwh"]  = elec_residual_after
    df["elec_eps_mwh"]       = elec_eps
    df["heat_load_mwh"]      = heat_load
    df["tes_e_mwh"]          = tes_e
    df["batt_e_mwh"]         = batt_e

    df["elec_ok"]               = elec_ok
    df["heat_chain_blocked"]    = heat_chain_blocked
    df["batt_full_or_cap"]      = batt_full_or_cap
    df["pv2b_at_cap"]           = cap_PV_to_build  # diagnostica

    df["export_justified"] = export_justified

    return df[df["export_mwh"] > tol]






    

#Prende in input l’oggetto rete post-ottimizzazione.
def calculate_summary(network) -> pd.DataFrame:
    # Pesi orari
    if not hasattr(network.snapshot_weightings, 'generators'):
        network.snapshot_weightings['generators'] = pd.Series(1.0, index=network.snapshots)
    w = network.snapshot_weightings.generators
    s = network.snapshots

    Z = pd.Series(0.0, index=s)

    # --- Flussi elettrici (MWh) ---
    imp_gross = network.links_t.p0.get('Grid Import', Z).clip(lower=0.0)
    imp_net   = (-network.links_t.p1.get('Grid Import', Z)).clip(lower=0.0)

    exp_col = 'PV Export' if 'PV Export' in network.links_t.p0.columns else None
    exp_gross = (network.links_t.p0.get(exp_col, Z)).clip(lower=0.0)
    exp_net   = ((-network.links_t.p1.get(exp_col, Z))).clip(lower=0.0)

    pv_to_build   = network.links_t.p0.get('PV Autoconsumo',  Z)
    pv_to_batt    = network.links_t.p0.get('Battery Charge',  Z)
    batt_to_build = network.links_t.p0.get('Battery Dispatch', Z)
    pv_to_hp_el   = network.links_t.p0.get('Heat Pump',       Z)  # input elettrico HP

    pv_gen   = network.generators_t.p.get('Rooftop PV', Z)

    # --- Flussi termici (MWh_th) ---
    hp_out_th     = (-network.links_t.p1.get('Heat Pump', Z)).clip(lower=0.0)  # già *COP
    hs_to_bld     = network.links_t.p0.get('Thermal Discharge', Z)
    hs_to_store   = network.links_t.p0.get('Thermal Charge',    Z)
    store_to_bld  = network.links_t.p0.get('Thermal Dispatch',  Z)
    dh_gen_th     = network.links_t.p0.get('Heat import', Z)  # per costi/CO2
    dh_to_build   = (-network.links_t.p1.get('Heat import', Z)).clip(lower=0.0)  # per i grafici
    heat_load     = network.loads_t.p_set.get('Building Heat Load', Z)
    effective_dh  = (heat_load - hs_to_bld - store_to_bld).clip(lower=0.0)

    # --- Emissioni & costi ---
    co2_grid_fac = network.links.at['Grid Import', 'co2_emission_factor_kg_per_mwh'] if 'Grid Import' in network.links.index else 0.0
    co2_dh_fac   = network.links.at['Heat import','co2_emission_factor_kg_per_mwh'] if 'Heat import'  in network.links.index else 0.0
    price_series = getattr(network, 'grid_price_series', pd.Series(0.0, index=s))

    co2_grid_kg  = ((imp_gross * co2_grid_fac) * w).sum()
    co2_dh_kg    = ((dh_gen_th * co2_dh_fac) * w).sum()
    co2_avoided  = ((exp_net * co2_grid_fac) * w).sum()

    grid_cost_eur = ((imp_gross * price_series) * w).sum()

    # --- Domande totali ---
    elec_load = network.loads_t.p_set.get('Building Elec Load', Z)
    total_elec = (elec_load * w).sum()
    total_heat = (heat_load * w).sum()

    return pd.DataFrame([{
        # elettrico
        'Total PV Generated (MWh)':               (pv_gen * w).sum(),
        'Total Grid Import (Net, MWh)':           (imp_net * w).sum(),
        'Total Grid Import (Gross, MWh)':         (imp_gross * w).sum(),
        'Total Grid Export (Net, MWh)':           (exp_net * w).sum(),
        'Total Grid Export (Gross, MWh)':         (exp_gross * w).sum(),
        'PV → Building Elec (MWh)':               (pv_to_build * w).sum(),
        'PV → Battery (MWh)':                     (pv_to_batt * w).sum(),
        'Battery → Building Elec (MWh)':          (batt_to_build * w).sum(),
        'PV → Heat Pump (MWh_el)':                (pv_to_hp_el * w).sum(),

        # termico
        'Heat Pump Production (MWh_th)':          (hp_out_th * w).sum(),
        'HeatSource → Building Heat (MWh_th)':    (hs_to_bld * w).sum(),
        'HeatSource → Thermal Storage (MWh_th)':  (hs_to_store * w).sum(),
        'Thermal Storage → Building (MWh_th)':    (store_to_bld * w).sum(),
        'Effective DH Import (MWh_th)':           (effective_dh * w).sum(),

        # domande
        'Total Elec Demand (MWh)':                total_elec,
        'Total Heat Demand (MWh_th)':             total_heat,

        # impatti/costi
        'CO2 Emissions Grid (kg)':                co2_grid_kg,
        'CO2 Emissions DH (kg)':                  co2_dh_kg,
        'CO2 Avoided on Export (kg)':             co2_avoided,
        'Grid Cost (EUR)':                        grid_cost_eur,
        'Total Operational Cost (EUR)':           grid_cost_eur + ((dh_gen_th * w).sum() * float(network.generators.at['Heating','marginal_cost']) if 'Heating' in network.generators.index else 0.0),
    }])



######La funzione generate_hourly_df(network) → pd.DataFrame costruisce un DataFrame “orario” che mette insieme, per ciascun timestamp (snapshot), i principali flussi di energia in mwh
def generate_hourly_df(network) -> pd.DataFrame:
    s = network.snapshots
    Z = pd.Series(0.0, index=s)

    # carichi
    elec = network.loads_t.p_set.get('Building Elec Load', Z)
    heat = network.loads_t.p_set.get('Building Heat Load', Z)

    # PV
    pv = network.generators_t.p.get('Rooftop PV', Z)

    # import/export (net/gross)
    imp_gross = network.links_t.p0.get('Grid Import', Z).clip(lower=0.0)
    imp_net   = (-network.links_t.p1.get('Grid Import', Z)).clip(lower=0.0)

    exp_col   = 'PV Export' if 'PV Export' in network.links_t.p0.columns else None
    exp_gross = (network.links_t.p0.get(exp_col, Z)).clip(lower=0.0)
    exp_net   = ((-network.links_t.p1.get(exp_col, Z))).clip(lower=0.0)

    # breakdown elettrico
    pv_to_build   = network.links_t.p0.get('PV Autoconsumo',  Z)
    pv_to_batt    = network.links_t.p0.get('Battery Charge',  Z)
    batt_to_build = network.links_t.p0.get('Battery Dispatch', Z)
    pv_to_hp_el   = network.links_t.p0.get('Heat Pump',       Z)

    # breakdown termico
    hp_out_th     = (-network.links_t.p1.get('Heat Pump', Z)).clip(lower=0.0)
    hs_to_bld     = network.links_t.p0.get('Thermal Discharge', Z)
    hs_to_store   = network.links_t.p0.get('Thermal Charge',    Z)
    store_to_bld  = network.links_t.p0.get('Thermal Dispatch',  Z)
    dh_gen_th     = network.links_t.p0.get('Heat import', Z)
    dh_to_build   = (-network.links_t.p1.get('Heat import', Z)).clip(lower=0.0)    
    effective_dh  = (heat - hs_to_bld - store_to_bld).clip(lower=0.0)

    # SoC
    battery_soc = network.stores_t.e.get('Battery Store', pd.Series(index=s, dtype=float))
    thermal_soc = network.stores_t.e.get('Thermal Store', pd.Series(index=s, dtype=float))

    # Emissioni orarie
    co2_grid_fac = network.links.at['Grid Import', 'co2_emission_factor_kg_per_mwh'] if 'Grid Import' in network.links.index else 0.0
    co2_dh_fac   = network.links.at['Heat import','co2_emission_factor_kg_per_mwh'] if 'Heat import'  in network.links.index else 0.0

    df = pd.DataFrame({
        'timestamp': s,
        'electricity_load_mwh':  elec.values,
        'heat_load_mwh':         heat.values,
        'pv_generation_mwh':     pv.values,

        'grid_import_mwh':       imp_net.values,
        'grid_import_gross_mwh': imp_gross.values,
        'grid_export_mwh':       exp_net.values,
        'grid_export_gross_mwh': exp_gross.values,

        'pv_to_building_mwh':    pv_to_build.values,
        'pv_to_battery_mwh':     pv_to_batt.values,
        'battery_to_building_mwh': batt_to_build.values,
        'pv_to_heat_pump_mwh_el':  pv_to_hp_el.values,

        'hp_output_mwh_th':        hp_out_th.values,
        'hs_to_building_mwh_th':   hs_to_bld.values,
        'hs_to_storage_mwh_th':    hs_to_store.values,
        'storage_to_building_mwh_th': store_to_bld.values,
        'dh_gen_mwh_th':           dh_gen_th.values,
        'dh_to_building_mwh_th':   dh_to_build.values,        
        'effective_dh_import_mwh_th': effective_dh.values,
        'battery_soc_mwh':        battery_soc.values if battery_soc is not None else None,
        'thermal_soc_mwh':        thermal_soc.values if thermal_soc is not None else None,

        'grid_import_emissions_kg': (imp_gross * co2_grid_fac).values,
        'dh_emissions_kg':          (dh_gen_th * co2_dh_fac).values,

        'grid_price_eur_per_mwh':   getattr(network, 'grid_price_series', pd.Series(0.0, index=s)).values,
    })
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

    # --- Validazione delle priorità (net/out_dir/scenario_name esistono qui) ---
    # Prima di chiamare i checker, valuta scala tipica (es. mediana del carico elettrico)
    scale = float(net.loads_t.p_set.get("Building Elec Load", pd.Series(0.0, index=net.snapshots)).abs().median() or 1.0)
    tol = max(1e-4, 1e-4 * scale)  # p.es. 0.01% della mediana, ma mai sotto 1e-4


    try:
        problems = validate_priorities(net, tol=tol)
        # salva un piccolo report CSV (facoltativo ma utile)
        pd.DataFrame(
            [(k, v[0], ";".join(map(str, v[1]))) for k, v in problems.items()],
            columns=["check", "violations", "examples"]
        ).to_csv(out_dir / f"{scenario_name}_priority_checks.csv", index=False)
    except Exception as e:
        print(f"Priority validation error: {e}")
    # --- Certificatore di export: tabella oraria con le cause ---
    try:
        df_export = explain_export(net, tol=tol)

        # Salva tabella completa (solo ore con export)
        csv_path  = out_dir / f"{scenario_name}_export_explanations.csv"
        html_path = out_dir / f"{scenario_name}_export_explanations.html"
        df_export.to_csv(csv_path)
        df_export.to_html(str(html_path))

        # Riassunto cause (conteggi e share)
        cause_cols = [
    "elec_ok",
    "heat_chain_blocked",
    "batt_full_or_cap",
    "pv2b_at_cap",  # opzionale: solo diagnostica
]

        if not df_export.empty:
            summary = (
                df_export[cause_cols]
                .sum()
                .sort_values(ascending=False)
                .rename("count")
                .to_frame()
            )
            summary["share"] = summary["count"] / len(df_export)
            summary_path = out_dir / f"{scenario_name}_export_causes_summary.csv"
            summary.to_csv(summary_path)

            # Eventuali export “non giustificati”
            unjustified = df_export[~df_export["export_justified"]]
            unjust_path = out_dir / f"{scenario_name}_export_unjustified.csv"
            unjustified.to_csv(unjust_path)

            print(f"[export checker] ore con export: {len(df_export)}")
            print(f"[export checker] tabella: {csv_path}")
            print(f"[export checker] riepilogo cause: {summary_path}")
            if not unjustified.empty:
                 print(f"[export checker] ATTENZIONE: export non giustificato in {len(unjustified)} ore -> {unjust_path}")
            else:
                 print("[export checker] nessun export non giustificato rilevato.")
        else:
            print("[export checker] nessuna ora con export > tol.")
    except Exception as e:
        print(f"[export checker] errore durante la generazione del report export: {e}")





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

        # --- breakdown 2×2 dello energy balance ---
        # nomi coerenti con generate_hourly_df(...)
        hp_total   = hourly_df['hp_output_mwh_th'].sum()
        dh_total   = hourly_df['dh_to_building_mwh_th'].sum()      # <-- usa la colonna "to building"
        storage_ch = hourly_df['hs_to_storage_mwh_th'].sum()
        storage_dc = hourly_df['storage_to_building_mwh_th'].sum()

        plot_energy_balance_breakdown({
            'pv_generation_mwh':           hourly_df['pv_generation_mwh'].sum(),
            'grid_import_mwh':             hourly_df['grid_import_mwh'].sum(),
            'grid_export_mwh':             hourly_df['grid_export_mwh'].sum(),
            'total_electric_load_mwh':     hourly_df['electricity_load_mwh'].sum(),
            'total_thermal_load_mwh':      hourly_df['heat_load_mwh'].sum(),
            'Heat Pump Production (MWh_th)':  hp_total,
            'DH Import to Building (MWh_th)': dh_total,            # <-- niente w qui
            'Thermal Store Charge (MWh_th)': storage_ch,
            'Thermal Store Discharge (MWh_th)': storage_dc,
            'co2_emissions_kg':            hourly_df['grid_import_emissions_kg'].sum(),
            'dh_emissions_kg':             hourly_df['dh_emissions_kg'].sum(),
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
