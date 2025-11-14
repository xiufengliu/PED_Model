# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Iterable, Optional
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ helpers numerici ------------------
def annuity_factor(r: float, n: float) -> float:
    r = float(r); n = float(n)
    if n <= 0: return 1.0
    if abs(r) < 1e-12: return 1.0 / n
    return r/(1.0 - (1.0+r)**(-n))

def annualized_from_upfront(upfront_eur: float, r: float, n_years: float) -> float:
    return float(upfront_eur) * annuity_factor(r, n_years)

def npv_of_stream(initial: float, annual: float, years: int, r: float) -> float:
    pv = initial
    for t in range(1, years+1):
        pv += annual/((1+r)**t)
    return pv

# ------------------ grafica base ------------------
def _bar_labels(ax, xlabels, values, unit: str = "", title: Optional[str] = None):
    xs = np.arange(len(xlabels))
    bars = ax.bar(xs, values)
    for i, b in enumerate(bars):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{values[i]:,.2f} {unit}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(xlabels, rotation=15, ha="right")
    if title: ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)

# ------------------ energy plots minimi ------------------
def plot_time_series(series: Dict[str, Iterable], scenario_name: str, output_dir: str):
    # opzionale: non lo usiamo nella card
    pass

def plot_energy_balance(vals: Dict[str,float], scenario_name: str, output_dir: str):
    labels = ["PV","Import rete","Export rete","Carico elettrico","Carico termico","CO₂ (kg)"]
    data = [
        vals.get('pv_generation_mwh',0.0),
        vals.get('grid_import_mwh',0.0),
        vals.get('grid_export_mwh',0.0),
        vals.get('total_electric_load_mwh',0.0),
        vals.get('total_thermal_load_mwh',0.0),
        vals.get('co2_emissions_kg',0.0),
    ]
    fig, ax = plt.subplots(figsize=(11,5))
    _bar_labels(ax, labels, data, unit="MWh / kg", title=f"Energy Balance — {scenario_name}")
    fig.tight_layout(); fig.savefig(f"{output_dir}/{scenario_name}_energy_balance.png"); plt.close(fig)

# ------------------ preparazione finanza ------------------
def prepare_financials(investment_df: pd.DataFrame,
                       summary_row: pd.DataFrame,
                       baseline_summary_row: Optional[pd.Series],
                       discount_rate: float, project_years: int,
                       lifetime_by_tech: Optional[dict]) -> tuple[pd.DataFrame, Dict[str,float]]:
    """
    Ritorna:
      pertech: CAPEX_upfront_EUR, CAPEX_ann_EUR, Simple_Payback_years (se baseline)
      kpis:    discount_rate, project_years, Total_CAPEX_upfront_EUR, Annual_OPEX_EUR,
               Project_NPV_EUR, Annual_Savings_vs_Baseline_EUR, Simple_Payback_total_years
    """
    pertech = investment_df.copy()
    if "Technology" not in pertech.columns:
        # rinomina la colonna tecnologia se necessario
        pertech = pertech.rename(columns={"technology":"Technology"})

    # CAPEX annualizzato già pronto nella colonna CAPEX_ann_EUR
    ann_opex = float(summary_row["Total Operational Cost (EUR)"].iloc[0])
    total_capex_ann = float(pertech["CAPEX_ann_EUR"].sum())

    # Se nel config capital_cost era upfront, l'investment_df lo ha già annualizzato,
    # ma qui vogliamo anche stimare l'UPFRONT equivalente per NPV e payback:
    # per semplicità stimiamo upfront = CAPEX_ann * (1/AF) per ogni tech, usando lifetime_by_tech.
    pertech["CAPEX_upfront_EUR"] = np.nan
    if lifetime_by_tech:
        pertech["CAPEX_upfront_EUR"] = pertech.apply(
            lambda r: (r["CAPEX_ann_EUR"] / annuity_factor(discount_rate,
                                                           lifetime_by_tech.get(_tech_key(r["Technology"]), 20))),
            axis=1
        )
    # se non abbiamo lifetimes, assumiamo che CAPEX_ann fosse già annualizzato da un upfront noto -> non stimiamo upfront
    total_upfront = float(pertech["CAPEX_upfront_EUR"].fillna(0.0).sum())

    # baseline → risparmi annui
    savings_ann = None
    if baseline_summary_row is not None:
        base_total = float(baseline_summary_row.get("Total Annual System Cost (EUR)",
                                                    baseline_summary_row.get("Total Operational Cost (EUR)", 0.0)))
        scen_total = float(summary_row.get("Total Annual System Cost (EUR)",
                                           summary_row.get("Total Operational Cost (EUR)", 0.0)))
        savings_ann = max(base_total - scen_total, 0.0)

    # NPV (costi): upfront + OPEX scontati
    npv_project = npv_of_stream(initial=-(total_upfront if total_upfront>0 else total_capex_ann),
                                annual=-ann_opex, years=project_years, r=discount_rate)

    # Payback semplice
    if savings_ann and savings_ann > 0 and total_upfront > 0:
        pertech["Savings_attributed_EUR_per_year"] = (pertech["CAPEX_upfront_EUR"] / total_upfront) * savings_ann
        pertech["Simple_Payback_years"] = pertech["CAPEX_upfront_EUR"] / pertech["Savings_attributed_EUR_per_year"]
        simple_payback_total = total_upfront / savings_ann
    else:
        pertech["Savings_attributed_EUR_per_year"] = np.nan
        pertech["Simple_Payback_years"] = np.nan
        simple_payback_total = np.nan

    kpis = {
        "discount_rate": discount_rate,
        "project_years": int(project_years),
        "Total_CAPEX_upfront_EUR": float(total_upfront if total_upfront>0 else 0.0),
        "Annual_OPEX_EUR": float(ann_opex),
        "Project_NPV_EUR": float(npv_project),
        "Annual_Savings_vs_Baseline_EUR": float(savings_ann) if savings_ann is not None else float("nan"),
        "Simple_Payback_total_years": float(simple_payback_total) if not math.isnan(simple_payback_total) else float("nan"),
    }
    return pertech, kpis

def _tech_key(label: str) -> str:
    l = label.lower()
    if "pv" in l: return "pv"
    if "heat pump" in l: return "hp"
    if "transformer" in l: return "transformer"
    if "export" in l: return "export"
    if "tes" in l: return "tes"
    if "battery" in l and "energy" in l: return "battery"  # accorpiamo
    if "battery" in l and "power" in l: return "battery"
    return l

# --------- grafici investimento (NPV, payback, CAPEX, dimensionamento) ---------
def plot_capex_upfront_bar(pertech: pd.DataFrame, scenario: str, out: str, years: int, r: float):
    fig, ax = plt.subplots(figsize=(12,5))
    _bar_labels(ax, pertech["Technology"], pertech["CAPEX_upfront_EUR"]/1e3, unit="k€",
                title=f"CAPEX upfront per tecnologia — {scenario}  (project={years}y, r={r:.2%})")
    fig.tight_layout(); fig.savefig(f"{out}/{scenario}_capex_upfront_by_tech.png"); plt.close(fig)

def plot_capex_ann_bar(investment_df: pd.DataFrame, scenario: str, out: str, years: int, r: float):
    fig, ax = plt.subplots(figsize=(12,5))
    _bar_labels(ax, investment_df["Technology"], investment_df["CAPEX_ann_EUR"]/1e3, unit="k€",
                title=f"CAPEX annualizzato per tecnologia — {scenario}  (project={years}y, r={r:.2%})")
    fig.tight_layout(); fig.savefig(f"{out}/{scenario}_capex_annualized_by_tech.png"); plt.close(fig)

def plot_investment_cost_pie(pertech: pd.DataFrame, scenario: str, out: str, years: int, r: float):
    fig, ax = plt.subplots(figsize=(7,7))
    vals = pertech["CAPEX_upfront_EUR"].values
    labels = pertech["Technology"].values
    if np.isclose(vals.sum(), 0): vals = np.ones_like(vals)
    ax.pie(vals, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title(f"Ripartizione CAPEX upfront — {scenario}  (project={years}y, r={r:.2%})")
    fig.tight_layout(); fig.savefig(f"{out}/{scenario}_capex_upfront_pie.png"); plt.close(fig)

def plot_payback_by_tech(pertech: pd.DataFrame, scenario: str, out: str, years: int, r: float):
    if pertech["Simple_Payback_years"].isna().all(): return
    fig, ax = plt.subplots(figsize=(12,5))
    _bar_labels(ax, pertech["Technology"], pertech["Simple_Payback_years"], unit="years",
                title=f"Payback semplice per tecnologia — {scenario}  (project={years}y, r={r:.2%})")
    fig.tight_layout(); fig.savefig(f"{out}/{scenario}_payback_by_tech.png"); plt.close(fig)

def plot_npv_waterfall(project_kpis: Dict[str,float], scenario: str, out: str):
    r = float(project_kpis["discount_rate"]); Y = int(project_kpis["project_years"])
    capex0 = float(project_kpis["Total_CAPEX_upfront_EUR"]); ann_opex = float(project_kpis["Annual_OPEX_EUR"])
    pv_opex = sum([ann_opex / ((1+r)**t) for t in range(1, Y+1)])
    has_base = not math.isnan(project_kpis.get("Annual_Savings_vs_Baseline_EUR", float("nan")))

    steps = []; labels = []
    if has_base:
        s = project_kpis["Annual_Savings_vs_Baseline_EUR"]
        base_ann = ann_opex + s
        pv_base_opex = sum([base_ann / ((1+r)**t) for t in range(1, Y+1)])
        steps = [pv_base_opex, -(pv_base_opex - pv_opex), capex0, pv_opex + capex0]
        labels= ["PV OPEX baseline (20y)", "PV risparmi (20y)", "CAPEX upfront", "PV OPEX+CAPEX scenario"]
    else:
        steps = [pv_opex, capex0, pv_opex + capex0]; labels = ["PV OPEX (20y)", "CAPEX upfront", "PV costi scenario (20y)"]

    fig, ax = plt.subplots(figsize=(12,5))
    bars = ax.bar(range(len(steps)), steps)
    for i, b in enumerate(bars):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{steps[i]/1e3:,.1f} k€",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(steps))); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title(f"NPV waterfall — {scenario}  (project={Y}y, r={r:.2%})")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(f"{out}/{scenario}_npv_waterfall.png"); plt.close(fig)

def plot_capacity_sticks(investment_df: pd.DataFrame, scenario: str, out: str):
    caps = investment_df[["Technology","Capacity"]].copy()
    fig, ax = plt.subplots(figsize=(11,5))
    _bar_labels(ax, caps["Technology"], caps["Capacity"], unit="MW / MWh",
                title=f"Dimensionamento ottimo — {scenario}")
    fig.tight_layout(); fig.savefig(f"{out}/{scenario}_optimal_capacities.png"); plt.close(fig)

def plot_finance_summary_card(project_kpis: Dict[str,float], scenario: str, out: str):
    r = project_kpis["discount_rate"]; Y = int(project_kpis["project_years"])
    capex0 = float(project_kpis.get("Total_CAPEX_upfront_EUR", 0.0))
    ann_opex = float(project_kpis.get("Annual_OPEX_EUR", 0.0))
    npv = float(project_kpis.get("Project_NPV_EUR", float("nan")))
    savings = project_kpis.get("Annual_Savings_vs_Baseline_EUR", float("nan"))
    payback = project_kpis.get("Simple_Payback_total_years", float("nan"))

    txt = [
        f"Project horizon: {Y} years   –   Discount rate: {r:.2%}",
        f"CAPEX upfront: {capex0:,.0f} €",
        f"Annual OPEX: {ann_opex:,.0f} € / year",
        f"NPV (present value of CAPEX+OPEX): {npv:,.0f} €",
        f"Annual savings vs baseline: {('n/a' if math.isnan(savings) else f'{savings:,.0f} € / year')}",
        f"Simple payback (total): {('n/a' if math.isnan(payback) else f'{payback:.1f} years')}",
    ]
    fig, ax = plt.subplots(figsize=(10,4)); ax.axis("off")
    ax.text(0.02, 0.95, f"Financial KPIs — {scenario}", fontsize=14, weight="bold", va="top")
    ax.text(0.02, 0.82, "\n".join(txt), fontsize=12, va="top")
    fig.tight_layout(); fig.savefig(f"{out}/{scenario}_finance_summary.png"); plt.close(fig)

# ================= Publication-quality extras (invest_opt) =================
def plot_economics_dashboard(economics: Dict, scenario_name: str, output_dir: str):
    # ... (puoi lasciare intatto quello che già avevi, o usare la tua versione preferita)
    pass

def plot_npv_sensitivity(economics: Dict, scenario_name: str, output_dir: str):
    # ... (idem; opzionale)
    pass
