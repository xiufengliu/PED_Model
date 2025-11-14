#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stacked bar charts (più leggibili) per:
  1) Import  = Import Elettrico + Import Termico
  2) Costi   = Costo Elettrico + Costo Termico (+ Ricavi Export come negativi)
  3) Emissioni = CO2 elettricità + CO2 teleriscaldamento
  4) SS sistema = quota % elettrica + quota % termica

Assume Excel con: righe=KPI, colonne=scenari, prima colonna = nome KPI.
"""

import argparse, math, re, unicodedata
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------- util ----------
def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", " ", s).strip().lower()
    return s

def find_row(df: pd.DataFrame, candidates: list[str]):
    idx_norm = {i: normalize(str(i)) for i in df.index}
    for cand in candidates:
        pat = re.compile(cand)
        for raw, normed in idx_norm.items():
            if pat.search(normed):
                return df.loc[raw]
    return None

def to_num(s: pd.Series | None):
    if s is None: return None
    return pd.to_numeric(s, errors="coerce")

def fmt(v, unit=None):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    if unit == "%": return f"{v:,.0f}%"
    if unit == "€": return f"€{v:,.0f}"
    return f"{v:,.2f}"

def hstack(ax, stacks: list[tuple[str, pd.Series]], title: str, unit: str|None=None):
    """barre orizzontali impilate, legenda fuori a destra."""
    if not stacks:
        ax.set_title(f"{title} (dati mancanti)")
        return
    scenarios = list(stacks[0][1].index.astype(str))

    left = pd.Series(0.0, index=scenarios, dtype=float)
    handles = []
    for label, series in stacks:
        vals = series.reindex(scenarios).astype(float).fillna(0.0)
        h = ax.barh(scenarios, vals.values, left=left.values)
        handles.append((h, label, vals))
        left = left + vals

    ax.xaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, pad=8, fontsize=14)
    ax.set_xlabel(f"Valore{f' [{unit}]' if unit else ''}")

    # etichette valori al termine di ciascuna fetta
    for h, _lab, vals in handles:
        for rect, v in zip(h, vals.values):
            if v == 0 or math.isnan(v): continue
            ax.annotate(
                fmt(v, "€" if unit=="€" else None),
                xy=(rect.get_x() + rect.get_width(), rect.get_y() + rect.get_height()/2),
                xytext=(6, 0),
                textcoords="offset points",
                va="center", ha="left", fontsize=9
            )

    ax.legend([h for h,_,_ in handles], [lab for _,lab,_ in handles],
              loc="center left", bbox_to_anchor=(1.0, 0.5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("excel")
    ap.add_argument("--sheet", default=None, help="nome o indice (default: primo)")
    ap.add_argument("--out", default="data/output/kpi_stacked_clean")
    ap.add_argument("--load-el", type=float, default=None)
    ap.add_argument("--load-th", type=float, default=None)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--w", type=float, default=12.0)
    ap.add_argument("--h", type=float, default=7.0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sheet = 0 if args.sheet is None else args.sheet
    df = pd.read_excel(args.excel, sheet_name=sheet)
    df = df.set_index(df.columns[0])
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    scenarios = list(df.columns.astype(str))

    # ----- pattern robusti (su stringhe normalizzate) -----
    pat_import_el = [r"\bimport\b.*\belett"]
    pat_import_th = [r"\bimport\b.*\bterm", r"\bimport\b.*\bdh\b", r"\bheat\b.*\bimport\b"]

    pat_cost_el   = [r"\bcost\b.*\belett", r"\bcosto\b.*\belett"]
    pat_cost_th   = [r"\bcost\b.*\bterm", r"\bcosto\b.*\bterm"]
    pat_rev_exp   = [r"\bricavi\b.*\bexport\b", r"\brevenue.*\bexport\b"]

    pat_co2_el    = [r"\bemissioni\b.*\belett", r"\bco2\b.*\belett"]
    pat_co2_th    = [r"\bemissioni\b.*\bterm", r"\bco2\b.*\b(th|dh)\b"]

    pat_load_el   = [r"\bload\b.*\belett"]
    pat_load_th   = [r"\bload\b.*\bterm", r"\bheat\b.*\bload\b"]

    # ----- estrazione -----
    import_el = to_num(find_row(df, pat_import_el))
    import_th = to_num(find_row(df, pat_import_th))

    cost_el   = to_num(find_row(df, pat_cost_el))
    cost_th   = to_num(find_row(df, pat_cost_th))
    rev_exp   = to_num(find_row(df, pat_rev_exp))

    co2_el    = to_num(find_row(df, pat_co2_el))
    co2_th    = to_num(find_row(df, pat_co2_th))

    load_el   = to_num(find_row(df, pat_load_el))
    load_th   = to_num(find_row(df, pat_load_th))

    if load_el is None and args.load_el is not None:
        load_el = pd.Series({s: args.load_el for s in scenarios})
    if load_th is None and args.load_th is not None:
        load_th = pd.Series({s: args.load_th for s in scenarios})

    # ----- PDF con tutte le figure -----
    pdf = PdfPages(out / "kpi_stacked_clean.pdf")

    # 1) Import
    if (import_el is not None) or (import_th is not None):
        fig = plt.figure(figsize=(args.w, args.h)); ax = plt.gca()
        stacks = []
        if import_el is not None: stacks.append(("Import Elettrico (MWh)", import_el.reindex(scenarios)))
        if import_th is not None: stacks.append(("Import Termico (MWh_th)", import_th.reindex(scenarios)))
        hstack(ax, stacks, "Import totali (stack)", "MWh / MWh_th")
        plt.tight_layout(); fig.savefig(out/"stack_import.png", dpi=args.dpi, bbox_inches="tight"); pdf.savefig(fig); plt.close(fig)
    else:
        print("⚠️ Import: KPI non trovati (controlla nomi).")

    # 2) Costi (ricavi export negativi)
    if (cost_el is not None) or (cost_th is not None) or (rev_exp is not None):
        fig = plt.figure(figsize=(args.w, args.h)); ax = plt.gca()
        stacks = []
        if cost_el is not None: stacks.append(("Costo Elettrico (€)", cost_el.reindex(scenarios)))
        if cost_th is not None: stacks.append(("Costo Termico (€)", cost_th.reindex(scenarios)))
        if rev_exp is not None: stacks.append(("Ricavi Export (€)", (-rev_exp).reindex(scenarios)))
        hstack(ax, stacks, "Costi (stack, ricavi export negativi)", "€")
        plt.tight_layout(); fig.savefig(out/"stack_costi.png", dpi=args.dpi, bbox_inches="tight"); pdf.savefig(fig); plt.close(fig)
    else:
        print("⚠️ Costi: KPI non trovati.")

    # 3) Emissioni
    if (co2_el is not None) or (co2_th is not None):
        fig = plt.figure(figsize=(args.w, args.h)); ax = plt.gca()
        stacks = []
        if co2_el is not None: stacks.append(("CO₂ Elettricità (t)", co2_el.reindex(scenarios)))
        if co2_th is not None: stacks.append(("CO₂ Teleriscaldamento (t)", co2_th.reindex(scenarios)))
        hstack(ax, stacks, "Emissioni (stack)", "t")
        plt.tight_layout(); fig.savefig(out/"stack_emissioni.png", dpi=args.dpi, bbox_inches="tight"); pdf.savefig(fig); plt.close(fig)
    else:
        print("⚠️ Emissioni: KPI non trovati. Controlla che le righe si chiamino tipo 'Emissioni CO2 Elettricità (t)' e 'Emissioni CO2 Teleriscaldamento (t)'.")

    # 4) SS di sistema (decomposta)
    if (load_el is not None) and (import_el is not None) and (load_th is not None) and (import_th is not None):
        eps = 1e-12
        ss_el = ((load_el - import_el) / load_el.replace(0, eps)) * 100.0
        ss_th = ((load_th - import_th) / load_th.replace(0, eps)) * 100.0
        tot_load = (load_el.fillna(0) + load_th.fillna(0)).replace(0, eps)

        comp_el = ss_el * (load_el.fillna(0) / tot_load)
        comp_th = ss_th * (load_th.fillna(0) / tot_load)

        fig = plt.figure(figsize=(args.w, args.h)); ax = plt.gca()
        stacks = [("SS elettrica (quota % su totale)", comp_el.reindex(scenarios)),
                  ("SS termica (quota % su totale)",   comp_th.reindex(scenarios))]
        hstack(ax, stacks, "Self-Sufficiency di sistema (decomposta, %)", "%")
        plt.tight_layout(); fig.savefig(out/"stack_ss_sistema.png", dpi=args.dpi, bbox_inches="tight"); pdf.savefig(fig); plt.close(fig)
    else:
        print("⚠️ SS sistema: servono Load_el, Import_el, Load_th, Import_th (usa --load-el/--load-th se mancano nel file).")

    pdf.close()
    print(f"✅ Salvati PNG + PDF in: {out.resolve()}")

if __name__ == "__main__":
    main()
