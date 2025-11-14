#!/usr/bin/env python3
"""
Pretty KPI plots from an Excel file where rows are KPIs and columns are scenarios.

Features
- Keeps column order from Excel as scenario order.
- Smart unit/label formatting (%, €, MWh, MW, t).
- Value labels on bars + horizontal grid.
- Saves one PNG per KPI + a single PDF with all charts.
- Optional heatmap summary (z-score per KPI).
- Optional sheet selection.

Usage
  python plot_kpis_pretty.py <excel_path> [--sheet SHEETNAME] [--out OUTDIR] [--no-heatmap]

Examples
  python plot_kpis_pretty.py data/input/timeseries/KPIs.xlsx
  python plot_kpis_pretty.py KPIs.xlsx --sheet KPI_2025 --out data/output/kpi_charts_pretty
"""

import argparse
import math, re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("excel", help="Path to the Excel file")
    p.add_argument("--sheet", default=None, help="Sheet name or index (default: first)")
    p.add_argument("--out", default="data/output/kpi_charts_pretty", help="Output directory")
    p.add_argument("--no-heatmap", action="store_true", help="Skip heatmap summary")
    return p.parse_args()

KPI_FORMATS = [
    (r"(?i)self[-_\s]*suff",    {"unit": "%",    "fmt": "{:,.0f}%", "scale": 1.0, "is_pct": True}),
    (r"(?i)self[-_\s]*cons",    {"unit": "%",    "fmt": "{:,.0f}%", "scale": 1.0, "is_pct": True}),
    (r"(?i)emissioni|CO2|CO₂",  {"unit": "t",    "fmt": "{:,.0f}",  "scale": 1.0, "is_pct": False}),
    (r"(?i)picco.*elettric",    {"unit": "MW",   "fmt": "{:,.2f}",  "scale": 1.0, "is_pct": False}),
    (r"(?i)picco.*termic",      {"unit": "MW_th","fmt": "{:,.2f}",  "scale": 1.0, "is_pct": False}),
    (r"(?i)MWh_th|termic",      {"unit": "MWh_th","fmt": "{:,.0f}", "scale": 1.0, "is_pct": False}),
    (r"(?i)MWh|elettric|PV",    {"unit": "MWh",  "fmt": "{:,.0f}",  "scale": 1.0, "is_pct": False}),
    (r"(?i)costo|ricavi|€|eur", {"unit": "€",    "fmt": "€{:,.0f}", "scale": 1.0, "is_pct": False}),
]

def infer_format(kpi_name: str):
    for pattern, props in KPI_FORMATS:
        if re.search(pattern, kpi_name):
            return props
    return {"unit": "", "fmt": "{:,.2f}", "scale": 1.0, "is_pct": False}

def number_fmt(v, fmt):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    try:
        return fmt.format(v)
    except Exception:
        return str(v)

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Read Excel
    sheet = 0 if args.sheet is None else args.sheet
    df = pd.read_excel(args.excel, sheet_name=sheet)
    first_col = df.columns[0]
    df = df.set_index(first_col)

    # 2) Numeric coercion
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")

    scenario_order = list(df.columns)

    # 3) Per-KPI plots + single PDF
    pdf_path = out_dir / "kpi_charts_pretty.pdf"
    saved = []

    with PdfPages(pdf_path) as pdf:
        for kpi_name, row in df.iterrows():
            s = row.reindex(scenario_order).dropna()
            if s.empty:
                continue

            props = infer_format(str(kpi_name))
            values = s.values.astype(float) * props["scale"]
            labels = s.index.astype(str)

            plt.figure(figsize=(11, 6))
            ax = plt.gca()
            bars = ax.bar(labels, values)
            ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)
            ax.set_axisbelow(True)
            ax.set_title(str(kpi_name), pad=8, fontsize=14)
            y_unit = props["unit"]
            ax.set_ylabel(f"Value{(' ['+y_unit+']') if y_unit else ''}")
            ax.set_xlabel("Scenario")
            plt.xticks(rotation=35, ha="right")

            for rect, v in zip(bars, values):
                if not (isinstance(v, float) and math.isnan(v)):
                    ax.annotate(
                        number_fmt(v, props["fmt"]),
                        xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9
                    )

            plt.tight_layout()
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(kpi_name))[:90]
            out_png = out_dir / f"{safe}.png"
            plt.savefig(out_png, dpi=200, bbox_inches="tight")
            pdf.savefig()
            plt.close()
            saved.append(out_png)

    # 4) Optional heatmap (z-score per KPI)
    if not args.no_heatmap:
        norm = df.copy()
        for idx in norm.index:
            row = norm.loc[idx]
            mu = row.mean(skipna=True)
            sd = row.std(skipna=True)
            if sd and sd > 0:
                norm.loc[idx] = (row - mu) / sd
            else:
                norm.loc[idx] = 0.0

        plt.figure(figsize=(max(8, len(df.columns)*0.6), max(6, len(df.index)*0.35)))
        im = plt.imshow(norm.values, aspect='auto')
        plt.colorbar(im, fraction=0.02, pad=0.02)
        plt.xticks(range(len(norm.columns)), norm.columns, rotation=35, ha="right")
        plt.yticks(range(len(norm.index)), norm.index)
        plt.title("KPI Heatmap (z-score per KPI)")
        plt.tight_layout()
        heatmap_path = out_dir / "kpi_heatmap.png"
        plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Saved PNGs + PDF in: {out_dir.resolve()}")
    print(f"Scenari: {', '.join(scenario_order)}")
    print(f"KPIs: {len(df.index)}")

if __name__ == "__main__":
    main()
