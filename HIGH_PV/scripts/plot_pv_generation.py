#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per generare grafici che mostrano l'influenza dei parametri meteorologici
sulla generazione fotovoltaica.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Set chart style
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Data paths
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
INPUT_PATH = os.path.join(DATA_PATH, 'input', 'timeseries')
OUTPUT_PATH = os.path.join(DATA_PATH, 'output', 'plots')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_pv_generation():
    """Load the photovoltaic generation profile."""
    filepath = os.path.join(INPUT_PATH, 'solar_pv_generation.csv')
    try:
        df = pd.read_csv(filepath)
        # Convert the first column to datetime
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        # Set the first column as index
        df.set_index(df.columns[0], inplace=True)
        # Rename the remaining column
        df.columns = ['PV Generation (normalized)']
        return df
    except Exception as e:
        print(f"Error loading PV profile: {e}")
        return None

def create_synthetic_weather_data(pv_data):
    """
    Crea dati meteorologici sintetici basati sul profilo di generazione PV.
    Questo è necessario perché non abbiamo i dati meteorologici reali.
    """
    # Crea un DataFrame con lo stesso indice del profilo PV
    weather = pd.DataFrame(index=pv_data.index)

    # Estrai mese e ora dal timestamp
    weather['Month'] = weather.index.month
    weather['Hour'] = weather.index.hour

    # Irradiazione solare (correlata direttamente con la generazione PV)
    weather['Solar Irradiance (W/m²)'] = pv_data['PV Generation (normalized)'] * 1000

    # Temperatura (più alta in estate, più bassa in inverno, picco nel pomeriggio)
    base_temp = np.sin(np.pi * (weather['Month'] - 1) / 12) * 15 + 10  # Oscillazione annuale
    daily_temp = np.sin(np.pi * (weather['Hour'] - 6) / 12) * 5  # Oscillazione giornaliera
    weather['Temperature (°C)'] = base_temp + daily_temp

    # Velocità del vento (casuale ma con stagionalità)
    wind_base = np.sin(np.pi * (weather['Month'] - 1) / 6) * 2 + 4  # Più vento in inverno
    weather['Wind Speed (m/s)'] = wind_base + np.random.normal(0, 1, len(weather))
    weather['Wind Speed (m/s)'] = weather['Wind Speed (m/s)'].clip(lower=0)

    # Copertura nuvolosa (inversamente correlata con la generazione PV)
    max_irradiance = weather.groupby('Hour')['Solar Irradiance (W/m²)'].transform('max')
    weather['Cloud Cover (%)'] = (1 - weather['Solar Irradiance (W/m²)'] / max_irradiance) * 80
    weather['Cloud Cover (%)'] = weather['Cloud Cover (%)'].clip(lower=0, upper=100)

    # Umidità relativa (correlata con la copertura nuvolosa)
    weather['Relative Humidity (%)'] = weather['Cloud Cover (%)'] * 0.5 + 40 + np.random.normal(0, 5, len(weather))
    weather['Relative Humidity (%)'] = weather['Relative Humidity (%)'].clip(lower=0, upper=100)

    # Precipitazioni (solo quando la copertura nuvolosa è alta)
    weather['Precipitation (mm)'] = np.where(weather['Cloud Cover (%)'] > 70,
                                            np.random.exponential(1, len(weather)), 0)

    # Rimuovi le colonne di supporto
    weather.drop(['Month', 'Hour'], axis=1, inplace=True)

    return weather

def plot_daily_profile(pv_data, weather_data, day):
    """
    Crea un grafico che mostra il profilo giornaliero di generazione PV
    e i parametri meteorologici per un giorno specifico.
    """
    # Filtra i dati per il giorno specificato
    day_start = pd.Timestamp(day)
    day_end = day_start + pd.Timedelta(days=1)
    pv_day = pv_data.loc[day_start:day_end]
    weather_day = weather_data.loc[day_start:day_end]

    # Crea la figura
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(4, 2, figure=fig)

    # Grafico principale: generazione PV
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(pv_day.index, pv_day['PV Generation (normalized)'], 'r-', linewidth=2)
    ax1.set_title(f'Profilo di generazione fotovoltaica - {day}')
    ax1.set_ylabel('Generazione normalizzata')
    ax1.set_xlim(day_start, day_end)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Irradiazione solare
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(weather_day.index, weather_day['Solar Irradiance (W/m²)'], 'orange', linewidth=2)
    ax2.set_title('Irradiazione solare')
    ax2.set_ylabel('W/m²')
    ax2.set_xlim(day_start, day_end)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Temperatura
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(weather_day.index, weather_day['Temperature (°C)'], 'g-', linewidth=2)
    ax3.set_title('Temperatura')
    ax3.set_ylabel('°C')
    ax3.set_xlim(day_start, day_end)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Velocità del vento
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(weather_day.index, weather_day['Wind Speed (m/s)'], 'b-', linewidth=2)
    ax4.set_title('Velocità del vento')
    ax4.set_ylabel('m/s')
    ax4.set_xlim(day_start, day_end)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Copertura nuvolosa
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(weather_day.index, weather_day['Cloud Cover (%)'], 'gray', linewidth=2)
    ax5.set_title('Copertura nuvolosa')
    ax5.set_ylabel('%')
    ax5.set_xlim(day_start, day_end)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Umidità relativa
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(weather_day.index, weather_day['Relative Humidity (%)'], 'c-', linewidth=2)
    ax6.set_title('Umidità relativa')
    ax6.set_ylabel('%')
    ax6.set_xlim(day_start, day_end)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Precipitazioni
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.bar(weather_day.index, weather_day['Precipitation (mm)'], color='blue', width=0.02)
    ax7.set_title('Precipitazioni')
    ax7.set_ylabel('mm')
    ax7.set_xlim(day_start, day_end)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f'pv_weather_day_{day}.png'), dpi=300)
    plt.close()

def plot_seasonal_comparison(pv_data, weather_data):
    """
    Crea un grafico che confronta la generazione PV e i parametri meteorologici
    per giorni rappresentativi di diverse stagioni.
    """
    # Seleziona giorni rappresentativi per ogni stagione
    winter_day = '2025-01-15'
    spring_day = '2025-04-15'
    summer_day = '2025-07-15'
    autumn_day = '2025-10-15'

    seasons = [
        ('Inverno', winter_day),
        ('Primavera', spring_day),
        ('Estate', summer_day),
        ('Autunno', autumn_day)
    ]

    # Crea la figura
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, (season, day) in enumerate(seasons):
        # Filtra i dati per il giorno specificato
        day_start = pd.Timestamp(day)
        day_end = day_start + pd.Timedelta(days=1)
        pv_day = pv_data.loc[day_start:day_end]

        # Plotta la generazione PV
        axes[i].plot(pv_day.index, pv_day['PV Generation (normalized)'], 'r-', linewidth=2)
        axes[i].set_title(f'{season} ({day})')
        axes[i].set_ylabel('Generazione PV normalizzata')
        axes[i].set_xlim(day_start, day_end)
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'pv_seasonal_comparison.png'), dpi=300)
    plt.close()

def plot_monthly_generation(pv_data):
    """
    Crea un grafico che mostra la generazione PV media mensile.
    """
    # Calcola la generazione media giornaliera per ogni mese
    pv_data['Month'] = pv_data.index.month
    pv_data['Day'] = pv_data.index.day
    monthly_data = pv_data.copy()
    monthly_avg = monthly_data.groupby('Month')['PV Generation (normalized)'].sum().reset_index()

    # Nomi dei mesi
    month_names = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']

    # Crea il grafico
    plt.figure(figsize=(12, 6))
    plt.bar(monthly_avg['Month'], monthly_avg['PV Generation (normalized)'], color='orange')
    plt.xticks(monthly_avg['Month'], month_names)
    plt.title('Generazione fotovoltaica mensile')
    plt.ylabel('Generazione cumulativa normalizzata')
    plt.xlabel('Mese')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'pv_monthly_generation.png'), dpi=300)
    plt.close()

def plot_baseline_vs_highpv():
    """
    Crea un grafico che confronta la generazione PV e i costi operativi tra gli scenari Baseline e High PV.
    Utilizza i valori esatti dai file di output.
    """
    # Valori esatti dai file di output
    baseline_import = 963.2845935510373  # MWh/anno
    highpv_import = 866.1122605038697  # MWh/anno

    baseline_pv = 64.1311737565145  # MWh/anno
    highpv_pv = 159.35536243754777  # MWh/anno

    baseline_cost = 170544.49872006284  # EUR/anno
    highpv_cost = 163666.6409869843  # EUR/anno

    # Costo del calore: 1467.99390776596 MWh_th × 69.73 EUR/MWh = 102363.22 EUR/anno
    heat_produced = 1467.99390776596  # MWh_th/anno
    heat_price = 69.73  # EUR/MWh
    heat_cost = heat_produced * heat_price  # EUR/anno

    baseline_elec_cost = baseline_cost - heat_cost
    highpv_elec_cost = highpv_cost - heat_cost

    # Carica il profilo di generazione PV per la distribuzione mensile
    pv_data = load_pv_generation()
    if pv_data is None:
        return

    # Calcola la generazione per i due scenari
    baseline_capacity = 65.0  # kWp
    highpv_capacity = 165.0  # kWp
    inverter_efficiency = 0.97

    # Calcola la generazione oraria
    pv_data['Baseline (kW)'] = pv_data['PV Generation (normalized)'] * baseline_capacity * inverter_efficiency
    pv_data['High PV (kW)'] = pv_data['PV Generation (normalized)'] * highpv_capacity * inverter_efficiency

    # Calcola la generazione giornaliera
    daily_data = pv_data.copy()
    daily_data['Date'] = daily_data.index.date
    # Raggruppa per data e somma i valori
    daily_generation = daily_data.groupby(daily_data['Date'])[['Baseline (kW)', 'High PV (kW)']].sum()

    # Crea il grafico
    plt.figure(figsize=(15, 12))

    # Chart 1: Annual PV generation comparison
    plt.subplot(2, 2, 1)
    scenarios = ['Baseline\n(65 kWp)', 'High PV\n(165 kWp)']
    pv_values = [baseline_pv, highpv_pv]

    plt.bar(scenarios, pv_values, color=['blue', 'red'])
    plt.title('Annual PV Generation')
    plt.ylabel('Generation (MWh/year)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Chart 2: Grid import comparison
    plt.subplot(2, 2, 2)
    import_values = [baseline_import, highpv_import]

    plt.bar(scenarios, import_values, color=['blue', 'red'])
    plt.title('Grid Import')
    plt.ylabel('Import (MWh/year)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Chart 3: Operational costs comparison
    plt.subplot(2, 2, 3)

    # Create stacked bar chart for costs
    width = 0.35
    x = np.arange(len(scenarios))

    plt.bar(x, [baseline_elec_cost, highpv_elec_cost], width, label='Electricity cost', color='orange')
    plt.bar(x, [heat_cost, heat_cost], width, bottom=[baseline_elec_cost, highpv_elec_cost], label='Heat cost', color='red')

    plt.title('Annual Operational Costs')
    plt.ylabel('Cost (EUR/year)')
    plt.xticks(x, scenarios)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Chart 4: Monthly PV generation comparison
    plt.subplot(2, 2, 4)

    # Calculate monthly distribution
    pv_data['Month'] = pv_data.index.month
    monthly_generation = pv_data.groupby('Month')[['Baseline (kW)', 'High PV (kW)']].sum()

    # Normalize monthly values to get correct distribution
    baseline_monthly = monthly_generation['Baseline (kW)'] / monthly_generation['Baseline (kW)'].sum() * baseline_pv
    highpv_monthly = monthly_generation['High PV (kW)'] / monthly_generation['High PV (kW)'].sum() * highpv_pv

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x = np.arange(len(month_names))
    width = 0.35

    plt.bar(x - width/2, baseline_monthly, width, label='Baseline (65 kWp)', color='blue')
    plt.bar(x + width/2, highpv_monthly, width, label='High PV (165 kWp)', color='red')

    plt.title('Monthly PV Generation')
    plt.ylabel('Generation (MWh/month)')
    plt.xlabel('Month')
    plt.xticks(x, month_names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'baseline_vs_highpv.png'), dpi=300)
    plt.close()

def plot_savings():
    """
    Crea un grafico che mostra il risparmio annuale ottenuto con lo scenario High PV.
    Utilizza i valori esatti dai file di output.
    """
    # Valori esatti dai file di output
    baseline_cost = 170544.49872006284  # EUR/anno
    highpv_cost = 163666.6409869843  # EUR/anno
    savings = baseline_cost - highpv_cost  # EUR/anno

    baseline_import = 963.2845935510373  # MWh/anno
    highpv_import = 866.1122605038697  # MWh/anno
    import_reduction = baseline_import - highpv_import  # MWh/anno

    # Calcola il tempo di ritorno dell'investimento
    additional_pv_capacity = 165 - 65  # kWp
    investment_cost = additional_pv_capacity * 900  # EUR (900 EUR/kWp)
    payback_time = investment_cost / savings  # anni

    # Crea il grafico
    plt.figure(figsize=(15, 10))

    # Chart 1: Annual savings
    plt.subplot(2, 2, 1)
    plt.bar(['Annual savings'], [savings], color='green')
    plt.title('Annual Savings with High PV')
    plt.ylabel('Savings (EUR/year)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.text(0, savings/2, f"{savings:.2f} EUR", ha='center', va='center', fontweight='bold')

    # Chart 2: Grid import reduction
    plt.subplot(2, 2, 2)
    plt.bar(['Grid import reduction'], [import_reduction], color='blue')
    plt.title('Grid Import Reduction')
    plt.ylabel('Reduction (MWh/year)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.text(0, import_reduction/2, f"{import_reduction:.2f} MWh", ha='center', va='center', fontweight='bold')

    # Chart 3: Payback period
    plt.subplot(2, 2, 3)
    plt.bar(['Payback period'], [payback_time], color='orange')
    plt.title('Investment Payback Period')
    plt.ylabel('Years')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.text(0, payback_time/2, f"{payback_time:.1f} years", ha='center', va='center', fontweight='bold')

    # Chart 4: Investment vs savings
    plt.subplot(2, 2, 4)
    labels = ['Additional\ninvestment', 'Savings\nover 10 years']
    values = [investment_cost, savings * 10]
    plt.bar(labels, values, color=['red', 'green'])
    plt.title('Investment vs. 10-Year Savings')
    plt.ylabel('EUR')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.text(0, investment_cost/2, f"{investment_cost:.0f} EUR", ha='center', va='center', fontweight='bold')
    plt.text(1, values[1]/2, f"{values[1]:.0f} EUR", ha='center', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'savings_analysis.png'), dpi=300)
    plt.close()

def plot_environmental_impact():
    """
    Creates a chart showing the environmental impact of the High PV scenario.
    Uses exact values from the output files and emission factors consistent with the baseline_summary.csv file.
    """
    # Exact values from output files
    baseline_import = 963.2845935510373  # MWh/year
    highpv_import = 866.1122605038697  # MWh/year
    import_reduction = baseline_import - highpv_import  # MWh/year
    heat_produced = 1467.99390776596  # MWh_th/year

    # Emission factors consistent with the baseline_summary.csv file
    # Calculated backwards from the provided emission values
    grid_emission_factor = 267.61 / baseline_import * 1000  # kg CO2/MWh (approx. 278 kg CO2/MWh)
    heat_emission_factor = 262.78 / heat_produced * 1000  # kg CO2/MWh (approx. 179 kg CO2/MWh)

    # Calculate avoided emissions (electricity only)
    emissions_avoided = import_reduction * grid_emission_factor / 1000  # tonnes CO2/year

    # Equivalents for avoided emissions
    trees_equivalent = emissions_avoided * 1000 / 25  # One tree absorbs about 25 kg CO2/year
    car_km_equivalent = emissions_avoided * 1000 / 0.12  # An average car emits about 0.12 kg CO2/km

    # Calculate total emissions
    baseline_elec_emissions = baseline_import * grid_emission_factor / 1000  # tonnes CO2/year
    highpv_elec_emissions = highpv_import * grid_emission_factor / 1000  # tonnes CO2/year
    heat_emissions = heat_produced * heat_emission_factor / 1000  # tonnes CO2/year (same for both scenarios)

    baseline_total_emissions = baseline_elec_emissions + heat_emissions  # tonnes CO2/year
    highpv_total_emissions = highpv_elec_emissions + heat_emissions  # tonnes CO2/year

    # Create the chart
    plt.figure(figsize=(15, 10))

    # Chart 1: Avoided emissions
    plt.subplot(2, 2, 1)
    plt.bar(['CO₂ emissions avoided'], [emissions_avoided], color='green')
    plt.title('CO₂ Emissions Avoided')
    plt.ylabel('Tonnes CO₂/year')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.text(0, emissions_avoided / 2, f"{emissions_avoided:.2f} t", ha='center', va='center', fontweight='bold')

    # Chart 2: Tree equivalent
    plt.subplot(2, 2, 2)
    plt.bar(['Equivalent in trees'], [trees_equivalent], color='green')
    plt.title('Equivalent in Planted Trees')
    plt.ylabel('Number of trees')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.text(0, trees_equivalent/2, f"{trees_equivalent:.0f} trees", ha='center', va='center', fontweight='bold')

    # Chart 3: Total emissions comparison
    plt.subplot(2, 2, 3)
    scenarios = ['Baseline', 'High PV']

    # Create stacked bar chart for emissions
    width = 0.7
    x = np.arange(len(scenarios))

    plt.bar(x, [baseline_elec_emissions, highpv_elec_emissions], width,
            label='Electricity emissions', color='blue')
    plt.bar(x, [heat_emissions, heat_emissions], width,
            bottom=[baseline_elec_emissions, highpv_elec_emissions],
            label='Heat emissions', color='red')

    plt.title('Total CO₂ Emissions Comparison')
    plt.ylabel('Tonnes CO₂/year')
    plt.xticks(x, scenarios)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add labels with total values
    for i, v in enumerate([baseline_total_emissions, highpv_total_emissions]):
        plt.text(i, v + 5, f"Total: {v:.2f} t", ha='center', va='bottom', fontweight='bold')

    # Add labels for electricity emissions
    for i, v in enumerate([baseline_elec_emissions, highpv_elec_emissions]):
        plt.text(i, v/2, f"{v:.2f} t", ha='center', va='center', fontweight='bold', color='white')

    # Add labels for heat emissions
    for i, v in enumerate([heat_emissions, heat_emissions]):
        y_pos = baseline_elec_emissions + v/2 if i == 0 else highpv_elec_emissions + v/2
        plt.text(i, y_pos, f"{v:.2f} t", ha='center', va='center', fontweight='bold', color='white')

    # Chart 4: Car kilometers equivalent
    plt.subplot(2, 2, 4)
    plt.bar(['Equivalent in car kilometers'], [car_km_equivalent / 1000], color='blue')
    plt.title('Equivalent in Car Kilometers Avoided')
    plt.ylabel('Thousands of km')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.text(0, car_km_equivalent / 2000, f"{car_km_equivalent / 1000:.0f} km", ha='center', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'environmental_impact.png'), dpi=300)
    plt.close()

def plot_self_consumption():
    """
    Creates a chart showing self-consumption and self-sufficiency parameters.
    """
    # Exact values from output files
    baseline_pv = 64.1311737565145  # MWh/year
    highpv_pv = 159.35536243754777  # MWh/year

    # Calculate total electricity demand
    baseline_import = 963.2845935510373  # MWh/year
    baseline_demand = baseline_import + baseline_pv  # MWh/year

    highpv_import = 866.1122605038697  # MWh/year
    highpv_demand = highpv_import + highpv_pv  # MWh/year

    # Calculate self-consumption and self-sufficiency
    # We assume that all PV generation is consumed on-site (self-consumption = 100%)
    baseline_self_consumption = 100.0  # %
    highpv_self_consumption = 100.0  # %

    baseline_self_sufficiency = baseline_pv / baseline_demand * 100  # %
    highpv_self_sufficiency = highpv_pv / highpv_demand * 100  # %

    # Create the chart
    plt.figure(figsize=(12, 6))

    # Bar chart for self-sufficiency
    scenarios = ['Baseline\n(65 kWp)', 'High PV\n(165 kWp)']
    x = np.arange(len(scenarios))
    width = 0.35

    plt.bar(x, [baseline_self_sufficiency, highpv_self_sufficiency], width, color='green')

    plt.title('Electricity Self-Sufficiency')
    plt.ylabel('Percentage (%)')
    plt.xticks(x, scenarios)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add labels with values
    for i, v in enumerate([baseline_self_sufficiency, highpv_self_sufficiency]):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha='center', va='bottom', fontweight='bold')

    # Add horizontal line for self-consumption (100%)
    plt.axhline(y=100, color='red', linestyle='--', label='Self-consumption (100%)')

    # Add legend
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'self_consumption.png'), dpi=300)
    plt.close()

def plot_annual_pv_demand_comparison():
    """
    Creates a chart showing how much of the annual electricity demand is covered by PV generation
    in the High PV scenario, with monthly breakdown.
    """
    # Load PV generation data
    pv_data = load_pv_generation()
    if pv_data is None:
        print("Unable to generate annual PV vs demand chart: PV profile not available.")
        return

    # Exact values from output files
    highpv_pv = 159.35536243754777  # MWh/year
    highpv_import = 866.1122605038697  # MWh/year
    highpv_demand = highpv_import + highpv_pv  # MWh/year

    # Calculate monthly values
    # First, create a datetime index for the data
    timestamps = pd.date_range(start='2025-01-01', periods=8760, freq='h')

    # Create a normalized PV profile (0-1)
    normalized_pv = pv_data['PV Generation (normalized)'].values

    # Scale to actual production values
    highpv_capacity = 165.0  # kWp
    inverter_efficiency = 0.97

    # Create a DataFrame with the hourly values
    hourly_data = pd.DataFrame({
        'PV Generation': normalized_pv * highpv_capacity * inverter_efficiency / 1000,  # Scale to 165 kWp and convert to MWh
    }, index=timestamps)

    # Calculate hourly demand (PV generation + a portion of grid import)
    # We distribute grid import evenly across all hours as a simplification
    hourly_data['Grid Import'] = highpv_import / 8760  # MWh per hour
    hourly_data['Total Demand'] = hourly_data['PV Generation'] + hourly_data['Grid Import']

    # Resample to monthly
    monthly_data = hourly_data.resample('M').sum()

    # Create the chart
    plt.figure(figsize=(15, 10))

    # Plot 1: Monthly comparison
    plt.subplot(2, 1, 1)

    # Convert index to month names
    months = [d.strftime('%b') for d in monthly_data.index]

    # Create the stacked bar chart
    plt.bar(months, monthly_data['Total Demand'], color='#ff7f0e', label='Grid Import')
    plt.bar(months, monthly_data['PV Generation'], color='#2ca02c', label='PV Generation')

    plt.title('Monthly Electricity Demand vs PV Generation (High PV Scenario)')
    plt.ylabel('Energy (MWh)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Add percentage labels
    for i, month in enumerate(months):
        pv_percentage = (monthly_data['PV Generation'].iloc[i] / monthly_data['Total Demand'].iloc[i]) * 100
        plt.text(i, monthly_data['PV Generation'].iloc[i] / 2, f"{pv_percentage:.1f}%",
                 ha='center', va='center', fontweight='bold', color='white')

    # Plot 2: Annual pie chart
    plt.subplot(2, 1, 2)

    # Create pie chart
    labels = ['PV Generation', 'Grid Import']
    sizes = [highpv_pv, highpv_import]
    colors = ['#2ca02c', '#ff7f0e']
    explode = (0.1, 0)  # explode the PV slice

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title(f'Annual Electricity Supply Sources (High PV Scenario)\nTotal Demand: {highpv_demand:.2f} MWh/year')

    # Add text with key metrics
    plt.figtext(0.5, 0.01,
                f"PV Capacity: 165 kWp | Annual PV Generation: {highpv_pv:.2f} MWh/year | Self-Sufficiency: {(highpv_pv/highpv_demand*100):.2f}%",
                ha='center', fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(OUTPUT_PATH, 'annual_pv_demand_comparison.png'), dpi=300)
    plt.close()

def main():
    """Main function."""
    print("Generating charts for photovoltaic generation analysis...")

    # Load PV generation profile
    pv_data = load_pv_generation()
    if pv_data is None:
        print("Unable to generate charts: PV profile not available.")
        return

    # Create synthetic weather data
    weather_data = create_synthetic_weather_data(pv_data)

    # Generate charts
    # Removed daily profile charts as requested
    plot_seasonal_comparison(pv_data, weather_data)
    plot_monthly_generation(pv_data)
    plot_baseline_vs_highpv()
    plot_savings()
    plot_environmental_impact()
    plot_self_consumption()
    plot_annual_pv_demand_comparison()  # New chart for annual PV vs demand comparison

    print(f"Charts successfully generated in directory: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
