"""
Thermal Energy Prices Analysis for Denmark

This script analyzes and visualizes the thermal energy prices data for Denmark.
It creates various visualizations to help understand the patterns and characteristics of the thermal energy prices.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Set plot style
plt.style.use('ggplot')
sns.set_palette("deep")

def load_thermal_prices_data(file_path):
    """Load thermal energy prices data from CSV file"""
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        print(f"Loaded thermal energy prices data from {file_path}")
        print(f"Data range: {df.index.min()} to {df.index.max()}")
        print(f"Number of data points: {len(df)}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_monthly_prices(df, output_dir):
    """Plot monthly average thermal energy prices"""
    # Resample to monthly
    monthly_data = df.resample('M').mean()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot monthly averages
    plt.bar(monthly_data.index, monthly_data['Price (EUR/MWh)'], color='firebrick')
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    # Add labels and title
    plt.xlabel('Month')
    plt.ylabel('Average Price (EUR/MWh)')
    plt.title('Monthly Average Thermal Energy Prices - Denmark')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_thermal_prices.png'), dpi=300)
    print(f"Saved monthly thermal prices plot to {os.path.join(output_dir, 'monthly_thermal_prices.png')}")
    plt.close()

def plot_price_trend(df, output_dir):
    """Plot thermal energy price trend over the year"""
    # Create figure
    plt.figure(figsize=(14, 6))
    
    # Plot price trend
    plt.plot(df.index, df['Price (EUR/MWh)'], 'r-', linewidth=2)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Price (EUR/MWh)')
    plt.title('Thermal Energy Price Trend - Denmark (2025)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add horizontal lines for seasonal averages
    winter_avg = df[df.index.month.isin([12, 1, 2])]['Price (EUR/MWh)'].mean()
    spring_avg = df[df.index.month.isin([3, 4, 5])]['Price (EUR/MWh)'].mean()
    summer_avg = df[df.index.month.isin([6, 7, 8])]['Price (EUR/MWh)'].mean()
    fall_avg = df[df.index.month.isin([9, 10, 11])]['Price (EUR/MWh)'].mean()
    
    plt.axhline(y=winter_avg, color='blue', linestyle='--', alpha=0.7, label=f'Winter Avg: {winter_avg:.2f} EUR/MWh')
    plt.axhline(y=spring_avg, color='green', linestyle='--', alpha=0.7, label=f'Spring Avg: {spring_avg:.2f} EUR/MWh')
    plt.axhline(y=summer_avg, color='red', linestyle='--', alpha=0.7, label=f'Summer Avg: {summer_avg:.2f} EUR/MWh')
    plt.axhline(y=fall_avg, color='orange', linestyle='--', alpha=0.7, label=f'Fall Avg: {fall_avg:.2f} EUR/MWh')
    
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'thermal_price_trend.png'), dpi=300)
    print(f"Saved thermal price trend plot to {os.path.join(output_dir, 'thermal_price_trend.png')}")
    plt.close()

def plot_seasonal_comparison(df, output_dir):
    """Plot seasonal comparison of thermal energy prices"""
    # Create seasonal data
    df = df.copy()
    df['month'] = df.index.month
    
    # Define seasons (meteorological seasons for Northern Hemisphere)
    winter_months = [12, 1, 2]
    spring_months = [3, 4, 5]
    summer_months = [6, 7, 8]
    fall_months = [9, 10, 11]
    
    # Filter data by season
    winter_data = df[df['month'].isin(winter_months)]
    spring_data = df[df['month'].isin(spring_months)]
    summer_data = df[df['month'].isin(summer_months)]
    fall_data = df[df['month'].isin(fall_months)]
    
    # Calculate seasonal statistics
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    avg_prices = [
        winter_data['Price (EUR/MWh)'].mean(),
        spring_data['Price (EUR/MWh)'].mean(),
        summer_data['Price (EUR/MWh)'].mean(),
        fall_data['Price (EUR/MWh)'].mean()
    ]
    
    min_prices = [
        winter_data['Price (EUR/MWh)'].min(),
        spring_data['Price (EUR/MWh)'].min(),
        summer_data['Price (EUR/MWh)'].min(),
        fall_data['Price (EUR/MWh)'].min()
    ]
    
    max_prices = [
        winter_data['Price (EUR/MWh)'].max(),
        spring_data['Price (EUR/MWh)'].max(),
        summer_data['Price (EUR/MWh)'].max(),
        fall_data['Price (EUR/MWh)'].max()
    ]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Set bar width
    bar_width = 0.25
    
    # Set positions for bars
    r1 = np.arange(len(seasons))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create bars
    plt.bar(r1, avg_prices, color='blue', width=bar_width, label='Average Price')
    plt.bar(r2, min_prices, color='green', width=bar_width, label='Minimum Price')
    plt.bar(r3, max_prices, color='red', width=bar_width, label='Maximum Price')
    
    # Add labels and title
    plt.xlabel('Season')
    plt.ylabel('Price (EUR/MWh)')
    plt.title('Seasonal Comparison of Thermal Energy Prices - Denmark (2025)')
    plt.xticks([r + bar_width for r in range(len(seasons))], seasons)
    plt.legend()
    
    # Add value labels on bars
    for i, v in enumerate(avg_prices):
        plt.text(i - 0.1, v + 2, f'{v:.2f}', color='blue', fontweight='bold')
    
    for i, v in enumerate(min_prices):
        plt.text(i + bar_width - 0.1, v + 2, f'{v:.2f}', color='green', fontweight='bold')
    
    for i, v in enumerate(max_prices):
        plt.text(i + 2*bar_width - 0.1, v + 2, f'{v:.2f}', color='red', fontweight='bold')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seasonal_price_comparison.png'), dpi=300)
    print(f"Saved seasonal price comparison plot to {os.path.join(output_dir, 'seasonal_price_comparison.png')}")
    plt.close()

def main():
    """Main function"""
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'input', 'timeseries', 'thermal_energy_prices_denmark.csv')
    output_dir = os.path.join(script_dir, 'thermal_prices_figures')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    thermal_prices_df = load_thermal_prices_data(data_path)
    if thermal_prices_df is None:
        return
    
    # Create visualizations
    plot_monthly_prices(thermal_prices_df, output_dir)
    plot_price_trend(thermal_prices_df, output_dir)
    plot_seasonal_comparison(thermal_prices_df, output_dir)
    
    print("Analysis complete. All visualizations saved to:", output_dir)

if __name__ == "__main__":
    main()
