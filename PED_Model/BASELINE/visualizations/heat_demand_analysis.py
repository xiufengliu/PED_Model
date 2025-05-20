"""
Heat Demand Analysis for Social Building in Denmark

This script analyzes and visualizes the heat demand data for a social building in Denmark.
It creates various visualizations to help understand the patterns and characteristics of the heat demand.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Set plot style
plt.style.use('ggplot')
sns.set_palette("deep")

def load_heat_demand_data(file_path):
    """Load heat demand data from CSV file"""
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        print(f"Loaded heat demand data from {file_path}")
        print(f"Data range: {df.index.min()} to {df.index.max()}")
        print(f"Number of data points: {len(df)}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_monthly_heat_demand(df, output_dir):
    """Plot monthly heat demand"""
    # Resample to monthly
    monthly_data = df.resample('M').mean()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot monthly averages
    plt.bar(monthly_data.index, monthly_data['heat_demand_kw'], color='firebrick')
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    # Add labels and title
    plt.xlabel('Month')
    plt.ylabel('Average Heat Demand (kW)')
    plt.title('Monthly Average Heat Demand - Social Building in Denmark')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_heat_demand.png'), dpi=300)
    print(f"Saved monthly heat demand plot to {os.path.join(output_dir, 'monthly_heat_demand.png')}")
    plt.close()

def plot_daily_profile_by_season(df, output_dir):
    """Plot daily heat demand profile by season"""
    # Define seasons
    df = df.copy()
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    
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
    
    # Calculate average daily profile for each season
    winter_profile = winter_data.groupby('hour')['heat_demand_kw'].mean()
    spring_profile = spring_data.groupby('hour')['heat_demand_kw'].mean()
    summer_profile = summer_data.groupby('hour')['heat_demand_kw'].mean()
    fall_profile = fall_data.groupby('hour')['heat_demand_kw'].mean()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot profiles
    plt.plot(winter_profile.index, winter_profile.values, 'b-', linewidth=2, label='Winter (Dec-Feb)')
    plt.plot(spring_profile.index, spring_profile.values, 'g-', linewidth=2, label='Spring (Mar-May)')
    plt.plot(summer_profile.index, summer_profile.values, 'r-', linewidth=2, label='Summer (Jun-Aug)')
    plt.plot(fall_profile.index, fall_profile.values, 'orange', linewidth=2, label='Fall (Sep-Nov)')
    
    # Add labels and title
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Heat Demand (kW)')
    plt.title('Daily Heat Demand Profile by Season - Social Building in Denmark')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(range(0, 24, 2))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'daily_profile_by_season.png'), dpi=300)
    print(f"Saved daily profile by season plot to {os.path.join(output_dir, 'daily_profile_by_season.png')}")
    plt.close()

def plot_heatmap(df, output_dir):
    """Plot heatmap of heat demand by month and hour"""
    # Prepare data
    df = df.copy()
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    
    # Create pivot table
    pivot_data = df.pivot_table(
        index='month', 
        columns='hour', 
        values='heat_demand_kw',
        aggfunc='mean'
    )
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list(
        'heat_demand', 
        ['#FFFFFF', '#FFF7BC', '#FEE391', '#FEC44F', '#FE9929', '#EC7014', '#CC4C02', '#993404', '#662506']
    )
    
    # Plot heatmap
    sns.heatmap(
        pivot_data, 
        cmap=cmap, 
        annot=False, 
        fmt='.1f', 
        linewidths=0.5,
        cbar_kws={'label': 'Heat Demand (kW)'}
    )
    
    # Add labels and title
    plt.xlabel('Hour of Day')
    plt.ylabel('Month')
    plt.title('Heat Demand by Month and Hour - Social Building in Denmark')
    
    # Adjust y-axis labels to month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.yticks(np.arange(0.5, 12.5), month_names)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heat_demand_heatmap.png'), dpi=300)
    print(f"Saved heat demand heatmap to {os.path.join(output_dir, 'heat_demand_heatmap.png')}")
    plt.close()

def main():
    """Main function"""
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'input', 'timeseries', 'social_building_heat_demand_denmark.csv')
    output_dir = os.path.join(script_dir, 'heat_demand_figures')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    heat_demand_df = load_heat_demand_data(data_path)
    if heat_demand_df is None:
        return
    
    # Create visualizations
    plot_monthly_heat_demand(heat_demand_df, output_dir)
    plot_daily_profile_by_season(heat_demand_df, output_dir)
    plot_heatmap(heat_demand_df, output_dir)
    
    print("Analysis complete. All visualizations saved to:", output_dir)

if __name__ == "__main__":
    main()
