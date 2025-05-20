#!/usr/bin/env python3
"""
Visualize the network structure of a scenario.

This script creates a visualization of the network structure for a given scenario,
showing buses, generators, loads, links, and other components.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the scenarios package
from scenarios import get_scenario_function

def create_network_graph(network):
    """
    Create a NetworkX graph from a PyPSA network.

    Args:
        network (pypsa.Network): The PyPSA network

    Returns:
        nx.Graph: A NetworkX graph representing the network structure
    """
    G = nx.Graph()

    # Add buses as nodes
    for bus_name in network.buses.index:
        carrier = network.buses.at[bus_name, 'carrier']
        G.add_node(bus_name, type='bus', carrier=carrier)

    # Add generators as nodes
    for gen_name in network.generators.index:
        bus = network.generators.at[gen_name, 'bus']
        carrier = network.generators.at[gen_name, 'carrier']
        p_nom = network.generators.at[gen_name, 'p_nom']
        G.add_node(gen_name, type='generator', carrier=carrier, p_nom=p_nom)
        G.add_edge(gen_name, bus, type='connection')

    # Add loads as nodes
    for load_name in network.loads.index:
        bus = network.loads.at[load_name, 'bus']
        G.add_node(load_name, type='load')
        G.add_edge(load_name, bus, type='connection')

    # Add links as edges
    for link_name in network.links.index:
        bus0 = network.links.at[link_name, 'bus0']
        bus1 = network.links.at[link_name, 'bus1']
        p_nom = network.links.at[link_name, 'p_nom']
        G.add_edge(bus0, bus1, type='link', name=link_name, p_nom=p_nom)

    # Add storage units as nodes
    for storage_name in network.storage_units.index:
        if network.storage_units.at[storage_name, 'p_nom'] > 0:
            bus = network.storage_units.at[storage_name, 'bus']
            p_nom = network.storage_units.at[storage_name, 'p_nom']
            G.add_node(storage_name, type='storage', p_nom=p_nom)
            G.add_edge(storage_name, bus, type='connection')

    # Add stores as nodes
    for store_name in network.stores.index:
        if network.stores.at[store_name, 'e_nom'] > 0:
            bus = network.stores.at[store_name, 'bus']
            e_nom = network.stores.at[store_name, 'e_nom']
            G.add_node(store_name, type='store', e_nom=e_nom)
            G.add_edge(store_name, bus, type='connection')

    return G

def visualize_network(network, output_file=None):
    """
    Visualize a PyPSA network using NetworkX and matplotlib.

    Args:
        network (pypsa.Network): The PyPSA network
        output_file (str, optional): Path to save the visualization
    """
    G = create_network_graph(network)

    # Create figure
    plt.figure(figsize=(12, 10))

    # Define positions for nodes
    pos = {}

    # Manual positioning for better visualization
    # Grid and building buses
    pos['Grid Connection'] = (0, 0)
    pos['Building Elec'] = (4, 0)
    pos['Heat Source'] = (0, -4)
    pos['Building Heat'] = (4, -4)

    # Generators
    pos['Grid'] = (-2, 0)
    pos['Rooftop PV'] = (4, 2)
    pos['Heat Source'] = (-2, -4)

    # Loads
    pos['Building Elec Load'] = (6, 0)
    pos['Building Heat Load'] = (6, -4)

    # Links
    # These are edges, not nodes, but we'll keep the positions for reference

    # Storage
    pos['Placeholder Battery'] = (2, 2)
    pos['Placeholder Thermal Storage'] = (2, -6)

    # For any nodes without manual positions, use spring layout
    missing_nodes = set(G.nodes()) - set(pos.keys())
    if missing_nodes:
        print(f"Warning: Missing positions for nodes: {missing_nodes}")
        temp_pos = nx.spring_layout(G.subgraph(missing_nodes))
        pos.update(temp_pos)

    # Draw nodes
    bus_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'bus']
    generator_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'generator']
    load_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'load']
    storage_nodes = [n for n, d in G.nodes(data=True) if d.get('type') in ['storage', 'store']]

    # Draw buses
    elec_buses = [n for n in bus_nodes if G.nodes[n].get('carrier') == 'electricity']
    heat_buses = [n for n in bus_nodes if G.nodes[n].get('carrier') == 'heat']

    nx.draw_networkx_nodes(G, pos, nodelist=elec_buses, node_color='skyblue',
                          node_size=800, label='Electricity Bus')
    nx.draw_networkx_nodes(G, pos, nodelist=heat_buses, node_color='indianred',
                          node_size=800, label='Heat Bus')

    # Draw generators
    elec_gens = [n for n in generator_nodes if G.nodes[n].get('carrier') == 'electricity']
    heat_gens = [n for n in generator_nodes if G.nodes[n].get('carrier') == 'heat']

    nx.draw_networkx_nodes(G, pos, nodelist=elec_gens, node_color='gold',
                          node_size=600, node_shape='s', label='Electricity Generator')
    nx.draw_networkx_nodes(G, pos, nodelist=heat_gens, node_color='firebrick',
                          node_size=600, node_shape='s', label='Heat Generator')

    # Draw loads
    nx.draw_networkx_nodes(G, pos, nodelist=load_nodes, node_color='lightgreen',
                          node_size=600, node_shape='v', label='Load')

    # Draw storage
    nx.draw_networkx_nodes(G, pos, nodelist=storage_nodes, node_color='orchid',
                          node_size=600, node_shape='h', label='Storage')

    # Draw edges
    connection_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'connection']
    link_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'link']

    nx.draw_networkx_edges(G, pos, edgelist=connection_edges, width=1.5, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edgelist=link_edges, width=2.5, alpha=0.7, edge_color='red')

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    # Add legend
    plt.legend(scatterpoints=1, loc='upper right', bbox_to_anchor=(1.0, 1.0))

    # Set title
    plt.title('Network Structure Visualization', fontsize=16)

    # Remove axis
    plt.axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
    else:
        plt.show()

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Visualize the network structure of a scenario.")
    parser.add_argument("-s", "--scenario", help="Name of the scenario to visualize", default="baseline")
    parser.add_argument("-c", "--config", help="Path to the main configuration file", default="config/config.yml")
    parser.add_argument("-p", "--params", help="Path to the component parameters file", default="config/component_params.yml")
    parser.add_argument("-o", "--output", help="Path to save the visualization", default=None)

    args = parser.parse_args()

    # Get the scenario function
    try:
        create_network = get_scenario_function(args.scenario)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Build the network
    print(f"Building {args.scenario} network...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/input')
    network = create_network(args.config, args.params, data_path)

    # Determine output file path
    output_file = args.output
    if not output_file:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 f'data/output/scenario_{args.scenario}')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{args.scenario}_network_structure.png')

    # Visualize the network
    print(f"Visualizing {args.scenario} network...")
    visualize_network(network, output_file)

    print("Visualization complete.")

if __name__ == "__main__":
    main()
