import torch
import networkx as nx
import pandas as pd


def build_graph(adj_filename, num_sensors):

    adj_data = pd.read_csv(adj_filename)

    G = nx.Graph()

    for sensor_id in range(num_sensors):
        G.add_node(sensor_id)

    for _, row in adj_data.iterrows():
        G.add_edge(row['from'], row['to'])

    return G


def build_cost_adj_matrix(num_sensors, adj_filename):
    """
    Generate a static adjacency matrix based on cost.

    output:
        (N, N)
    """
    G = build_graph(adj_filename, num_sensors)

    cost_adj_matrix = nx.to_numpy_array(G, nodelist=range(num_sensors))
    cost_adj_matrix = torch.tensor(cost_adj_matrix, dtype=torch.float32)

    return cost_adj_matrix
