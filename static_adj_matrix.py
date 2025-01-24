import os
import torch
from lib import config_loader
from lib.cost_adj_matrix import build_cost_adj_matrix
from lib.similarity_adj_matrix import build_similarity_adj_matrix

args = config_loader.setTerminal()
data_config, train_config = config_loader.getConfig(args)

adj_filename = data_config['adj_filename']
traffic_filename = data_config['traffic_filename']
dataset_name = data_config['dataset_name']
points_per_hour = int(data_config['points_per_hour'])
num_sensors = int(data_config['num_sensors'])

cost_adj_matrix = build_cost_adj_matrix(num_sensors, adj_filename)
similarity_adj_matrix = build_similarity_adj_matrix(traffic_filename, points_per_hour, num_sensors)

file = dataset_name
dirpath = os.path.dirname(adj_filename)
filename_cost_adj_matrix = os.path.join(dirpath, file + '_cost_adj_matrix.pt')
filename_similarity_adj_matrix = os.path.join(dirpath, file + '_similarity_adj_matrix.pt')

torch.save(cost_adj_matrix, filename_cost_adj_matrix)
torch.save(similarity_adj_matrix, filename_similarity_adj_matrix)
