import numpy as np
import torch
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def build_similarity_adj_matrix(traffic_filename, points_per_hour, num_sensors):

    data = np.load(traffic_filename)['data']

    data_mean = np.mean([data[24 * points_per_hour * i: 24 * points_per_hour * (i + 1)]
                         for i in range(data.shape[0] // (24 * points_per_hour))], axis=1)

    similarity_adj_matrix = np.zeros((num_sensors, num_sensors))

    for i in (range(num_sensors)):
        for j in range(i, num_sensors):
            similarity_adj_matrix[i][j], _ = fastdtw(data_mean[:, i, :], data_mean[:, j, :], dist=euclidean)

    for i in range(num_sensors):
        for j in range(i):
            similarity_adj_matrix[i][j] = similarity_adj_matrix[j][i]

    std = np.std(similarity_adj_matrix)
    similarity_adj_matrix = similarity_adj_matrix / std
    similarity_adj_matrix = np.exp(-1 * similarity_adj_matrix)

    similarity_adj_matrix = torch.tensor(similarity_adj_matrix, dtype=torch.float32)

    return similarity_adj_matrix
