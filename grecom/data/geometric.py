from scipy.sparse import coo_matrix
from sklearn.neighbors import kneighbors_graph
import numpy as np
import torch
from torch_scatter import scatter_add


def create_similarity_graph(features, k=10):
    """From a feature matrix, construct the k-neighbors graph using euclidean
    distance as similarity metric. Gaussian kernel weights are used.

    :param features: Feature matrix
    :type features: numpy.array
    :param k: Number of neighbours, defaults to 10
    :type k: int, optional
    :return: Edge index and edge weights of similarity graph
    :rtype: tuple of numpy.array
    """
    C = kneighbors_graph(features, k, n_jobs=-1).toarray()
    D = kneighbors_graph(features, k, n_jobs=-1, mode='distance').toarray()
    sigma = 1 / 3 * D[C == 1].mean()
    A = coo_matrix(C * np.exp(-D ** 2 / (2 * sigma ** 2)))
    return np.stack([A.row, A.col]), A.data


def conv_norm(edge_index, num_nodes,
              edge_weight=None, dtype=None, symmetric=True):
    """Return graph convolution normalization constants.

    :param edge_index: Edge indices of the graph in COO format
    :type edge_index: torch.Tensor
    :param num_nodes: Number of nodes in the graph
    :type num_nodes: int
    :param edge_weight: Weights of the graph, if None, the graph is unweighted
    :type edge_weight: torch.Tensor, optional
    :param dtype: Weights dtype, defaults to None
    :type dtype: object, optional
    :param symmetric: If True, use symmetric normalization. If False, use left
    normalization, defaults to True
    :type symmetric: bool, optional
    :return: Convolution normalization constants
    :rtype: torch.Tensor
    """
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    left_deg = deg_inv_sqrt[row] * edge_weight
    if symmetric:
        return left_deg * deg_inv_sqrt[col]  # symmetric norm
    return left_deg
