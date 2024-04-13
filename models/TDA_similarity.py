import numpy as np
from utils import *
from topology import *
import gudhi as gd
import networkx as nx
from persim import wasserstein
import citation
from keras.preprocessing.sequence import pad_sequences
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj
import pickle

# load data, e.g., cora
##dataset = 'cora'
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = citation.load_data(dataset)

# adj_array = adj.toarray().astype(np.float32)
# print('ADJ:', adj.shape)
# print(adj_array)

# k-hop subgraphs for whole nodes in the network
# k_hop_subgraphs = k_th_order_subgraph(adj_array, k = 2)
# for i, graph in enumerate(k_hop_subgraphs):
# print(f'Shape of graph {i}:', graph.shape)
# print(graph)


# Load the ENZYMES dataset
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

# List to store adjacency matrices
k_hop_subgraphs = []

# Extract adjacency matrix for each graph
for data in dataset:
    adj_matrix_tensor = to_dense_adj(data.edge_index)[0]  # This is a PyTorch tensor

    # Ensure the tensor is on the CPU then convert it to a NumPy array
    adj_matrix_np = adj_matrix_tensor.cpu().detach().numpy()
    k_hop_subgraphs.append(adj_matrix_np)

# Now, k_hop_subgraphs contains NumPy arrays of adjacency matrices on the CPU
for i, graph in enumerate(k_hop_subgraphs):
    print(f'Shape of graph {i}:', graph.shape)
    print(graph)

# Save the list of adjacency matrices using pickle
with open('enzymes_adjacency_matrices.pkl', 'wb') as f:
    pickle.dump(k_hop_subgraphs, f)

maxscale = 10  # maxscale can be tuned
degree_sublevel_filtration_dgms = list()
for i in range(len(k_hop_subgraphs)):
    print(i)
    k_hop_subgraph = k_hop_subgraphs[i]
    # numpy_subgraphs = [subgraph.numpy() for subgraph in k_hop_subgraphs]
    # nodes_degree = [np.sum(subgraph, axis=1) for subgraph in numpy_subgraphs]
    nodes_degree = np.sum(k_hop_subgraph, axis=1)
    # contruct simplicial complex
    stb = gd.SimplexTree()
    (xs, ys) = np.where(np.triu(k_hop_subgraph))
    for j in range(k_hop_subgraph.shape[0]):
        stb.insert([j], filtration=-1e10)

    for idx, x in enumerate(xs):
        stb.insert([x, ys[idx]], filtration=-1e10)

    for j in range(k_hop_subgraph.shape[0]):
        # Now pass the scalar value to assign_filtration
        # stb.assign_filtration([j], degree_value)

        stb.assign_filtration([j], nodes_degree[j])

    stb.make_filtration_non_decreasing()
    dgm = stb.persistence()
    pd = [dgm[i][1] if dgm[i][1][1] < maxscale else (dgm[i][1][0], maxscale) for i in np.arange(0, len(dgm), 1)]
    degree_sublevel_filtration_dgms.append(pd)

padded_dgms = pad_sequences(degree_sublevel_filtration_dgms, padding='post', value=0)
sublevel_dgms = np.array(padded_dgms)
np.save('enzy_10_sublevel_dgms', sublevel_dgms)
# sublevel_dgms = np.array(degree_sublevel_filtration_dgms)
# np.save('sublevel_dgms', sublevel_dgms)

A = (600, 600)
# compute the similarity between any pair of nodes
# wasserstein distances calculation
wasserstein_distances = np.zeros(A, dtype=np.float32)
for i in range(A[0] - 1):
    for j in range(i + 1, A[0]):
        wasserstein_distances[i, j] = wasserstein(sublevel_dgms[i], sublevel_dgms[j])
        wasserstein_distances[j, i] = wasserstein(sublevel_dgms[j],
                                                  sublevel_dgms[i])  # wasserstein distance is not symmetric

# np.save(dataset + '_maxscale_' + str(maxscale) + '_sublevel_filtration_wasserstein_distances', wasserstein_distances)
dataset_name = 'ENZYMES'  # Replace 'ENZYMES' with the actual name of your dataset if different
filename = f'{dataset_name}_maxscale_{maxscale}_sublevel_filtration_wasserstein_distances.npy'
np.save(filename, wasserstein_distances)

# recommend for large network
# wasserstein distances calculation
# k = 1 # only consider calculating the distances between node u and its k-hop neighborhood
# G = nx.from_numpy_matrix(adj_array)
# wasserstein_distances = np.zeros(adj_array.shape, dtype = np.float32)
# for i in range(adj_array.shape[0] - 1):
# v_labels = [name for name, value in nx.single_source_shortest_path_length(G, i, cutoff=k).items()]
# for j in v_labels:
# wasserstein_distances[i, j] = wasserstein(sublevel_dgms[i], sublevel_dgms[j])
# wasserstein_distances[j, i] = wasserstein(sublevel_dgms[j], sublevel_dgms[i]) # wasserstein distance is not symmetric
# print(wasserstein_distances[i, j])

# np.savez_compressed(dataset + '_' +  str(k) + '_hop_'+ 'maxscale_' + str(maxscale) + '_sublevel_filtration_wasserstein_distances', wasserstein_distances)
