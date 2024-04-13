import networkx as nx
import numpy as np
import gudhi as gd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

## graph
def separate_TUDataset(graph_list, PI_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [int(graph.y) for graph in graph_list]
    train_idx, test_idx = list(skf.split(np.zeros(len(labels)), labels))[fold_idx]

    train_graph_list = [graph_list[int(i)] for i in train_idx]
    train_PI_list = [PI_list[int(i)] for i in train_idx]

    test_graph_list = [graph_list[int(i)] for i in test_idx]
    test_PI_list = [PI_list[int(i)] for i in test_idx]

    return train_graph_list, train_PI_list, test_graph_list, test_PI_list

## PI loader
def separate_TUDataset_with_calibration_validation(graph_list, PI_list, seed, test_size=0.20,
                                                   calibration_validation_size=0.30, calibration_ratio=0.7):
    assert 0 <= test_size < 1, "test_size must be between 0 and 1."
    assert 0 <= calibration_validation_size < 1, "calibration_validation_size must be between 0 and 1."
    assert 0 <= calibration_ratio <= 1, "calibration_ratio must be between 0 and 1."

    labels = [int(graph.y) for graph in graph_list]

    # Split into initial train+calibration+validation and test sets
    sss_initial = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    initial_train_val_idx, test_idx = next(sss_initial.split(np.zeros(len(labels)), labels))

    # Further split the initial train+calibration+validation set into train and calibration+validation sets
    calibration_validation_ratio = calibration_validation_size / (1 - test_size)
    sss_second = StratifiedShuffleSplit(n_splits=1, test_size=calibration_validation_ratio, random_state=seed)
    train_idx, calibration_validation_idx = next(
        sss_second.split(np.zeros(len(initial_train_val_idx)), np.array(labels)[initial_train_val_idx]))

    # Split calibration+validation set into calibration and validation sets based on calibration_ratio
    num_calibration = int(len(calibration_validation_idx) * calibration_ratio)
    calibration_idx = calibration_validation_idx[:num_calibration]
    validation_idx = calibration_validation_idx[num_calibration:]

    train_graph_list = [graph_list[i] for i in initial_train_val_idx[train_idx]]
    train_PI_list = [PI_list[i] for i in initial_train_val_idx[train_idx]]

    test_graph_list = [graph_list[i] for i in test_idx]
    test_PI_list = [PI_list[i] for i in test_idx]

    calibration_graph_list = [graph_list[i] for i in initial_train_val_idx[calibration_idx]]
    calibration_PI_list = [PI_list[i] for i in initial_train_val_idx[calibration_idx]]

    validation_graph_list = [graph_list[i] for i in initial_train_val_idx[validation_idx]]
    validation_PI_list = [PI_list[i] for i in initial_train_val_idx[validation_idx]]

    return train_graph_list, train_PI_list, test_graph_list, test_PI_list, calibration_graph_list, calibration_PI_list, validation_graph_list, validation_PI_list

def separate_TUDataset_with_calibration_validation_indices(graph_list, PI_list, seed, test_size=0.20,
                                                           calibration_validation_size=0.05, calibration_ratio=0.7):
    assert 0 <= test_size < 1, "test_size must be between 0 and 1."
    assert 0 <= calibration_validation_size < 1, "calibration_validation_size must be between 0 and 1."
    assert 0 <= calibration_ratio <= 1, "calibration_ratio must be between 0 and 1."

    labels = [int(graph.y) for graph in graph_list]

    # Split into initial train+calibration+validation and test sets
    sss_initial = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for initial_train_val_idx, test_idx in sss_initial.split(np.zeros(len(labels)), labels):
        break

    # Further split the initial train+calibration+validation set into train and calibration+validation sets
    calibration_validation_ratio = calibration_validation_size / (1 - test_size)
    sss_second = StratifiedShuffleSplit(n_splits=1, test_size=calibration_validation_ratio, random_state=seed)
    for train_idx, calibration_validation_idx in sss_second.split(np.zeros(len(initial_train_val_idx)),
                                                                  np.array(labels)[initial_train_val_idx]):
        break

    # Remapping calibration+validation indices to global indices
    calibration_validation_idx = initial_train_val_idx[calibration_validation_idx]
    train_idx = initial_train_val_idx[train_idx]

    # Split calibration+validation set into calibration and validation sets based on calibration_ratio
    num_calibration = int(len(calibration_validation_idx) * calibration_ratio)
    calibration_idx = calibration_validation_idx[:num_calibration]
    validation_idx = calibration_validation_idx[num_calibration:]

    # Extracting lists based on global indices
    train_graph_list = [graph_list[i] for i in train_idx]
    train_PI_list = [PI_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]
    test_PI_list = [PI_list[i] for i in test_idx]
    calibration_graph_list = [graph_list[i] for i in calibration_idx]
    calibration_PI_list = [PI_list[i] for i in calibration_idx]
    validation_graph_list = [graph_list[i] for i in validation_idx]
    validation_PI_list = [PI_list[i] for i in validation_idx]

    # Return the subsets and their indices
    return ((train_graph_list, train_PI_list, train_idx),
            (test_graph_list, test_PI_list, test_idx),
            (calibration_graph_list, calibration_PI_list, calibration_idx),
            (validation_graph_list, validation_PI_list, validation_idx))


def pass_data_iteratively(model, graphs, PIs, minibatch_size = 64):
    model.eval()
    outputs = []
    attn_features = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        batch_graph = [graphs[j] for j in sampled_idx]
        batch_PI = torch.stack([PIs[j] for j in sampled_idx])
        output,attn_feature,_ = model(batch_graph,batch_PI)
        outputs.append(output.detach())
        attn_features.append(attn_feature.detach())
    return torch.cat(outputs, 0), torch.cat(attn_features, 0)
def similarity(vali_feature, calib_feature):
    return np.linalg.norm(vali_feature - calib_feature)

def truncate_number(num):
    return max(0, min(1, num))
def first_k_sorted_values_and_indices(arr, k):
    if len(arr) <= 1 or k <= 0:
        return [], []

    # Step 1: Pair the remaining elements with their original indices
    indexed_arr = [(value, index) for index, value in enumerate(arr)]

    # Step 2: Sort these pairs based on the values
    sorted_pairs = sorted(indexed_arr, key=lambda x: x[0])

    # Step 3: Select the first k pairs after sorting
    first_k_pairs = sorted_pairs[:k]

    # Step 4: Extract the values and their original indices
    values, original_indices = zip(*first_k_pairs) if first_k_pairs else ([], [])

    return list(values), list(original_indices)

def plot_roc(result):
    fpr0, tpr0, _ = roc_curve(result['Y'], result['p_hat'])
    roc_auc0 = auc(fpr0, tpr0)

    fpr1, tpr1, _ = roc_curve(result['Y'], result['fix_sen_upper_p'])
    roc_auc_p1 = auc(fpr1, tpr1)

    fpr2, tpr2, _ = roc_curve(result['Y'], result['fix_sen_lower_p'])
    roc_auc_fix1 = auc(fpr2, tpr2)

    fpr3, tpr3, _ = roc_curve(result['Y'], result['fix_spe_upper_p'])
    roc_auc_p = auc(fpr3, tpr3)

    fpr4, tpr4, _ = roc_curve(result['Y'], result['fix_spe_lower_p'])
    roc_auc_fix = auc(fpr4, tpr4)
    print(roc_auc0, roc_auc_p1, roc_auc_fix1, roc_auc_p, roc_auc_fix)

    plt.figure()
    lw = 2  # Line width
    plt.plot(fpr0, tpr0, color='black', lw=lw, label='ROC curve 2 (area = %0.2f)' % roc_auc0)
    plt.plot(fpr1, tpr1, color='orange', lw=lw, label='ROC curve 1 (area = %0.2f)' % roc_auc_p1)
    plt.plot(fpr2, tpr2, color='blue', lw=lw, label='ROC curve 2 (area = %0.2f)' % roc_auc_fix1)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('sen.jpg')
    plt.show()

    plt.figure()
    lw = 2  # Line width
    plt.plot(fpr0, tpr0, color='black', lw=lw, label='ROC curve 2 (area = %0.2f)' % roc_auc0)
    plt.plot(fpr3, tpr3, color='orange', lw=lw, label='ROC curve 1 (area = %0.2f)' % roc_auc_p)
    plt.plot(fpr4, tpr4, color='blue', lw=lw, label='ROC curve 2 (area = %0.2f)' % roc_auc_fix)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('spe.jpg')
    plt.show()

def sublevel_persistence_diagram(A, method, max_scale=50):
    assert method in ['degree','betweenness','communicability','eigenvector','closeness']
    
    G = nx.from_numpy_array(A)
    if method == 'degree':
        node_features = np.sum(A, axis=1)
    elif method == 'betweenness':
        node_features_dict = nx.betweenness_centrality(G)
        node_features = [i for i in node_features_dict.values()]
    elif method == 'communicability':
        node_features_dict = nx.communicability_betweenness_centrality(G)
        node_features = [i for i in node_features_dict.values()]
    elif method == 'eigenvector':
        node_features_dict = nx.eigenvector_centrality(G,max_iter=10000)
        node_features = [i for i in node_features_dict.values()]
    elif method == 'closeness':
        node_features_dict = nx.closeness_centrality(G)
        node_features = [i for i in node_features_dict.values()]
    
    stb = gd.SimplexTree()
    (xs, ys) = np.where(np.triu(A))
    for j in range(A.shape[0]):
        stb.insert([j], filtration=-1e10)

    for idx, x in enumerate(xs):
        stb.insert([x, ys[idx]], filtration=-1e10)

    for j in range(A.shape[0]):
        stb.assign_filtration([j], node_features[j])

    stb.make_filtration_non_decreasing()
    dgm = stb.persistence()
    pd = [dgm[i][1] if dgm[i][1][1] != np.inf else (dgm[i][1][0], max_scale) for i in np.arange(0, len(dgm), 1)]

    return np.array(pd)

def persistence_images(dgm, resolution = [50,50], return_raw = False, normalization = True, bandwidth = 1., power = 1.):
    PXs, PYs = dgm[:, 0], dgm[:, 1]
    xm, xM, ym, yM = PXs.min(), PXs.max(), PYs.min(), PYs.max()
    x = np.linspace(xm, xM, resolution[0])
    y = np.linspace(ym, yM, resolution[1])
    X, Y = np.meshgrid(x, y)
    Zfinal = np.zeros(X.shape)
    X, Y = X[:, :, np.newaxis], Y[:, :, np.newaxis]

    P0, P1 = np.reshape(dgm[:, 0], [1, 1, -1]), np.reshape(dgm[:, 1], [1, 1, -1])
    weight = np.abs(P1 - P0)
    distpts = np.sqrt((X - P0) ** 2 + (Y - P1) ** 2)

    if return_raw:
        lw = [weight[0, 0, pt] for pt in range(weight.shape[2])]
        lsum = [distpts[:, :, pt] for pt in range(distpts.shape[2])]
    else:
        weight = weight ** power
        Zfinal = (np.multiply(weight, np.exp(-distpts ** 2 / bandwidth))).sum(axis=2)

    output = [lw, lsum] if return_raw else Zfinal

    max_output = np.max(output)
    min_output = np.min(output)
    if normalization and (max_output != min_output):
        norm_output = (output - min_output)/(max_output - min_output)
    else:
        norm_output = output

    return norm_output

def compute_PI_tensor(graph_list,PI_dim,sublevel_filtration_methods=['degree','betweenness','communicability','eigenvector','closeness']):
        
        PI_list = []
        for graph in graph_list:
            adj = nx.adjacency_matrix(to_networkx(graph)).todense()
            PI_list_i = [] 
            # PI tensor
            for j in range(len(sublevel_filtration_methods)):
                pd = sublevel_persistence_diagram(adj,sublevel_filtration_methods[j])
                pi = torch.FloatTensor(persistence_images(pd,resolution=[PI_dim]*2))
                PI_list_i.append(pi)
            PI_tensor_i = torch.stack(PI_list_i)
            PI_list.append(PI_tensor_i)

        PI_concat = torch.stack(PI_list)
        return PI_concat