import numpy as np
import torch
import torch.nn as nn
from util import similarity, first_k_sorted_values_and_indices, pass_data_iteratively, truncate_number
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

def tps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    cal_scores = 1-cal_smx[np.arange(n),cal_labels]
    q_level = np.ceil((n+1)*(1-alpha))/n
    qhat = np.quantile(cal_scores, q_level, method='higher')
    prediction_sets = val_smx >= (1-qhat)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff
def aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha = 0.1):
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]
    qhat = np.quantile(
        cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher"
    )
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff

def raps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    lam_reg = 0.01
    k_reg = min(5, cal_smx.shape[1])
    disallow_zero_sets = False
    rand = True
    reg_vec = np.array(k_reg*[0,] + (cal_smx.shape[1]-k_reg)*[lam_reg,])[None,:]

    cal_pi = cal_smx.argsort(1)[:,::-1];
    cal_srt = np.take_along_axis(cal_smx,cal_pi,axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:,None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n),cal_L]
    # Get the score quantile
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')
    # Deploy
    n_val = val_smx.shape[0]
    val_pi = val_smx.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_smx,val_pi,axis=1)
    val_srt_reg = val_srt + reg_vec
    val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
    indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
    if disallow_zero_sets: indicators[:,0] = True
    prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff

def conformal_test_unconditional(model, device, calib_graphs, calib_PIs, vali_graphs, vali_PIs, conformal_alpha = 0.1, conf_score = 'tps'):
    model.eval()
    print("--------Enter unconditional conformal test---------")

    smx = nn.Softmax(dim=1)
    prob_output, _ = pass_data_iteratively(model, calib_graphs, calib_PIs, 16)
    prob_cal = smx(prob_output).detach().cpu().numpy()
    if len(vali_graphs) == 1:
        PIs = torch.stack(vali_PIs)
        output,_,_ = model(vali_graphs,PIs)
    else:
        output, a = pass_data_iteratively(model, vali_graphs, vali_PIs, 16)

    prob_val = smx(output).detach().cpu().numpy()
    cal_labels = torch.LongTensor([graph.y for graph in calib_graphs]).detach().cpu().numpy()
    val_labels = torch.LongTensor([graph.y for graph in vali_graphs]).detach().cpu().numpy()

    if conf_score == 'aps':
        prediction_sets, cov, eff = aps(prob_cal,prob_val,cal_labels,val_labels, len(calib_graphs), conformal_alpha)
    elif conf_score == 'raps':
        prediction_sets, cov, eff = raps(prob_cal, prob_val, cal_labels, val_labels, len(calib_graphs), conformal_alpha)
    elif conf_score == 'tps':
        prediction_sets, cov, eff = tps(prob_cal, prob_val, cal_labels, val_labels, len(calib_graphs), conformal_alpha)

    return prediction_sets, cov, eff

def conformal_test_conditional(model, device, calib_graphs, calib_PIs, calib_idx, vali_graphs, vali_PIs, vali_idx, TDA_feature,
                               conformal_alpha = 0.1, conf_score = 'tps', k = 50, trained_TDA_similarity = False):
    model.eval()
    print("--------Enter conditional conformal test---------")

    smx = nn.Softmax(dim=1)

    if not trained_TDA_similarity:
        print("Using original TDA feature for similarity")
    else:
        print("Using trained TDA feature for similarity")
    prediction_sets = []
    prob_output_candidate, feature_output = pass_data_iteratively(model, calib_graphs, calib_PIs, 4)
    for i, vali_graph in enumerate(vali_graphs):
        output, vali_attn_feature, _ = model([vali_graph], vali_PIs[i].unsqueeze(0))
        prob_val = smx(output).detach().cpu().numpy()
        val_labels = torch.LongTensor(vali_graph.y).detach().cpu().numpy()

        n = len(calib_graphs)
        if len(calib_graphs) <= k:
            prob_output, _ = pass_data_iteratively(model, calib_graphs, calib_PIs, 4)
            similar_graphs_labels = torch.LongTensor(
                [similar_graph.y for similar_graph in calib_graphs]).detach().cpu().numpy()
            prob_temp = prob_output
        else:
            if not trained_TDA_similarity:
                similarity_scores = TDA_feature[vali_idx[i]][calib_idx]
                _, similar_graphs_indices = first_k_sorted_values_and_indices(similarity_scores, k)
            else:
                similarity_scores = [(i, similarity(vali_attn_feature.detach().cpu().numpy(), feature.detach().cpu().numpy())) for i, feature in enumerate(feature_output)]
                similarity_scores.sort(key=lambda x: x[1])
                similar_graphs_indices = [index for index, _ in similarity_scores[:k]]
            similar_graphs = [calib_graphs[index] for index in similar_graphs_indices]
            prob_output = [prob_output_candidate[index] for index in similar_graphs_indices]
            similar_graphs_labels = torch.LongTensor(
                [similar_graph.y for similar_graph in similar_graphs]).detach().cpu().numpy()
            n = k
            prob_temp = torch.stack(prob_output)
        prob_cal = smx(prob_temp).detach().cpu().numpy()

        if conf_score == 'aps':
            prediction_set, _, _ = aps(prob_cal, prob_val, similar_graphs_labels, val_labels, n,
                                            conformal_alpha)
        elif conf_score == 'raps':
            prediction_set, _, _ = raps(prob_cal, prob_val, similar_graphs_labels, val_labels, n,
                                             conformal_alpha)
        elif conf_score == 'tps':
            prediction_set, _, _ = tps(prob_cal, prob_val, similar_graphs_labels, val_labels, n,
                                            conformal_alpha)
        prediction_set = prediction_set.flatten()
        prediction_sets.append(prediction_set)

    prediction_sets = np.array(prediction_sets)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]), val_labels].mean()
    eff = np.sum(prediction_sets) / len(prediction_sets)
    # print(prediction_sets)
    sd = np.std(prediction_sets)

    return prediction_sets, cov, eff, sd

def conformal_test_ROC(model, device, graphs, PIs, tst_index, obs_index, distance_data, train_size=0.5, seed=42, number_nearest=100, exchangeable = True):
    model.eval()
    if exchangeable: print("--------Enter Exchangeable ROC conformal test---------")
    else: print("--------Enter Non-Exchangeable ROC conformal test---------")
    smx = nn.Softmax(dim=1)
    np.random.seed(seed)
    # Assuming data is a NumPy array with 2 columns: ["p_hat", "Y"]
    prob, _ = pass_data_iteratively(model, graphs, PIs, 4)
    prob1 = smx(prob)[:, 1].detach().cpu().numpy() # Get probability for class 1
    y = np.array([graph.y for graph in graphs]).squeeze()
    data = np.vstack((prob1, y)).T

    tst_data = data[tst_index, :]
    seed_pool = np.random.randint(1, 1000000, size=len(tst_index))
    tst_output = np.zeros((len(tst_index), 6))

    for i, idx in enumerate(tst_index):
        # Split the observation data for each test instance using the provided obs_index
        train_indices, cal_indices = train_test_split(obs_index, train_size=train_size, random_state=seed_pool[i])

        if not exchangeable:
            tst_cal_dis_data = distance_data[idx, cal_indices]
            nearest_indices = np.argsort(tst_cal_dis_data)[:number_nearest]
            selected_cal_indices = cal_indices[nearest_indices]
            cal_indices = selected_cal_indices

        Cs = np.zeros((len(cal_indices), 2))
        for j, cal_idx in enumerate(cal_indices):
            # Use distance_data to find nearest neighbors
            distances = distance_data[cal_idx, train_indices]
            nearest_indices = np.argsort(distances)[:number_nearest]

            p_hat = data[cal_idx, 0]
            p_t = np.mean(data[train_indices][nearest_indices, 0])
            temp_Cs = p_t - p_hat
            Cs[j, 0] = temp_Cs
            Cs[j, 1] = data[cal_idx, 1]

        # Calculate quantiles for sensitivity and specificity adjustments
        fix_sen_quantile_95 = np.percentile(Cs[Cs[:, 1] == 0.0][:, 0], 95)
        fix_sen_quantile_5 = np.percentile(Cs[Cs[:, 1] == 0.0][:, 0], 5)
        fix_spe_quantile_95 = np.percentile(Cs[Cs[:, 1] == 1.0][:, 0], 95)
        fix_spe_quantile_5 = np.percentile(Cs[Cs[:, 1] == 1.0][:, 0], 5)

        # Update the test output array
        tst_output[i, 0] = tst_data[i, 0]
        tst_output[i, 1] = truncate_number(tst_data[i, 0] + fix_sen_quantile_95)
        tst_output[i, 2] = truncate_number(tst_data[i, 0] + fix_sen_quantile_5)
        tst_output[i, 3] = truncate_number(tst_data[i, 0] + fix_spe_quantile_95)
        tst_output[i, 4] = truncate_number(tst_data[i, 0] + fix_spe_quantile_5)
        tst_output[i, 5] = tst_data[i, 1]

    senCI_len = np.mean(tst_output[:, 1] - tst_output[:, 2])
    condition = np.logical_and(tst_output[:, 1] > tst_output[:, 0], tst_output[:, 2] < tst_output[:, 0])
    cov_sen = np.sum(condition) / len(tst_index)
    speCI_len = np.mean(tst_output[:, 3] - tst_output[:, 4])
    condition = np.logical_and(tst_output[:, 3] > tst_output[:, 0], tst_output[:, 4] < tst_output[:, 0])
    cov_spe = np.sum(condition) / len(tst_index)
    tst_output_df = pd.DataFrame(tst_output, columns=["p_hat", "fix_sen_upper_p", "fix_sen_lower_p", "fix_spe_upper_p", "fix_spe_lower_p", "Y"])

    return tst_output_df, senCI_len, cov_sen, speCI_len, cov_spe

# Define or adjust the pass_data_iteratively function and any other necessary components to fit this implementation.
