import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.datasets import TUDataset

from tqdm import trange

from util import separate_TUDataset, compute_PI_tensor, pass_data_iteratively, plot_roc
from util import separate_TUDataset_with_calibration_validation_indices as CPDataset2
from models.tensorgcn import TenGCN
from models.conformal import conformal_test_unconditional, conformal_test_conditional, conformal_test_ROC
import matplotlib.pyplot as plt
criterion = nn.CrossEntropyLoss()
def train(args, model, device, train_graphs, train_PIs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = trange(total_iters, unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        batch_PI = torch.stack([train_PIs[idx] for idx in selected_idx])
        output,_,_ = model(batch_graph,batch_PI)

        labels = torch.LongTensor([graph.y for graph in batch_graph]).to(device)

        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()         
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    return average_loss

def test(model, device, test_graphs, test_PIs):
    model.eval()
    smx = nn.Softmax(dim=1)
    output,_= pass_data_iteratively(model, test_graphs, test_PIs)
    output = smx(output)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.y for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))
    
    print("accuracy test: %f" % acc_test)
    return acc_test

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="DD",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--conf_alpha', type=float, default=0.1,
                        help='user-defined coverage for conformal prediction')
    parser.add_argument('--conf_score', type=str, default='tps',
                        help='method to compute non-conformity score')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=7,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of GCN layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    parser.add_argument('--conditional', action="store_true", default=True,
                        help='whether use conditional conformal prediction or  not')
    parser.add_argument('--ROC_exchangeable', action="store_true", default=True,
                        help='whether use exchangeable ROC conformal prediction or  not')
    parser.add_argument('--k', type=int, default=100,
                        help='number of neighbors when doing non-exchangeable conformal prediction(default: 100)')
    parser.add_argument('--trained_TDA_similarity', action="store_true", default=True,
                        help='use trained TDA feature for similarity computation or not when doing the conditional conformal prediction')
    parser.add_argument('--use_test_CP', action="store_true", default=True,
                        help='use test set for conformal prediction')
    # below are new model specific arguments
    parser.add_argument('--tensor_layer_type', type = str, default = "TCL", choices=["TCL","TRL"],
                                        help='Tensor layer type: TCL/TRL')
    parser.add_argument('--node_pooling', action="store_false",
    					help='node pooling based on node scores')
    # NOTE
    # PROTEINS: ['degree','betweenness','closeness']
    # ENZYMES: ['degree','betweenness','eigenvector','closeness']
    # DD: ['degree','betweenness','eigenvector','closeness']
    parser.add_argument('--sublevel_filtration_methods', nargs='+', type=str, default=['degree','betweenness','eigenvector','closeness'],
    					help='Methods for sublevel filtration on PDs')
    parser.add_argument('--PI_dim', type=int, default=50,
                        help='PI size: PI_dim * PI_dim')
    args = parser.parse_args()

    random_seed = 49
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    if args.trained_TDA_similarity:
        TDA_feature = None
    else:
        TDA_feature = np.load('{}_maxscale_10_sublevel_filtration_wasserstein_distances.npy'.format(args.dataset))

    graphs = TUDataset(root='/tmp/' + args.dataset, name=args.dataset)
    num_classes = graphs.num_classes

    ## NOTE: compute graph PI tensor if necessary
    #PIs = compute_PI_tensor(graphs,args.PI_dim,args.sublevel_filtration_methods).to(device)
    #torch.save(PIs,'{}_{}_PI.pt'.format(args.dataset,args.PI_dim))

    ## load pre-computed PIs
    PIs = torch.load('{}_{}_PI.pt'.format(args.dataset,args.PI_dim)).to(device)
    print('finished loading PI for dataset {} with PI_dim = {}'.format(args.dataset,args.PI_dim))

    #data split
    (train_graphs, train_PIs, train_idx), (test_graphs, test_PIs, test_idx),(calib_graphs, calib_PIs, calib_idx),(vali_graphs, vali_PIs, vali_idx) = CPDataset2(graphs, PIs, args.seed)

    ## Training
    model = TenGCN(args.num_layers, args.num_mlp_layers, train_graphs[0].x.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.tensor_layer_type, args.node_pooling, args.PI_dim, args.sublevel_filtration_methods, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    max_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print("Current epoch is:", epoch)

        avg_loss = train(args, model, device, train_graphs, train_PIs, optimizer, epoch)
        scheduler.step()
        acc_test = test(model, device, test_graphs, test_PIs)

        max_acc = max(max_acc, acc_test)

        if not args.filename == "":
            with open(args.filename, 'a+') as f:
                f.write("%f %f %f" % (avg_loss, acc_test))
                f.write("\n")
    save_path = f'model_pth/model_{args.dataset}_epoch{args.epochs}_batch{args.batch_size}.pth'
    torch.save(model.state_dict(), save_path)

    #load trained model
    # model_state = torch.load('model_pth\model_PROTEINS_epoch80_batch16.pth', map_location=torch.device('cuda'))
    # model.load_state_dict(model_state)


    ####CP test
    # target_graphs, target_PIs, target_idx = vali_graphs, vali_PIs, vali_idx
    # if num_classes == 2: #####ROC test
    #     df_obs_indices = np.concatenate((train_idx, vali_idx, calib_idx))
    #     result, senCI_len, cov_sen, speCI_len, cov_spe = conformal_test_ROC(model, device, graphs, PIs, test_idx,
    #                                                                         df_obs_indices, distance_data=TDA_feature,
    #                                                                         exchangeable=args.ROC_exchangeable, number_nearest= args.k)
    #     plot_roc(result)
    #     result.to_csv(f'ROC_result/model_{args.dataset}_epoch{args.epochs}_batch{args.batch_size}_exchangeable{args.ROC_exchangeable}_knn{args.k}.csv', index=False)
    #     print(f"coverage rate for sensitivity is: {cov_sen}, for specificity is: {cov_spe}")
    #     print(f"The average length of the confidence interval for sensitivity is: {senCI_len}, for specificity is: {speCI_len}")
    # else:
    #     if args.use_test_CP:
    #         print("Use test set for conformal prediction:")
    #         target_graph, target_PIs, target_idx = test_graphs, test_PIs, test_idx
    #     if args.conditional:
    #         prediction_sets, cov, eff, se = conformal_test_conditional(model, device, calib_graphs, calib_PIs, calib_idx, target_graphs,
    #                                                                  target_PIs, target_idx, TDA_feature, args.conf_alpha, args.conf_score, args.k, args.trained_TDA_similarity)
    #     else:
    #         prediction_sets, cov, eff = conformal_test_unconditional(model, device, calib_graphs, calib_PIs, target_graphs,
    #                                                                target_PIs, args.conf_alpha, args.conf_score)
    #
    #     print(f"The proportion of validation samples whose true labels are included in their prediction sets is: {cov}, under the {(1- args.conf_alpha)*100}% coverage")
    #     print(f"The average size of the prediction sets across all validation samples is: {eff,se}")

    with open('acc_results.txt', 'a+') as f:
        f.write(str(args.dataset) + ' ' + str(args.fold_idx) + ' ' + str(max_acc) + '\n')

if __name__ == '__main__':
    main()