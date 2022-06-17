# copied from https://github.com/ASU-APG/adam-stage/tree/main/processing
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import yaml
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import glob
import pickle as pickle
from MPNN import MPNN, MPNN_Linear
from utils import accuracy, get_stroke_data, LinearModel, load_data, STRING_OBJECT_LABELS


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_path",
        type=Path,
        help="The model path to use",
    )
    parser.add_argument(
        "curriculum_path",
        type=Path,
        help="Directory we should read the curriculum from.",
    )
    args = parser.parse_args()

    if not args.model_path.is_file():
        raise ValueError(f"Cannot load model from nonexistent file {args.model_path}.")

    "Processing data from image to stroke graph"
    print("Loading test data...")
    test_coords, test_adj, test_label = get_stroke_data(args.curriculum_path, "test")
    print("Done loading data.")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # data = from_pickle('./data.pkl')
    # train_coords = data['train_coords']
    # train_adj = data['train_adj']
    # train_label = data['train_label']
    # test_coords = data['test_coords']
    # test_adj = data['test_adj']
    # test_label = data['test_label']
    """
    Converting stroke graph data for graph node/edge.
    """
    nodes_t, edges_t, num_nodes_t = load_data(test_coords, test_adj)
    test_node_features = np.zeros([len(test_coords), num_nodes_t, 100])
    test_adjacency_matrices = np.zeros([len(test_coords), num_nodes_t, num_nodes_t])
    test_edge_features = np.zeros([len(test_coords), num_nodes_t, num_nodes_t, 100])

    for i in range(len(test_coords)):
        num_node = len(test_adj[i])
        test_node_features[i, :num_node, :] = nodes_t[i]
        test_adjacency_matrices[i, :num_node, :num_node] = np.array(test_adj[i])
        for edge in edges_t[i].keys():
            test_edge_features[i, edge[0], edge[1], :] = edges_t[i][edge]
            test_edge_features[i, edge[1], edge[0], :] = edges_t[i][edge]
    test_node_features = (
        torch.tensor(test_node_features).type(torch.FloatTensor).to(device)
    )
    test_edge_features = (
        torch.tensor(test_edge_features).type(torch.FloatTensor).to(device)
    )
    test_adjacency_matrices = (
        torch.tensor(test_adjacency_matrices).type(torch.FloatTensor).to(device)
    )
    test_label = torch.tensor(np.array(test_label)).type(torch.LongTensor).to(device)

    "MPNN model"

    model_base = MPNN([100, 100], hidden_state_size=100,
                      message_size=20, n_layers=1, l_target=50,
                      type='regression')
    model = LinearModel(model_base, 50, len(STRING_OBJECT_LABELS)).to(device)
    model.load_state_dict(torch.load(args.model_path))
    evaluation = accuracy
    model.eval()
    losses = []
    best_acc = 0

    "Inference"
    test_acc = Variable(
        evaluation(
            (
                nn.LogSoftmax()(
                    model(test_adjacency_matrices, test_node_features, test_edge_features)
                )
            ).data,
            test_label.data,
            topk=(1,),
        )[0]
    )
    print("test acc :{}".format(test_acc))


if __name__ == "__main__":
    main()
