# copied from https://github.com/ASU-APG/adam-stage/tree/main/processing
# original code by Sheng Cheng
from argparse import ArgumentParser
import logging
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
        "train_curriculum_path",
        type=Path,
        help="Train curriculum. Where to read the curriculum from and write the outputs to.",
    )
    parser.add_argument(
        "eval_curriculum_path",
        type=Path,
        help="Test curriculum. Where to read the curriculum from and write the outputs to.",
    )
    parser.add_argument(
        "save_model_to",
        type=Path,
        help="Where to save the model state dict.",
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=1000,
        help="Number of steps or epochs to train the model for. (In this script steps = epochs "
        "because our batch size is 'all of the input data'.)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    "Processing data from image to stroke graph"
    logging.info("Loading training data...")
    train_coords, train_adj, train_label = get_stroke_data(
        args.train_curriculum_path, "train"
    )
    logging.info("Loading test data...")
    test_coords, test_adj, test_label = get_stroke_data(args.eval_curriculum_path, "test")
    logging.info("Done loading data.")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    "Converting stroke graph data for graph node/edge. "
    logging.info("Converting data to new format...")
    nodes, edges, num_nodes = load_data(train_coords, train_adj)
    train_node_features = np.zeros([len(train_coords), num_nodes, 100])
    train_adjacency_matrices = np.zeros([len(train_coords), num_nodes, num_nodes])
    train_edge_features = np.zeros([len(train_coords), num_nodes, num_nodes, 100])
    for sample_index in range(len(train_coords)):
        num_node = len(train_adj[sample_index])
        train_node_features[sample_index, :num_node, :] = nodes[sample_index]
        train_adjacency_matrices[sample_index, :num_node, :num_node] = np.array(
            train_adj[sample_index]
        )
        for edge in edges[sample_index].keys():
            train_edge_features[sample_index, edge[0], edge[1], :] = edges[sample_index][
                edge
            ]
            train_edge_features[sample_index, edge[1], edge[0], :] = edges[sample_index][
                edge
            ]
    train_node_features = (
        torch.tensor(train_node_features).type(torch.FloatTensor).to(device)
    )
    train_edge_features = (
        torch.tensor(train_edge_features).type(torch.FloatTensor).to(device)
    )
    train_adjacency_matrices = (
        torch.tensor(train_adjacency_matrices).type(torch.FloatTensor).to(device)
    )
    label = torch.tensor(np.array(train_label)).type(torch.LongTensor).to(device)
    nodes_t, edges_t, num_nodes_t = load_data(test_coords, test_adj)
    test_node_features = np.zeros([len(test_coords), num_nodes_t, 100])
    test_adjacency_matrices = np.zeros([len(test_coords), num_nodes_t, num_nodes_t])
    test_edge_features = np.zeros([len(test_coords), num_nodes_t, num_nodes_t, 100])
    for sample_index in range(len(test_coords)):
        num_node = len(test_adj[sample_index])
        test_node_features[sample_index, :num_node, :] = nodes_t[sample_index]
        test_adjacency_matrices[sample_index, :num_node, :num_node] = np.array(
            test_adj[sample_index]
        )
        for edge in edges_t[sample_index].keys():
            test_edge_features[sample_index, edge[0], edge[1], :] = edges_t[sample_index][
                edge
            ]
            test_edge_features[sample_index, edge[1], edge[0], :] = edges_t[sample_index][
                edge
            ]
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
    logging.info("Done converting data format.")
    "MPNN model"
    logging.info("Optimizing model...")
    model_base = MPNN(
        [100, 100], hidden_state_size=100, message_size=20, n_layers=1, l_target=50, type='regression'
        )
    model = LinearModel(model_base, 50, len(STRING_OBJECT_LABELS)).to(device)
    criterion = nn.NLLLoss()
    evaluation = accuracy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    losses = []
    best_acc = 0
    "Training"
    for step in range(args.num_train_steps):
        train_g, train_h, train_e, train_target = (
            Variable(train_adjacency_matrices),
            Variable(train_node_features),
            Variable(train_edge_features),
            Variable(label),
        )
        optimizer.zero_grad()
        # output should have shape (N_samples, n_classes)
        output = model(train_g, train_h, train_e)
        train_loss = criterion(nn.LogSoftmax(dim=1)(output), train_target)
        losses.append(train_loss.item())
        acc = Variable(evaluation(output.data, train_target.data, topk=(1,))[0])
        test_acc = Variable(
            evaluation(
                (
                    nn.LogSoftmax(dim=1)(
                        model(
                            test_adjacency_matrices,
                            test_node_features,
                            test_edge_features,
                        )
                    )
                ).data,
                test_label.data,
                topk=(1,),
            )[0]
        )
        if test_acc > best_acc:
            best_acc = test_acc
        train_loss.backward()
        optimizer.step()
        logging.info(
            "{}, loss: {:.4e}, acc: {}, test acc :{}".format(
                step, train_loss.item(), acc, test_acc
            )
        )
    logging.info("Best test acc is {}".format(best_acc))
    logging.info(f"Saving model state dict to {args.save_model_to}")
    torch.save(model.state_dict(), args.save_model_to)
    logging.info("Model saved.")


if __name__ == "__main__":
    main()
