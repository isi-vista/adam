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
from utils import accuracy, LinearModel, load_data
from shape_stroke_extraction import Stroke_Extraction

string_labels = [
    'triangleblock',
    'banana',
    'book',
    'ball',
    'toytruck',
    'floor',
    'sofa',
    'orange',
    'window',
    'table',
    'cubeblock',
    'toysedan',
    'box',
    'cup',
    'sphereblock',
    'chair',
    'desk',
    'apple',
    'paper'

]

phases = ['train', 'test']




if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "curriculum_path",
        type=Path,
        help="Where to read the curriculum from and write the outputs to.",
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
    base_path = args.curriculum_path
    for phase in phases:
        if phase == 'train':
            num_sample = 10
            train_coords = []
            train_adj = []
            train_label = []
        else:
            num_sample = 5
            test_coords = []
            test_adj = []
            test_label = []
        for integer_label, string_label in enumerate(string_labels):
            for object_view in range(1):
                for sample_id in range(num_sample):
                    if not os.path.exists(
                            os.path.join(base_path, 'feature', 'feature_{}_{}_{}_{}.yaml'.format(phase, string_label, object_view, sample_id))):
                        Extractor = Stroke_Extraction(obj_type="{}_{}".format(phase, string_label), obj_id=sample_id, obj_view=object_view,
                                                      base_path=base_path, vis=True, save_output=True)
                        out = Extractor.get_strokes()
                    else:
                        with open(os.path.join(base_path, 'feature', 'feature_{}_{}_{}_{}.yaml'.format(phase, string_label, object_view, sample_id)),
                                  'r') as stream:
                            out = yaml.safe_load(stream)
                    num_obj = len(out)
                    if num_obj == 0:
                        continue
                    coords = []
                    adj = []
                    count = [0]
                    for object_number in range(num_obj):
                        coords.append(out[object_number]['stroke_graph']['strokes_normalized_coordinates'])
                        adj.append(out[object_number]['stroke_graph']['adjacency_matrix'])
                        count.append(len(out[object_number]['stroke_graph']['adjacency_matrix']) + count[-1])
                    coords = np.asarray(coords)
                    coords = np.concatenate(coords, 0)
                    real_adj = np.zeros([count[-1], count[-1]])
                    for object_number in range(num_obj):
                        real_adj[count[object_number]:count[object_number + 1], count[object_number]:count[object_number + 1]] = adj[object_number]
                    if phase == 'train':
                        train_coords.append(coords)
                        train_adj.append(real_adj)
                        train_label.append(integer_label)
                    else:
                        test_coords.append(coords)
                        test_adj.append(real_adj)
                        test_label.append(integer_label)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # data = from_pickle('./data.pkl')
    # train_coords = data['train_coords']
    # train_adj = data['train_adj']
    # train_label = data['train_label']
    # test_coords = data['test_coords']
    # test_adj = data['test_adj']
    # test_label = data['test_label']

    "Converting stroke graph data for graph node/edge. "
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

    "MPNN model"

    model_base = MPNN([100, 100], hidden_state_size=100,
                      message_size=20, n_layers=1, l_target=50,
                      type='regression')
    model = LinearModel(model_base, 50, len(string_labels)).to(device)
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
        output = model(train_g, train_h, train_e)
        train_loss = criterion(nn.LogSoftmax()(output), train_target)
        losses.append(train_loss.item())
        acc = Variable(evaluation(output.data, train_target.data, topk=(1,))[0])
        test_acc = Variable(
            evaluation(
                (
                    nn.LogSoftmax()(
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
            ),
            flush=True,
        )
    logging.info("Best test acc is {}".format(best_acc))
    logging.info(f"Saving model state dict to {args.save_model_to}")
    torch.save(model.state_dict(), args.save_model_to)
    logging.info("Model saved.")
