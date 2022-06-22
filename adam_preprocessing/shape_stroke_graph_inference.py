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

label_name = [
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

phase = ['test']




if __name__ == "__main__":
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
    base_path = args.curriculum_path
    for i in phase:
        if i == 'train':
            num_sample = 10
            train_coords = []
            train_adj = []
            train_label = []
        else:
            num_sample = 5
            test_coords = []
            test_adj = []
            test_label = []
            test_camera = []
        for l, j in enumerate(label_name):
            for p in range(3):
                for q in range(num_sample):

                    "If infomation saved, load it"
                    if not os.path.exists(
                            os.path.join(base_path, 'feature', 'feature_{}_{}_{}_{}.yaml'.format(i, j, p, q))):
                        try:
                            Extractor = Stroke_Extraction(obj_type="{}_{}".format(i, j), obj_id=q, obj_view=p,
                                                          base_path=base_path, vis=True, save_output=True)
                            out = Extractor.get_strokes()
                        except:
                            print("no strokes")
                            out = []
                    else:
                        with open(os.path.join(base_path, 'feature', 'feature_{}_{}_{}_{}.yaml'.format(i, j, p, q)),
                                  'r') as stream:
                            out = yaml.safe_load(stream)
                    num_obj = len(out)
                    if num_obj == 0:
                        continue
                    coords = []
                    adj = []
                    count = [0]

                    "Merge multiple stroke graph if only one object in the image"

                    for ii in range(num_obj):
                        coords.append(out[ii]['stroke_graph']['strokes_normalized_coordinates'])
                        adj.append(out[ii]['stroke_graph']['adjacency_matrix'])
                        count.append(len(out[ii]['stroke_graph']['adjacency_matrix']) + count[-1])
                    coords = np.asarray(coords)
                    coords = np.concatenate(coords, 0)
                    real_adj = np.zeros([count[-1], count[-1]])
                    for iii in range(num_obj):
                        real_adj[count[iii]:count[iii + 1], count[iii]:count[iii + 1]] = adj[iii]
                    if i == 'train':
                        train_coords.append(coords)
                        train_adj.append(real_adj)
                        train_label.append(l)
                    else:
                        test_coords.append(coords)
                        test_adj.append(real_adj)
                        test_label.append(l)
                        test_camera.append(p)

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
    model = LinearModel(model_base, 50, len(label_name)).to(device)
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
