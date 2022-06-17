# copied from https://github.com/ASU-APG/adam-stage/tree/main/processing
from argparse import ArgumentParser
import logging
from pathlib import Path
from typing import Sequence, Tuple

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

string_labels = [
    'pyramid_block',
    'banana',
    'book',
    'ball',
    'toy_truck',
    'floor',
    'sofa',
    'orange',
    'window',
    'table',
    'cube_block',
    'toy_sedan',
    'box',
    'cup',
    'sphere_block',
    'chair',
    'desk',
    'apple',
    'paper',
    'mug',
]


def get_stroke_data(curriculum_path: Path, train_or_test: str) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[int]]:
    # copied and edited from phase3_load_from_disk() -- see adam.curriculum.curriculum_from_files
    with open(curriculum_path / "info.yaml", encoding="utf=8") as curriculum_info_yaml:
        curriculum_params = yaml.safe_load(curriculum_info_yaml)

    curriculum_coords = []
    curriculum_adjs = []
    curriculum_labels = []
    for situation_num in range(curriculum_params["num_dirs"]):
        situation_dir = curriculum_path / f"situation_{situation_num}"
        language_tuple: Tuple[str, ...] = tuple()
        if (situation_dir / "description.yaml").exists():
            with open(
                situation_dir / "description.yaml", encoding="utf-8"
            ) as situation_description_file:
                situation_description = yaml.safe_load(situation_description_file)
            language_tuple = tuple(situation_description["language"].split(" "))
        elif train_or_test == "train":
            raise ValueError(
                f"Training situations must provide a description, but situation {situation_num} "
                f"in {curriculum_path} does not."
            )

        integer_label, string_label = label_from_object_language_tuple(language_tuple, situation_num)
        feature_yamls = sorted(situation_dir.glob("feature*"))
        if len(feature_yamls) == 1:
            with open(situation_dir / feature_yamls[0], encoding="utf-8") as feature_yaml_in:
                features = yaml.safe_load(feature_yaml_in)

            num_obj = len(features["objects"])
            if num_obj == 0:
                continue

            # Sometimes the object detection/stroke extraction process picks up on multiple objects.
            # When that happens, we want to combine them into one mega-example.
            # In this mega-example, each distinct object is treated as its own connected component
            # in a larger graph.
            #
            # First we'll grab the coordinates since that's the cleaner part.
            # Concatenate together the coords for each connected component to get a single
            # (n_strokes_overall, 2, n_control_points) coordinates array
            coords = np.concatenate(
                [
                    np.asarray(object_["stroke_graph"]["strokes_normalized_coordinates"])
                    for object_ in features["objects"]
                ],
                axis=0
            )

            # Now create together the big adjacency matrix
            n_nodes_overall = coords.shape[0]
            # jac: the above should do the same thing but I'm asserting to make sure it does.
            # for clarity's sake, I should delete this once I verify it works.
            assert n_nodes_overall == sum(
                len(object_["stroke_graph"]["adjacency_matrix"]) for object_ in features["objects"]
            )
            adj = np.zeros([n_nodes_overall, n_nodes_overall])

            nodes_seen = 0
            for idx, object_ in enumerate(features["objects"]):
                # Set the appropriate submatrix of adj equal to this object's adjacency matrix.
                n_nodes = len(object_["stroke_graph"]["adjacency_matrix"])
                adj[
                    nodes_seen:nodes_seen + n_nodes,
                    nodes_seen:nodes_seen + n_nodes,
                ] = np.asarray(object_["stroke_graph"]["adjacency_matrix"])

            curriculum_coords.append(coords)
            curriculum_adjs.append(adj)
            curriculum_labels.append(integer_label)

        elif train_or_test == "train":
            raise ValueError(f"Situation number {situation_num} has more than one feature file.")
        # jac: Need to deal with this when we deal with decode for actions, which will probably
        # require changing the return type here somehow... or other drastic changes.
        else:
            raise NotImplementedError(
                f"Don't know how to do decode when situation number {situation_num} has more than "
                f"one feature file."
            )

    return curriculum_coords, curriculum_adjs, curriculum_labels


def label_from_object_language_tuple(language_tuple: Tuple[str, ...], situation_num: int) -> Tuple[int, str]:
    if len(language_tuple) > 2:
        raise ValueError(f"Don't know how to deal with long object name: {language_tuple}")
    elif not language_tuple:
        raise ValueError(f"Can't extract label from empty object name: {language_tuple}")
    elif len(language_tuple) == 1:
        logging.warning(
            "Language tuple for object situation number %d is shorter than expected; this "
            "might cause problems.",
            situation_num,
        )

    string_label = language_tuple[-1]
    if string_label not in string_labels:
        raise ValueError(f"Unrecognized label: {string_label}")
    return string_labels.index(string_label), string_label


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
    train_coords, train_adj, train_label = get_stroke_data(
        args.train_curriculum_path, "train"
    )
    test_coords, test_adj, test_label = get_stroke_data(args.eval_curriculum_path, "test")
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
    model_base = MPNN(
        [100, 100], hidden_state_size=100, message_size=20, n_layers=1, l_target=50, type='regression'
        )
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


if __name__ == "__main__":
    main()
