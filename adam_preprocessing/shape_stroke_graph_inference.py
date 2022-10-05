# copied from https://github.com/ASU-APG/adam-stage/tree/main/processing
# original code by Sheng Cheng
from argparse import ArgumentParser
from copy import deepcopy
import logging
from pathlib import Path
from typing import Sequence

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


def update_features_yaml(features_yaml, *, predictions_by_object: Sequence[Sequence[str]]):
    # jac: not terribly efficient to do a full deepcopy, but who cares, this should be a small dict
    # anyway... also this is probably much less expensive than the GNN inference itself.
    result = deepcopy(features_yaml)
    for object_, labels in zip(result["objects"], predictions_by_object):
        if len(labels) == 1:
            object_["stroke_graph"]["concept_name"] = labels[0]
            if "concept_names" in object_["stroke_graph"]:
                del object_["stroke_graph"]["concept_names"]
        else:
            object_["stroke_graph"]["concept_names"] = labels
            if "concept_name" in object_["stroke_graph"]:
                del object_["stroke_graph"]["concept_name"]
    return result


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
    parser.add_argument(
        "--save_outputs_to",
        type=Path,
        default=None,
        help="Directory where we should write the feature outputs to. Outputs are structured as if "
        "the output path is a curriculum directory.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Top k decodes to retrieve from the GNN",
    )
    parser.add_argument(
        "--dir-num",
        type=int,
        default=None,
        help="A specific situation directory to decode.",
    )
    parser.add_argument(
        "--compute-accuracy",
        action="store_true",
        help="Automatically compute GNN accuracy."
    )
    parser.add_argument(
        "--multi-object",
        action="store_true",
        help="Allow for multiple objects in each situation"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not args.model_path.is_file():
        raise ValueError(f"Cannot load model from nonexistent file {args.model_path}.")

    "Processing data from image to stroke graph"
    logging.info("Loading test data...")
    situation_number_to_object_indices, test_coords, test_adj, test_label = get_stroke_data(
        args.curriculum_path,
        "test",
        dir_num=args.dir_num,
        int_curriculum_labels=args.compute_accuracy,
        multi_object=args.multi_object,
    )
    logging.info("Done loading data.")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    """
    Converting stroke graph data for graph node/edge.
    """
    logging.info("Converting data...")
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
    logging.info("Data reformatted.")

    "MPNN model"

    logging.info("Loading model...")
    model_base = MPNN([100, 100], hidden_state_size=100,
                      message_size=20, n_layers=1, l_target=50,
                      type='regression')
    model = LinearModel(model_base, 50, len(STRING_OBJECT_LABELS)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    logging.info("Model loaded.")
    logging.info("Predicting...")
    evaluation = accuracy
    model.eval()
    losses = []
    best_acc = 0
    outputs = None

    "Inference"
    if len(test_coords) > 0:
        outputs = nn.LogSoftmax(dim=1)(
            model(test_adjacency_matrices, test_node_features, test_edge_features)
        )
    else:
        logging.warning("Can't give reasonable output for empty input.")
    if args.compute_accuracy and outputs is not None:
        test_acc = Variable(
            evaluation(
                outputs.data,
                test_label.data,
                topk=(1,),
            )[0]
        )
        logging.info("test acc :{}".format(test_acc))
    logging.info("Done predicting.")

    if args.save_outputs_to:
        logging.info("Saving outputs to %s...", args.save_outputs_to)
        # copied and edited from phase3_load_from_disk() -- see adam.curriculum.curriculum_from_files
        if args.dir_num is None:
            with open(
                args.curriculum_path / "info.yaml", encoding="utf=8"
            ) as curriculum_info_yaml:
                curriculum_params = yaml.safe_load(curriculum_info_yaml)

        n_saved = 0
        objs_seen = 0
        for situation_num in range(curriculum_params["num_dirs"]) if args.dir_num is None else [args.dir_num]:
            input_situation_dir = args.curriculum_path / f"situation_{situation_num}"
            feature_yamls = sorted(input_situation_dir.glob("feature*"))
            if len(feature_yamls) == 1:
                # Load features, update them, then save
                with open(
                    feature_yamls[0], encoding="utf-8"
                ) as feature_yaml_in:
                    features = yaml.safe_load(feature_yaml_in)

                if outputs is None:
                    predictions_by_object = [["unknown"]]
                else:
                    _, predicted_label_ints = outputs.topk(args.top_k)
                    predictions_by_object = [
                        [
                            STRING_OBJECT_LABELS[predicted_label_ints[object_idx][i]]
                            for i in range(args.top_k)
                        ] for object_idx in situation_number_to_object_indices[situation_num]
                    ]
                    objs_seen += len(features['objects'])

                updated_features = update_features_yaml(
                    features,
                    predictions_by_object=predictions_by_object
                )

                output_situation_dir = args.save_outputs_to / f"situation_{situation_num}"
                output_situation_dir.mkdir(exist_ok=True, parents=True)
                with open(
                    output_situation_dir / feature_yamls[0].name,
                    mode="w",
                    encoding="utf-8",
                ) as feature_yaml_out:
                    yaml.safe_dump(updated_features, feature_yaml_out)

            # jac: Need to deal with this when we deal with decode for actions, which will probably
            # require changing the return type here somehow... or other drastic changes.
            else:
                raise NotImplementedError(
                    f"Don't know how to do decode when situation number {situation_num} has more than "
                    f"one feature file."
                )
            n_saved += 1
        logging.info("Saved %d outputs to %s.", n_saved, args.save_outputs_to)
        logging.info("Done saving outputs to %s.", args.save_outputs_to)


if __name__ == "__main__":
    main()
