# copied from https://github.com/ASU-APG/adam-stage/tree/main/processing
# original code by Sheng Cheng
import logging
from pathlib import Path
from typing import Sequence, Tuple, Optional
import yaml

import torch.nn as nn
import pickle
import numpy as np
import torch


STRING_OBJECT_LABELS = [
    "pyramid_block",
    "banana",
    "book",
    "ball",
    "toy_truck",
    "floor",
    "sofa",
    "orange",
    "window",
    "table",
    "cube_block",
    "toy_sedan",
    "box",
    "cup",
    "sphere_block",
    "chair",
    "desk",
    "apple",
    "paper",
    "mug",
]


def get_stroke_data(
    curriculum_path: Path, train_or_test: str, *, dir_num: Optional[int] = None, int_curriculum_labels: bool = True, multi_object: bool = False
) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[int]]:
    # copied and edited from phase3_load_from_disk() -- see adam.curriculum.curriculum_from_files
    if dir_num is None:
        with open(curriculum_path / "info.yaml", encoding="utf=8") as curriculum_info_yaml:
            curriculum_params = yaml.safe_load(curriculum_info_yaml)

    curriculum_coords = []
    curriculum_adjs = []
    curriculum_labels = []
    for situation_num in range(curriculum_params["num_dirs"]) if dir_num is None else [dir_num]:
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

        feature_yamls = sorted(situation_dir.glob("feature*"))
        if len(feature_yamls) == 1:
            with open(
                feature_yamls[0], encoding="utf-8"
            ) as feature_yaml_in:
                features = yaml.safe_load(feature_yaml_in)

            num_obj = len(features["objects"])
            if num_obj == 0:
                continue

            # If in multi-object mode, get each (coords, adj, label) for all
            # objects across the curriculum
            if multi_object:
                for object_ in features["objects"]:
                    coords = np.asarray(object_["stroke_graph"]["strokes_normalized_coordinates"])
                    curriculum_coords.append(coords)

                    adj = np.asarray(object_["stroke_graph"]["adjacency_matrix"])
                    curriculum_adjs.append(adj)

                    if int_curriculum_labels:
                        integer_label, _ = label_from_object_language_tuple(
                            language_tuple, situation_num
                        )
                        curriculum_labels.append(integer_label)

            # If in single-object mode, get 1 set of (coords, adj, label) for
            # each situation
            else:
                # Combine multiple objects into one mega-example. Concatenate
                # together the coords for each connected component to get a
                # single (n_strokes_overall, 2, n_control_points) coordinates
                # array
                coords = np.concatenate(
                    [
                        np.asarray(object_["stroke_graph"]["strokes_normalized_coordinates"])
                        for object_ in features["objects"]
                    ],
                    axis=0,
                )

                # Now create together the big adjacency matrix
                n_nodes_overall = coords.shape[0]
                # jac: the above should do the same thing but I'm asserting
                # to make sure it does. for clarity's sake, I should delete
                # this once I verify it works.
                assert n_nodes_overall == sum(
                    len(object_["stroke_graph"]["adjacency_matrix"])
                    for object_ in features["objects"]
                )
                adj = np.zeros([n_nodes_overall, n_nodes_overall])

                nodes_seen = 0
                for idx, object_ in enumerate(features["objects"]):
                    # Set the appropriate submatrix of adj equal to this
                    # object's adjacency matrix.
                    n_nodes = len(object_["stroke_graph"]["adjacency_matrix"])
                    adj[
                        nodes_seen: nodes_seen + n_nodes,
                        nodes_seen: nodes_seen + n_nodes,
                    ] = np.asarray(object_["stroke_graph"]["adjacency_matrix"])
                    nodes_seen += n_nodes

                curriculum_coords.append(coords)
                curriculum_adjs.append(adj)

                if int_curriculum_labels:
                    integer_label, _ = label_from_object_language_tuple(
                        language_tuple, situation_num
                    )
                    curriculum_labels.append(integer_label)

        elif train_or_test == "train":
            raise ValueError(
                f"Situation number {situation_num} has more than one feature file."
            )
        # jac: Need to deal with this when we deal with decode for actions,
        # which will probably require changing the return type here
        # somehow... or other drastic changes.
        else:
            raise NotImplementedError(
                f"Don't know how to do decode when situation number {situation_num} has more than "
                f"one feature file."
            )

    return curriculum_coords, curriculum_adjs, curriculum_labels


def label_from_object_language_tuple(
    language_tuple: Tuple[str, ...], situation_num: int
) -> Tuple[int, str]:
    if len(language_tuple) > 2:
        raise ValueError(
            f"Don't know how to deal with long object name: {language_tuple}"
        )
    elif not language_tuple:
        raise ValueError(f"Can't extract label from empty object name: {language_tuple}")
    elif len(language_tuple) == 1:
        logging.warning(
            "Language tuple for object situation number %d is shorter than expected; this "
            "might cause problems.",
            situation_num,
        )

    string_label = language_tuple[-1]
    if string_label not in STRING_OBJECT_LABELS:
        raise ValueError(f"Unrecognized label: {string_label}")
    return STRING_OBJECT_LABELS.index(string_label), string_label


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred = pred.type_as(target)
    target = target.type_as(pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class LinearModel(nn.Module):
    def __init__(self, model, out, target):
        super(LinearModel, self).__init__()
        self.model = model
        self.linear = nn.Linear(in_features=out, out_features=target)
        torch.nn.init.eye_(self.linear.weight)
        self.feature = None

    def forward(self, g, h, e):
        x = self.model(g, h, e)
        self.feature = x
        x = self.linear(x)
        return x


def to_pickle(thing, path):  # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path):  # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing


def load_data(data, adj):
    n = 10
    node_info = []
    edge_info = []
    num_nodes = 0
    for item in range(len(data)):

        g = np.asmatrix(adj[item])

        strokes = np.array(data[item])
        strokes = strokes.transpose(0, 2, 1)
        inner_dis = []
        num_nodes = np.maximum(num_nodes, len(strokes))
        for jj in range(len(strokes)):
            stroke = strokes[jj]
            stroke1_1 = stroke[:, :1]
            stroke1_2 = stroke[:, 1:]

            distance_map = np.square(stroke1_1).repeat(n, 1) + np.square(stroke1_1.T).repeat(n, 0) - 2 * np.dot(
                stroke1_1, stroke1_1.T)
            distance_map += np.square(stroke1_2).repeat(n, 1) + np.square(stroke1_2.T).repeat(n,
                                                                                              0) - 2 * np.dot(
                stroke1_2, stroke1_2.T)
            distance_map += 1e-13
            distance_map = np.sqrt(distance_map)
            distance_map = distance_map.flatten()
            inner_dis.append(distance_map)

        inner_dis = np.array(inner_dis)
        h = inner_dis
        node_info.append(h)
        e = {}
        ee = np.where(g == 1)
        for jj in range(len(ee[0])):
            if ee[0][jj] >= ee[1][jj]:
                stroke1 = strokes[ee[0][jj]]
                stroke2 = strokes[ee[1][jj]]
                stroke1_1 = stroke1[:, :1]
                stroke1_2 = stroke1[:, 1:]
                stroke2_1 = stroke2[:, :1]
                stroke2_2 = stroke2[:, 1:]

                distance_map = np.square(stroke1_1).repeat(n, 1) + np.square(stroke2_1.T).repeat(n,
                                                                                                 0) - 2 * np.dot(
                    stroke1_1, stroke2_1.T)
                distance_map += np.square(stroke1_2).repeat(n, 1) + np.square(stroke2_2.T).repeat(n,
                                                                                                  0) - 2 * np.dot(
                    stroke1_2, stroke2_2.T)

                distance_map += 1e-13
                distance_map = np.sqrt(distance_map)
                distance_map = distance_map.flatten()

                e[(ee[0][jj], ee[1][jj])] = distance_map
        edge_info.append(e)
    return node_info, edge_info, num_nodes
