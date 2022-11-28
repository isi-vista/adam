"""
Code for doing stroke extraction on a curriculum.

Based on original code written by Sheng Cheng, found at
https://github.com/ASU-APG/adam-stage/tree/main/processing
"""
from itertools import cycle, islice
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, NamedTuple

logger = logging.getLogger(__name__)

try:
    import matlab
except ImportError:
    logger.warning("Couldn't import MATLAB API; creating a placeholder object instead.")
    matlab = object()
import matplotlib.pyplot as plt
from matplotlib.rcsetup import cycler as mpl_cycler
import matplotlib.patheffects as patheffects
import scipy.io
from tqdm import tqdm

try:
    import matlab.engine
except ImportError:
    logger.warning("Couldn't import MATLAB engine; setting it to None.")
    matlab.engine = None
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import SpectralClustering
import yaml
import os
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import pairwise_distances
import networkx as nx

from utils import label_from_object_language_tuple


MIN_MASK_SIZE = 10  # Minimum number of pixels for unique mask to be saved
TOUCHING_DISTANCE_THRESHOLD = 50.0


class ExtractionResult(NamedTuple):
    """
    Stroke extraction results for one segmentation file.

    File may or may not contain more than 1 mask.
    """
    num_obj: int
    reduced_strokes: np.ndarray
    stroke_obj_ids: np.ndarray
    adj: np.ndarray
    colors: np.ndarray


def vectorized_bspline_coeff(vi, vs):
    C = np.zeros(vi.shape)
    sel1 = np.logical_and((vs >= vi), (vs < vi + 1))
    C[sel1] = (1 / 6) * np.power(vs[sel1] - vi[sel1], 3)

    sel2 = np.logical_and((vs >= vi + 1), (vs < vi + 2))
    C[sel2] = (1 / 6) * (
        -3 * np.power(vs[sel2] - vi[sel2] - 1, 3)
        + 3 * np.power(vs[sel2] - vi[sel2] - 1, 2)
        + 3 * (vs[sel2] - vi[sel2] - 1)
        + 1
    )

    sel3 = np.logical_and((vs >= vi + 2), (vs < vi + 3))
    C[sel3] = (1 / 6) * (
        3 * np.power(vs[sel3] - vi[sel3] - 2, 3)
        - 6 * np.power(vs[sel3] - vi[sel3] - 2, 2)
        + 4
    )

    sel4 = np.logical_and((vs >= vi + 3), (vs < vi + 4))
    C[sel4] = (1 / 6) * np.power(1 - (vs[sel4] - vi[sel4] - 3), 3)

    return C


def get_bsplines_matrix(neval=10, nland=10):
    lb = 2
    ub = nland + 1
    sval = np.linspace(lb, ub, neval)
    ns = len(sval)
    sval = sval[:, np.newaxis]
    S = np.repeat(sval, nland, axis=1)
    L = nland
    I = np.repeat(np.arange(L)[np.newaxis, :], ns, axis=0)
    A = vectorized_bspline_coeff(I, S)
    sumA = A.sum(1)
    Coef = A / np.repeat(sumA[:, np.newaxis], L, 1)
    return Coef


def reduced_stroke(stroke, coef_inv):
    neval = len(stroke)
    coef = get_bsplines_matrix(neval, coef_inv.shape[1])
    P = np.dot(coef.T, coef)
    P = np.linalg.solve(P, coef.T)
    q = np.dot(coef_inv, P)
    return np.dot(q, stroke)


def kp2stroke(strokeinfo):
    """
    Given an "adjacency list" giving a stroke graph's edges, build a
    """
    n = len(strokeinfo)
    connection = np.zeros([n, n])
    for i, node_ind in enumerate(strokeinfo):
        # con is n-long array that is true for all coordinates on an axis-aligned line with node_ind
        # and false for the rest.
        con = np.logical_or(strokeinfo == node_ind[0], strokeinfo == node_ind[1])
        # con is
        con = np.logical_or(con[:, 0], con[:, 1])
        connection[i, np.where(con)[0]] = 1
        if node_ind[0] == node_ind[1]:
            connection[i, i] = 1
        else:
            connection[i, i] = 0
    return connection


def objects_touching(object_one, object_two):
    """Determine if two objects should be considered 'touching' based on the
    euclidean (absolute) distance of their closest two points.
    """
    object_one_points = np.concatenate(object_one['stroke_graph']['original_reduced_strokes'], axis=0)
    object_two_points = np.concatenate(object_two['stroke_graph']['original_reduced_strokes'], axis=0)

    # Check euclidean distance between every pair of points in the two objects:
    return np.any(
        np.sum(
            (object_one_points[:, np.newaxis] - object_two_points) ** 2, axis=-1
        ) ** 0.5 < TOUCHING_DISTANCE_THRESHOLD
    )


def merge_small_strokes(
    strokes: Sequence[Sequence[Tuple[float, float]]], adj: np.ndarray
) -> Tuple[Sequence[Sequence[Tuple[float, float]]], np.ndarray]:
    """
    Given a sequence of strokes and their adjacency matrix, merge small strokes.

    Parameters:
        strokes: The list of strokes.
        adj: The adjacency matrix over strokes.

    Return:
        A tuple (new_strokes, new_adj) where we have merged as many small strokes as possible.
    """
    stroke_to_root_id = cluster_small_strokes(strokes, adj)
    unique = np.unique(stroke_to_root_id)
    root_id_to_new_stroke_idx = {
        root_id: idx for idx, root_id in enumerate(unique)
    }

    # Build inverse map of root to children
    root_to_cluster = {}
    for stroke_id, root_id in enumerate(stroke_to_root_id):
        root_to_cluster.setdefault(root_id, []).append(stroke_id)

    # New strokes are built by merging all key points/control points of the strokes each cluster, in
    # sorted order
    new_strokes: List[List[Tuple[float, float]]] = [[] for _ in unique]
    for idx, (root, cluster) in enumerate(root_to_cluster.items()):
        cluster_strokes = [strokes[stroke_id] for stroke_id in cluster]
        new_strokes[idx] = [
            point
            for i, stroke in enumerate(strokes_end_to_end(cluster_strokes))
            for point in (stroke[: -1] if i + 1 < len(cluster_strokes) else stroke)
        ]

    # jac: I swear there has to be a more NumPy way to do this, but it eludes me. For now, avoiding
    # premature optimization.
    new_adj = np.zeros([len(unique), len(unique)])
    for root1, cluster1 in root_to_cluster.items():
        for root2, cluster2 in root_to_cluster.items():
            new_stroke_idx1 = root_id_to_new_stroke_idx[stroke_to_root_id[root1]]
            new_stroke_idx2 = root_id_to_new_stroke_idx[stroke_to_root_id[root2]]
            new_adj[new_stroke_idx1, new_stroke_idx2] = np.logical_and(
                root1 != root2,
                np.any(adj[cluster1, :][:, cluster2])
            )

    return new_strokes, np.array(new_adj)


def cluster_small_strokes(
    strokes: Sequence[Sequence[Tuple[float, float]]], adj: np.ndarray
) -> np.ndarray:
    """
    Given a sequence of strokes and their adjacency matrix, produce a clustering of small strokes.

    Parameters:
        strokes: The list of strokes.
        adj: The adjacency matrix over strokes.

    Return:
        An array c with one entry per stroke identifying cluster to which that stroke belongs.
        Additionally we force c[c[i]] = c[i] for all i.
    """

    def get_cluster_id(cluster: Sequence[int]) -> int:
        return sum(2 ** stroke_id_ for stroke_id_ in cluster)

    # jac: Given the representation above, we technically don't need this, but it saves time.
    # If needed we could reconstruct the list from cluster_id_ as follows:
    #     [x for x in range(len(strokes) if cluster_id_ & 1 << x == 1]
    # But I don't think we are so desperate for memory/space.
    cluster_to_stroke_ids = {get_cluster_id([idx]): [idx] for idx in range(len(strokes))}
    # Each stroke starts in its own cluster
    stroke_to_cluster_id = {idx: get_cluster_id([idx]) for idx in range(len(strokes))}

    def cluster_len(cluster_id: int) -> int:
        strokes_in_cluster = cluster_to_stroke_ids[cluster_id]
        return sum(
            len(strokes[stroke_id]) for stroke_id in strokes_in_cluster
            # If there are N strokes in the cluster, there should be N - 1 points where they overlap.
            # Those control points will be removed when we merge the strokes, so we subtrac them out
            # when calculating the cluster length.
        ) - len(strokes_in_cluster) + 1

    def neighbors(cluster_id: int) -> Sequence[int]:
        strokes_in_cluster = cluster_to_stroke_ids[cluster_id]
        adjacent_strokes = np.max(
            np.atleast_2d(adj[strokes_in_cluster]),
            axis=0,
        )
        # Ignore strokes in the cluster
        adjacent_strokes[strokes_in_cluster] = 0
        return [stroke_to_cluster_id[stroke_id] for stroke_id, is_neighbor in enumerate(adjacent_strokes) if is_neighbor]

    def cluster_degree(cluster_id: int) -> int:
        return len(neighbors(cluster_id))

    while True:
        smallest_clusters_first = sorted(cluster_to_stroke_ids.keys(), key=cluster_len)
        should_stop = True
        for cluster_id_ in smallest_clusters_first:
            if cluster_len(cluster_id_) < 10:
                mergeable_neighbors = [
                    neighbor for neighbor in neighbors(cluster_id_)
                    if (cluster_degree(cluster_id_) == 2 or cluster_degree(neighbor) == 2)
                    # They must also not share a neighbor -- this is to prevent connected T-shaped
                    # stroke intersections from merging together.
                    #
                    # For example, say you have strokes like:
                    #
                    #   \ /
                    #    .
                    # _./
                    #
                    # where straight lines are strokes and periods are key points. Then your stroke
                    # graph looks like a triangle with a hanger-on:
                    #
                    #    s1--s2
                    #     \ /
                    # s4---s3
                    #
                    # That's a problem because s1 and s2 are both degree 2, so going by degree alone
                    # we're allowed to merge either with s3. Allowing this can lead to merging
                    # strokes in a non-orientable way. Say you merge s1 with s3. Then you have a
                    # graph like:
                    #
                    #        s'2
                    #       /
                    # s'4--s'3
                    #
                    # But now s'3 (representing the tail and the top-left stroke) can be merged with
                    # s'2 (representing the top-right stroke), because it is degree 2! That's a
                    # problem because there is no reasonable way to orient these three strokes so
                    # that they form a connected path.
                    #
                    # To prevent this, we disallow merging strokes with common neighbors.
                    and not any(
                        shared_neighbor for shared_neighbor in neighbors(cluster_id_)
                        if shared_neighbor in neighbors(neighbor)
                    )
                ]
                if mergeable_neighbors:
                    # We're still making progress, so don't stop yet!
                    should_stop = False

                    # Get the shortest neighbor
                    shortest_neighbor = min(mergeable_neighbors, key=cluster_len)

                    # Remove old clusters from mappings
                    old_cluster = cluster_to_stroke_ids.pop(cluster_id_)
                    old_neighbor_cluster = cluster_to_stroke_ids.pop(shortest_neighbor)

                    # Set up new cluster
                    new_cluster_strokes = old_cluster + old_neighbor_cluster
                    new_cluster_id = get_cluster_id(new_cluster_strokes)
                    cluster_to_stroke_ids[new_cluster_id] = new_cluster_strokes

                    for stroke in new_cluster_strokes:
                        stroke_to_cluster_id[stroke] = new_cluster_id

                    break
            else:
                break

        if should_stop:
            break

    return np.array([
        min(cluster_to_stroke_ids[stroke_to_cluster_id[stroke_idx]])
        for stroke_idx in range(len(strokes))
    ])


def strokes_end_to_end(strokes: Sequence[Sequence[Tuple[float, float]]]) -> Sequence[Sequence[Tuple[float, float]]]:
    """
    Given a sequence of strokes, return the strokes in an end-to-end ordering if possible.

    This tries to achieve an ordering that is as close to end-to-end as possible.

    Parameters:
        strokes: The list of strokes.

    Return:
        A sequence of strokes with consistent orientation, ordered end to end.
    """
    if len(strokes) <= 1:
        return strokes

    def kings_move_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate the king's move (Chebyshev) distance between two points."""
        x0, y0 = point1
        x1, y1 = point2
        return max(abs(x1 - x0), abs(y1 - y0))

    def overlap(stroke1: Sequence[Tuple[float, float]], stroke2: Sequence[Tuple[float, float]]) -> bool:
        """Check if two strokes overlap at their endpoints."""
        return any(
            kings_move_distance(stroke1[s1_endpoint], stroke2[s2_endpoint]) == 0
            for s1_endpoint in [0, -1]
            for s2_endpoint in [0, -1]
        )

    def consistent_orientation(stroke1: Sequence[Tuple[float, float]], stroke2: Sequence[Tuple[float, float]]) -> bool:
        """Check if two overlapping strokes are oriented consistently."""
        return (kings_move_distance(stroke1[0], stroke2[-1]) == 0) or (kings_move_distance(stroke1[-1], stroke2[0]) == 0)

    # Create a graph where the nodes are strokes and edges represent "overlap at endpoints"
    g = nx.Graph()
    g.add_nodes_from(range(len(strokes)))
    for idx1, stroke1 in enumerate(strokes):
        for idx2, stroke2 in enumerate(strokes[idx1 + 1 :], start=idx1 + 1):
            if overlap(stroke1, stroke2):
                g.add_edge(idx1, idx2)

    if not nx.is_connected(g):
        raise ValueError("Can't order strokes end to end: Strokes are not connected enough.")

    # The result should be a path graph. Equivalently (https://en.wikipedia.org/wiki/Path_graph):
    # It should have no cycles (= it's a tree), and no vertex with degree > 2.
    if nx.is_tree(g) and all(g.degree(n) <= 2 for n in g.nodes):
        terminals = [n for n in g.nodes if g.degree(n) == 1]
        last_node = None
        cur_node = terminals[0]
        ordering = [terminals[0]]
        # Always preserve the first stroke's orientation -- because that is our root/starting point
        preserve_orientation = [True]
        while cur_node != terminals[1]:
            next_node = [n for n in g.neighbors(cur_node) if n != last_node][0]
            ordering.append(next_node)
            preserve_orientation.append(
                # We want to reverse the next stroke *iff* it is not oriented consistently with the
                # starting stroke. We can make this iterative as follows:
                #   1. Reverse the next stroke iff it is orientated consistently with the current stroke
                #      and the current stroke should have its orientation reversed.
                #   2. Preserve the next stroke's orientation iff it is oriented consistently with the
                #      current stroke and the current stroke should have its orientation preserved.
                # That is, whether to preserve the next stroke's orientation is given by:
                consistent_orientation(strokes[cur_node], strokes[next_node])
                == preserve_orientation[-1]
            )
            last_node = cur_node
            cur_node = next_node

        result = [
            strokes[stroke_idx] if preserve_orientation else list(reversed(strokes[stroke_idx]))
            for stroke_idx, preserve_orientation in zip(ordering, preserve_orientation)
        ]
        # Reverse the output list if needed so that strokes can be concatenated in list order
        if len(result) > 1 and kings_move_distance(result[0][-1], result[1][0]) > 0:
            result = list(reversed(result))
        assert len(result) == len(strokes)
        return result
    else:
        raise ValueError("Strokes don't connect in a line, so can't order them end to end.")


def plot_oriented_strokes(ax: plt.Axes, strokes: Sequence[Sequence[Tuple[float, float]]]) -> None:
    """
    Plot oriented strokes on the given axes.

    This creates a line chart with one color per stroke where each line has arrows pointing in the
    direction that the given stroke is oriented.

    For strokes with just one point, we plot a circle instead.

    Parameters:
        ax: The axes to plot on.
        strokes: The strokes to plot.
    """
    colors = mpl_cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    for idx, (stroke, color) in enumerate(zip(strokes, cycle(colors))):
        ys, xs = zip(*stroke)
        if len(stroke) > 1:
            ax.plot(
                xs,
                ys,
                label=idx,
                path_effects=[
                    patheffects.withTickedStroke(angle=45, length=0.5),
                    patheffects.withTickedStroke(angle=-45, length=0.5),
                ],
                **color,
            )
        else:
            ax.scatter(xs, ys, marker='s', label=idx, **color)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def plot_stroke_graph(ax: plt.Axes, strokes: Sequence[Sequence[Tuple[float, float]]], adj: 'np.ndarray[np.int]') -> None:
    """
    Plot stroke graph on the given axes.

    We try to position the vertices as close as possible to the mean of the stroke's points.

    Parameters:
        ax: The axes to plot on.
        strokes: The strokes to plot.
        adj: The adjacency matrix over stroke indices.
    """
    if not np.all(adj.T == adj):
        raise ValueError("Expected symmetric adjacency matrix (undirected graph) as input.")

    g = nx.Graph()
    if adj.shape == (1, 1):
        g.add_node(0)
    else:
        for p in range(len(adj)):
            g.add_node(p)
            for q in range(p, len(adj)):
                if adj[p][q] == 1:
                    g.add_edge(p, q)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    nx.draw(
        g,
        ax=ax,
        pos={idx: np.mean(np.asarray(stroke), axis=0)[[1, 0]] for idx, stroke in enumerate(strokes)},
        node_color=list(islice(cycle(colors), len(strokes))),
        with_labels=True,
    )
    ax.invert_yaxis()


class Stroke_Extraction:
    """
    obj_type: object name + train/test
    obj_id: the id of the object
    obj_view: the id of the camera
    segmentation_img_path: the path to the object segmentation image
    rgb_img_path: the path to the RGB image
    stroke_img_save_path: where to save the stroke image (if saved)
    stroke_graph_img_save_path: where to save the stroke graph image (if saved)
    features_save_path: where to save the YAML file of features (if saved)
    vis: whether to save the visualization(graph structure and strokes)
    save_output: whether to save the output

    output: data:
    dict(
        object_name: obj_id,
        subobject_id: "0",
        viewpoint_id: obj_view,
        distance: distance between stroke graphs,
        stroke_graph=dict(
            adjacency_matrix,
            stroke_mean_x,
            stroke_mean_y,
            stroke_std,
            strokes_normalized_coordinates: normalized stroke coordinates,
            concept_name: predicted/given object name,
            confidence_score: the confidence score for prediction, 1 for training data,
        ),
        color: mean color of the object,
        texture=None,
        sub_part=None
    )

    the file structure:
    {base_path}/{obj_type}/cam{obj_view}/rgb__{obj_id}.png

    Examples:
    extractor = Stroke_Extraction()
    data = extractor.get_strokes()

    """

    def __init__(
        self,
        *,
        obj_type: str = "outputs",
        obj_id: Optional[str] = "1",
        obj_view: Optional[str] = "1",
        clustering_seed: int = 42,
        segmentation_img_path: str,
        rgb_img_path: str,
        stroke_img_save_path: str,
        stroke_graph_img_save_path: str,
        output_dir: str,
        process_masks_independently: bool = False,
        should_merge_small_strokes: bool = False,
        debug_vis: bool = False,
        debug_matlab_stroke_img_save_path: str,
        vis: bool = True,
        save_output: bool = True,
    ):
        self.segmentation_paths: List[str] = []
        self.obj_view = obj_view
        self.obj_type = obj_type
        self.obj_id = obj_id
        self.clustering_seed = clustering_seed
        self.vis = vis
        self.debug_vis = debug_vis
        self.should_merge_small_strokes = should_merge_small_strokes
        self.save_output = save_output
        self.path = segmentation_img_path
        self.img_bgr = img_bgr = cv2.imread(rgb_img_path)
        self.img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.debug_matlab_stroke_img_save_path = debug_matlab_stroke_img_save_path
        self.stroke_img_save_path = stroke_img_save_path
        self.stroke_graph_img_save_path = stroke_graph_img_save_path
        self.output_dir = output_dir
        self.features_save_path = os.path.join(output_dir, "feature.yaml")
        self.process_masks_independently = process_masks_independently

    @staticmethod
    def from_m5_objects_curriculum_base_path(
        *,
        obj_type: str = "outputs",
        obj_id: str = "1",
        obj_view: str = "1",
        base_path: str,
        vis: bool = True,
        save_output: bool = True,
    ) -> "Stroke_Extraction":
        """
        Set up to do stroke extraction on M5 objects raw curriculum format.

        This is the format used before ISI reformatting.
        """
        save_dir = os.path.join(base_path, "feature")
        return Stroke_Extraction(
            obj_type=obj_type,
            obj_id=obj_id,
            obj_view=obj_view,
            segmentation_img_path=os.path.join(
                base_path,
                obj_type,
                "cam{}".format(obj_view),
                "semantic_{}.png".format(obj_id),
            ),
            rgb_img_path=os.path.join(
                base_path, obj_type, "cam{}".format(obj_view), "rgb_{}.png".format(obj_id)
            ),
            vis=vis,
            save_output=save_output,
            stroke_img_save_path=os.path.join(
                save_dir,
                "stroke_{}_{}_{}.png".format(obj_type, obj_view, obj_id),
            ),
            stroke_graph_img_save_path=os.path.join(
                save_dir,
                "stroke_graph_{}_{}_{}.png".format(obj_type, obj_view, obj_id),
            ),
            features_save_path=os.path.join(
                save_dir,
                "feature_{}_{}_{}.yaml".format(obj_type, obj_view, obj_id),
            ),
        )

    def stroke_extraction_from_matlab(self, segmentation_path: str):
        """
        Get the raw stroke extraction results from Matlab using the BPL code.

        Note that stroke extraction is performed on the object segmentation image, i.e. the
        semantic_j.png file. It is *not* done on the raw RGB image.

        The output is a pair [S, E] where:

        - S is a k x 1 sequence of "edge paths in the image"
        - E is a k x 2 sequence of "graph edges"
        """
        eng = matlab.engine.start_matlab()
        file_dir = os.path.dirname(__file__)
        eng.addpath(eng.genpath(file_dir), nargout=0)
        out = eng.ske(segmentation_path, nargout=2)
        eng.close()
        return out

    def remove_strokes(self, segmentation_path: Path) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process Matlab stroke extraction results into labels.

        In this function we:

        - construct the adjacency matrix
        - clean up the strokes, removing those with fewer than 10 key points (?)
        - split the stroke graph into connected components
        - determine the number of objects in the scene as the number of components
        - label each stroke according to which object it is a part of

        The outputs are returned as:

        - num_objs
        - strokes
        - stroke_obj_ids
        - adj
        """
        out_s, out_e = self.stroke_extraction_from_matlab(str(segmentation_path))
        strokes = []
        removed_ind = []
        adj = kp2stroke(np.array(out_e))
        if self.debug_vis:
            fig, axs = plt.subplots(ncols=2)
            plot_oriented_strokes(axs[0], out_s)
            plot_stroke_graph(axs[1], out_s, adj)
            fig.tight_layout()
            fig.savefig(self.debug_matlab_stroke_img_save_path)

        new_s, adj = merge_small_strokes(out_s, adj) if self.should_merge_small_strokes else (out_s, adj)
        for i in range(len(new_s)):
            s = np.asarray(matlab.double(new_s[i]))
            if len(s) < 10:
                removed_ind.append(i)
                continue
            strokes.append(s)
        # Make an array of the strokes.
        # Because each stroke in the list may have more than 10 control points, this has to be a
        # "ragged" array. Thus we specify dtype=object to avoid a NumPy deprecation warning.
        #
        # jac: Maybe we can get away with making this rectangular? But given we still need to
        # downsample strokes at this point, I wouldn't be on it -- probably that would mess things
        # up somehow.
        strokes = np.array(strokes, dtype=object)
        # Do initial breakup of the scene into objects by looking at the connected components of the graph.
        n_components, labels = connected_components(adj, directed=True)
        if len(removed_ind) > 0:
            adj = np.delete(adj, np.array(removed_ind), 0)
            adj = np.delete(adj, np.array(removed_ind), 1)
        if len(removed_ind) > 0:
            reduced_labels = np.delete(labels, np.array(removed_ind))
            for i in range(0, len(strokes)):
                for j in range(i + 1, len(strokes)):
                    dis = euclidean_distances(strokes[i], strokes[j])
                    if dis.min() < 15 and reduced_labels[i] == reduced_labels[j]:
                        adj[i, j] = 1.0
                        adj[j, i] = 1.0
        # jac: hack to get things running
        num_obj = min(n_components, len(adj))
        # If we have at least two strokes and two objects, see if we can break up the connected components further to
        # get more fine-grained objects.
        #
        # jac: I'm guessing this might be meant to cope with occlusion/objects visually overlapping each other?
        # jac: I don't think the "is not" here will ever get triggered. Oh well.
        # jac: hack: changed to num_obj > 1. I think this is fine.
        if adj is not [[1.0]] and num_obj > 1:
            clustering = SpectralClustering(
                random_state=self.clustering_seed,
                n_clusters=num_obj,
                affinity="precomputed",
            ).fit(adj)
            stroke_obj_ids_ = clustering.labels_
        # If there's only one object, things are easy -- just assign all strokes to that one object.
        elif num_obj == 1:
            stroke_obj_ids_ = np.zeros(len(adj))
        # Otherwise, adj is [[1.0]] and num_obj is not 1. If num_obj is zero, this sets labels_ to an empty 1D array.
        # If num_obj is greater than 1, it assigns all num_obj strokes to a single object.
        #
        # jac: The num_obj == 0 case is not interesting -- it's the case where we don't have any strokes to begin with
        # so there's nothing to assign and we don't assign anything.
        #
        # jac: In the num_obj > 1 case I think what happened is that some of the strokes had fewer than 10 points so we
        # deleted them from the adjacency matrix, but only after we already calculated the number of connected
        # components. So adj is np.eye(1) but num_obj is greater than 1.
        else:
            if num_obj == 0:
                logger.warning(
                    "No objects detected for segmentation image %s.", segmentation_path
                )
            stroke_obj_ids_ = np.zeros(num_obj)

        return num_obj, strokes, stroke_obj_ids_, adj

    def strokes2controlpoints(self, strokes) -> np.ndarray:
        """
        Downsamples the known strokes to 10 points using B-splines.

        Because of the processing logic in self.remove_strokes(), we know that each stroke has at
        least 10 points such that it is possible to do this.

        Outputs the downsampled strokes to self.reduced_strokes. This should be an N x 10 x 2 array,
        where N is the number of strokes. The last dimension is the x vs y coords; the middle is the
        control point number.

        Return:
            reduced_strokes
        """
        reduced_strokes = []
        coef_inv = get_bsplines_matrix(10)
        for i in range(len(strokes)):
            reduced_strokes.append(reduced_stroke(strokes[i], coef_inv))
        return np.array(reduced_strokes, dtype=np.float64)

    def plot_strokes(self, extractions: Sequence[ExtractionResult]):
        """
        Create stroke image, saving the result to self.stroke_img_save_path.
        """
        stroke_colors_list = []
        n_rows, n_cols = len(extractions), max(e.num_obj for e in extractions)
        plt.figure(0, figsize=(6.4 / 3 * n_cols, 4.8 * n_rows))
        for e_idx, extraction in enumerate(extractions):
            stroke_colors = []
            for j in range(extraction.num_obj):
                reduced_obj = extraction.reduced_strokes[np.where(extraction.stroke_obj_ids == j)[0]]
                plt.subplot(n_rows, n_cols, (e_idx * n_cols) + j + 1)
                s_c = []
                for i in range(len(reduced_obj)):
                    p = plt.plot(reduced_obj[i][:, 1], reduced_obj[i][:, 0], linewidth=3)
                    s_c.append(p[0].get_color())
                    plt.scatter(reduced_obj[i][:, 1], reduced_obj[i][:, 0], c="r")
                stroke_colors.append(s_c)
                plt.gca().invert_yaxis()
                plt.gca().set_aspect("equal", adjustable="box")
                plt.axis("off")
            stroke_colors_list.append(stroke_colors)
        plt.savefig(self.stroke_img_save_path)
        plt.close()
        return stroke_colors_list

    def plot_graph(self, extractions: Sequence[ExtractionResult], stroke_colors_list):
        """
        Create stroke *graph* image in self.stroke_graph_img_save_path.
        """
        n_rows, n_cols = len(extractions), max(e.num_obj for e in extractions)
        plt.figure(0, figsize=(6.4 / 3 * n_cols, 4.8 * n_rows))
        for e_idx, (extraction, stroke_colors) in enumerate(zip(extractions, stroke_colors_list)):
            for i in range(extraction.num_obj):
                G = nx.Graph()
                if extraction.adj is [[0.0]]:
                    G.add_node("s0")
                else:
                    ind = np.where(extraction.stroke_obj_ids == i)[0]
                    adj_obj = extraction.adj[ind, :]
                    adj_obj = adj_obj[:, ind]

                    for p in range(len(adj_obj)):
                        G.add_node("s_{}".format(int(p + 1)))
                        for q in range(p, len(adj_obj)):
                            if adj_obj[p, q] == 1:
                                G.add_edge(
                                    "s_{}".format(int(p + 1)), "s_{}".format(int(q + 1))
                                )
                plt.subplot(n_rows, n_cols, (e_idx * n_cols) + i + 1)
                nx.draw(G, with_labels=True, node_color=stroke_colors[i])
        plt.savefig(self.stroke_graph_img_save_path)
        plt.close()

    def get_colors(self, segmentation_path: Path, num_obj: int) -> np.ndarray:
        """
        Collect the average color for each object in the image together with the object "centers."

        This uses the object segmentation image to identify which pixels in the RGB image "belong
        to" each object. This segmentation image file is usually named semantic_{stuff}.png
        """
        # jac: img_seg is a 256x256 integer array? gray-values/pixel values are distinct iff the
        # objects for those locations are (considered) distinct. I think that is the convention
        # here.
        img_seg = cv2.cvtColor(cv2.imread(str(segmentation_path)), cv2.COLOR_BGR2GRAY)
        # jac: obj_area is a tuple of two equal-length arrays xs, ys such that for all i,
        # img_seg[xs[i], ys[i]] != 0.
        obj_area = np.where(img_seg != 0)
        colors = np.zeros([num_obj, 5])
        for i in range(num_obj):
            # If we've gone beyond the number of unique objects identified in the object
            # segmentation image, stop.
            #
            # jac: This can happen when stroke extraction splits one object into more than one
            # connected stroke graph component, I think.
            if i >= len(np.unique(img_seg[obj_area])):
                break
            # reminder: np.unique(img_seg[obj_area]) means "the ith unique object detected by object
            # segmentation," so this means "make a matrix of the pixel locations that are part of
            # this ith object."
            #
            # seg_i is a 2 x (N_pixels(i)) matrix -- the number of columns varies with the number of
            # pixels.  row 0 is the row coordinate, row 1 is the column coordinate.
            seg_i = np.array(np.where(img_seg == np.unique(img_seg[obj_area])[i]))
            colors[i, :3] = self.img_rgb[seg_i[0], seg_i[1], :].mean(0)
            # Store the mean coordinates for the object in the last two color columns
            colors[i, 3:] = seg_i.mean(1)
        return colors

    def save_distinct_masks(self) -> List[Path]:
        """
        Save distinct masks in separate files.
        """
        img_seg = cv2.imread(self.path)
        all_seg_pixels = np.reshape(img_seg, (-1, img_seg.shape[-1]))
        all_unique_masks = np.unique(all_seg_pixels, axis=0)

        # Ignore black (background) pixels
        unique_masks = all_unique_masks[
            np.where((all_unique_masks != [0, 0, 0]).any(axis=-1))
        ]
        segmentation_paths: List[Path] = []
        num_masks: int = 0
        for mask_color in unique_masks:
            matches_color = (img_seg == mask_color).all(axis=-1)[..., np.newaxis]
            if matches_color.sum() >= MIN_MASK_SIZE:
                segmentation_path = Path(
                    self.output_dir) / f"{Path(self.path).stem}_mask_{num_masks}.png"
                segmentation_paths.append(segmentation_path)
                distinct_mask = np.where(matches_color, img_seg, [0, 0, 0])
                cv2.imwrite(str(segmentation_path), distinct_mask)
                num_masks += 1
        return segmentation_paths

    def get_strokes(self):
        """
        Extract strokes from the image at self.path.

        This performs stroke extraction as well as reformatting the outputs for use in ADAM.

        This writes a YAML file to self.features_save_path if self.save_output is true.
        """
        segmentation_paths: List[Path]
        if self.process_masks_independently:
            segmentation_paths = self.save_distinct_masks()
        else:
            segmentation_paths = [Path(self.path)]

        # Extract strokes, downsample, and add color data
        extractions: List[ExtractionResult] = []
        for segmentation_path in segmentation_paths:
            extractions.append(self.get_strokes_for_file(segmentation_path))
        if self.vis:
            stroke_colors_list = self.plot_strokes(extractions)
            self.plot_graph(extractions, stroke_colors_list)

        # Get combined stroke extraction outputs for all masks
        combined_extraction: ExtractionResult = self.combine_extractions(extractions)

        objects = []
        # Translate the extracted (downsampled strokes) and color data into the ADAM stroke
        # extraction data format.
        for i in range(combined_extraction.num_obj):
            # Using self.label, pick out the IDs/indices for the strokes belonging to object i.
            # Use this to grab the relevant stroke samples and the relevant submatrix of self.adj.
            ind = np.where(combined_extraction.stroke_obj_ids == i)[0]
            # shape: N_i x 10 x 2 where N_i is the number of strokes in the ith object
            reduced_obj = combined_extraction.reduced_strokes[ind]
            # shape: N_i x N_i
            adj_obj = combined_extraction.adj[ind, :]
            adj_obj = adj_obj[:, ind]

            # Calculate the overall mean coordinates across all strokes and coordinates.
            # This is a 2D array with 2 entries -- one per coordinate.
            m = reduced_obj.mean((0, 1))
            # Get pairwise distances between the stroke coordinate mean (treated as a 1x2 array) and
            # the pixel-based object centers.
            dd = pairwise_distances(m[None, :], combined_extraction.colors[:, 3:])
            # Choose the color using the nearest pixel-based object center.
            color = combined_extraction.colors[dd[0].argmin(0), :3]
            # We normalize using a standard deviation s that is calculated using both the deviations
            # from the x-mean and the deviations from the y-mean, as if these have identical
            # distributions. This is sort of weird.
            s = (reduced_obj - m).std()
            reduced_strokes_norm = (reduced_obj - m) / s

            # Calculate the dimensions of a 2D box around the object
            max_x, max_y = reduced_obj.max((0, 1))
            min_x, min_y = reduced_obj.min((0, 1))
            x_size, y_size = max_x-min_x, max_y-min_y
            box_area = x_size * y_size

            # Calculate pixel-space-distance from this object to every other object in the image.
            # We use the Euclidean distance between the two objects' stroke coordinate means as our
            # measure of distance between objects.
            distance = dict()
            relative_distance = dict()
            for j in range(combined_extraction.num_obj):
                other_object_name = "object" + str(j)
                if i == j:
                    continue
                else:
                    ind_ = np.where(combined_extraction.stroke_obj_ids == j)[0]
                    reduced_obj_ = combined_extraction.reduced_strokes[ind_]
                    m_ = reduced_obj_.mean((0, 1))

                    # Get x & y offsets, calculate distance
                    offsets = m - m_
                    distance[other_object_name] = (
                        np.sqrt((offsets ** 2).sum())
                    ).item()

                    # "Relative distance" includes offsets along axes; this must
                    # later be normalized by the product of the sizes of the two
                    # objects.
                    relative_distance[other_object_name] = dict(
                        x_offset=offsets[0].item(),
                        y_offset=offsets[1].item(),
                        euclidean_distance=distance[other_object_name]
                    )
            if len(distance.keys()) == 0:
                distance = None
                relative_distance = None

            objects.append(
                dict(
                    object_name="object" + str(i),
                    subobject_id="0",
                    viewpoint_id=self.obj_view,
                    distance=distance,
                    relative_distance=relative_distance,
                    size=dict(
                        # FIXME: calling .item() causes crash (it says they
                        #  are type `float`), but if they aren't cast to
                        #  `float`, they get serialized as complex objects,
                        #  not primitive floats.
                        width=float(x_size),
                        height=float(y_size),
                        box_area=float(box_area)
                    ),
                    stroke_graph=dict(
                        adjacency_matrix=adj_obj.tolist(),
                        stroke_mean_x=reduced_obj.mean((0, 1)).tolist()[1],
                        stroke_mean_y=reduced_obj.mean((0, 1)).tolist()[0],
                        stroke_std=reduced_obj.std((0, 1, 2)).tolist(),
                        strokes_normalized_coordinates=reduced_strokes_norm.transpose(
                            0, 2, 1
                        ).tolist(),
                        concept_name=self.obj_type,
                        confidence_score=1.0,
                        # FIXME: I'm storing this for calculating touching;
                        #  should do this in a more elegant way
                        original_reduced_strokes=reduced_obj.tolist(),
                    ),
                    color=color.tolist(),
                    texture=None,
                    sub_part=None,
                )
            )

        # Calculate relative sizes using already-calculated absolute sizes:
        for i in range(len(objects)):
            relative_size = dict()
            for j in range(len(objects)):
                if i == j:
                    continue
                else:
                    other_object_name = objects[j]['object_name']
                    relative_size[other_object_name] = dict()

                    for axis, dimension in (('x', 'width'), ('y', 'height')):
                        # Relative size is a discrete ('bigger_than') relation
                        object_dimension = objects[i]['size'][dimension]
                        other_object_dimension = objects[j]['size'][dimension]
                        relative_size[other_object_name][f'{dimension}_greater_than'] = object_dimension > other_object_dimension

                        # Normalize relative distances by the relevant axis
                        # sizes of both objects
                        objects[i]['relative_distance'][other_object_name][f'{axis}_offset'] /= (object_dimension * other_object_dimension)

                    # Normalize euclidean distance by product of areas of both
                    # objects
                    object_area = objects[i]['size']['box_area']
                    other_object_area = objects[j]['size']['box_area']
                    objects[i]['relative_distance'][other_object_name]['euclidean_distance'] /= (object_area * other_object_area)
            objects[i]['relative_size'] = relative_size

        # Determine touching relations based on closest-point distance
        touching = list()
        for i in range(len(objects)):
            object_name = objects[i]['object_name']
            for j in range(i+1, len(objects)):
                other_object_name = objects[j]['object_name']
                if objects_touching(objects[i], objects[j]):
                    touching.append([object_name, other_object_name])

        # FIXME: This attribute is never used
        self.data = objects
        if self.save_output:
            with open(
                self.features_save_path,
                "w",
            ) as file:
                yaml.dump({"objects": objects, "touching": touching}, file)
                file.close()
        # FIXME: This return value is never used
        return objects

    def combine_extractions(self, extractions: Sequence[ExtractionResult]) -> ExtractionResult:
        """
        Combine all extractions into one (mostly by concatenation).
        """
        num_obj = sum(e.num_obj for e in extractions)
        total_strokes_seen = 0
        total_obj_seen = 0
        stroke_obj_ids = []

        total_strokes = sum(e.adj.shape[0] for e in extractions)
        adj = np.zeros((total_strokes, total_strokes))
        for e in extractions:
            num_strokes = e.adj.shape[0]
            stroke_obj_ids.append(e.stroke_obj_ids + total_obj_seen)
            adj[
                total_strokes_seen:total_strokes_seen + num_strokes,
                total_strokes_seen:total_strokes_seen + num_strokes
            ] = e.adj
            total_obj_seen += e.num_obj
            total_strokes_seen += num_strokes
        stroke_obj_ids = np.concatenate(stroke_obj_ids, axis=0)
        # Below: must skip strokeless extractions, or concatenation won't work
        reduced_strokes = np.concatenate(
            [e.reduced_strokes for e in extractions if e.reduced_strokes.shape != (0,)], axis=0)
        colors = np.concatenate([e.colors for e in extractions], axis=0)

        return ExtractionResult(num_obj, reduced_strokes, stroke_obj_ids, adj, colors)

    def get_strokes_for_file(self, segmentation_path: Path) -> ExtractionResult:
        num_obj, strokes, stroke_obj_ids, adj = self.remove_strokes(segmentation_path)
        reduced_strokes = self.strokes2controlpoints(strokes)
        colors = self.get_colors(segmentation_path, num_obj)

        return ExtractionResult(num_obj, reduced_strokes, stroke_obj_ids, adj, colors)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "curriculum_path",
        type=Path,
        help="The curriculum path to process.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="The output curriculum dir.",
    )
    parser.add_argument(
        "--dir-num",
        default=None,
        help="A specific situation directory number to process. If provided only this directory is processed.",
    )
    parser.add_argument(
        "--merge-small-strokes",
        action="store_true",
        help="Merge small strokes before filtering."
    )
    parser.add_argument(
        "--no-merge-small-strokes",
        action="store_false",
        dest="merge_small_strokes",
        help="Don't merge small strokes before filtering."
    )
    parser.add_argument(
        "--use-segmentation-type",
        choices=["semantic", "color-refined"],
        default="semantic",
        help="Extract strokes from the color-refined segmentation.",
    )
    parser.add_argument(
        "--tolerate-errors",
        action="store_true",
        help="If passed, will attempt to continue processing of the input curriculum when "
        "extraction fails for a given situation."
    )
    parser.add_argument(
        "--no-tolerate-errors",
        action="store_false",
        dest="tolerate_errors",
        help="If passed, will halt immediately when extraction fails for a given situation."
    )
    parser.add_argument(
        "--process-masks-independently",
        action="store_true",
        help="If passed, will not merge adjacent masks of different colors."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # copied and edited from phase3_load_from_disk() -- see adam.curriculum.curriculum_from_files
    if args.dir_num:
        logger.info(f"Processing directory number {args.dir_num}")
    else:
        with open(
            args.curriculum_path / "info.yaml", encoding="utf=8"
        ) as curriculum_info_yaml:
            curriculum_params = yaml.safe_load(curriculum_info_yaml)
        logger.info(
            "Input curriculum has %d dirs/situations.", curriculum_params["num_dirs"]
        )

    for situation_num in tqdm(
        range(curriculum_params["num_dirs"]) if args.dir_num is None else [args.dir_num],
        desc="Situations processed",
        total=curriculum_params["num_dirs"] if args.dir_num is None else 100,
    ):
        situation_dir = args.curriculum_path / f"situation_{situation_num}"
        output_situation_dir = args.output_dir / f"situation_{situation_num}"
        output_situation_dir.mkdir(exist_ok=True, parents=True)
        if args.use_segmentation_type == "semantic":
            segmentation_imgname = "semantic_0.png"
        elif args.use_segmentation_type == "color-refined":
            segmentation_imgname = "combined_color_refined_semantic_0.png"
        else:
            raise ValueError(f"Unrecognized segmentation type: {args.segmentation_type}.")
        try:
            S = Stroke_Extraction(
                segmentation_img_path=str(situation_dir / segmentation_imgname),
                rgb_img_path=str(situation_dir / "rgb_0.png"),
                debug_matlab_stroke_img_save_path=str(
                    output_situation_dir / "matlab_stroke_0.png"),
                stroke_img_save_path=str(output_situation_dir / "stroke_0.png"),
                stroke_graph_img_save_path=str(
                    output_situation_dir / "stroke_graph_0.png"),
                output_dir=output_situation_dir,
                should_merge_small_strokes=args.merge_small_strokes,
                obj_type="",
                # TODO create issue: in the reorganized curriculum format, we don't store the
                #  information about the object view/ID in an authoritative global-per-sample way such
                #  that we could retrieve it here in the stroke extraction code.
                obj_view="-1",
                obj_id="-1",
                process_masks_independently=args.process_masks_independently,
            )
            S.get_strokes()
        except ValueError as e:
            if args.tolerate_errors:
                logger.warning("Couldn't process situation %d; continuing...", situation_num)
                logger.debug("Error in situation %d was %s", situation_num, e)
            else:
                raise
    print("out")


if __name__ == "__main__":
    main()
