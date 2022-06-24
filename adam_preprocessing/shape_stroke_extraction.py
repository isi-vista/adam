"""
Code for doing stroke extraction on a curriculum.

Based on original code written by Sheng Cheng, found at
https://github.com/ASU-APG/adam-stage/tree/main/processing
"""
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import matlab
except ImportError:
    logger.warning("Couldn't import MATLAB API; creating a placeholder object instead.")
    matlab = object()
import matplotlib.pyplot as plt
import scipy.io

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
        features_save_path: str,
        vis: bool = True,
        save_output: bool = True,
    ):
        self.obj_view = obj_view
        self.obj_type = obj_type
        self.obj_id = obj_id
        self.clustering_seed = clustering_seed
        self.vis = vis
        self.save_output = save_output
        self.path = segmentation_img_path
        self.img_bgr = img_bgr = cv2.imread(rgb_img_path)
        self.img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.stroke_img_save_path = stroke_img_save_path
        self.stroke_graph_img_save_path = stroke_graph_img_save_path
        self.features_save_path = features_save_path

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

    def stroke_extraction_from_matlab(self):
        """
        Get the raw stroke extraction results from Matlab using the BPL code.

        Note that stroke extraction is performed on the object segmentation image, i.e. the
        semantic_j.png file. It is *not* done on the raw RGB image.

        The output is a pair [S, E] where:

        - S is a k x 1 sequence of "edge paths in the image"
        - E is a k x 2 sequence of "graph edges"
        """
        eng = matlab.engine.start_matlab()
        s = eng.genpath("./lightspeed")
        eng.addpath(s, nargout=0)
        s = eng.genpath("./BPL")
        eng.addpath(s, nargout=0)
        out = eng.ske(self.path, nargout=2)
        eng.close()
        return out

    def remove_strokes(self):
        """
        Process Matlab stroke extraction results into labels.

        In this function we:

        - construct the adjacency matrix
        - clean up the strokes, removing those with fewer than 10 key points (?)
        - split the stroke graph into connected components
        - determine the number of objects in the scene as the number of components
        - label each stroke according to which object it is a part of

        The outputs are stored in:

        - self.num_obj
        - self.strokes
        - self.adj
        - self.labels
        """
        out_s, out_e = self.stroke_extraction_from_matlab()
        strokes = []
        removed_ind = []
        adj = kp2stroke(np.array(out_e))
        for i in range(len(out_s)):
            s = np.array(out_s[i])
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
        self.num_obj = num_obj = min(n_components, len(adj))
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
            labels_ = clustering.labels_
        # If there's only one object, things are easy -- just assign all strokes to that one object.
        elif num_obj == 1:
            labels_ = np.zeros(len(adj))
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
                    "No objects detected for segmentation image %s.", self.path
                )
            labels_ = np.zeros(num_obj)
        self.label = labels_
        self.strokes = strokes
        self.adj = adj

    def strokes2controlpoints(self):
        """
        Downsamples the known strokes to 10 points using B-splines.

        Because of the processing logic in self.remove_strokes(), we know that each stroke has at
        least 10 points such that it is possible to do this.

        Outputs the downsampled strokes to self.reduced_strokes. This should be an N x 10 x 2 array,
        where N is the number of strokes. The last dimension is the x vs y coords; the middle is the
        control point number.
        """
        reduced_strokes = []
        coef_inv = get_bsplines_matrix(10)
        for i in range(len(self.strokes)):
            reduced_strokes.append(reduced_stroke(self.strokes[i], coef_inv))
        self.reduced_strokes = reduced_strokes = np.array(reduced_strokes)

    def plot_strokes(self):
        """
        Create stroke image, saving the result to self.stroke_img_save_path.
        """
        stroke_colors = []
        for j in range(self.num_obj):
            reduced_obj = self.reduced_strokes[np.where(self.label == j)[0]]
            plt.subplot(1, self.num_obj, j + 1)
            s_c = []
            for i in range(len(reduced_obj)):
                p = plt.plot(reduced_obj[i][:, 1], reduced_obj[i][:, 0], linewidth=3)
                s_c.append(p[0].get_color())
                plt.scatter(reduced_obj[i][:, 1], reduced_obj[i][:, 0], c="r")
            stroke_colors.append(s_c)
            plt.gca().invert_yaxis()
            plt.gca().set_aspect("equal", adjustable="box")
            plt.axis("off")
        plt.savefig(self.stroke_img_save_path)
        plt.close()
        self.stroke_colors = stroke_colors

    def plot_graph(self):
        """
        Create stroke *graph* image in self.stroke_graph_img_save_path.
        """
        for i in range(self.num_obj):
            plt.figure(i)
            G = nx.Graph()
            if self.adj is [[0.0]]:
                G.add_node("s0")
            else:
                ind = np.where(self.label == i)[0]
                adj_obj = self.adj[ind, :]
                adj_obj = adj_obj[:, ind]

                for p in range(len(adj_obj)):
                    G.add_node("s_{}".format(int(p + 1)))
                    for q in range(p, len(adj_obj)):
                        if adj_obj[p, q] == 1:
                            G.add_edge(
                                "s_{}".format(int(p + 1)), "s_{}".format(int(q + 1))
                            )
            plt.subplot(1, self.num_obj, i + 1)
            nx.draw(G, with_labels=True, node_color=self.stroke_colors[i])
        plt.savefig(self.stroke_graph_img_save_path)
        plt.close()

    def get_colors(self):
        """
        Collect the average color for each object in the image together with the object "centers."

        This uses the object segmentation image to identify which pixels in the RGB image "belong
        to" each object. This segmentation image file is usually named semantic_{stuff}.png
        """
        # jac: img_seg is a 256x256 integer array? gray-values/pixel values are distinct iff the
        # objects for those locations are (considered) distinct. I think that is the convention
        # here.
        img_seg = cv2.imread(self.path)
        img_seg = cv2.cvtColor(img_seg, cv2.COLOR_BGR2GRAY)
        # jac: obj_area is a tuple of two equal-length arrays xs, ys such that for all i,
        # img_seg[xs[i], ys[i]] != 0.
        obj_area = np.where(img_seg != 0)
        img_tmp = np.zeros([256, 256, 3], dtype=np.uint8)
        img_tmp[obj_area[0], obj_area[1], :] = self.img_rgb[obj_area[0], obj_area[1], :]
        colors = np.zeros([self.num_obj, 5])
        for i in range(self.num_obj):
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
        self.colors = colors

    def get_strokes(self):
        """
        Extract strokes from the image at self.path.

        This performs stroke extraction as well as reformatting the outputs for use in ADAM.

        This writes a YAML file to self.features_save_path if self.save_output is true.
        """
        # Extract strokes, downsample, and add color data
        self.remove_strokes()
        self.strokes2controlpoints()
        self.get_colors()
        if self.vis:
            self.plot_strokes()
            self.plot_graph()
        data = []
        if len(self.strokes) == 0:
            return data
        # Translate the extracted (downsampled strokes) and color data into the ADAM stroke
        # extraction data format.
        for i in range(self.num_obj):
            # Using self.label, pick out the IDs/indices for the strokes belonging to object i.
            # Use this to grab the relevant stroke samples and the relevant submatrix of self.adj.
            ind = np.where(self.label == i)[0]
            # shape: N_i x 10 x 2 where N_i is the number of strokes in the ith object
            reduced_obj = self.reduced_strokes[ind]
            # shape: N_i x N_i
            adj_obj = self.adj[ind, :]
            adj_obj = adj_obj[:, ind]

            # Calculate the overall mean coordinates across all strokes and coordinates.
            # This is a 2D array with 2 entries -- one per coordinate.
            m = reduced_obj.mean((0, 1))
            # Get pairwise distances between the stroke coordinate mean (treated as a 1x2 array) and
            # the pixel-based object centers.
            dd = pairwise_distances(m[None, :], self.colors[:, 3:])
            # Choose the color using the nearest pixel-based object center.
            color = self.colors[dd[0].argmin(0), :3]
            # We normalize using a standard deviation s that is calculated using both the deviations
            # from the x-mean and the deviations from the y-mean, as if these have identical
            # distributions. This is sort of weird.
            s = (reduced_obj - m).std()
            reduced_strokes_norm = (reduced_obj - m) / s

            # Calculate pixel-space-distance from this object to every other object in the image.
            # We use the Euclidean distance between the two objects' stroke coordinate means as our
            # measure of distance between objects.
            distance = dict()
            for j in range(self.num_obj):
                if i == j:
                    continue
                else:
                    ind_ = np.where(self.label == j)[0]
                    reduced_obj_ = self.reduced_strokes[ind_]
                    m_ = reduced_obj_.mean((0, 1))
                    distance["object" + str(j)] = (
                        np.sqrt(((m - m_) ** 2).sum())
                    ).tolist()
            if len(distance.keys()) == 0:
                distance = None

            data.append(
                dict(
                    object_name="object" + str(i),
                    subobject_id="0",
                    viewpoint_id=self.obj_view,
                    distance=distance,
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
                    ),
                    color=color.tolist(),
                    texture=None,
                    sub_part=None,
                )
            )
        self.data = data
        if self.save_output:
            with open(
                self.features_save_path,
                "w",
            ) as file:
                yaml.dump({"objects": data}, file)
                file.close()
        return data


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "base_path",
        help="The raw curriculum path to process.",
    )
    parser.add_argument(
        "--object_types",
        help="The object types to process.",
        nargs="+",
        default=("test_small_single_mug",),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    for object_type in args.object_types:
        for i in range(3):
            for j in range(5):
                S = Stroke_Extraction.from_m5_objects_curriculum_base_path(
                    base_path=args.base_path,
                    obj_type=object_type,
                    obj_view=str(i),
                    obj_id=str(j),
                )
                out = S.get_strokes()
    print("out")


if __name__ == "__main__":
    main()
