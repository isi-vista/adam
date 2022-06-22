# copied from https://github.com/ASU-APG/adam-stage/tree/main/processing
# code by Sheng Cheng
import logging
import cv2
import numpy as np
try:
    import matlab
except ImportError:
    logging.warning("Couldn't import MATLAB API; creating a placeholder object instead.")
    matlab = object()
import matplotlib.pyplot as plt
import scipy.io
try:
    import matlab.engine
except ImportError:
    logging.warning("Couldn't import MATLAB engine; setting it to None.")
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
            -3 * np.power(vs[sel2] - vi[sel2] - 1, 3) + 3 * np.power(vs[sel2] - vi[sel2] - 1, 2) + 3 * (
            vs[sel2] - vi[sel2] - 1) + 1)

    sel3 = np.logical_and((vs >= vi + 2), (vs < vi + 3))
    C[sel3] = (1 / 6) * (3 * np.power(vs[sel3] - vi[sel3] - 2, 3) - 6 * np.power(vs[sel3] - vi[sel3] - 2, 2) + 4)

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
    n = len(strokeinfo)
    connection = np.zeros([n, n])
    for i in range(n):
        node_ind = strokeinfo[i]
        con = np.logical_or(strokeinfo == node_ind[0], strokeinfo == node_ind[1])
        con = np.logical_or(con[:, 0], con[:, 1])
        connection[i, np.where(con)[0]] = 1
        if node_ind[0] == node_ind[1]:
            connection[i, i] = 1
        else:
            connection[i, i] = 0
    return connection


class Stroke_Extraction():
    """
    obj_type: object name + train/test
    obj_id: the id of the object
    obj_view: the id of the camera
    base_path : the root path for the folder
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

    def __init__(self, obj_type="outputs", obj_id="1", obj_view="1",
                 base_path='/Users/cs/Desktop/Research/darpa_3d/curriculum_basic_v1_1/curriculum/test/phase3-m4-core-eval/generation_yaml/',
                 vis=True, save_output=True):
        self.obj_view = obj_view
        self.obj_type = obj_type
        self.obj_id = obj_id
        self.vis = vis
        self.save_output = save_output
        self.path = os.path.join(base_path, obj_type, 'cam{}'.format(obj_view), 'semantic_{}.png'.format(obj_id))
        rgb_path = os.path.join(base_path, obj_type, 'cam{}'.format(obj_view), 'rgb_{}.png'.format(obj_id))
        self.img_bgr = img_bgr = cv2.imread(rgb_path)
        self.img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.save_dir = os.path.join(base_path, 'feature')

    def stroke_extraction_from_matlab(self):
        eng = matlab.engine.start_matlab()
        s = eng.genpath('./BPL')
        eng.addpath(s, nargout=0)
        out = eng.ske(self.path, nargout=2)
        eng.close()
        return out

    def remove_strokes(self):
        out = self.stroke_extraction_from_matlab()
        strokes = []
        removed_ind = []
        adj = kp2stroke(np.array(out[1]))
        for i in range(len(out[0])):
            s = np.array(out[0][i])
            if len(s) < 10:
                removed_ind.append(i)
                continue
            strokes.append(s)
        strokes = np.array(strokes)
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
                        adj[i, j] = 1.
                        adj[j, i] = 1.
        self.num_obj = num_obj = n_components
        if adj is not [[1.]] and num_obj != 1:
            clustering = SpectralClustering(n_clusters=num_obj, affinity='precomputed').fit(adj)
            labels_ = clustering.labels_
        elif num_obj == 1:
            labels_ = np.zeros(len(adj))
        else:
            labels_ = np.zeros(num_obj)
        self.label = labels_
        self.strokes = strokes
        self.adj = adj

    def strokes2controlpoints(self):
        reduced_strokes = []
        coef_inv = get_bsplines_matrix(10)
        for i in range(len(self.strokes)):
            reduced_strokes.append(reduced_stroke(self.strokes[i], coef_inv))
        self.reduced_strokes = reduced_strokes = np.array(reduced_strokes)

    def plot_strokes(self):
        stroke_colors = []
        for j in range(self.num_obj):
            reduced_obj = self.reduced_strokes[np.where(self.label == j)[0]]
            plt.subplot(1, self.num_obj, j + 1)
            s_c = []
            for i in range(len(reduced_obj)):
                p = plt.plot(reduced_obj[i][:, 1], reduced_obj[i][:, 0], linewidth=3)
                s_c.append(p[0].get_color())
                plt.scatter(reduced_obj[i][:, 1], reduced_obj[i][:, 0], c='r')
            stroke_colors.append(s_c)
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis("off")
        plt.savefig(
            os.path.join(self.save_dir, 'stroke_{}_{}_{}.png'.format(self.obj_type, self.obj_view, self.obj_id)))
        plt.close()
        self.stroke_colors = stroke_colors

    def plot_graph(self):
        for i in range(self.num_obj):
            plt.figure(i)
            G = nx.Graph()
            if self.adj is [[0.]]:
                G.add_node("s0")
            else:
                ind = np.where(self.label == i)[0]
                adj_obj = self.adj[ind, :]
                adj_obj = adj_obj[:, ind]

                for p in range(len(adj_obj)):
                    G.add_node("s_{}".format(int(p + 1)))
                    for q in range(p, len(adj_obj)):
                        if adj_obj[p, q] == 1:
                            G.add_edge("s_{}".format(int(p + 1)), "s_{}".format(int(q + 1)))
            plt.subplot(1, self.num_obj, i + 1)
            nx.draw(G, with_labels=True, node_color=self.stroke_colors[i])
        plt.savefig(
            os.path.join(self.save_dir, 'stroke_graph_{}_{}_{}.png'.format(self.obj_type, self.obj_view, self.obj_id)))
        plt.close()

    def get_colors(self):
        img_seg = cv2.imread(self.path)
        img_seg = cv2.cvtColor(img_seg, cv2.COLOR_BGR2GRAY)
        obj_area = np.where(img_seg != 0)
        img_tmp = np.zeros([256, 256, 3], dtype=np.uint8)
        img_tmp[obj_area[0], obj_area[1], :] = self.img_rgb[obj_area[0], obj_area[1], :]
        colors = np.zeros([self.num_obj, 5])
        for i in range(self.num_obj):
            if i >= len(np.unique(img_seg[obj_area])):
                continue
            seg_i = np.array(np.where(img_seg == np.unique(img_seg[obj_area])[i]))
            colors[i, :3] = self.img_rgb[seg_i[0], seg_i[1], :].mean(0)
            colors[i, 3:] = seg_i.mean(1)
        self.colors = colors

    def get_strokes(self):
        self.remove_strokes()
        self.strokes2controlpoints()
        self.get_colors()
        if self.vis:
            self.plot_strokes()
            self.plot_graph()
        data = []
        if len(self.strokes) == 0:
            return data
        for i in range(self.num_obj):
            ind = np.where(self.label == i)[0]
            reduced_obj = self.reduced_strokes[ind]
            adj_obj = self.adj[ind, :]
            adj_obj = adj_obj[:, ind]
            m = reduced_obj.mean((0, 1))
            dd = pairwise_distances(m[None, :], self.colors[:, 3:])
            color = self.colors[dd[0].argmin(0), :3]
            s = (reduced_obj - m).std()
            reduced_strokes_norm = (reduced_obj - m) / s
            distance = dict()
            for j in range(self.num_obj):
                if i == j:
                    continue
                else:
                    ind_ = np.where(self.label == j)[0]
                    reduced_obj_ = self.reduced_strokes[ind_]
                    m_ = reduced_obj_.mean((0, 1))
                    distance["object" + str(j)] = (np.sqrt(((m - m_) ** 2).sum())).tolist()
            if len(distance.keys()) == 0:
                distance = None
            data.append(dict(
                object_name="object" + str(i),
                subobject_id="0",
                viewpoint_id=self.obj_view,
                distance=distance,
                stroke_graph=dict(
                    adjacency_matrix=adj_obj.tolist(),
                    stroke_mean_x=reduced_obj.mean((0, 1)).tolist()[1],
                    stroke_mean_y=reduced_obj.mean((0, 1)).tolist()[0],
                    stroke_std=reduced_obj.std((0, 1, 2)).tolist(),
                    strokes_normalized_coordinates=reduced_strokes_norm.transpose(0, 2, 1).tolist(),
                    concept_name=self.obj_type,
                    confidence_score=1.,
                ),
                color=color.tolist(),
                texture=None,
                sub_part=None
            ))
        self.data = data
        if self.save_output:
            with open(os.path.join(self.save_dir,
                                   'feature_{}_{}_{}.yaml'.format(self.obj_type, self.obj_view, self.obj_id)),
                      'w') as file:
                documents = yaml.dump(data, file)
                file.close()
        return data


def main():
    for i in range(3):
        for j in range(5):
            S = Stroke_Extraction(base_path='/home/scheng53/Desktop/darpa_3d/adam_single_objv0', obj_type='test_small_single_mug', obj_view=str(i), obj_id=str(j))
            out = S.get_strokes()
    print("out")


if __name__ == '__main__':
    main()
