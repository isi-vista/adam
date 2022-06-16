import torch.nn as nn
import pickle
import numpy as np
import torch


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
        # x = nn.ReLU()(x)
        x = self.linear(x)
        return x


#         return nn.LogSoftmax()(x)

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

                # e[(ee[0][jj], ee[1][jj])] = angle
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
                # e[(ee[0][jj], ee[1][jj])] = np.concatenate([distance_map, angle])
        edge_info.append(e)
    return node_info, edge_info, num_nodes
