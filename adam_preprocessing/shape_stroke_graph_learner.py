# copied from https://github.com/ASU-APG/adam-stage/tree/main/processing
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

phase = ['train', 'test']




if __name__ == "__main__":

    "Processing data from image to stroke graph"
    base_path = '/Users/cs/Desktop/Research/darpa_3d/adam_single_objv0/'
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
        for l, j in enumerate(label_name):
            for p in range(1):
                for q in range(num_sample):
                    if not os.path.exists(
                            os.path.join(base_path, 'feature', 'feature_{}_{}_{}_{}.yaml'.format(i, j, p, q))):
                        Extractor = Stroke_Extraction(obj_type="{}_{}".format(i, j), obj_id=q, obj_view=p,
                                                      base_path=base_path, vis=True, save_output=True)
                        out = Extractor.get_strokes()
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
    h = np.zeros([len(train_coords), num_nodes, 100])
    g = np.zeros([len(train_coords), num_nodes, num_nodes])
    e = np.zeros([len(train_coords), num_nodes, num_nodes, 100])

    for i in range(len(train_coords)):
        num_node = len(train_adj[i])
        h[i, :num_node, :] = nodes[i]
        g[i, :num_node, :num_node] = np.array(train_adj[i])
        for edge in edges[i].keys():
            e[i, edge[0], edge[1], :] = edges[i][edge]
            e[i, edge[1], edge[0], :] = edges[i][edge]
    h = torch.tensor(h).type(torch.FloatTensor).to(device)
    e = torch.tensor(e).type(torch.FloatTensor).to(device)
    g = torch.tensor(g).type(torch.FloatTensor).to(device)
    label = torch.tensor(np.array(train_label)).type(torch.LongTensor).to(device)

    nodes_t, edges_t, num_nodes_t = load_data(test_coords, test_adj)
    test_h = np.zeros([len(test_coords), num_nodes_t, 100])
    test_g = np.zeros([len(test_coords), num_nodes_t, num_nodes_t])
    test_e = np.zeros([len(test_coords), num_nodes_t, num_nodes_t, 100])

    for i in range(len(test_coords)):
        num_node = len(test_adj[i])
        test_h[i, :num_node, :] = nodes_t[i]
        test_g[i, :num_node, :num_node] = np.array(test_adj[i])
        for edge in edges_t[i].keys():
            test_e[i, edge[0], edge[1], :] = edges_t[i][edge]
            test_e[i, edge[1], edge[0], :] = edges_t[i][edge]
    test_h = torch.tensor(test_h).type(torch.FloatTensor).to(device)
    test_e = torch.tensor(test_e).type(torch.FloatTensor).to(device)
    test_g = torch.tensor(test_g).type(torch.FloatTensor).to(device)
    test_label = torch.tensor(np.array(test_label)).type(torch.LongTensor).to(device)

    "MPNN model"

    model_base = MPNN([100, 100], hidden_state_size=100,
                      message_size=20, n_layers=1, l_target=50,
                      type='regression')
    model = LinearModel(model_base, 50, len(label_name)).to(device)
    criterion = nn.NLLLoss()
    evaluation = accuracy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    losses = []
    best_acc = 0

    "Training"
    for i in range(1000):
        train_g, train_h, train_e, train_target = Variable(g), Variable(h), Variable(e), Variable(
            label)
        optimizer.zero_grad()
        output = model(train_g, train_h, train_e)
        train_loss = criterion(nn.LogSoftmax()(output), train_target)
        losses.append(train_loss.item())
        acc = Variable(evaluation(output.data, train_target.data, topk=(1,))[0])
        test_acc = Variable(
            evaluation((nn.LogSoftmax()(model(test_g, test_h, test_e))).data, test_label.data, topk=(1,))[0])
        if test_acc > best_acc:
            best_acc = test_acc
        train_loss.backward()
        optimizer.step()
        print("{}, loss: {:.4e}, acc: {}, test acc :{}".format(i, train_loss.item(), acc, test_acc))
    print("Best test acc is {}".format(best_acc))
