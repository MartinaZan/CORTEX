from os.path import join

import json
import numpy as np
import pandas as pd

import random

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance

class GCS(Generator):
    def init(self):
        base_path = self.local_config['parameters']['data_dir']
        self._data_file_path = join(base_path, self.local_config['parameters']['data_file_name'])
        self._data_label_name = self.local_config['parameters']['data_label_name']

        self.dataset.node_features_map = None
        self.dataset.edge_features_map = None#rdk_edge_features_map()
        self.generate_dataset()

    def check_configuration(self):
        super().check_configuration()
        local_config=self.local_config

        if 'data_file_name' not in local_config['parameters']:
            raise Exception("The name of the data file must be given.")
       
        if 'data_label_name' not in local_config['parameters']:
            raise Exception("The name of the label column must be given.")

    def generate_dataset(self):
        if not len(self.dataset.instances):
            self._read(self._data_file_path)

    def _read(self, path):
        data = json.load(open(path))
        # data.keys() --> dict_keys(['edge_mapping', 'node_ids', 'time_periods', 'target', 'series'])
        #   In 'target' ho la label della classificazione
        #   In 'series' ho i valori della serie (da mettere come feature dei nodi)

        self.dataset.node_features_map = {'attribute_0': 0} #data['node_ids']
        # self.dataset.edge_features_map =  # change me

        random.shuffle(data['time_periods']) #####
        
        for t in data['time_periods']:
            # OLD
            #A,X,W = corr2graph(t, data['edge_mapping']['edge_index'][str(t)], data['edge_mapping']['edge_weight'][str(t)], data["series"][t-49], self.dataset)
            #y = data["target"][t-49]
            #if (t-49) % 3 == 0:
            #    y = 1 

            # NEW
            A,X,W = corr2graph(t, data['edge_mapping']['edge_index'][str(t)], data['edge_mapping']['edge_weight'][str(t)], data["series"][t], self.dataset)
            y = data["target"][t]

            g = GraphInstance(
                id = t,
                label = y,          # crisi/no crisi
                data = A,           # data: n × n matrix where n is the number of nodes (i.e., it is the binary adjacency matrix)
                node_features = X,  # node_features: n × d matrix where d is the number of node features (per ora considero d = 1)
                edge_weights = W,   # edge_weights: n × n matrix containing the weight of each edge (i.e., it is the weighted adjacency matrix)
                graph_features = {}
            )
            self.dataset.instances.append(g)
    
def corr2graph(id, data, weight, series, dataset):
    # n_map = dataset.node_features_map
    # e_map = dataset.edge_features_map = {k:1 for k in range(376)}
    
    n = 24

    A = np.zeros((n, n))
    W = np.zeros((n, n))

    X = np.array(series).reshape(24, 1)

    for p, edge in enumerate(data):
        #if np.abs(weight[p]) > 0.3:
        W[edge[0], edge[1]] = np.abs(weight[p])
        A[edge[0], edge[1]] = 1 if weight[p] != 0 else 0

    W = W[W != 0].flatten()

    return A,X,W
