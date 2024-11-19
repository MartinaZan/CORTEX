from os.path import join

import json
import numpy as np
import pandas as pd

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
        # data.keys() = { 'node_ids', 'edge_mapping', 'time_periods', 'y' }
        data_labels = data[self._data_label_name]
        self.dataset.node_features_map = data['node_ids']
        # self.dataset.edge_features_map =  # change me

        for t in data['time_periods']:
            A,X,W = corr2graph(t, data['edge_mapping']['edge_index'][str(t)], data['edge_mapping']['edge_weight'][str(t)], data_labels[t-49], self.dataset)
            g = GraphInstance(
                id=t,
                label=1, # change me
                data=A,
                node_features=X,
                edge_features=W,
                graph_features={}
            )
            self.dataset.instances.append(g)


    
def corr2graph(id, data, weight, label, dataset):
    n_map = dataset.node_features_map
    e_map = dataset.edge_features_map = {k:1 for k in range(376)}
    n1, n2 = np.array(data).max(0)
    A = np.zeros((n1+1, n2+1))
    X = np.zeros((n1+1, len(n_map)))
    W = np.zeros((n1+1, len(e_map)))

    for i, n in enumerate(n_map):
        X[i, n_map[n]-1] = label[i]

    for p, edge in enumerate(data):
        A[edge[0], edge[1]] = 1

        W[edge[0], edge[1]] = weight[p]
    
    return A,X,W

