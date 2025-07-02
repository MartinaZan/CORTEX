from os.path import join

import json
import numpy as np
import random

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance

class EEG_CORR(Generator):
    def init(self):
        base_path = self.local_config['parameters']['data_dir']
        file_names = self.local_config['parameters']['data_file_name']

        self._data_file_path = [join(base_path, file_name) for file_name in file_names]
        self._data_label_name = self.local_config['parameters']['data_label_name']

        self.num_node_features = None

        self.dataset.node_features_map = None
        self.dataset.edge_features_map = None

        self.generate_dataset()

    def check_configuration(self):
        super().check_configuration()
        local_config=self.local_config

        if 'data_file_name' not in local_config['parameters']:
            raise Exception("The name of the data file must be given.")
       
        if 'data_label_name' not in local_config['parameters']:
            raise Exception("The name of the label column must be given.")

    # Function for generating the dataset, which calls the function to read data from the JSON file
    def generate_dataset(self):
        if not len(self.dataset.instances):
            for item in self._data_file_path:
                self._read(item)

    def _read(self, path):
        if not len(self.dataset.instances):
            idx = 0
        else:
            idx = len(self.dataset.instances)

        # Opening the JSON file containing graph data.
        # 'target' contains the true labels for classification
        # 'series' contains the multivariate time series values, used to define node features
        data = json.load(open(path))

        self.num_node_features = len(data["series"][0])

        # Definition of node and edge feature maps. These are dictionaries containing as many entries as features.
        self.dataset.node_features_map = {f'attribute_{i}': i for i in range(self.num_node_features)}
        # self.dataset.edge_features_map = ...

        # Create a graph for each time step
        for t in data['record_time_ids']:
            A,X,W = data2graph(t, data['edge_mapping']['edge_index'][str(t)], data['edge_mapping']['edge_weight'][str(t)], data["series"][t], data["target"][t], self.dataset)
            
            # Read the true label at time (seizure / no seizure)
            y = data["target"][t]
            
            time_stamp = data["record_time_stamps"][t]
            patient_id = data["patient_id"]
            record_id = data["record_id"]

            g = GraphInstance(
                id = idx,                   # global graph id
                label = y,                  # label seizure / no seizure
                data = A,                   # data: n x n binary adjacency matrix
                node_features = X,          # node_features: n x d node feature matrix
                edge_weights = W,           # edge_weights: n x n weighted adjacency matrix
                graph_features = {},        # graph-level features (currently none)
                time_id = t,                # id temporale (sarebbe id singolo paziente)
                time_stamp = time_stamp,    # Vero tempo
                patient_id = patient_id,    # patient_id
                record_id = record_id,      # record_id
            )
            self.dataset.instances.append(g)

            idx = idx + 1
    
def data2graph(id, data, weight, series, target, dataset):
    n_map = dataset.node_features_map
    # e_map = dataset.edge_features_map
    
    # Numero di nodi
    n = len(series[0])

    A = np.zeros((n, n))
    W = np.zeros((n, n))

    ##########################################################################

    if len(n_map) == 1:
        X = np.array(series).reshape(n, 1)
    else:
        X = np.fliplr(np.array(series).T)

    ##########################################################################

    # Scorre tutti gli archi e imposta matrice di adiacenza pesata (W) e binaria (A)
    for p, edge in enumerate(data):
        if np.abs(weight[p]) > 0 : # 0.3: # (condizione per threshold)
            W[edge[0], edge[1]] = np.abs(weight[p])
            A[edge[0], edge[1]] = 1 if weight[p] != 0 else 0

    # W deve essere un vettore, quindi lo flatteniamo
    W = W[W != 0].flatten()

    return A,X,W
