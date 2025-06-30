import copy
import os
import sys
import heapq
import pickle

from src.core.explainer_base import Explainer
import numpy as np
import torch

from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import  init_dflts_to_of 
from utils_martina.phase_trajectory_analyzer import PhaseTrajectoryAnalyzer

class TemporalDCESExplainer(Explainer):
    """The Temporal Distribution Compliant Explanation Search Explainer identifies
    the top-k counterfactual instances by searching within the subset of stable
    instances in the dataset.
    The selection is based on minimizing a metric that includes:
    1. A measure of dissimilarity between the counterfactual and the original
       instance, computed as the Frobenius norm of the difference between their graph
       Laplacians multiplied by the node features;
    2. A measure of temporal distance between the instance and the counterfactual;
    3. A measure of instability of the counterfactual (linked to permanence time
       in the stability region).
    This approach ensures that selected counterfactuals are not only structurally and
    temporally meaningful, but also lie within regions of stable behavior.
    """

    def check_configuration(self):
        super().check_configuration()

        # Dissimilarity metric
        dst_metric = 'src.evaluation.evaluation_metric_laplacian.LaplacianMetric'

        # Check if the dissimilarity metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)


    def init(self):
        super().init()

        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])
        
        # Optional parameter for number the number of counterfactuals to select
        self.top_k = self.local_config['parameters'].get('top_k', 10)

        # Optional parameters for \alpha, \beta and \gamma to weight the terms
        self.alpha = self.local_config['parameters'].get('dissimilarity_weight', 8)
        self.beta = self.local_config['parameters'].get('time_weight', 1.5)
        self.gamma = self.local_config['parameters'].get('stability_weight', 0.5)


    def explain(self, instance, return_list=False):
        # Get label of the current instance
        input_label = self.oracle.predict(instance)

        # We only want explanations for class 1
        if input_label != 1:
            return copy.deepcopy(instance)
    
        # Get ids of candidates in set C and their permanence time
        candidates_idx, candidates_permanence = self.get_candidates_and_permanence(instance.patient_id, instance.record_id)
        values_stability = ( 1 - (candidates_permanence - min(candidates_permanence)) / (max(candidates_permanence) - min(candidates_permanence)) )
        candidates_dict = {t: val for t, val in zip(candidates_idx, values_stability)}

        ids = []
        M_dissim = []
        M_time = []
        M_instab = []

        candidates_map = {}

        # Consider only same patient, same record, previous time, and stable candidates
        relevant_instances = [
            inst for inst in self.dataset.instances
            if inst.patient_id == instance.patient_id and
            inst.record_id == instance.record_id and
            inst.time <= instance.time and
            inst.time in candidates_dict
        ]

        # Iterating over all relevant instances
        for ctf_candidate in relevant_instances:
            # Get label of the current candidate
            candidate_label = self.oracle.predict(ctf_candidate)

            # If the candidate is valid, compute the terms in the metric
            if input_label != candidate_label:
                ids.append(ctf_candidate.time)
                M_dissim.append(self.distance_metric.evaluate(instance, ctf_candidate, self.oracle))
                M_time.append(instance.time - ctf_candidate.time)
                M_instab.append(candidates_dict[ctf_candidate.time])
                candidates_map[ctf_candidate.time] = ctf_candidate

        # If there is no counterfactual, return current instance
        if not ids:
            return copy.deepcopy(instance)
        
        # Normalize values to control their contribution
        M_dissim = self.normalize(M_dissim)
        M_time = self.normalize(M_time)
        M_instab = self.normalize(M_instab)

        # Compute M
        M_values = self.alpha * M_dissim + self.beta * M_time + self.gamma * M_instab

        # Sort candidates by ascending M (lower is better) and select top k
        sorted_indices = np.argsort(M_values)
        top_k_indices = sorted_indices[:self.top_k]
            
        # Retrieve the corresponding instances
        top_k_candidates = []
        for i in top_k_indices:
            t = ids[i]
            candidate_instance = candidates_map.get(t)
            if candidate_instance:
                top_k_candidates.append( (M_values[i], candidate_instance) )

        if not top_k_candidates:
            return copy.deepcopy(instance)

        if return_list: # Return all top k counterfactuals
            return top_k_candidates
        else: # Return only top counterefactual
            return top_k_candidates[0][1]
    

    # Get stable candidates and permanence time from phase trajectory analyzer
    def get_candidates_and_permanence(self, patient_id, record_id):
        time, series = self.get_time_and_softmax(patient_id, record_id)

        analyzer = PhaseTrajectoryAnalyzer(time, series, L=10, percentile=0.25)
        candidates_idx, candidates_permanence = analyzer.get_idx_candidates_and_permanence()

        return candidates_idx, candidates_permanence


    # Get time and softmax associates to current record
    # TO DO: optimize it
    def get_time_and_softmax(self, patient_id, record_id):
        with open(f"EEG_data\EEG_data_params_{patient_id}_{record_id}.pkl", "rb") as f:
            loaded_variables = pickle.load(f)

        indices = loaded_variables["indici"]
        Start = loaded_variables["Start"]

        softmax = []
        time = []

        for g1 in self.dataset.instances:
            if g1.patient_id == patient_id and g1.record_id == record_id:
                logits = self.oracle.predict_proba(g1)
                softmax.append(torch.softmax(logits, dim=0))
                time.append(indices[g1.time]/256 + Start)

        time = np.array(time)
        softmax = np.array(softmax)[:,1]

        return time, softmax
    
    
    # Normalize values
    @staticmethod
    def normalize(arr):
        arr = np.array(arr)
        min_val = arr.min()
        max_val = arr.max()

        if max_val - min_val == 0:
            return np.zeros_like(arr)
        
        return (arr - min_val) / (max_val - min_val)