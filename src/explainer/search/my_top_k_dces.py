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

class DCESExplainer(Explainer):
    """The Distribution Compliant Explanation Search Explainer performs a search of 
    the minimum counterfactual instance in the original dataset instead of generating
    a new instance. Here, we look for the top-k counterfactuals."""

    # CAMBIARE QUESTO COMMENTO :)
    # The explainer finds a counterfactual by minimizing a quantity which includes:
    #  - Dissimilarity metric (distance node embeddings)
    #  - Time penalty (how far the counterfactual is from the original instance in time)
    #  - Stability penalty (linked to permanence time in stability region)

    def check_configuration(self):
        super().check_configuration()

        # Dissimilarity metric to use for the distance between instances
        # dst_metric = 'src.evaluation.evaluation_metric_embeddings.EmbeddingMetric'
        dst_metric = 'src.evaluation.evaluation_metric_laplacian.LaplacianMetric'

        # Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)


    def init(self):
        super().init()

        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])
        
        # Optional parameter for number of top_k counterfactuals to be selected
        self.top_k = self.local_config['parameters'].get('top_k', 10)

        self.dissimilarity_weight = self.local_config['parameters'].get('metric_coefficient', 8)

        # Optional parameters for magnitude penalties (0 to ignore penalties)
        self.time_weight = self.local_config['parameters'].get('max_time_penalty', 1.5)

        # Optional parameters for magnitude penalties (0 to ignore penalties)
        self.stability_weight = self.local_config['parameters'].get('stability_penalty', 0.5)


    def explain(self, instance, return_list=False):
        # Get label of the current instance
        input_label = self.oracle.predict(instance)

        if input_label != 1:
            return copy.deepcopy(instance)
    
        # Bounds of the interval within which the counterfactual search can take place
        candidates_idx, candidates_stability = self.get_candidates_and_stability(instance.patient_id, instance.record_id)
        values_stability = ( 1 - (candidates_stability - min(candidates_stability)) / (max(candidates_stability) - min(candidates_stability)) )

        ids = []
        dissimilarity_terms = []
        time_terms = []
        stability_terms = []

        # Iterating over all the instances of the dataset
        for ctf_candidate in self.dataset.instances:
            # Consider only same patient, same record and previous time
            if ctf_candidate.time <= instance.time and ctf_candidate.time in set(candidates_idx) and \
                ctf_candidate.patient_id == instance.patient_id and ctf_candidate.record_id == instance.record_id:

                candidate_label = self.oracle.predict(ctf_candidate)

                if input_label != candidate_label:
                    # Compute terms
                    ids.append(ctf_candidate.time)
                    # dissimilarity_terms.append(self.distance_metric.evaluate(instance, ctf_candidate, self.oracle).cpu().numpy())
                    dissimilarity_terms.append(self.distance_metric.evaluate(instance, ctf_candidate, self.oracle))
                    time_terms.append(instance.time - ctf_candidate.time)
                    stability_terms.append(values_stability[np.where(candidates_idx == ctf_candidate.time)[0][0]])

        if not ids:
            return copy.deepcopy(instance)
        
        dissimilarity_terms = self.normalize(dissimilarity_terms)
        time_terms = self.normalize(time_terms)
        stability_terms = self.normalize(stability_terms)

        total_scores = self.dissimilarity_weight * dissimilarity_terms + self.time_weight * time_terms + self.stability_weight * stability_terms

        # Sort candidates by ascending score (lower is better) and select top k
        sorted_indices = np.argsort(total_scores)
        top_k_indices = sorted_indices[:self.top_k]
            
        # Retrieve the corresponding instances
        top_k_candidates = []
        for i in top_k_indices:
            t = ids[i]
            candidate_instance = next((c for c in self.dataset.instances if c.patient_id == instance.patient_id and c.record_id == instance.record_id and c.time == t), None)
            if candidate_instance:
                top_k_candidates.append( (total_scores[i], candidate_instance) )

        if not top_k_candidates:
            return copy.deepcopy(instance)

        if return_list:
            return top_k_candidates
        else:
            return top_k_candidates[0][1]
    

    def get_candidates_and_stability(self, patient_id, record_id):

        time, series = self.get_time_and_softmax(patient_id, record_id)

        analyzer = PhaseTrajectoryAnalyzer(time, series, L=10, percentile=0.25)
        candidates_idx, candidates_stability = analyzer.get_idx_candidates_and_permanence()

        return candidates_idx, candidates_stability


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
    
    
    @staticmethod
    def normalize(arr):
        arr = np.array(arr)
        min_val = arr.min()
        max_val = arr.max()

        if max_val - min_val == 0:
            return np.zeros_like(arr)
        
        return (arr - min_val) / (max_val - min_val)