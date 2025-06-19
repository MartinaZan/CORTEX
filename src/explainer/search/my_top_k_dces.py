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
from utils_martina.phase_space_analyzer import PhaseSpaceAnalyzer

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
        dst_metric = 'src.evaluation.evaluation_metric_embeddings.EmbeddingMetric'

        # Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)


    def init(self):
        super().init()

        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])
        
        # Optional parameter for number of top_k counterfactuals to be selected
        self.top_k = self.local_config['parameters'].get('top_k', 10)

        self.metric_coefficient = self.local_config['parameters'].get('metric_coefficient', 10)

        # Optional parameters for magnitude penalties (0 to ignore penalties)
        self.max_time_penalty = self.local_config['parameters'].get('max_time_penalty', 9)

        # Optional parameters for magnitude penalties (0 to ignore penalties)
        self.stability_penalty = self.local_config['parameters'].get('stability_penalty', 2.5)


    def explain(self, instance, return_list=False):
        # Get label of the current instance
        input_label = self.oracle.predict(instance)

        if input_label == 1:
            # Bounds of the interval within which the counterfactual search can take place
            candidates_idx, candidates_stability = self.get_candidates_and_stability(instance.patient_id, instance.record_id)
            values_stability = ( 1 - (candidates_stability - min(candidates_stability)) / (max(candidates_stability) - min(candidates_stability)) )

            # print(f"Paziente: {instance.record_id}")
            # print(f"Tempo: {instance.time}")
            # print(f"Set: {set(candidates_idx)}")

            top_k_heap = []

            values = None

            # Iterating over all the instances of the dataset
            for ctf_candidate in self.dataset.instances:
                # Considera solo stesso paziente, stesso record e tempo precedente (in caso cambia <= con <)
                if ctf_candidate.time <= instance.time and ctf_candidate.time in set(candidates_idx) and \
                    ctf_candidate.patient_id == instance.patient_id and ctf_candidate.record_id == instance.record_id:

                    candidate_label = self.oracle.predict(ctf_candidate)

                    if input_label != candidate_label:
                        # GED metric (or other distance metric)
                        metric = metric_coefficient * self.distance_metric.evaluate(instance, ctf_candidate, self.oracle).cpu().numpy()

                        # Time penalty
                        time_penalty = self.max_time_penalty * ((instance.time - ctf_candidate.time) / instance.time) ** 2

                        # Stability penalty
                        stability_penalty = self.stability_penalty * values_stability[np.where(candidates_idx == ctf_candidate.time)[0][0]]
                        
                        # Add all penalties to the metric
                        ctf_distance = metric + time_penalty + stability_penalty

                        # print(time_penalty)
                        # print(stability_penalty)

                        candidate_copy = copy.deepcopy(ctf_candidate)
                        
                        if len(top_k_heap) < self.top_k:
                            heapq.heappush(top_k_heap, (-ctf_distance, candidate_copy))
                        else:
                            if ctf_distance < -top_k_heap[0][0]:
                                heapq.heappushpop(top_k_heap, (-ctf_distance, candidate_copy))
                                
                                values = {'metric': metric,
                                          'time_penalty': time_penalty,
                                          'stability_penalty': stability_penalty,
                                          'ctf_distance': ctf_distance,
                                          'candidate_time': ctf_candidate.time,
                                          'candidate_id': ctf_candidate.patient_id,
                                          'candidate_record': ctf_candidate.record_id
                                          }
            
            if not top_k_heap:
                # No counterfactual found: return original instance
                return copy.deepcopy(instance)

            # print(values)
            # print('---')

            # Sort from best to worst
            top_k_sorted = sorted([(-dist, ctf) for dist, ctf in top_k_heap], key=lambda x: x[0])

            # If return_list is True, return the top_k sorted list
            # Otherwise, return only the best counterfactual
            if return_list:
                return top_k_sorted
            else:
                return top_k_sorted[0][1]
        else:
            return copy.deepcopy(instance)
    

    def get_candidates_and_stability(self, patient_id, record_id):

        time, series = self.get_time_and_softmax(patient_id, record_id)

        analyzer = PhaseSpaceAnalyzer(time, series, L=10, percentile=0.25)
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