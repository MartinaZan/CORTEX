import copy
import pickle
import numpy as np
import torch

from src.core.explainer_base import Explainer
from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import init_dflts_to_of
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
        dst_metric = 'src.evaluation.evaluation_metric_dissimilarity.M_dissim_metric'

        # Check if the dissimilarity metric exists or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)


    def init(self):
        super().init()

        # Set distance metric
        self.distance_metric = get_instance_kvargs(
            self.local_config['parameters']['distance_metric']['class'],
            self.local_config['parameters']['distance_metric']['parameters']
        )

        # Optional parameter for number the number of counterfactuals to select
        self.top_k = self.local_config['parameters'].get('top_k', 10)

        # Optional parameters for \alpha, \beta and \gamma to weight the terms
        self.alpha = self.local_config['parameters'].get('dissimilarity_weight', 8)
        self.beta = self.local_config['parameters'].get('time_weight', 1.5)
        self.gamma = self.local_config['parameters'].get('stability_weight', 0.5)

        # Cache to store candidates and permanence keyed by (patient_id, record_id)
        self.candidates_cache = {}

        # Cache to avoid repeated file access and recomputation
        self.softmax_cache = {}

        # Preload candidates and permanence for all patient_id, record_id pairs
        self.preload_candidates()


    def preload_candidates(self):
        for inst in self.dataset.instances:
            key = (inst.patient_id, inst.record_id)
            if key not in self.candidates_cache:
                time, softmax = self.get_time_and_softmax(*key)
                analyzer = PhaseTrajectoryAnalyzer(time, softmax, L=10, percentile=0.25)
                idx, permanence = analyzer.get_idx_candidates_and_permanence()
                self.candidates_cache[key] = (idx, permanence)


    def explain(self, instance, return_list=False):
        # Get label of the current instance
        input_label = self.oracle.predict(instance)

        # Added to get explanation only for class 1 (remove it if not needed) --> speed
        if instance.label != 1:
            return copy.deepcopy(instance)

        # Get ids of candidates in set C and their permanence time
        candidates_idx, permanence = self.get_candidates_and_permanence(instance.patient_id, instance.record_id)

        if candidates_idx.size == 0:
            return copy.deepcopy(instance)

        # Normalized stability score for each candidate
        stability_scores = self.compute_stability_scores(candidates_idx, permanence)

        # Consider only same patient, same record, previous time, and stable candidates
        relevant = self.get_relevant_candidates(instance, candidates_idx)

        ids, M_dissim, M_time, M_instab = [], [], [], []
        candidates_map = {}

        # Iterating over all relevant instances
        for candidate in relevant:
            # Get label of the current candidate
            candidate_label = self.oracle.predict(candidate)

            # If the candidate is valid, compute the terms in the metric
            if input_label != candidate_label:
                ids.append(candidate.time_id)
                dissim, time_dist, instab = self.compute_metric_components(instance, candidate, stability_scores)
                M_dissim.append(dissim)
                M_time.append(time_dist)
                M_instab.append(instab)
                candidates_map[candidate.time_id] = candidate

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
                top_k_candidates.append((M_values[i], candidate_instance))

        if not top_k_candidates:
            return copy.deepcopy(instance)

        if return_list:  # Return all top k counterfactuals
            return top_k_candidates
        else:  # Return only top counterefactual
            return top_k_candidates[0][1]


    # Get M_dissim, M_time, M_instab for the pair (instance, candidate)
    def compute_metric_components(self, instance, candidate, stability_scores=None):
        if stability_scores is None:
            candidates_idx, permanence = self.get_candidates_and_permanence(instance.patient_id, instance.record_id)
            if permanence.max() - permanence.min() == 0:
                stability_scores = {t: np.nan for t in candidates_idx}
            else:
                normalized = 1 - ((permanence - permanence.min()) / (permanence.max() - permanence.min()))
                stability_scores = dict(zip(candidates_idx, normalized))

        M_dissim = self.distance_metric.evaluate(instance, candidate, self.oracle)
        M_time = (instance.time_stamp - candidate.time_stamp) ** 2
        M_instab = stability_scores.get(candidate.time_id, np.nan)

        return M_dissim, M_time, M_instab


    # Get stable candidates and permanence time from phase trajectory analyzer
    def get_candidates_and_permanence(self, patient_id, record_id):
        idx, perm = self.candidates_cache.get((patient_id, record_id), ([], []))
        return np.array(idx), np.array(perm)


    # Get time and softmax associates to current record
    def get_time_and_softmax(self, patient_id, record_id):
        key = (patient_id, record_id)
        if key in self.softmax_cache:
            return self.softmax_cache[key]

        softmax = []
        time = []

        for g1 in self.dataset.instances:
            if g1.patient_id == patient_id and g1.record_id == record_id:
                logits = self.oracle.predict_proba(g1)
                softmax_val = torch.softmax(logits, dim=0)[1].item()
                softmax.append(softmax_val)
                time.append(g1.time_stamp)

        time = np.array(time)
        softmax = np.array(softmax)

        self.softmax_cache[key] = (time, softmax)
        return time, softmax

    # Return only valid candidate instances
    def get_relevant_candidates(self, instance, candidates_idx):
        return [
            inst for inst in self.dataset.instances
            if inst.patient_id == instance.patient_id
            and inst.record_id == instance.record_id
            and inst.time_id <= instance.time_id
            and inst.time_id in candidates_idx
        ]

    # Compute normalized stability score
    def compute_stability_scores(self, candidates_idx, permanence):
        if permanence.max() - permanence.min() == 0:
            return {t: 0 for t in candidates_idx}
        normalized = 1 - ((permanence - permanence.min()) / (permanence.max() - permanence.min()))
        return dict(zip(candidates_idx, normalized))


    # Normalize values
    @staticmethod
    def normalize(arr):
        arr = np.array(arr)
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val == 0:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)