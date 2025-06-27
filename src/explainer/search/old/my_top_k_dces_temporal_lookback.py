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

class DCESExplainer(Explainer):
    """The Distribution Compliant Explanation Search Explainer performs a search of 
    the minimum counterfactual instance in the original dataset instead of generating
    a new instance. Here, we look for the top-k counterfactuals."""

    # The explainer finds a counterfactual by minimizing a quantity which includes:
    #  - Dissimilarity metric (e.g., Graph Edit Distance)
    #  - Time penalty (how far the counterfactual is from the original instance in time)
    #  - Softmax penalty (how far the counterfactual is from the original class)
    #  - Softmax variation penalty (how stable is the counterfactual in the next steps)

    def check_configuration(self):
        super().check_configuration()

        # Dissimilarity metric to use for the distance between instances
        dst_metric='src.evaluation.evaluation_metric_ged.GraphEditDistanceMetric'

        # Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)


    def init(self):
        super().init()

        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])
        
        # Optional parameter for number of top_k counterfactuals to be selected
        self.top_k = self.local_config['parameters'].get('top_k', 5)

        # Optional parameters for lookback explainer (length window and frequency threshold)
        self.lookback = self.local_config['parameters'].get('lookback', 75)
        self.threshold = self.local_config['parameters'].get('threshold', 0.9)

        # Optional parameters for magnitude penalties (0 to ignore penalties)
        self.max_time_penalty = self.local_config['parameters'].get('max_time_penalty', 10)
        self.max_softmax_penalty = self.local_config['parameters'].get('max_softmax_penalty', 20)
        self.max_variation_penalty = self.local_config['parameters'].get('max_variation_penalty', 20)
        self.steps_variation_penalty = self.local_config['parameters'].get('steps_variation_penalty', 5)

        # Store intervals for oracle lookcback
        self.intervals = self.lookback_oracle(lookback=self.lookback, threshold=self.threshold)


    def explain(self, instance, return_list=False):
        # Get label of the current instance
        input_label = self.oracle.predict(instance)

        # Bounds of the interval within which the counterfactual search can take place
        a, b = self.intervals[instance.id]

        print(instance.id)
        print(instance.record_id)
        print(instance.time)
        print(a)
        print(b)
        print('---')

        top_k_heap = []

        # Iterating over all the instances of the dataset
        for ctf_candidate in self.dataset.instances:
            # Considera solo stesso paziente, stesso record e tempo precedente (in caso cambia <= con <)
            # if ctf_candidate.patient_id == instance.patient_id and ctf_candidate.record_id == instance.record_id and ctf_candidate.time <= instance.time:
            if ctf_candidate.time <= instance.time and (a <= ctf_candidate.id <= b) and \
                ctf_candidate.patient_id == instance.patient_id and ctf_candidate.record_id == instance.record_id:

                candidate_label = self.oracle.predict(ctf_candidate)

                if input_label != candidate_label:
                    # GED metric (or other distance metric)
                    metric = self.distance_metric.evaluate(instance, ctf_candidate, self.oracle)

                    # Time penalty
                    time_penalty = self.max_time_penalty * ((ctf_candidate.id - b) / (b - a))**2
                    
                    # Softmax penalty (height)
                    softmax = torch.softmax(self.oracle.predict_proba(ctf_candidate), dim=0)[1].item()
                    softmax_penalty = self.max_softmax_penalty * softmax
                    
                    # Softmax penalty (variation next 5 points)
                    softmax_variation = 0
                    old = softmax
                    for i in range(1, self.steps_variation_penalty):
                        if ctf_candidate.id + i < len(self.dataset.instances) and self.dataset.instances[ctf_candidate.id + i].record_id == instance.record_id:
                            new = torch.softmax(self.oracle.predict_proba(self.dataset.instances[ctf_candidate.id + i]), dim=0)[1].item()
                            softmax_variation += np.abs(new - old)
                            old = new
                    variation_penalty = self.max_variation_penalty * softmax_variation

                    # Add all penalties to the metric
                    ctf_distance = metric + time_penalty + softmax_penalty + variation_penalty

                    candidate_copy = copy.deepcopy(ctf_candidate)
                    
                    if len(top_k_heap) < self.top_k:
                        heapq.heappush(top_k_heap, (-ctf_distance, candidate_copy))
                    else:
                        if ctf_distance < -top_k_heap[0][0]:
                            heapq.heappushpop(top_k_heap, (-ctf_distance, candidate_copy))
                            # print(metric)
                            # print(time_penalty)
                            # print(softmax_penalty)
                            # print(variation_penalty)
                            # print('---')
        
        if not top_k_heap:
            # No counterfactual found: return original instance
            return copy.deepcopy(instance)

        # Sort from best to worst
        top_k_sorted = sorted([(-dist, ctf) for dist, ctf in top_k_heap], key=lambda x: x[0])

        # If return_list is True, return the top_k sorted list
        # Otherwise, return only the best counterfactual
        if return_list:
            return top_k_sorted
        else:
            return top_k_sorted[0][1]


    def get_ids_and_oracle_outputs(self):
        ids = []
        outputs = []

        for instance in self.dataset.instances:
            ids.append(instance.id)
            outputs.append(self.oracle.predict(instance))
        
        return ids, outputs


    def lookback_oracle(self, lookback, threshold):
        times, outputs = self.get_ids_and_oracle_outputs()

        result = {} # Dictionary to store detected intervals
        i = 0       # Index to iterate through the outputs list

        while i < len(outputs):

            # Beginning of a seizure condition
            if outputs[i] == 1:

                # Find the continuous segment of 1s starting at index i
                j = i
                while j < len(outputs) and outputs[j] == 1:
                    j += 1

                # Set the initial 'end' of the lookback window
                end = i - 1

                # Move the lookback window backward until it contains enough zeros
                start = end - lookback + 1
                while start >= 0 and outputs[start:end + 1].count(0) < int(lookback * threshold):
                    end -= 1
                    start = end - lookback + 1
                    if end < 0:
                        break

                # Ensure the start doesn't go below 0
                if start < 0:
                    start = 0

                # Save the interval (start, end) for each index between i and j
                for k in range(i, j):
                    result[times[k]] = (start, end)

                # Move the index to the end of the current seizure (segment of 1s)
                i = j
            else:
                # Continue scanning if current value is not 1
                result[times[i]] = (i, i)
                i += 1

        return result
