import copy
import sys

from src.core.explainer_base import Explainer
import numpy as np
import torch

from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import  init_dflts_to_of 

class DCESExplainer(Explainer):
    """The Distribution Compliant Explanation Search Explainer performs a search of 
    the minimum counterfactual instance in the original dataset instead of generating
    a new instance"""

    def check_configuration(self):
        super().check_configuration()

        dst_metric='src.evaluation.evaluation_metric_ged.GraphEditDistanceMetric'  ### ATTUALE (metrica GED)
        # dst_metric='src.evaluation.embedding_metrics.EmbeddingMetric'            ### PROVA (metrica embeddings)

        # Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)


    def init(self):
        super().init()

        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])
        
        # Parametri opzionali lookback explainer
        self.lookback = self.local_config['parameters'].get('lookback', 75)
        self.threshold = self.local_config['parameters'].get('threshold', 0.9)

        # Parametro opzionale penalty explainer (0 per non considerare penalità)
        self.max_time_penalty = self.local_config['parameters'].get('max_time_penalty', 10)
        self.max_softmax_penalty = self.local_config['parameters'].get('max_softmax_penalty', 20)

        # Get intervals for lookback oracle
        self.intervals = self.lookback_oracle(lookback=self.lookback, threshold=self.threshold)

    def explain(self, instance):
        # Get label of the current instance
        input_label = self.oracle.predict(instance)

        # Bounds of the interval within which the counterfactual search can take place
        a, b = self.intervals[instance.id]

        # if the method does not find a counterfactual example returns the original graph
        min_ctf = instance

        # Iterating over all the instances of the dataset
        min_ctf_dist = sys.float_info.max
        for ctf_candidate in self.dataset.instances:
            # Considera solo stesso paziente, stesso record e tempo precedente (in caso cambia <= con <)
            # if ctf_candidate.patient_id == instance.patient_id and ctf_candidate.record_id == instance.record_id and ctf_candidate.time <= instance.time:
            if ctf_candidate.time <= instance.time and (a <= ctf_candidate.id <= b) and \
                ctf_candidate.patient_id == instance.patient_id and ctf_candidate.record_id == instance.record_id:

                # OLD
                # candidate_label = self.oracle.predict(ctf_candidate) ## Chiama oracle_base.predict

                # OPPURE
                logits = self.oracle.predict_proba(ctf_candidate)
                softmax = torch.softmax(logits, dim=0)[1].item()
                # candidate_label = 1 if softmax > 0.5 else 0
                candidate_label = self.oracle.predict(ctf_candidate)

                if input_label != candidate_label:
                    metric = self.distance_metric.evaluate(instance, ctf_candidate, self.oracle)

                    time_penalty = self.max_time_penalty * ((ctf_candidate.id - b) / (b - a))**2
                    softmax_penalty = self.max_softmax_penalty * (softmax) # ** 0.5)

                    ctf_distance = metric + time_penalty + softmax_penalty
                    
                    if ctf_distance < min_ctf_dist:
                        min_ctf_dist = ctf_distance
                        min_ctf = ctf_candidate

                        # print(metric)
                        # print(time_penalty)
                        # print(softmax_penalty)
                        # print('---')

        result = copy.deepcopy(min_ctf)
        result.id = instance.id

        return result
    
    def get_ids_and_oracle_outputs(self):
        ids = []
        outputs = []

        for instance in self.dataset.instances:
            ids.append(instance.id)
            outputs.append(self.oracle.predict(instance))
        
        return ids, outputs
    
    # def get_ids_and_softmax(self):
    #     ids = []
    #     softmax = []

    #     for instance in self.dataset.instances:
    #         ids.append(instance.id)

    #         logits = self.oracle.predict_proba(instance)
    #         prob = torch.softmax(logits, dim=0)[1].item()
    #         softmax.append(prob)
        
    #     return ids, softmax


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
