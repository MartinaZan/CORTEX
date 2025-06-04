import copy
import sys

from src.core.explainer_base import Explainer
import numpy as np

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
        self.lookback = self.local_config['parameters'].get('lookback', 50)
        self.threshold = self.local_config['parameters'].get('threshold', 0.9)

        # Parametro opzionale penalty explainer (0 per non considerare penalità)
        self.max_penalty = self.local_config['parameters'].get('max_penalty', 10)

    def explain(self, instance):
        intervals = self.lookback_oracolo(lookback=self.lookback, threshold=self.threshold)

        input_label = self.oracle.predict(instance)

        # Bounds of the interval within which the counterfactual search can take place
        a, b = intervals[instance.id]

        # if the method does not find a counterfactual example returns the original graph
        min_ctf = instance

        # Iterating over all the instances of the dataset
        min_ctf_dist = sys.float_info.max
        for ctf_candidate in self.dataset.instances:
            # Considera solo stesso paziente, stesso record e tempo precedente (in caso cambia <= con <)
            # if ctf_candidate.patient_id == instance.patient_id and ctf_candidate.record_id == instance.record_id and ctf_candidate.time <= instance.time:
            if  ctf_candidate.time <= instance.time and (a <= ctf_candidate.id <= b) and \
                ctf_candidate.patient_id == instance.patient_id and ctf_candidate.record_id == instance.record_id:

                candidate_label = self.oracle.predict(ctf_candidate) ## Chiama oracle_base.predict

                if input_label != candidate_label:
                    ctf_distance = self.distance_metric.evaluate(instance, ctf_candidate, self.oracle)

                    penalization = self.max_penalty * ((ctf_candidate.id - b) / (b - a))**2
                    ctf_distance = ctf_distance + penalization
                    
                    if ctf_distance < min_ctf_dist:
                        min_ctf_dist = ctf_distance
                        min_ctf = ctf_candidate
        
        # if b != a:
        #     print(f"{min_ctf.id} in ({a},{b})")
        #     print(self.max_penalty * ((min_ctf.id - b) / (b - a))**2)

        result = copy.deepcopy(min_ctf)
        result.id = instance.id

        return result
    
    def get_eligible_times_and_outputs(self):
        times = []
        outputs = []

        for instance in self.dataset.instances:
            times.append(instance.id)
            outputs.append(self.oracle.predict(instance))
        
        return times, outputs


    def lookback_oracolo(self, lookback, threshold):
        times, outputs = self.get_eligible_times_and_outputs()

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
