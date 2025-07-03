from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
import numpy as np
from .evaluation_metric_dissimilarity import M_dissim_metric

class ImplausibilityMetric(EvaluationMetric):
    """Metric to compute the implausibility of a counterfactual.
    
    Implausibility is defined as the minimum dissimilarity (M_dissim) between 
    the counterfactual (instance_2) and all prior real instances (inst) in the
    dataset, for the same record, up to and including the current time_id."""

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'implausibility'
        self.M_dissim = M_dissim_metric()

    def evaluate(self, instance_1, instance_2, oracle: Oracle=None, explainer: Explainer=None, dataset=None):
        if dataset is None or len(dataset) == 0:
            raise ValueError("Dataset must be provided for implausibility evaluation.")

        min_dissimilarity = float('inf')

        for inst in dataset:
            if inst.time_id <= instance_1.time_id and inst.patient_id == instance_1.patient_id and inst.record_id == instance_1.record_id:
                if instance_2.num_nodes != inst.num_nodes:
                    return np.nan
                
                if instance_2 == inst:
                    return 0
                
                dissimilarity = self.M_dissim.evaluate(inst, instance_2, dataset=dataset)

                if dissimilarity < min_dissimilarity:
                    min_dissimilarity = dissimilarity

        return min_dissimilarity