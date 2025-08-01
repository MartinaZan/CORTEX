from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
import numpy as np
import networkx as nx

class M_dissim_metric(EvaluationMetric):
    """Metric to compute M_dissim: a dissimilarity measure of attributed graphs based on
    normalized Laplacian and node features."""

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'M_dissim_metric'

    def evaluate(self, instance_1 , instance_2 , oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        X1 = instance_1.node_features
        X2 = instance_2.node_features

        L1 = nx.normalized_laplacian_matrix(instance_1.get_nx()).toarray()
        L2 = nx.normalized_laplacian_matrix(instance_2.get_nx()).toarray()

        G1 = L1 @ X1
        G2 = L2 @ X2

        if G1.shape != G2.shape:
            return np.nan
        
        diff = G1 - G2
        result = np.linalg.norm(diff, ord='fro')

        return result