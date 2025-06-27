from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
import numpy as np
import networkx as nx

# TO DO: Da rinominare !!!
class LaplacianMetric(EvaluationMetric):
    """Classe di prova per metrica dissimilarity"""

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'laplacian_metric'

    def evaluate(self, instance_1 , instance_2 , oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        # instance_1 è il grafo originale
        # instance_2 è la spiegazione

        X1 = instance_1.node_features
        X2 = instance_2.node_features

        L1 = nx.normalized_laplacian_matrix(instance_1.get_nx()).toarray()
        L2 = nx.normalized_laplacian_matrix(instance_2.get_nx()).toarray()

        G1 = L1 @ X1
        G2 = L2 @ X2
        
        diff = G1 - G2
        
        result = np.linalg.norm(diff, ord='fro')

        return result