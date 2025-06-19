from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
import torch

class EmbeddingMetric(EvaluationMetric):
    """
    Graph dissimilarity metric based on the distance of the embeddings produced by the GCN model.
    This metric computes the L2 norm (Euclidean distance) between the embeddings of two instances.
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Embedding_Distance'

    def evaluate(self, instance_1 , instance_2 , oracle : Oracle=None, explainer : Explainer=None, dataset = None):

        _, embeddings_1 = oracle.predict(instance_1,return_embeddings=True)
        _, embeddings_2 = oracle.predict(instance_2,return_embeddings=True)

        result = torch.norm(embeddings_1 - embeddings_2)

        # ADDED:
        result = (result / torch.norm(embeddings_1))

        return result