from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer


class EmbeddingMetric(EvaluationMetric):
    """Classe di prova per definizione distanza embeddings
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'embedding_metrics'

    def evaluate(self, instance_1 , instance_2 , oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        # instance_1 è il grafo originale
        # instance_2 è la spiegazione

        _, embeddings_1 = oracle.predict(instance_1,return_embeddings=True)
        _, embeddings_2 = oracle.predict(instance_2,return_embeddings=True)

        result = sum(sum((embeddings_1 - embeddings_2) ** 2)) ** 0.5    # Ossia, torch.norm(embeddings_1 - embeddings_2)

        return result