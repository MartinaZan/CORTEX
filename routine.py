import pickle

from src.evaluation.evaluator_manager import EvaluatorManager
from src.evaluation.evaluator_manager_do import EvaluatorManager as PairedEvaluatorManager
from src.evaluation.evaluator_manager_triplets import EvaluatorManager as TripletsEvaluatorManager

from src.utils.context import Context

from utils_martina.my_utils import *

#######################################################################################################

import torch
import random
import numpy as np

#######################################################################################################

def remove_cache_dataset():
    folder = ".\data\cache\datasets"

    for file in os.listdir(folder):
        if file.startswith("GCS-"):  # Checks if the file begins with "GCS-"
            cache_path = os.path.join(folder, file)
            if os.path.isfile(cache_path):
                os.remove(cache_path)
                print(f"Removed: {cache_path}")

def remove_cache_oracle():
    folder = ".\data\cache\\oracles"

    for file in os.listdir(folder):
        if file.startswith("GCS-"):  # Checks if the file begins with "GCS-"
            cache_path = os.path.join(folder, file)
            if os.path.isdir(cache_path):
                shutil.rmtree(cache_path)  # Rimuove la cartella e il suo contenuto
                print(f"Removed folder and its contents: {cache_path}")

def copia_file(cartella_origine, cartella_destinazione):
    # Scorre tutti i file nella cartella di origine
    for elemento in os.listdir(cartella_origine):
        percorso_elemento = os.path.join(cartella_origine, elemento)
        
        if os.path.isfile(percorso_elemento):
            # Sposta il file nella cartella di destinazione (sostituisce il file se ce n'è già un altro con lo stesso nome)
            shutil.copy(percorso_elemento, cartella_destinazione)
            print(f"      File {elemento} copiato in {cartella_destinazione}")

def routine(patient,lag_nodes,num_node_features,observations):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    remove_cache_dataset()
    remove_cache_oracle()

    cartella_origine = f"EEG_data\{patient}\Distanza {lag_nodes}\{num_node_features} node feature"
    cartella_destinazione = "EEG_data"
    
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    copia_file(cartella_origine, cartella_destinazione)

    create_dataset_json(observations)

    context = Context.get_context('config/GCS-GCN.jsonc')
    context.run_number = -1

    eval_manager = PairedEvaluatorManager(context)

    eval_manager.evaluate()