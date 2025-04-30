import pickle

from src.evaluation.evaluator_manager_do_CICLO_EXPLAINER import EvaluatorManager as PairedEvaluatorManager

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

def remove_cache_explainer():
    folder = ".\data\cache\\explainers"

    for file in os.listdir(folder):
        if file.startswith("GCS-"):  # Checks if the file begins with "GCS-"
            cache_path = os.path.join(folder, file)
            if os.path.isdir(cache_path):
                shutil.rmtree(cache_path)  # Rimuove la cartella e il suo contenuto
                print(f"Removed folder and its contents: {cache_path}")

def routine():

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    remove_cache_dataset()
    remove_cache_oracle()
    remove_cache_explainer()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    context = Context.get_context('config/GCS-GCN.jsonc')
    context.run_number = -1

    eval_manager = PairedEvaluatorManager(context)

    eval_manager.evaluate()