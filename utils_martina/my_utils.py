import os
import re
import ast
import json
import shutil

################################################################################################################

def get_most_recent_file(folder_path):
    # Get the name of the most recent file in folder_path

    return max(os.listdir(folder_path), key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))

################################################################################################################

def create_dataset_json(observations):
    # Set dataset json files based on the name of the record

    data_file_names = [f"grafo_corr_sliding_{i}.json" for i in range(len(observations))]

    for i in range(len(observations)):
        shutil.copyfile(f"EEG_data/dataset_{observations[i]}.json", f"data/datasets/gcs/{data_file_names[i]}")

    file_json = {
        "class": "src.dataset.dataset_base.Dataset",
        "parameters": {
            "generator": {
                "class": "src.dataset.generators.gcs.GCS",
                "parameters": {
                    "data_dir": "data/datasets/gcs/",
                    "data_file_name": data_file_names,
                    "data_label_name": "y"
                }
            }
        }
    }

    with open("config\snippets\datasets\GCS.json", "w") as f:
        json.dump(file_json, f, indent=4)

################################################################################################################

def get_embeddings_and_outputs_all(eval_manager,index_evaluator=0):
    # Get embeddings of all elements

    result = []
    expl = eval_manager._evaluators[index_evaluator]._explainer
    
    for element in expl.dataset.instances:
        result.append(expl.oracle.get_oracle_info(element))
    
    return result

def get_embeddings_and_outputs_test(eval_manager,index_evaluator=0):
    # Get embeddings for the test set

    result = []
    expl = eval_manager._evaluators[index_evaluator]
    ids_list = [explanation.id for explanation in eval_manager._evaluators[index_evaluator].explanations]

    for element in expl.dataset.instances:
        if element.id in ids_list:
            result.append(expl._explainer.oracle.get_oracle_info(element))

    return result

################################################################################################################

# Funzione per estrarre l'elemento desiderato in base all'indice
def extract_dictionary_from_runtime_metric(content,metric,i=0):

    # Trova tutte le occorrenze del dizionario
    occurrences = content.split(metric)[1:]

    # Converti ogni occorrenza in una lista di dizionari
    records = []
    for occ in occurrences:
        # Cerca la parte che inizia con '[{' (lista di dizionari) e termina con '}]'
        start_idx = occ.find("[{")
        end_idx = occ.find("}]") + 2  # Include la parentesi di chiusura ']}'
        
        if start_idx != -1 and end_idx != -1:
            dict_part = occ[start_idx:end_idx]
            # Usa ast.literal_eval per trasformare il testo in un oggetto Python (lista di dizionari)
            records.append(ast.literal_eval(dict_part))
    
    # Estrai l'elemento corrispondente all'indice i
    if i < len(records) and i >= 0:
        list = records[i]
        dict = {item['id']: item['value'] for item in list}
        return dict
    else:
        return None  # Se l'indice è fuori dai limiti

# Funzione per estrarre l'elemento desiderato in base all'indice
def extract_time_and_record(content,i=0):

    # Trova tutte le occorrenze del dizionario
    occurrences = content.split("RuntimeMetric")[1:]

    # Converti ogni occorrenza in una lista di dizionari
    records = []
    for occ in occurrences:
        # Cerca la parte che inizia con '[{' (lista di dizionari) e termina con '}]'
        start_idx = occ.find("[{")
        end_idx = occ.find("}]") + 2  # Include la parentesi di chiusura ']}'
        
        if start_idx != -1 and end_idx != -1:
            dict_part = occ[start_idx:end_idx]
            # Usa ast.literal_eval per trasformare il testo in un oggetto Python (lista di dizionari)
            records.append(ast.literal_eval(dict_part))
    
    # Estrai l'elemento corrispondente all'indice i
    if i < len(records) and i >= 0:
        list = records[i]
        
        # Create a dictionary
        dict_time = {item['id']: item['time'] for item in list}
        dict_record = {item['id']: item['record'] for item in list}

        return dict_time, dict_record
    else:
        return None  # Se l'indice è fuori dai limiti
    
# Funzione per estrarre l'elemento desiderato in base all'indice
def get_other_info(content,metric,i=0):

    # Trova tutte le occorrenze del dizionario
    occurrences = content.split(metric)[1:]

    # Converti ogni occorrenza in una lista di dizionari
    records = []
    for occ in occurrences:
        # Cerca la parte che inizia con '[{' (lista di dizionari) e termina con '}]'
        start_idx = occ.find("[{")
        end_idx = occ.find("}]") + 2  # Include la parentesi di chiusura ']}'
        
        if start_idx != -1 and end_idx != -1:
            dict_part = occ[start_idx:end_idx]
            # Usa ast.literal_eval per trasformare il testo in un oggetto Python (lista di dizionari)
            records.append(ast.literal_eval(dict_part))
    
    # Estrai l'elemento corrispondente all'indice i
    if i < len(records) and i >= 0:
        list = records[i]
        return list
    else:
        return None  # Se l'indice è fuori dai limiti
    
def get_oracle_and_explainer_names(eval_manager,index_evaluator):
    string = eval_manager.evaluators[index_evaluator].name

    oracle_name = re.search(r'using_(.*?)Oracle', string).group(1)
    if oracle_name == '': # It should be ok...
        oracle_name = 'GCN'
    explainer_name = re.search(r'for_(.*?)Explainer', string)
    if explainer_name == None:
        explainer_name = 'RSGG'
    else:
        explainer_name = explainer_name.group(1)

    return f"{oracle_name} oracle - {explainer_name} explainer"